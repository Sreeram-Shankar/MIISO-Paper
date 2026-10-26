#The outlier count (we call it k) from an IGA setup tuple
#(problem, bc, p) for 1d, or (problem, bc, p, dim, n) for tensor product.
#returns how many spectrum outliers sit above the physical band
function outlier_count(setup)
    problem, bc, p = setup[1], setup[2], Int(setup[3])

    function end_k(bci)
        if problem == "bar"
            if bci == "fixed"
                return (p - 1) ÷ 2
            end
            if bci == "free"
                return p ÷ 2
            end
        elseif problem == "beam"
            if bci == "fixed"
                return max((p - 2) ÷ 2, 0)
            end
            if bci == "pinned"
                return (p - 1) ÷ 2
            end
            if bci == "free"
                return p ÷ 2
            end
        end
        error("unknown problem/bc: $(repr(problem)), $(repr(bci))")
    end

    function axis_k(bci)
        left, right =
            bci isa Tuple || bci isa AbstractVector ? (bci[1], bci[2]) : (bci, bci)
        return end_k(left) + end_k(right)
    end

    length(setup) == 3 && return axis_k(bc)
    dim, n = Int(setup[4]), setup[5]
    bcs = bc isa AbstractVector ? bc : fill(bc, dim)
    ns = n isa Tuple || n isa AbstractVector ? collect(Int, n) : fill(Int(n), dim)
    N = Nhat = 1
    for (ni, bci) in zip(ns, bcs)
        N *= ni
        Nhat *= ni - axis_k(bci)
    end
    return N - Nhat
end

#thin wrapper so Arpack eigs can take a matrix-free operator.
#f is the matvec, n is the dimension
struct MatvecOp
    f::Any
    n::Int
end
Base.size(A::MatvecOp) = (A.n, A.n)
Base.size(A::MatvecOp, ::Int) = A.n
Base.eltype(::MatvecOp) = Float64
LinearAlgebra.issymmetric(::MatvecOp) = false
Base.:*(A::MatvecOp, x::AbstractVector) = A.f(x)
LinearAlgebra.mul!(y::AbstractVector, A::MatvecOp, x::AbstractVector) =
    (copyto!(y, A.f(x)); y)
asvec(v::Vector{Float64}) = v
asvec(v) = vec(Float64.(v))
matvec(J::AbstractMatrix, v) = J * asvec(v)
matvec(J::MatvecOp, v) = J * asvec(v)
matvec(J, v) = J(asvec(v))

#Jacobian-vector product when the user does not pass J.
#f is the rhs, t and y are the linearization point
struct JacVec
    f::Any
    t::Float64
    y::Vector{Float64}
end

function JacVec(f, t, y)
    y = y isa Vector{Float64} ? y : vec(Float64.(y))
    return JacVec(f, float(t), y)
end

const JVPDual = ForwardDiff.Dual{ForwardDiff.Tag{JacVec,Float64},Float64,1}

#writes J*x into w using one ForwardDiff dual sweep through f
function jvp!(w, J::JacVec, x, yd)
    y = J.y
    @inbounds for i in eachindex(y)
        yd[i] = ForwardDiff.Dual{ForwardDiff.Tag{JacVec,Float64}}(y[i], x[i])
    end
    fd = vec(J.f(J.t, yd))
    @inbounds for i in eachindex(w)
        w[i] = ForwardDiff.partials(fd[i], 1)
    end
    return w
end

#one workspace per parallel branch, allocated in miiso_solve and passed in.
#V and H are the Arnoldi basis and Hessenberg matrix.
#w, hcol, tmp are scratch for matvecs and projections.
#yd is the ForwardDiff dual buffer for JacVec.
#Yp is the column-chunk scratch for threaded sparse mul_hJ!.
#dy, y2, y3, d2, d3, u3, u4, z0 are stage / defect / combo buffers
#reused by miiso3–6 so they do not allocate every step.
#phi_pool is a bump allocator for phi-action outputs in one step
#(reset with reset_phi_pool!)
mutable struct KrylovWS
    V::Matrix{Float64}
    H::Matrix{Float64}
    w::Vector{Float64}
    hcol::Vector{Float64}
    tmp::Vector{Float64}
    yd::Vector{JVPDual}
    Yp::Matrix{Float64}
    dy::Vector{Float64}
    y2::Vector{Float64}
    y3::Vector{Float64}
    d2::Vector{Float64}
    d3::Vector{Float64}
    u3::Vector{Float64}
    u4::Vector{Float64}
    z0::Vector{Float64}
    phi_pool::Vector{Vector{Float64}}
    phi_npool::Int
end

#allocate a fresh Krylov workspace for state size n and Arnoldi cap m
function krylov_ws_new(n::Int, m::Int)
    m = max(Int(m), 1)
    return KrylovWS(
        zeros(n, m),
        zeros(m, m),
        zeros(n),
        zeros(m),
        zeros(n),
        Vector{JVPDual}(undef, n),
        zeros(n, 0),
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Vector{Float64}[],
        0,
    )
end

#resize a Float64 buffer in place if needed
function ensure!(v::Vector{Float64}, n::Int)
    length(v) == n || resize!(v, n)
    return v
end

#start a new step's phi bump allocation from the front of the pool
function reset_phi_pool!(ws::KrylovWS)
    ws.phi_npool = 0
    return ws
end

#hand out the next phi-output vector of length n from the bump pool
function take_phi_out!(ws::KrylovWS, n::Int)
    ws.phi_npool += 1
    if ws.phi_npool > length(ws.phi_pool)
        push!(ws.phi_pool, zeros(n))
    end
    return ensure!(ws.phi_pool[ws.phi_npool], n)
end

#stage defect D = f(t + c h, U) - f - J(U - y) - c h f_t
function defect!(D, wsi, f, t, c, h, U, y, fn, ft, J)
    dU = ensure!(wsi.dy, length(y))
    @. dU = U - y
    apply_J!(wsi.tmp, J, dU, wsi.yd)
    fU = asvec(f(t + c * h, U))
    @inbounds for i in eachindex(D)
        D[i] = fU[i] - fn[i] - wsi.tmp[i] - c * h * ft[i]
    end
    return D
end

#apply J to x into w for various forms and callable thingsies
function apply_J!(w, J::JacVec, x, yd)
    jvp!(w, J, asvec(x), yd)
    return w
end
function apply_J!(w, J::Union{AbstractMatrix,MatvecOp}, x, yd)
    mul!(w, J, asvec(x))
    return w
end
function apply_J!(w, J, x, yd)
    copyto!(w, asvec(J(asvec(x))))
    return w
end

#run two independent pieces at once. split the BLAS threads across them
function parallel_pair(f, g)
    Threads.nthreads() < 2 && return f(), g()
    old = BLAS.get_num_threads()
    BLAS.set_num_threads(max(1, old ÷ 2))
    tf = Threads.@spawn f()
    tg = Threads.@spawn g()
    try
        return fetch(tf), fetch(tg)
    finally
        BLAS.set_num_threads(old)
    end
end

#same idea for any number of independent pieces (MIISO5 stages / phi-actions)
function parallel_n(fs...)
    Threads.nthreads() < 2 && return map(g -> g(), fs)
    old = BLAS.get_num_threads()
    BLAS.set_num_threads(max(1, old ÷ length(fs)))
    tasks = map(g -> Threads.@spawn(g()), fs)
    try
        return map(fetch, tasks)
    finally
        BLAS.set_num_threads(old)
    end
end

#w = h * J * x for dense matrices
function mul_hJ!(w, J::AbstractMatrix, x, h, ws::KrylovWS)
    mul!(w, J, x)
    rmul!(w, h)
    return w
end
#column-wise product into private columns of Yp, then a sum.
#worth threading only when there is a lot of work per row
function mul_hJ!(w, J::SparseMatrixCSC, x, h, ws::KrylovWS)
    nt = Threads.nthreads()
    nnzJ = nnz(J)
    if nt == 1 || nnzJ < 200_000 || nnzJ < 16 * size(J, 2)
        mul!(w, J, x)
    else
        nt = min(nt, size(J, 2))
        if size(ws.Yp, 2) < nt
            ws.Yp = zeros(size(J, 1), nt)
        end
        colptr, rowval, nzval = J.colptr, J.rowval, J.nzval
        ncols = size(J, 2)
        tasks = Vector{Task}(undef, nt)
        for t = 1:nt
            j0 = 1 + (t - 1) * ncols ÷ nt
            j1 = t * ncols ÷ nt
            tasks[t] = Threads.@spawn begin
                yt = @view ws.Yp[:, t]
                fill!(yt, 0)
                @inbounds for j = j0:j1
                    xj = x[j]
                    for p = colptr[j]:(colptr[j+1]-1)
                        yt[rowval[p]] += nzval[p] * xj
                    end
                end
            end
        end
        foreach(wait, tasks)
        copyto!(w, @view ws.Yp[:, 1])
        for t = 2:nt
            @inbounds for i in eachindex(w)
                w[i] += ws.Yp[i, t]
            end
        end
    end
    rmul!(w, h)
    return w
end
function mul_hJ!(w, J::JacVec, x, h, ws::KrylovWS)
    jvp!(w, J, x, ws.yd)
    rmul!(w, h)
    return w
end
function mul_hJ!(w, J::MatvecOp, x, h, ws::KrylovWS)
    copyto!(ws.tmp, x)
    mul!(w, J, ws.tmp)
    rmul!(w, h)
    return w
end
function mul_hJ!(w, J, x, h, ws::KrylovWS)
    copyto!(ws.tmp, x)
    copyto!(w, asvec(J(ws.tmp)))
    rmul!(w, h)
    return w
end

#wrap whatever J the user gave into something Arpack can matvec
function as_matvec(J, n)
    if J isa AbstractMatrix
        return J
    elseif J isa JacVec
        yd, w = Vector{JVPDual}(undef, n), zeros(n)
        return MatvecOp(v -> jvp!(w, J, v, yd), n)
    elseif J isa MatvecOp
        return J
    else
        return MatvecOp(v -> asvec(matvec(J, v)), n)
    end
end
