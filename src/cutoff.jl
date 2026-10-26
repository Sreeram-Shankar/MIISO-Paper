#this caches stores the information needed
#for the cutoff between the resolved and outliers.
#phi and psi are the normalized right and left eigenvectors
#of the jacobian, lam_ref is the original cutoff eigenvalue,
#corresponding to J0. J0phi_term is the term that is used to
#update the cutoff eigenvalue. w_re_buf and w_im_buf are the
#real and imaginary parts of the eigenvector, respectively.
#h is the time step
mutable struct CutoffCache
    phi::Vector{ComplexF64}
    psi::Vector{ComplexF64}
    lam_ref::ComplexF64
    J0phi_term::ComplexF64
    w_re_buf::Vector{Float64}
    w_im_buf::Vector{Float64}
    h::Float64
end

#Matrix free placement of the outlier cutoff eigenvalue. For second-order J = [0 I; −A 0]
#Implicitly restarted Arnoldi exploits the block structure to solve for the cutoff easily
#If not, it falls back to the full Jacobian with IRAM
function cutoff_eigenpair(J0, k, n, h; m_setup = nothing, probe = nothing)
    if n >= 4 && iseven(n)
        lam_ref, phi, psi = spatial_cutoff_setup(J0, k, n, h; m_setup, probe)
        lam_ref !== nothing && return lam_ref, phi, psi
    end
    return fullJ_cutoff_setup(J0, k, n, h; m_setup, probe)
end

#if the system does not have the second order structure, we use the full Jacobian
#and run the IRAM method
function fullJ_cutoff_setup(J0, k, n, h; m_setup = nothing, probe = nothing)
    nev = min(max(2k, 1), max(n - 2, 1))
    ncv = m_setup === nothing ? min(n, max(2nev + 1, 20)) : min(n, Int(m_setup))
    ncv = max(ncv, nev + 2)
    kw = probe === nothing ? (;) : (; v0 = Float64.(real.(asvec(probe))))
    λ, X = eigs(
        as_matvec(J0, n);
        nev = nev,
        ncv = ncv,
        which = :LM,
        ritzvec = true,
        maxiter = 5000,
        kw...,
    )
    order = sortperm(abs.(λ); rev = true)
    idx = order[min(2k, length(order))]
    lam_ref = ComplexF64(float(h) * λ[idx])
    phi = ComplexF64.(vec(X[:, idx]))
    phi ./= norm(phi)
    return lam_ref, phi, phi
end

#if it is, we exploit the block structure to solve for the cutoff in a more cheap way
function spatial_cutoff_setup(J0, k, n, h; m_setup = nothing, probe = nothing)
    N = n ÷ 2
    yd = Vector{JVPDual}(undef, n)
    buf = zeros(n)
    z = zeros(n)
    fill!(z, 0)
    z[1:N] .= randn(N)
    apply_J!(buf, J0, z, yd)
    ntop, nbot = norm(@view(buf[1:N])), norm(@view(buf[(N+1):end]))
    # [q;v]: J[φ;0] = [0; −Aφ].  [v;q]: J[0;φ] = [−Aφ; 0]
    q_first = nbot > 1e-8 * (ntop + nbot) && ntop <= 1e-6 * nbot
    if !q_first
        fill!(z, 0)
        z[(N+1):end] .= randn(N)
        apply_J!(buf, J0, z, yd)
        ntop, nbot = norm(@view(buf[1:N])), norm(@view(buf[(N+1):end]))
        ntop > 1e-8 * (ntop + nbot) && nbot <= 1e-6 * ntop ||
            return nothing, nothing, nothing
    end
    function apply_A(x)
        fill!(z, 0)
        if q_first
            copyto!(@view(z[1:N]), x)
            apply_J!(buf, J0, z, yd)
            return -buf[(N+1):end]
        else
            copyto!(@view(z[(N+1):end]), x)
            apply_J!(buf, J0, z, yd)
            return -buf[1:N]
        end
    end
    nev = min(max(k, 1) + 1, max(N - 2, 1))
    ncv = m_setup === nothing ? min(N, max(2nev + 1, 20)) : min(N, Int(m_setup))
    ncv = max(ncv, nev + 2)
    v0 = if probe === nothing
        nothing
    else
        pr = Float64.(real.(asvec(probe)))
        q_first ? pr[1:N] : pr[(N+1):end]
    end
    kw = v0 === nothing ? (;) : (; v0 = v0)
    λ, X = eigs(
        MatvecOp(apply_A, N);
        nev = nev,
        ncv = ncv,
        which = :LM,
        ritzvec = true,
        maxiter = 5000,
        kw...,
    )
    i = sortperm(real.(λ))[min(2, length(λ))]
    ω = sqrt(max(real(λ[i]), 0.0))
    φA = collect(Float64, real.(@view X[:, i]))
    nrm = norm(φA)
    nrm < 1e-30 && return nothing, nothing, nothing
    φA ./= nrm
    phi =
        q_first ? ComplexF64.(vcat(φA, (im * ω) .* φA)) :
        ComplexF64.(vcat((im * ω) .* φA, φA))
    psi =
        q_first ? ComplexF64.(vcat((-im * ω) .* φA, φA)) :
        ComplexF64.(vcat(φA, (-im * ω) .* φA))
    phi ./= norm(phi)
    return ComplexF64(float(h) * im * ω), phi, psi
end

#complex handling of the Jacobian
function apply_J_complex!(w_re, w_im, J, phi_re, phi_im, yd_buf)
    apply_J!(w_re, J, phi_re, yd_buf)
    apply_J!(w_im, J, phi_im, yd_buf)
    return w_re, w_im
end

#initialization of the cutoff cache
function init_cutoff(J0, k, n, h; m_setup = nothing, probe = nothing)
    lam_ref, phi, psi = cutoff_eigenpair(J0, k, n, h; m_setup, probe)
    w_re, w_im = zeros(n), zeros(n)
    apply_J_complex!(w_re, w_im, J0, real.(phi), imag.(phi), Vector{JVPDual}(undef, n))
    Jphi = complex.(w_re, w_im)
    J0phi_term = h * dot(psi, Jphi) / dot(psi, phi)
    return CutoffCache(phi, psi, lam_ref, J0phi_term, w_re, w_im, float(h))
end

#Rayleigh update of the frozen cutoff mode, omega_c = |h lambda_hat| (left–right quotient update)
function step_cutoff!(cache::CutoffCache, J, yd_buf)
    phi_re, phi_im = real.(cache.phi), imag.(cache.phi)
    w_re, w_im = apply_J_complex!(cache.w_re_buf, cache.w_im_buf, J, phi_re, phi_im, yd_buf)
    Jphi = complex.(w_re, w_im)
    lam_hat = dot(cache.psi, Jphi) / dot(cache.psi, cache.phi)
    return abs(cache.lam_ref + (lam_hat * cache.h - cache.J0phi_term))
end
