"""
MIISO2 — Modal Integrator for Isogeometric Stabilization of Outliers (order 2).

Spectral map (z = h λ, ω = |z|):
  ω₀ = r·ω_c,  θ(ω) = 0 / s^p / 1 on [0,ω₀] / (ω₀,ω_c) / [ω_c,∞)
  R(z) = |R_D|^θ |R_C|^(1−θ) · exp(i[(1−θ) arg R_C + θ arg R_D])

R_C = Padé (1,1), R_D = Padé (0,2)
ω_c from Omega_c, else h√λ_{N-k+1} of (K,M) via shift-invert, else
|hλ|_(k) of hJ via LM eigs every step (k from setup).

Update:  y⁺ = y + h φ₁(hJ) f + (h²/2) f_t ,  φ₁ = (R−1)/z
φ₁ via Arnoldi + Ritz map (non-holomorphic). Cost O(N m² + m³).

Hiemstra outlier count: CMAME 387 (2021) 114115.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import scipy.linalg as la
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh, eigs


def _outlier_end_k(p, problem, bc):
    if problem == "bar":
        if bc == "fixed":
            return (p - 1) // 2
        if bc == "free":
            return p // 2
    elif problem == "beam":
        if bc == "fixed":
            return max((p - 2) // 2, 0)
        if bc == "pinned":
            return (p - 1) // 2
        if bc == "free":
            return p // 2
    raise ValueError(f"unknown problem/bc: {problem!r}, {bc!r}")


def _outlier_axis_k(p, problem, bci):
    if isinstance(bci, (tuple, list)):
        left, right = bci
    else:
        left = bci
        right = bci
    return _outlier_end_k(p, problem, left) + _outlier_end_k(p, problem, right)


#Hiemstra outlier count from IGA setup (problem, bc, p[, dim, n])
def outlier_count(setup):
    problem = setup[0]
    bc = setup[1]
    p = int(setup[2])
    if len(setup) == 3:
        return _outlier_axis_k(p, problem, bc)
    dim = int(setup[3])
    n = setup[4]
    if isinstance(bc, list):
        bcs = bc
    else:
        bcs = [bc] * dim
    if isinstance(n, (tuple, list)):
        ns = list(n)
    else:
        ns = [int(n)] * dim
    N = 1
    Nhat = 1
    for ni, bci in zip(ns, bcs):
        ki = _outlier_axis_k(p, problem, bci)
        N *= int(ni)
        Nhat *= int(ni) - ki
    return N - Nhat


#conservative 2nd order 1/1 Pade (A-stable, non-dissipative) for the physical modes
def R_C(z):
    z = complex(z)
    return (1.0 + 0.5 * z) / (1.0 - 0.5 * z)

#dissipative 2nd order 0/2 Pade (L-stable) for the outlier modes
def R_D(z):
    z = complex(z)
    return 1.0 / (1.0 - z + 0.5 * z ** 2)


#apply J to a vector, handling dense/sparse/operator/callable forms
def matvec(J, v):
    v = np.asarray(v, float).reshape(-1)
    if isinstance(J, (LinearOperator, np.ndarray)) or sparse.issparse(J):
        return np.asarray(J @ v, float)
    if callable(J):
        return np.asarray(J(v), float)
    return np.asarray(np.asarray(J, float) @ v, float)


#build an m-dim Krylov basis of mv around v via double Gram-Schmidt
def arnoldi(mv, v, m):
    v = np.asarray(v, float).reshape(-1)
    n = v.size
    m = int(min(max(m, 1), n))
    beta = float(np.linalg.norm(v))
    if beta < 1e-30:
        return np.zeros((n, 1)), np.zeros((1, 1)), 0.0
    V = np.zeros((n, m))
    H = np.zeros((m, m))
    V[:, 0] = v / beta
    for j in range(m):
        w = np.asarray(mv(V[:, j]), float).reshape(-1)
        Vj = V[:, : j + 1]
        for _ in range(2):
            hcol = Vj.T @ w
            H[: j + 1, j] += hcol
            w = w - Vj @ hcol
        hn = float(np.linalg.norm(w))
        if j + 1 < m:
            H[j + 1, j] = hn
            if hn < 1e-14:
                return V[:, : j + 1], H[: j + 1, : j + 1], beta
            V[:, j + 1] = w / hn
    return V, H, beta


#per-mode blend of conservative R_C and dissipative R_D across the taper band [omega0, omega_c]
def R_blend(z_vals, omega0, omega_c, p=4):

    scalar = np.ndim(z_vals) == 0
    z_vals = np.asarray(z_vals, complex).reshape(-1)
    n = z_vals.size
    if n == 0:
        return z_vals.copy()
    omega0 = float(omega0)
    omega_c = float(omega_c)
    th = np.empty(n, float)
    for i, z in enumerate(z_vals):
        w = abs(z)
        if w <= omega0:
            th[i] = 0.0
        elif w >= omega_c:
            th[i] = 1.0
        else:
            th[i] = ((w - omega0) / (omega_c - omega0)) ** p
    out = np.empty(n, complex)
    only_c = th <= 0.0
    only_d = th >= 1.0
    mid = (th > 0.0) & (th < 1.0)
    if np.any(only_c):
        idx = np.where(only_c)[0]
        c_vals = np.empty(idx.size, complex)
        for j in range(idx.size):
            c_vals[j] = R_C(z_vals[idx[j]])
        out[idx] = c_vals
    if np.any(only_d):
        idx = np.where(only_d)[0]
        d_vals = np.empty(idx.size, complex)
        for j in range(idx.size):
            d_vals[j] = R_D(z_vals[idx[j]])
        out[idx] = d_vals
    if np.any(mid):
        idx = np.where(mid)[0]
        zm = z_vals[idx]
        thm = th[idx]
        Rc = np.empty(zm.size, complex)
        Rd = np.empty(zm.size, complex)
        for j in range(zm.size):
            Rc[j] = R_C(zm[j])
            Rd[j] = R_D(zm[j])
        order = np.argsort(np.abs(zm))
        arg_c = np.empty(idx.size, float)
        arg_d = np.empty(idx.size, float)
        arg_c[order] = np.unwrap(np.angle(Rc[order]))
        arg_d[order] = np.unwrap(np.angle(Rd[order]))
        mag = (np.abs(Rd) ** thm) * (np.abs(Rc) ** (1.0 - thm))
        out[idx] = mag * np.exp(1j * ((1.0 - thm) * arg_c + thm * arg_d))
    if scalar:
        return complex(out[0])
    return out


#phi_1(hJ) v via one Arnoldi and a tapered Ritz-value map
def apply_phi(J, v, h, omega0, omega_c, krylov_m, p=4):
    """φ₁(hJ) v ≈ β V φ₁(H) e₁ via Arnoldi + Ritz map."""
    v = np.asarray(v, float).reshape(-1)
    n = v.size
    h = float(h)
    def _hJ(x):
        return h * matvec(J, x)
    V, H, beta = arnoldi(_hJ, v, int(min(max(int(krylov_m), 1), n)))
    if beta < 1e-30:
        return np.zeros(n)
    vals, X = la.eig(H)
    R = R_blend(vals, omega0, omega_c, p=p)
    phi = np.empty(vals.size, complex)
    for i in range(vals.size):
        z = vals[i]
        if abs(z) < 1e-14:
            phi[i] = 1.0 + 0.5 * z
        else:
            phi[i] = (R[i] - 1.0) / z
    e1 = np.zeros(H.shape[0], complex)
    e1[0] = 1.0
    Xinv_e1 = la.solve(X, e1)
    coeff = phi * Xinv_e1
    vec = X @ coeff
    return np.real(beta * (V @ vec))


def miiso2_step(f, t, y, h, setup=None, k=None, Omega_c=None, J=None, K=None, M=None, ft=None, krylov_m="auto", omega_frac=0.99, taper_order=4):
    """One step: y ← y + h φ₁(hJ) f + (h²/2) f_t.

    Cutoff: ``Omega_c``; else ``setup``/``k`` with ``K``, ``M`` (preferred for
    IGA); else LM eigs of ``hJ`` every step. Operators: ``J``, ``K``, ``M``,
    ``ft``. Then ``krylov_m``. Leave ``omega_frac`` / ``taper_order`` at
    defaults unless you know you need otherwise.
    """

    #basic setup 
    y = np.asarray(y, float)
    fn = np.asarray(f(t, y), float)
    n = y.size

    #get J and ft
    if J is None:
        y0 = jnp.asarray(y, dtype=jnp.float64)
        def _f_y(yy):
            return jnp.asarray(f(t, yy), dtype=jnp.float64)
        J = np.asarray(jax.jacobian(_f_y)(y0), dtype=float)
    if ft is None:
        y0 = jnp.asarray(y, dtype=jnp.float64)
        t0 = jnp.asarray(t, dtype=jnp.float64)
        def _f_t(tt):
            return jnp.asarray(f(tt, y0), dtype=jnp.float64)
        ft = np.asarray(jax.jacobian(_f_t)(t0), dtype=float)
    else:
        ft = np.asarray(ft, float)

    # cutoff: Omega_c, else (K,M) pencil, else LM eigs of hJ
    if setup is not None:
        k = outlier_count(setup)
    elif k is None:
        k = 0
    else:
        k = int(k)
    if krylov_m is None:
        km = n
    elif krylov_m == "auto":
        km = min(40, n)
    else:
        km = int(krylov_m)
    if Omega_c is not None:
        oc = float(Omega_c)
    elif K is not None and M is not None:
        if callable(K):
            Kraw = K(t, y)
        else:
            Kraw = K
        Kk = Kraw.tocsc()
        if sparse.issparse(M):
            Mm = M.tocsc()
        else:
            Mm = M
        nK = Kk.shape[0]
        kk = int(min(max(k, 0), nK))
        if kk == 0:
            raise ValueError("need Omega_c, or setup/k with (K, M) or J")
        evals = np.sort(np.real(eigsh(Kk, k=min(kk + 1, nK - 2), M=Mm, which="LA", return_eigenvectors=False)))
        oc = float(h) * np.sqrt(max(float(evals[1]), 0.0))
    elif k > 0:
        if isinstance(J, LinearOperator):
            op = J
        else:
            def _op_matvec(v):
                return matvec(J, v)
            op = LinearOperator((n, n), matvec=_op_matvec, dtype=float)
        oc = float(np.min(np.abs(h * eigs(op, k=min(k, n - 2), which="LM", return_eigenvectors=False))))
    else:
        raise ValueError("need Omega_c, or setup/k with (K, M) or J")

    #take a step of the MIISO2 integrator
    phi_term = h * apply_phi(J, fn, h, float(omega_frac) * oc, oc, km, p=taper_order)
    return y + phi_term + 0.5 * h ** 2 * ft


def miiso2_solve(f, y0, T, h, t0=0.0, setup=None, k=None, Omega_c=None, J=None, K=None, M=None, ft=None, krylov_m="auto", omega_frac=0.99, taper_order=4):
    """Integrate y'=f from t0 to T with fixed step h.

    Parameters
    ----------
    f : callable
        Right-hand side f(t, y).
    y0 : array_like
        Initial state.
    T, h : float
        Final time and fixed step (adjusted slightly so an integer number of
        steps ends at T).
    t0 : float, default=0
        Initial time.

    Cutoff (for IGA prefer setup + K, M)
    ------------------------------------
    setup : tuple
        IGA setup (problem, bc, p) or (problem, bc, p, dim, n).
        Preferred way to get the Hiemstra outlier count k.
    k : int
        Outlier count if setup is not given.
    Omega_c : float
        Fixed dimensionless cutoff ω_c (no eigensolve). With K and M,
        also seeds the first shift-invert of the live pencil.
        If omitted and K, M are omitted, ω_c is taken from LM eigs of
        hJ every step (requires setup or k).

    Operators
    ---------
    J : array, sparse, LinearOperator, or callable, optional
        Jacobian of f. Callable J(t, y) is refreshed each step; a
        matrix/operator is frozen. If omitted, JAX differentiates f.
        Used for the step and, without K/M, for the hJ cutoff.
    K, M : sparse matrix or callable, optional
        Structural pencil for ω_c = h√λ_{N-k+1}. K may be K(t, y)
        (live tangent); M is the mass. Preferred nonlinear IGA cutoff.
    ft : array_like or callable, optional
        ∂f/∂t. Callable ft(t, y) refreshed each step; array frozen.
        If omitted, JAX differentiates f.

    Krylov
    ------
    krylov_m : {"auto", None} or int, default="auto"
        Arnoldi dimension: "auto" → min(40, n), None → full n.

    Taper (user-facing but leave at defaults)
    --------------------------------------
    omega_frac : float, default=0.99
        Start of the taper as a fraction of ω_c.
    taper_order : int, default=4
        Taper exponent.
    """

    y = np.asarray(y0, float).copy()
    t = float(t0)
    nsteps = max(int(np.round((float(T) - t) / float(h))), 0)
    if nsteps == 0:
        return y
    h = (float(T) - t) / nsteps
    n = y.size
    if krylov_m is None:
        km = n
    elif krylov_m == "auto":
        km = min(40, n)
    else:
        km = int(krylov_m)
    if setup is not None:
        k = outlier_count(setup)
    elif k is None:
        k = 0
    else:
        k = int(k)

    J_call = callable(J) and not isinstance(J, (LinearOperator, np.ndarray)) and not sparse.issparse(J)
    ft_call = callable(ft) and not isinstance(ft, np.ndarray)
    K_call = callable(K) and not sparse.issparse(K)
    use_pencil = K is not None and M is not None
    use_hJ = (not use_pencil) and (Omega_c is None)
    if use_hJ and int(k) <= 0:
        raise ValueError("need Omega_c, or setup/k with (K, M) or J")
    if use_pencil and int(k) <= 0:
        raise ValueError("pencil cutoff needs setup or k>0")

    if use_pencil and Omega_c is not None:
        lam0 = (float(Omega_c) / h) ** 2
    else:
        lam0 = None
    if Omega_c is not None:
        oc = float(Omega_c)
    else:
        oc = None
    if use_pencil:
        Omega_c = None  # live pencil; Omega_c was seed only

    for istep in range(nsteps):
        if J_call:
            Jj = J(t, y)
        elif J is None:
            yj = jnp.asarray(y, dtype=jnp.float64)
            def _f_y(yy):
                return jnp.asarray(f(t, yy), dtype=jnp.float64)
            Jj = np.asarray(jax.jacobian(_f_y)(yj), dtype=float)
        else:
            Jj = J

        if ft_call:
            ftj = ft(t, y)
        elif ft is None:
            yj = jnp.asarray(y, dtype=jnp.float64)
            tt = jnp.asarray(t, dtype=jnp.float64)
            def _f_t(s):
                return jnp.asarray(f(s, yj), dtype=jnp.float64)
            ftj = np.asarray(jax.jacobian(_f_t)(tt), dtype=float)
        else:
            ftj = np.asarray(ft, float)

        if use_pencil:
            if K_call:
                Kraw = K(t, y)
            else:
                Kraw = K
            Kk = Kraw.tocsc()
            if sparse.issparse(M):
                Mm = M.tocsc()
            else:
                Mm = M
            nK = Kk.shape[0]
            if lam0 is None:
                evals = np.sort(np.real(eigsh(Kk, k=min(max(k, 1) + 1, nK - 2), M=Mm, which="LA", return_eigenvectors=False)))
                lam0 = float(evals[1])
            else:
                ev = np.real(eigsh(Kk, k=1, M=Mm, sigma=float(lam0) * (1.0 + 1e-10), which="LM", return_eigenvectors=False)[0])
                lam0 = float(ev)
            oc = float(h) * np.sqrt(max(lam0, 0.0))
            y = miiso2_step(f, t, y, h, Omega_c=oc, J=Jj, ft=ftj, krylov_m=km, omega_frac=omega_frac, taper_order=taper_order)
        elif use_hJ:
            y = miiso2_step(f, t, y, h, k=k, J=Jj, ft=ftj, krylov_m=km, omega_frac=omega_frac, taper_order=taper_order)
        else:
            y = miiso2_step(f, t, y, h, Omega_c=oc, J=Jj, ft=ftj, krylov_m=km, omega_frac=omega_frac, taper_order=taper_order)
        t += h
    return y
