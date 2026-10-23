"""Initial state: Gaussian axial envelope over sites, Boltzmann distribution within a site.

For each atom:
  1. height z ~ N(z_c, sigma_z) truncated to the trap region; site j = nearest site.
  2. (n, x, y) from  P(n, x, y | j) ∝ exp(-V_n(x, y; j) / kT)  on the disk
     rho <= rho_factor * w(z_j) around the lattice centre line.  This includes bands above
     the barrier (initially free atoms) and radially unbound positions.
  3. (vx, vy) Maxwellian at T0.
  4. Bound band -> BOUND; otherwise FREE with  1/2 m vz^2 = E_band + U_lat/2  (random sign).

Step 2 is exact rejection sampling.  Sites are grouped in slabs (default 40 µm ≈ 75 sites);
each slab gets a proposal built on a transverse grid of its central site: for every grid cell
and band, Ṽ = min(V_n at the 4 cell corners) - margin.  A cell/band pair is drawn with
probability ∝ exp(-Ṽ/kT) * area, a point uniformly in the cell, and the exact V_n at the
atom's own site is accepted with probability exp(-(V_n - Ṽ)/kT).  Points with V_n < Ṽ
(margin too small) are counted as ``violations`` and should be ~0.
"""
from __future__ import annotations

import numpy as np

from .physics.potentials import SemiclassicalTrap
from .state import ParticleState, Status


def sample_sites(trap: SemiclassicalTrap, rng, N, z_c, sigma, z_half):
    z = np.empty(0)
    while len(z) < N:
        zz = rng.normal(z_c, sigma, 2 * (N - len(z)) + 10)
        z = np.concatenate([z, zz[np.abs(zz - z_c) <= z_half]])
    return trap.site_index_at(z[:N])


class _SlabProposal:
    def __init__(self, trap: SemiclassicalTrap, j_rep, rho_max, kT, G, margin_kT, cut_kT):
        T = trap
        z_rep = float(T.site_z_axis(np.array([j_rep]))[0])
        self.x0 = float(T.x_center(z_rep))
        g = np.linspace(-rho_max, rho_max, G)
        X, Y = np.meshgrid(self.x0 + g, g, indexing="ij")
        R = z_rep * (1 + (T.lat.up.zR / z_rep) ** 2)
        zg = z_rep - ((X - self.x0) ** 2 + Y ** 2).ravel() / (2 * R)
        z, L = T.lat.site_surface(X.ravel(), Y.ravel(), j_rep, z_guess=zg)
        s = T._s(L.Ulat)
        base = L.Voff + T.mg * (z - z_rep)       # per-site constant m g z_j(0) removed
        V_list = []
        V_min = None
        for n in range(T.tab.n_bands):
            Vn = base + T.ER * T.tab.energy(s, np.full(s.shape, n))
            if V_min is None:
                V_min = Vn.min()
            if (Vn.min() - V_min) / kT > cut_kT:
                break
            V_list.append(Vn.reshape(G, G))
        self.n_cut = len(V_list)
        V = np.stack(V_list)                                   # (n, G, G)
        corners = np.minimum(np.minimum(V[:, :-1, :-1], V[:, 1:, :-1]),
                             np.minimum(V[:, :-1, 1:], V[:, 1:, 1:]))
        self.h = g[1] - g[0]
        self.cell_x = self.x0 + g[:-1]                         # lower-left corners
        self.cell_y = g[:-1]
        cx = (self.cell_x + 0.5 * self.h - self.x0)[:, None]
        cy = (self.cell_y + 0.5 * self.h)[None, :]
        inside = (cx ** 2 + cy ** 2) <= (rho_max + self.h) ** 2
        self.V_tilde = corners - margin_kT * kT                # (n, G-1, G-1)
        logw = -(self.V_tilde - V_min) / kT
        logw = np.where(inside[None], logw, -np.inf)
        w = np.exp(logw - logw.max())
        self.p = (w / w.sum()).ravel()
        self.shape = w.shape
        self.V_min = V_min
        self.kT = kT

    def draw(self, rng, M):
        k = rng.choice(len(self.p), size=M, p=self.p)
        n, ix, iy = np.unravel_index(k, self.shape)
        x = self.cell_x[ix] + self.h * rng.random(M)
        y = self.cell_y[iy] + self.h * rng.random(M)
        return n, x, y, self.V_tilde[n, ix, iy]


def sample_initial(trap: SemiclassicalTrap, cfg, rng, n_test=None, sites=None, T0=None,
                   distribution=None, escape=None):
    """Return (ParticleState, info dict).

    ``sites`` overrides the envelope (tests).  ``distribution`` overrides
    cfg.cloud.distribution ("boltzmann" or "truncated"; the latter needs ``escape``, an
    EscapeTable, and resamples (n, x, y, v) at fixed site until E < E_esc)."""
    N = int(n_test or cfg.numerics.n_test)
    kT = (cfg.cloud.kT0 if T0 is None else T0 * 1.380649e-23)
    distribution = distribution or cfg.cloud.distribution
    if sites is None:
        sites = sample_sites(trap, rng, N, cfg.cloud.z_center, cfg.cloud.sigma_z, cfg.region.z_half)
    sites = np.asarray(sites, np.int64)
    st = ParticleState.empty(N)
    st.j[:] = sites
    cache = {}                     # slab proposals are reused across resampling rounds
    info = _sample_in_sites(trap, cfg, rng, st, np.arange(N), kT, cache)
    if distribution == "truncated":
        if escape is None:
            raise ValueError("truncated distribution needs an EscapeTable")
        info["truncation_rounds"] = 0
        bad = _unbound_mask(trap, st, np.arange(N), escape)
        while bad.any():
            idx = np.nonzero(bad)[0]
            _sample_in_sites(trap, cfg, rng, st, idx, kT, cache)
            bad[idx] = _unbound_mask(trap, st, idx, escape)
            info["truncation_rounds"] += 1
            if info["truncation_rounds"] > 200:
                raise RuntimeError("truncated sampling did not converge")
        info["n_bound"] = int(np.count_nonzero(st.status == Status.BOUND))
        info["n_free"] = int(np.count_nonzero(st.status == Status.FREE))
    elif distribution != "boltzmann":
        raise ValueError(f"unknown initial distribution {distribution!r}")
    info["distribution"] = distribution
    return st, info


def _unbound_mask(trap, st, idx, escape):
    """True where the atom is FREE or its energy is at/above its band's escape energy."""
    E = total_energy(trap, st, idx)
    Eesc = escape(st.j[idx], st.n[idx])
    return (st.status[idx] != Status.BOUND) | ~(E < Eesc)


def _sample_in_sites(trap, cfg, rng, st, which, kT, cache=None):
    """Fill (n, x, y, z, vx, vy, vz, status) of atoms ``which`` given their sites st.j."""
    m = trap.m
    num = cfg.numerics
    sites = st.j[which]

    sps = max(1, int(round(num.init_slab_um * 1e-6 / trap.d)))
    slab = sites // sps                      # absolute slab index: slab b = sites [b*sps, (b+1)*sps)
    which = np.asarray(which)
    cache = {} if cache is None else cache
    info = {"violations": 0, "proposed": 0, "accepted": 0, "n_slabs": 0, "n_cut_max": 0}
    for b in np.unique(slab):
        idx = which[slab == b]
        j_lo, j_hi = b * sps, (b + 1) * sps - 1
        if b not in cache:
            z_slab = trap.site_z_axis(np.array([j_lo, j_hi]))
            rho_max = float(trap.rho_region(z_slab).max())
            prop = _SlabProposal(trap, j_lo + sps // 2, rho_max, kT, num.init_grid,
                                 num.init_margin_kT, num.init_band_cut_kT)
            j_axis = np.arange(j_lo, j_hi + 1)
            cache[b] = (prop, j_axis, trap.site_z_axis(j_axis))
        prop, j_axis, z_axis = cache[b]
        info["n_slabs"] += 1
        info["n_cut_max"] = max(info["n_cut_max"], prop.n_cut)
        todo = idx
        while len(todo):
            M = len(todo)
            n, x, y, Vt = prop.draw(rng, M)
            j = st.j[todo]
            e = trap.bound_eval(x, y, j, n, force=False)
            rho = trap.rho_from_center(x, y, e.z)
            ok_disk = rho <= trap.rho_region(e.z)
            V_rel = e.V - trap.mg * z_axis[np.searchsorted(j_axis, j)]
            dV = (V_rel - Vt) / kT
            info["violations"] += int(np.count_nonzero(ok_disk & (dV < 0)))
            acc = ok_disk & (rng.random(M) < np.exp(-np.maximum(dV, 0.0)))
            info["proposed"] += M
            a = todo[acc]
            info["accepted"] += len(a)
            st.n[a], st.x[a], st.y[a], st.z[a] = n[acc], x[acc], y[acc], e.z[acc]
            free = ~e.bound[acc]
            st.status[a] = np.where(free, Status.FREE, Status.BOUND)
            vz = np.sqrt(trap.release_vz2(e.E_band[acc], e.Ulat[acc]))
            sign = np.where(rng.random(len(a)) < 0.5, -1.0, 1.0)
            st.vz[a] = np.where(free, sign * vz, 0.0)
            st.released_band[a] = np.where(free, n[acc], -1)
            todo = todo[~acc]

    sig_v = np.sqrt(kT / m)
    st.vx[which] = rng.normal(0.0, sig_v, len(which))
    st.vy[which] = rng.normal(0.0, sig_v, len(which))
    info["acceptance"] = info["accepted"] / max(info["proposed"], 1)
    info["n_bound"] = int(np.count_nonzero(st.status == Status.BOUND))
    info["n_free"] = int(np.count_nonzero(st.status == Status.FREE))
    return info


def total_energy(trap: SemiclassicalTrap, st: ParticleState, idx=None):
    """Mechanical energy (J) of BOUND / FREE / IN_TRANSIT atoms (nan for others)."""
    idx = np.arange(st.N) if idx is None else np.asarray(idx)
    E = np.full(len(idx), np.nan)
    b = st.status[idx] == Status.BOUND
    if b.any():
        i = idx[b]
        E[b] = trap.bound_energy(st.x[i], st.y[i], st.vx[i], st.vy[i], st.j[i], st.n[i],
                                 z_guess=st.z[i])
    f = np.isin(st.status[idx], [Status.FREE, Status.IN_TRANSIT])
    if f.any():
        i = idx[f]
        E[f] = trap.free_energy(st.r(i), st.v(i))
    return E
