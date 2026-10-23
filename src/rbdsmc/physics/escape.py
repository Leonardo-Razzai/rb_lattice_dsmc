"""Escape energies of bound atoms: the lowest energy at which an atom in band n of site j
can leave without collisions.

An atom in band n moves radially in V_n(x, y; j).  It is lost when it reaches the contour
where band n stops being bound (it is then released axially and falls), or the edge of the
trap region if that comes first.  Along each ray from the lattice centre the barrier is
the maximum of V_n between the centre and that point; the escape energy is the minimum
over rays (mountain pass).  Without collisions an atom with E < E_esc stays forever, so
E_esc also defines the bound / unbound classification and predicts the collisionless
plateau of N(t).

Escape energies vary slowly with the site, apart from the trivial gravity offset
m g z_j(0).  ``EscapeTable`` therefore stores E_esc - m g z_j(0) for the central site of
each slab and interpolates linearly in z_j(0).
"""
from __future__ import annotations

import numpy as np


def escape_energies(trap, j: int, n_max: int, n_dirs: int = 24, n_pts: int = 300,
                    n_bisect: int = 30):
    """Escape energy (J, absolute) and release radius for bands 0..n_max-1 of site j."""
    T = trap
    z0 = float(T.site_z_axis(np.array([j]))[0])
    x0 = float(T.x_center(z0))
    rho_max = float(T.rho_region(z0))
    phi = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    rho = np.linspace(0, rho_max, n_pts)
    X = x0 + np.outer(np.cos(phi), rho)
    Y = np.outer(np.sin(phi), rho)
    R = z0 * (1 + (T.lat.up.zR / z0) ** 2)
    zg = z0 - (rho ** 2 / (2 * R))[None, :] * np.ones((n_dirs, 1))
    z, L = T.lat.site_surface(X.ravel(), Y.ravel(), j, z_guess=zg.ravel())
    s = T._s(L.Ulat)
    base = (L.Voff + T.mg * z).reshape(n_dirs, n_pts)
    E_esc = np.full(n_max, np.nan)
    rho_rel = np.full(n_max, np.nan)
    for n in range(n_max):
        nn = np.full(s.shape, n)
        V = base + T.ER * T.tab.energy(s, nn).reshape(n_dirs, n_pts)
        bound = T.band_is_bound(s, nn).reshape(n_dirs, n_pts)
        if not bound[:, 0].all():          # band not bound even at the centre
            E_esc[n] = -np.inf
            rho_rel[n] = 0.0
            continue
        # first unbound point along each ray (n_pts if bound up to the region edge)
        has_ub = (~bound).any(axis=1)
        k = np.where(has_ub, np.argmax(~bound, axis=1), n_pts)
        ray_E = np.array([V[d, :k[d]].max() for d in range(n_dirs)])
        ray_r = np.full(n_dirs, rho_max)
        dsel = np.nonzero(has_ub)[0]
        if len(dsel):
            lo, hi = rho[k[dsel] - 1], rho[k[dsel]]
            c, sn = np.cos(phi[dsel]), np.sin(phi[dsel])
            nn_d = np.full(len(dsel), n)
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                e = T.bound_eval(x0 + mid * c, mid * sn, j, nn_d,
                                 z_guess=z0 - mid ** 2 / (2 * R), force=False)
                lo = np.where(e.bound, mid, lo)
                hi = np.where(e.bound, hi, mid)
            e = T.bound_eval(x0 + lo * c, lo * sn, j, nn_d, z_guess=z0 - lo ** 2 / (2 * R),
                             force=False)
            ray_E[dsel] = np.maximum(ray_E[dsel], e.V)
            ray_r[dsel] = lo
        i = np.argmin(ray_E)
        E_esc[n] = ray_E[i]
        rho_rel[n] = ray_r[i]
    return E_esc, rho_rel, z0


class EscapeTable:
    """E_esc(j, n) by linear interpolation between slab-centre sites (gravity offset removed)."""

    def __init__(self, trap, j_min: int, j_max: int, n_max: int, slab_sites: int = 75, **kw):
        self.trap = trap
        js = np.arange(j_min, j_max + slab_sites, slab_sites)
        self.j_nodes = js
        self.n_max = n_max
        rel = np.empty((len(js), n_max))
        zs = np.empty(len(js))
        for i, j in enumerate(js):
            E, _, z0 = escape_energies(trap, int(j), n_max, **kw)
            rel[i] = E - trap.mg * z0
            zs[i] = z0
        self.E_rel = rel
        self.z_nodes = zs

    def __call__(self, j, n):
        """Escape energy (J) for sites j and bands n (arrays).  Bands beyond the table or
        unbound at the centre get -inf (always unbound)."""
        j = np.asarray(j, np.int64)
        n = np.asarray(n, np.int64)
        z0 = self.trap.site_z_axis(np.unique(j))
        zmap = dict(zip(np.unique(j).tolist(), z0.tolist()))
        zj = np.array([zmap[v] for v in j.tolist()]) if len(j) else np.empty(0)
        out = np.full(len(j), -np.inf)
        ok = n < self.n_max
        f = np.clip((j - self.j_nodes[0]) / (self.j_nodes[1] - self.j_nodes[0])
                    if len(self.j_nodes) > 1 else np.zeros(len(j)), 0, len(self.j_nodes) - 1)
        i0 = np.minimum(np.floor(f).astype(int), len(self.j_nodes) - 2) if len(self.j_nodes) > 1 \
            else np.zeros(len(j), int)
        t = f - i0
        if len(self.j_nodes) > 1:
            a = self.E_rel[i0, np.minimum(n, self.n_max - 1)]
            b = self.E_rel[i0 + 1, np.minimum(n, self.n_max - 1)]
            fin = np.isfinite(a) & np.isfinite(b)
            val = np.full(len(j), -np.inf)
            val[fin] = (1 - t[fin]) * a[fin] + t[fin] * b[fin]
        else:
            val = self.E_rel[0, np.minimum(n, self.n_max - 1)]
        out[ok] = val[ok] + self.trap.mg * zj[ok]
        return out


def escape_table_for(trap, cfg, cache_dir="results/cache", n_max=None, slab_sites=75):
    """Build (or load from cache) the escape table covering the trap region.  By default
    n_max covers every band that can be bound anywhere in the region (deepest point + 1)."""
    import hashlib
    import json
    from pathlib import Path
    if n_max is None:
        zz = np.linspace(trap.z_c - trap.z_half, trap.z_c + trap.z_half, 201)
        L = trap.lat.local(np.column_stack([trap.x_center(zz), 0 * zz, zz]))
        s_deep = float(np.max(L.Ulat) / trap.ER) * 1.05
        n_max = int(trap.tab.n_bound(np.array([min(s_deep, trap.s_hi)]), trap.tilt)[0]) + 1
    n_max = min(n_max, trap.tab.n_bands)
    j_min = int(trap.site_index_at(trap.z_c - trap.z_half)[0])
    j_max = int(trap.site_index_at(trap.z_c + trap.z_half)[0])
    d = cfg.to_dict()
    trap_inputs = {"optics": d["optics"], "region": d["region"], "bands": d["bands"],
                   "gravity": d["physics"]["gravity"], "z_center_mm": d["cloud"]["z_center_mm"]}
    key_src = {"trap": trap_inputs, "n_max": n_max, "j": [j_min, j_max], "v": 2,
               "slab": slab_sites}
    key = hashlib.sha256(json.dumps(key_src, sort_keys=True).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"escape_{key}.npz"
    if path.exists():
        zf = np.load(path)
        tab = EscapeTable.__new__(EscapeTable)
        tab.trap, tab.j_nodes, tab.n_max = trap, zf["j_nodes"], int(zf["n_max"])
        tab.E_rel, tab.z_nodes = zf["E_rel"], zf["z_nodes"]
        return tab
    tab = EscapeTable(trap, j_min, j_max, n_max, slab_sites=slab_sites)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, j_nodes=tab.j_nodes, n_max=tab.n_max, E_rel=tab.E_rel, z_nodes=tab.z_nodes)
    return tab
