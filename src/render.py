"""Turn predicted per-gene factors into whole-number counts for a group of generated cells.

All renderers start from real control cells of the target context and return a sparse
(cells x genes) count matrix. `factor` is one multiplicative change per gene (2 ** lfc).

resample   draw cells, change each count by the factor (binomial thinning / Poisson adding)
exact      like resample, but the factor is corrected so each gene's group total lands on
           n_cells x control_mean x factor (removes the chance difference of the drawn cells)
exactNN    like exact, with NN% of cells left untouched (cells where the knockdown failed)
poolK      each generated cell is the average profile of K control cells, scaled to that average
           depth, with exact group totals and random rounding to whole numbers.
           WARNING: tested 2026-10-19 on context A with no change applied: about half of all expressed
           genes came out "significant" in a Wilcoxon test against real controls, because averaged
           cells have a different distribution shape (fewer zeros). Kept for reference, not for use.
tilt       two targets per gene: `factor` for the per-cell average (what the DE tests and fold changes
           see, every cell weighted equally) and `bulk_factor` for the group total (what mse and pds see,
           deeper cells weigh more). The change is applied more strongly in shallow cells and more
           weakly in deep cells so both targets are met; group totals are then made exact.
"""
import numpy as np
from scipy import sparse


def apply_effect(X, f, rng, per_value=False):
    """X: sparse counts (cells x genes); f: one factor per gene (or, with per_value, one per stored
    value of X). Returns new sparse counts."""
    X = X.copy()
    counts = X.data.astype(np.int64)
    f_of_value = f if per_value else f[X.indices]   # the factor of each stored value's gene
    down, up = f_of_value < 1, f_of_value > 1
    counts[down] = rng.binomial(counts[down], f_of_value[down])
    counts[up] += rng.poisson(counts[up] * (f_of_value[up] - 1))
    X.data = counts.astype(np.float32)
    X.eliminate_zeros()                             # counts thinned to 0 are not stored
    return X


class Renderer:
    """Holds one context's control cells and makes groups of generated cells from them."""

    def __init__(self, controls, mode, rng, n_cells=400, max_factor=16.0):
        self.C = sparse.csr_matrix(controls, dtype=np.float32)
        self.mode, self.rng, self.n, self.max_factor = mode, rng, n_cells, max_factor
        self.mean = np.asarray(self.C.mean(axis=0)).ravel()            # average counts per cell
        self.lib = np.asarray(self.C.sum(axis=1)).ravel()
        if mode.startswith("pool"):
            self.k = int(mode[4:])
            self.props = sparse.diags(1.0 / np.maximum(self.lib, 1)) @ self.C   # per-cell profile
            self.props = self.props.tocsr()
        elif mode.startswith("exact"):
            self.off = round(int(mode[5:] or 0) / 100 * n_cells)
        elif mode == "tilt":
            pass
        else:
            assert mode == "resample", mode

    def group(self, factor, bulk_factor=None):
        factor = np.minimum(factor, self.max_factor)
        if self.mode == "tilt":
            return self._tilted(factor, np.minimum(bulk_factor, self.max_factor))
        if self.mode == "resample":
            return apply_effect(self._draw(self.n), factor, self.rng)
        if self.mode.startswith("exact"):
            return self._exact(factor)
        return self._pooled(factor)

    def _draw(self, m):
        return self.C[self.rng.integers(0, self.C.shape[0], m)]

    def _exact(self, factor):
        picked = self._draw(self.n)
        target = self.n * self.mean * factor                            # what the group total should be
        off, on = picked[:self.off], picked[self.off:]
        off_total = np.asarray(off.sum(axis=0)).ravel()
        on_total = np.asarray(on.sum(axis=0)).ravel()
        # factor for the responding cells that lands the group total on target
        f_on = np.where(on_total > 0, (target - off_total) / np.maximum(on_total, 1), factor)
        f_on = np.clip(f_on, 0.0, self.max_factor)
        return sparse.vstack([off, apply_effect(on, f_on, self.rng)], format="csr")

    def _tilted(self, f_cell, f_bulk):
        picked = self._draw(self.n)
        depth = np.asarray(picked.sum(axis=1)).ravel()
        z = (depth - depth.mean()) / max(depth.std(), 1e-9)
        cv = depth.std() / max(depth.mean(), 1e-9)                        # spread of depths
        # per cell c and gene g: f = f_cell + (f_bulk - f_cell) * z_c / cv
        #   the plain average over cells is f_cell (z averages to 0); the depth-weighted average,
        #   which is what the group total follows, is f_bulk
        rows = np.repeat(np.arange(picked.shape[0]), np.diff(picked.indptr))
        g = picked.indices
        f = f_cell[g] + (f_bulk[g] - f_cell[g]) * z[rows] / cv
        f = np.clip(f, 0.0, self.max_factor)
        out = apply_effect(picked, f, self.rng, per_value=True)
        # clipping and random counts move the totals a little: make group totals exact
        target = self.n * self.mean * f_bulk
        total = np.asarray(out.sum(axis=0)).ravel()
        ratio = np.where(total > 0, target / np.maximum(total, 1), 1.0)
        return apply_effect(out, np.clip(ratio, 0.0, self.max_factor), self.rng)

    def _pooled(self, factor):
        n, k = self.n, self.k
        idx = self.rng.integers(0, self.C.shape[0], n * k)
        avg = sparse.csr_matrix((np.full(n * k, 1.0 / k), (np.repeat(np.arange(n), k), np.arange(n * k))),
                                shape=(n, n * k))
        profile = (avg @ self.props[idx]).tocsr()                       # average profile of k cells
        depth = avg @ self.lib[idx]                                     # their average depth
        expected = (sparse.diags(depth) @ profile @ sparse.diags(factor)).tocsr()
        # exact group totals: scale each gene so the expected total is n x control mean x factor
        total = np.asarray(expected.sum(axis=0)).ravel()
        target = n * self.mean * factor
        scale = np.where(total > 0, target / np.maximum(total, 1e-12), 0.0)
        expected = (expected @ sparse.diags(scale)).tocsr()
        # random rounding keeps the expected value: 2.3 -> 2 (70%) or 3 (30%)
        low = np.floor(expected.data)
        expected.data = (low + (self.rng.random(low.size) < expected.data - low)).astype(np.float32)
        expected.eliminate_zeros()
        return expected
