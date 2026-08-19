import pyobs
import numpy as np

# Exercises observable __getitem__ / __setitem__ across index forms:
# integers, slices, ndarrays and Ellipsis (... ), both reading and writing.

rng = pyobs.random.generator("slicing")
val = np.arange(1, 7) * 1.0
sig = val * 0.01
data = rng.markov_chain(val, sig ** 2, 4.0, 2000).flatten()


def fresh():
    o = pyobs.observable()
    o.create("a", data, shape=(2, 3))
    return o


def same(a, b):
    va, ea = a.error()
    vb, eb = b.error()
    return np.allclose(va, vb) and np.allclose(ea, eb)


obs = fresh()

# ---- __getitem__: ellipsis forms equal their explicit-slice counterparts ----
assert same(obs[...], obs[:, :]), "obs[...] != obs[:, :]"
assert same(obs[..., 0], obs[:, 0:1]), "obs[..., 0] != obs[:, 0:1]"
assert same(obs[0, ...], obs[0:1, :]), "obs[0, ...] != obs[0:1, :]"

# shapes follow numpy: a selected axis collapses to length 1 (dims preserved)
assert obs[...].shape == (2, 3)
assert obs[..., 0].shape == (2, 1)
assert obs[0, ...].shape == (1, 3)

# ---- __getitem__: ndarray indexing equals the equivalent slice ----
idx = np.array([0, 1])
assert same(obs[idx, :], obs[0:2, :]), "obs[idx, :] != obs[0:2, :]"
assert same(obs[idx, ...], obs[0:2, :]), "obs[idx, ...] != obs[0:2, :]"
assert same(obs[:, np.array([0, 2])], obs[:, 0:3:2]), "ndarray column index"

# ---- __setitem__: writing the read-back block is a no-op (round-trip) ----
for key in [
    (0, slice(None)),
    (slice(None), 1),
    Ellipsis,
    (Ellipsis, 0),
    (0, Ellipsis),
    (idx, slice(None)),
    (1, 2),
]:
    o = fresh()
    before = o.error()
    o[key] = o[key]
    after = o.error()
    assert np.allclose(before[0], after[0]) and np.allclose(
        before[1], after[1]
    ), f"round-trip changed the observable for index {key}"

# ---- __setitem__: writes the targeted block and leaves the rest untouched ----
# overwrite row 0 with row 1 (same ensemble)
o = fresh()
v0, e0 = o.error()
o[0, :] = o[1, :]
v, e = o.error()
assert np.allclose(v[0], v0[1]) and np.allclose(e[0], e0[1]), "row 0 not set to row 1"
assert np.allclose(v[1], v0[1]) and np.allclose(e[1], e0[1]), "row 1 was modified"

# the same via ellipsis on the column axis: column 0 <- column 1
o = fresh()
v0, e0 = o.error()
o[..., 0] = o[..., 1]
v, e = o.error()
assert np.allclose(v[:, 0], v0[:, 1]) and np.allclose(e[:, 0], e0[:, 1]), "col 0 not set"
assert np.allclose(v[:, 2], v0[:, 2]) and np.allclose(e[:, 2], e0[:, 2]), "col 2 changed"

# scalar element assignment
o = fresh()
v0, e0 = o.error()
o[1, 2] = o[0, 0]
v, e = o.error()
assert np.allclose(v[1, 2], v0[0, 0]) and np.allclose(e[1, 2], e0[0, 0]), "element set"
assert np.allclose(v[0, 1], v0[0, 1]) and np.allclose(e[0, 1], e0[0, 1]), "neighbor kept"


# test .iloc for numpy like slicing (dropping axes index by a single integer)
obs = fresh()

assert obs.iloc[0, :].shape == (3,), "iloc[0, :] should drop axis 0"
assert obs.iloc[:, 0].shape == (2,), "iloc[:, 0] should drop axis 1"
assert obs.iloc[..., 0].shape == (2,), "iloc[..., 0] should drop the last axis"
assert obs.iloc[0, ...].shape == (3,), "iloc[0, ...] should drop axis 0"

# a length-1 slice should not be dropped (matching numpy)
assert obs.iloc[0:1, :].shape == (1, 3), "iloc[0:1, :] must keep the sliced axis"

# an ndarray index should not be dropped
assert obs.iloc[np.array([0]), :].shape == (1, 3), "iloc ndarray axis kept"

# if no index is integer .iloc must match __getitem__ 
assert obs.iloc[:, :].shape == (2, 3)
assert obs.iloc[...].shape == (2, 3)

# check for other int types 
assert obs.iloc[np.int64(0), :].shape == (3,), "numpy.int64 index not squeezed"

# values match the dims-preserved read, just without the trivial axis
assert same(obs.iloc[0, :], obs[0, :]), "iloc[0, :] value differs from obs[0, :]"
assert same(obs.iloc[1, 2], obs[1, 2]), "iloc scalar value differs"

print("slicing: getitem/setitem ok for int, slice, ndarray and ellipsis")
print("slicing: iloc drops integer-indexed axes (numpy-style)")
