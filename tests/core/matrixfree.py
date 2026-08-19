import pyobs
import numpy as np 

# =============================================================================
# pyobs.set_matrixfreegrad(mode):
#   False -> always build the (Na x Ni) gradient matrix (default / current behaviour)
#   True  -> never build it; apply the differential column-by-column
#   N:int -> matrix-free only when Na*Ni >= N exceeds the threshold 
#
# Note that this choice is saved when the gradient is initialized. 
# This means that the derived observable must be rebuilt in order to compare the errors. 
# =============================================================================


def test(name, func, obs): 
    # compute the derived observable once with the gradient matrix and once without 
    # the results should match up to machine precision 
    pyobs.set_matrixfreegrad(False)
    res_wmatrix = func(*obs)
    pyobs.set_matrixfreegrad(True)
    res_mfree  = func(*obs)
    pyobs.set_matrixfreegrad(False)

    m0, err0 = res_wmatrix.error() 
    m1, err1 = res_mfree.error()
    assert np.all(abs(m0 - m1) < np.finfo(float).eps), f"\ntest={name} failed\nwith matrix: {res_wmatrix}\nwithout matrix:{res_mfree}"
    assert np.all(abs(err0/err1 - 1) < 1e-10), f"\ntest={name} failed\nwith matrix: {res_wmatrix}\nwithout matrix:{res_mfree}"
    print(f"test {name} passed")


# some fake observables 
rng = pyobs.random.generator("matrixfree-core")

val = [2,0.5,0.5,3.5]
cov = (np.array(val)*0.05)**2
N=100
tau=2.0

n   = 2

obs_scalar = pyobs.observable.from_data(rng.markov_chain(val[0], cov[0], tau, N))
obs_A      = pyobs.observable.from_data(rng.markov_chain(val, cov, tau, N).flatten(), shape=(n,n))
obs_B      = pyobs.observable.from_data(rng.markov_chain(val, cov, tau, N).flatten(), shape=(n,n))
print(f"scalar observable={obs_scalar}")
print(f"A={obs_A}")
print(f"B={obs_B}")

# ----------------------------------------------------------------------------- 
# 0. Check that matrix free gradient approach actually engages
# -----------------------------------------------------------------------------

pyobs.set_matrixfreegrad(True)
g = pyobs.gradient(lambda x: x @ obs_B.mean, obs_A.mean)
assert g.matrixfree and not hasattr(g, "grad"), "matrix full mode is running"

pyobs.set_matrixfreegrad(False)
g = pyobs.gradient(lambda x: x @ obs_B.mean, obs_A.mean)
assert (not g.matrixfree) and hasattr(g, "grad"), "matrix free mode is running"

# check threshold mode and if matrixfree is actually engaged in a real computation, e.g. __matmul__
pyobsgradient = pyobs.gradient
matrixfree_yesno = False
def spy(*args, **kwargs): 
    global matrixfree_yesno
    g = pyobsgradient(*args, **kwargs)
    matrixfree_yesno = getattr(g, "matrixfree", None)
    return g 
pyobs.gradient = spy 

# 2x2=4 so A @ B has Na = 4 and Ni=4 
pyobs.set_matrixfreegrad(16) 
_ = obs_A @ obs_B
assert matrixfree_yesno == True # 16 is above threshold of 16

pyobs.set_matrixfreegrad(17)
_ = obs_A @ obs_B
assert matrixfree_yesno == False # 16 is below threshold of 17

pyobs.gradient = pyobsgradient


# -----------------------------------------------------------------------------
# 1. Check that matrix free gradient approach is identical to eager matrix construction
# -----------------------------------------------------------------------------

# test that for gtype="full" results are identical 
test("matmul", lambda A, B: A@B, [obs_A, obs_B])
test("tr[AB]", lambda A, B: pyobs.einsum("ij,ji", A, B), [obs_A, obs_B])
test("stack", lambda A, B: pyobs.stack([A, B]), [obs_A, obs_B])
test("A*c", lambda A, c: A*c, [obs_A, obs_scalar])
test("A_{ij}B_{jk}B_{ki}", lambda A, B, C: pyobs.einsum("ij,jk,ki", A, B, C), [obs_A, obs_B, obs_B])

# test that for gtype="diag" results are identical (set_matrixfree has no effect in this case)
test("A + B", lambda A, B: A + B, [obs_A, obs_B])
test("c + c", lambda a, b: a + b, [obs_scalar, obs_scalar])
test("sin(c)", lambda c: pyobs.sin(c), [obs_scalar])


# -----------------------------------------------------------------------------
# 2. Index alignment: multiple ensembles, replicas, holes
# -----------------------------------------------------------------------------

obs_ensB = pyobs.observable.from_data(rng.markov_chain(val, cov, tau, N).flatten(), shape=(n,n), ename="EnsB")
test("two ensembles A @ B", lambda A, B: A@B, [obs_A, obs_ensB])

obs_repl2 = pyobs.observable.from_data(rng.markov_chain(val, cov, tau, N).flatten(), shape=(n,n), rname="2")
test("two replicas A @ B", lambda A, B: A@B, [obs_A, obs_repl2])

obs_wholes = pyobs.observable.from_data(rng.markov_chain(val, cov, tau, N//2).flatten(), shape=(n,n), icnfg=range(0, N, 2))
test("with holes A @ B", lambda A, B: A@B, [obs_A, obs_wholes])


# -----------------------------------------------------------------------------
# 3. cdata-backed inputs (cdata is real-valued only)
# -----------------------------------------------------------------------------
obs_cov = pyobs.observable() 
obs_cov.create_from_cov("FVE", val, (np.array(val)/10)**2)
test("with cdata", lambda x, y: x*y, [obs_scalar, obs_cov])
test("with cdata", lambda x, y: x.multiply(y), [obs_scalar, obs_cov])
test("with cdata", lambda x, y: y.multiply(x), [obs_scalar, obs_cov])


# -----------------------------------------------------------------------------
# 4. complex inputs/outputs/gradients
# -----------------------------------------------------------------------------

obs_complex = pyobs.observable.from_data(rng.markov_chain(val, cov, tau, N).flatten() - 0.5j*rng.markov_chain(val, cov, tau, N).flatten(), 
                              shape=(n, n))
test("complex matmul", lambda A, B: A@B, [obs_A, obs_complex])
test("eigLR: lambda", lambda A: pyobs.linalg.eigLR(A)[0], [obs_complex])
test("eigLR: eigr", lambda A: pyobs.linalg.eigLR(A)[1], [obs_complex])
test("eigLR: eigl", lambda A: pyobs.linalg.eigLR(A)[2], [obs_complex])
test("inv", lambda A: A.inv(), [obs_complex])