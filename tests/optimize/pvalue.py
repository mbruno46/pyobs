import pyobs
import numpy
import math

# ---------------------------------------------------------------------------
# Regression tests for the p-value of pyobs.mfit and for real spectra obtained
# from numpy.linalg.eig.
#
# Background
#   * numpy.linalg.eig returns a complex array for *every* real input since
#     NumPy 2.0, also when the spectrum is real. Accumulating those values into
#     a real array raised UFuncOutputCastingError in mfit.pvalue and in the
#     eig/eigLR gradients.
#   * mfit.pvalue compared the simulated quadratic form with
#     0 after subtracting self.c2/n from each eigenvalue; that is not the
#     distribution of the test statistic, and it returned p = 0 for chi2 >= n
#     (where the exact p-value is still finite).
# ---------------------------------------------------------------------------

print("=== chi2 survival function used as reference (no scipy dependency) ===")


def chi2_sf(x, dof):
    """Regularized upper incomplete gamma Q(dof/2, x/2) for integer dof."""
    a = dof / 2.0
    z = x / 2.0
    if z <= 0.0:
        return 1.0
    q = math.erfc(math.sqrt(z))  # Q(1/2,z)
    for i in range(int(round(a - 0.5))):
        ai = 0.5 + i
        q += z**ai * math.exp(-z) / math.gamma(ai + 1)  # Q(ai+1,z)
    return q


# reference values (scipy.stats.chi2.sf) to guard the helper itself
for x, dof, ref in [
    (2.5, 5, 0.776495),
    (5.0, 5, 0.415880),
    (8.0, 5, 0.156236),
    (16.0, 5, 0.0068440739),
]:
    assert abs(chi2_sf(x, dof) - ref) < 1e-5, (x, dof, chi2_sf(x, dof), ref)

# ---------------------------------------------------------------------------
# 1) Exact Gaussian linear problem: the null law of the test statistic is
#    chi-square with dof = n - p parameters. The covariance is supplied exactly
#    (identity) so that the check isolates the statistic from the estimation of
#    the covariance matrix.
# ---------------------------------------------------------------------------
print("=== p-value against the exact chi-square law ===")

n, p = 8, 3
nmc = 10000
alpha = 1e-3
targets = [2.5, 5.0, 8.0, 16.0]
hoeffding = math.sqrt(math.log(2 * len(targets) / alpha) / (2 * nmc))


class GaussianRNG:
    """Minimal replacement of pyobs.random.generator: normal draws only."""

    def __init__(self, seed):
        self.rng = numpy.random.default_rng(seed)

    def sample_normal(self, size):
        return self.rng.normal(size=size)


def exact_fit(chi2):
    y = numpy.zeros(n)
    y[p] = numpy.sqrt(chi2)
    obs = pyobs.observable()
    obs.create_from_cov("oracle", y, numpy.eye(n))
    fun = lambda x, a, b, c: a * (x == 0) + b * (x == 1) + c * (x == 2)
    dfun = lambda x, a, b, c: [float(x == 0), float(x == 1), float(x == 2)]
    fit = pyobs.mfit(numpy.arange(n), numpy.eye(n), fun, dfun)

    # exact coordinate projection: bypass the minimizer, keep the propagation
    class Backend:
        def __call__(self, f, p0, jac, hess):
            from pyobs.optimize.minimize import outputformat

            z = numpy.zeros(p)
            return outputformat(z, f(z), 0, "exact")

    fit(obs, p0=numpy.zeros(p), min_search=Backend())
    # exact covariance: replace the estimator by the known identity
    obs.covariance_matrix = lambda errinfo: [numpy.eye(n), numpy.zeros((n, n))]
    return fit, obs


print(f"{'chi2':>6} {'p_value':>9} {'chi2_sf':>9} {'mc error':>9} {'|diff|':>8}")
for k, chi2 in enumerate(targets):
    fit, obs = exact_fit(chi2)
    pv, dp, cexp = fit.pvalue(GaussianRNG(k), obs, {}, nmc=nmc)
    ref = chi2_sf(fit.c2, n - p)
    print(f"{fit.c2:6.2f} {pv:9.4f} {ref:9.4f} {dp:9.4f} {abs(pv - ref):8.4f}")
    assert abs(pv - ref) <= hoeffding, (chi2, pv, ref, hoeffding)
    # regression: the previous implementation returned exactly 0 for chi2 >= n
    assert pv > 0.0, f"p-value vanished for chi2 = {chi2}"
    # the returned sample must be the quadratic form itself, not a shifted copy
    assert abs(numpy.mean(cexp) - (n - p)) < 0.15 * n, numpy.mean(cexp)

# ---------------------------------------------------------------------------
# 2) End-to-end: a real Markov-chain ensemble exercises the Gamma covariance
#    path and the complex-dtype regression in the same call.
# ---------------------------------------------------------------------------
print("=== p-value from a Markov-chain ensemble (end-to-end) ===")

a, m, T = 1.0, 0.2, 16
y = numpy.array([a * numpy.exp(-m * t) for t in range(T)])
dy2 = (y * 0.1) ** 2
rng = pyobs.random.generator("pvalue")
data = rng.markov_chain(y, dy2, 1.0, 1000)
corr = pyobs.observable()
corr.create("test", data.flatten(), shape=(T,))

[f, df, _] = pyobs.symbolic.diff("a*exp(-m*x)", "x", "a,m")
[_, dc] = corr.error()
fit = pyobs.mfit(numpy.arange(T), 1.0 / dc**2, f, df)
pars = fit(corr)
assert abs(pars[0].mean - a) < 3 * pars[0].error()[1]
assert abs(pars[1].mean - m) < 3 * pars[1].error()[1]

pv, dp, cexp = fit.pvalue(rng, corr, {"test": pyobs.errinfo(W=10)}, nmc=2000)
print(f"chi2 = {fit.c2:.3f}, dof = {T - 2}, p = {pv:.3f} +- {dp:.3f}")
assert numpy.isfinite(pv) and 0.0 <= pv <= 1.0
assert numpy.all(numpy.isfinite(cexp))
# the data are generated from the fitted model: a vanishing p-value would signal
# the broken statistic (chi2 < n here, so the old code was biased low, and for
# chi2 >= n it returned exactly zero)
assert pv > 0.01, pv

# ---------------------------------------------------------------------------
# 3) Real spectra returned as complex arrays (numpy.linalg.eig, NumPy >= 2)
# ---------------------------------------------------------------------------
print("=== real spectra from numpy.linalg.eig ===")

sym = numpy.array([[2.0, 0.5, 0.0], [0.5, 3.5, 0.25], [0.0, 0.25, 1.0]])
w_raw, _ = numpy.linalg.eig(sym)
w = pyobs.real_spectrum(w_raw, sym)
assert not numpy.iscomplexobj(w)
assert numpy.allclose(numpy.sort(w), numpy.linalg.eigvalsh(sym))

# a matrix similar to a symmetric one (positive definite congruence) has a real
# spectrum even though it is not symmetric itself
L = numpy.linalg.cholesky(numpy.array([[2.0, 0.3], [0.3, 1.0]]))
nonsym = L @ sym[:2, :2] @ L.T
w_ns, _ = numpy.linalg.eig(nonsym)
assert pyobs.real_spectrum(w_ns, nonsym).dtype == numpy.float64

# a genuinely complex spectrum must be rejected, not silently truncated
rot = numpy.array([[0.0, -1.0], [1.0, 0.0]])
try:
    pyobs.real_spectrum(numpy.linalg.eig(rot)[0], rot)
    raise AssertionError("complex spectrum was not rejected")
except pyobs.PyobsError:
    pass

# eig/eigLR on a real spectrum must produce real-valued observables again
mat = numpy.array([2.0, 0.5, 0.5, 3.5])
cov = (mat * 0.05) ** 2
mdata = rng.markov_chain(mat, cov, 1.0, 2000)
# symmetrise each configuration: pyobs.linalg.eig requires an exactly symmetric
# central value, which independent noise on the entries would violate
msym = 0.5 * (mdata.reshape(-1, 2, 2) + mdata.reshape(-1, 2, 2).transpose(0, 2, 1))
assert numpy.array_equal(numpy.mean(msym, axis=0), numpy.mean(msym, axis=0).T)
omat = pyobs.observable()
omat.create("test", msym.reshape(-1), shape=(2, 2))
[wobs, vobs] = pyobs.linalg.eig(omat)
assert not numpy.iscomplexobj(wobs.mean)
err = wobs.error()[1]
assert numpy.all(numpy.isfinite(err)) and numpy.all(err > 0)

[lobs, vobs, wobs_l] = pyobs.linalg.eigLR(omat)  # real spectrum, non-symmetric input
assert not numpy.iscomplexobj(lobs.mean)
assert numpy.all(numpy.isfinite(lobs.error()[1]))

print("All p-value tests passed")
