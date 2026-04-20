import pyobs
import numpy as np

rng = pyobs.random.generator('test=1')
data = rng.markov_chain(np.array([1,2]), np.array([0.1,0.02])**2, 10, 500).flatten()

c = pyobs.tensormap('name')
c.append(data, name='myname')
assert np.sum(c['myname'] - data)==0

obs_ref = []
obs = pyobs.tensormap('tag0','tag1')
for name in ['a', 'b','c']:
    for p in [np.array([1,2]), np.array([2,3])]:
        _data = rng.markov_chain(p, np.array([0.1,0.02])**2, 10, 500)
        _o = pyobs.observable()
        _o.create('test', _data.flatten(), shape=(2,))
        obs.append(_o, tag0=name, tag1=p)
        obs_ref.append(_o)

obs.peek()
print(obs)

i=0
for t0 in obs.tag0:
    for t1 in obs.tag1:
        print(f'Checking tags {t0} {t1}')
        assert np.sum(obs[t0, t1].mean - obs_ref[i].mean)==0
        i += 1

print(obs['a', :])