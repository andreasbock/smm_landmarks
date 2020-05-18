from lib import *
import pickle
import pylab as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# The following data contain lists of tuples of (velocity_norm, penalty_norm)
# for the tests given in `test_names`
test_names = ['criss_cross', 'pent_to_tri', 'squeeze']
lddmm = pickle.load(open("lddmm.pickle", "rb"))
mm = pickle.load(open("mm.pickle", "rb"))
custom_nu = pickle.load(open("custom_nu.pickle", "rb"))

for test_name in test_names:
    l, m, c = lddmm[test_name], mm[test_name], custom_nu[test_name]
    width = .8
    fig = plt.figure()
    idx = np.arange(3)
    v = plt.bar(idx, [np.float(l[0]), c[0], np.float(m[0])], width, alpha=.5)
    p = plt.bar(idx, [np.float(l[1]), c[1], np.float(m[1])], width,
        bottom=[np.float(l[0]), c[0], np.float(m[0])])
    plt.legend((v[0], p[0]), ('Kinetic energy', 'Penalty term'))
    plt.xticks(idx, ('LDDMM', 'SMM', 'MM'))
    plt.yscale('symlog')
    plt.ylabel('Functional value')
    plt.savefig('functionals_' + test_name + '.pdf', bbox_inches='tight')

for k in lddmm:
    print(k, lddmm[k])
print("\n")
for k in mm:
    print(k, mm[k])
print("\n")
for k in custom_nu:
    print(k, custom_nu[k])
