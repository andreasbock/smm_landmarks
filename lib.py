import numpy as np
import pylab as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def sample_circle(num_landmarks, scale=1, shift=0):
    thetas = np.linspace(0, 2*np.pi, num=num_landmarks+1)[:-1]
    positions = scale * np.array([[np.cos(x), np.sin(x)] for x in thetas]) + shift
    return positions

def criss_cross(num_landmarks):
    test_name = 'criss_cross'
    q0x = np.linspace(-1,1,num_landmarks)
    q0 = -np.ones([num_landmarks,2])
    q0[:,1] = q0x
    q1x = np.linspace(-1,1,num_landmarks)
    q1 = np.ones([num_landmarks,2])
    q1[:,1] = q1x[::-1]
    return q0, q1, test_name

def pringle(num_landmarks):
    test_name = 'pringle'
    scale = 1
    num_top = num_landmarks // 2
    if num_landmarks % 2 == 0:
        num_bot = num_top
    else:
        num_bot = num_top + 1

    thetas = np.linspace(0, np.pi, num=num_top+1)[:-1]
    positions_top = scale * np.array([[np.cos(x), np.sin(x)] for x in thetas])

    thetas = np.linspace(np.pi, 2*np.pi, num=num_bot+1)[:-1]
    positions_bot = scale * np.array([[np.cos(x), np.sin(x)] for x in thetas])[::-1]

    q0 = sample_circle(num_landmarks)
    q1 = np.append(positions_top, positions_bot, axis=0)
    return q0, q1, test_name

def squeeze(num_landmarks):
    test_name = 'squeeze'

    scale = 1

    # test 1
    eps = 1e-04 # for 8 LMS, goes to one center
    y_map = lambda y: y

    # test 2: 16 (line singularity 2*3) LMS
    eps = .5
    y_map = lambda y: y

    # test 3: 16 (one big singularity at zero) LMS
    eps = .5
    y_map = lambda y: 0

    thetas = np.linspace(0, 2*np.pi, num=num_landmarks+1)[:-1]
    q1 = scale * np.array([[np.cos(x), np.sin(x)] for x in thetas])

    k = 0
    for p in q1:
        x, y = p
        if abs(y) < eps:
            q1[k] = (0., y_map(y))
        k += 1

    k = 0
    pert = 2.
    for p in q1:
        x, y = p
        if abs(x) > 2.5:
            q1[k] = (x + - pert*np.sign(x), y)
        k += 1

    q0 = sample_circle(num_landmarks)

    return q0, q1, test_name

def triangle_flip(num_landmarks):
    test_name = 'triangle_flip'

    if num_landmarks % 3 != 0:
        print("Want a nice image, so satisfy 'num_landmarks % 3 == 0' !")
        exit()
    scale = 1

    a = np.array([1e-06, .7])  # lazy with sign(0) in reflection
    b = np.array([-.7, 0])
    c = np.array([.7, 0])

    # interpolate between them to generate points
    ss = num_landmarks // 3
    ss0 = 1
    ss1 = 4
    q0_a = [(1-s/ss0) * a + s/ss0 * b for s in range(ss0)] # [`a` -> `b`)
    q0_b = [(1-s/ss1) * b + s/ss1 * c for s in range(ss1)] # [`b` -> `c`)
    q0_c = [(1-s/ss0) * c + s/ss0 * a for s in range(ss0)] # [`c` -> `a`)

    q0 = np.array(q0_a + q0_b + q0_c)

    # flip to generate q1 (reflect about x_reflect)
    q1 = np.copy(q0)
    k = 0
    x_reflect=0
    for k in range(num_landmarks):
        x, y = q1[k]
        dist = 2 * np.sqrt((y - x_reflect)**2)
        q1[k] = x, y - np.sign(y - x_reflect) * dist
        k += 1

    return q0, q1, test_name

def triangle_rot(num_landmarks):
    test_name = 'triangle_rot'
    q0, _, _ = triangle_flip(num_landmarks)

    # rotate
    theta = np.pi / 2
    rot_m = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])

    new_origin = (a + b + c)/ 3. - np.array([-2.5, 0])
    q1 = np.array(list(map(lambda p: np.dot(rot_m, p), np.copy(q0)))) + new_origin
    return q0, q1, test_name

def pent_to_square(num_landmarks):
    test_name = 'pent_to_square'

    if num_landmarks % 5 != 0:
        print("Want a nice image, so satisfy 'num_landmarks % 5 == 0' !")
        exit()
    scale = 1

    a = np.array([0, 0])
    b = np.array([-.5, .5])
    c = np.array([0, 1])
    d = np.array([1, 1])
    e = np.array([1, 0])

    # interpolate between them to generate points
    ss = num_landmarks // 5

    pts = lambda x,y : [(1-s/ss) * x + s/ss * y for s in range(ss)]
    q0 = np.array(pts(a, b) + pts(b, c) + pts(c, d) + pts(d, e) + pts(e, a))

    # collapse B into C
    q1 = np.array(ss*[a] + pts(a, c) + pts(c, d) + pts(d, e) + pts(e, a))
    return q0, q1, test_name

def pent_to_tri(num_landmarks):
    test_name = 'pent_to_tri'

    if num_landmarks % 5 != 0:
        print("Want a nice image, so satisfy 'num_landmarks % 5 == 0' !")
        exit()
    scale = 1

    a = np.array([0, 0])
    b = np.array([-.5, .5])
    c = np.array([0, 1])
    d = np.array([1, 1])
    e = np.array([1, 0])

    # interpolate between them to generate points
    ss = num_landmarks // 5

    pts = lambda x,y : [(1-s/ss) * x + s/ss * y for s in range(ss)]
    q0 = np.array(pts(a, b) + pts(b, c) + pts(c, d) + pts(d, e) + pts(e, a))

    # collapse A&C into B
    q1 = np.array(2*ss*[b] + pts(b, c) + pts(c, a) + pts(a, b))

    return q0, q1, test_name

def plot_target(q0, q1, fname=None):
    fig = plt.figure()
    q0_plt = np.append(q0, [q0[0,:]], axis=0)
    q1_plt = np.append(q1, [q1[0,:]], axis=0)

    plt.plot(q0_plt[:,0], q0_plt[:,1], 'm*-', label='$q_0$')
    plt.plot(q1_plt[:,0], q1_plt[:,1], 'bo-', label='$q_1$')
    plt.legend(loc='best')

    if fname:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()

class map_estimator():
    def __init__(self, xs, centers):
        self.xs = np.copy(xs)
        self.centers = centers

def trace_plot(fnls, log_dir):
    plt.figure()
    plt.xlabel('Iteration')
    plt.ylabel('Functional')
    plt.plot(range(len(fnls)), fnls, 'r*-')
    plt.savefig(log_dir + 'functional_traceplot.pdf', bbox_inches='tight')

def centroid_plot(c_samples, log_dir, x_min, x_max, y_min, y_max):
    for i in range(np.shape(np.array(c_samples))[1]):
        cs = np.array(c_samples)[:, i, :]
        cx, cy = cs[:, 0], cs[:, 1]

        plt.figure()
        plt.xlabel('Iteration')
        plt.ylabel('Centroid position')
        plt.plot(cx[0], cy[0], 'r>-', alpha=1)
        plt.plot(cx[-1], cy[-1], 'b<-', alpha=1)
        plt.plot(cx, cy, 'go-', alpha=0.3)
        plt.axis([x_min, x_max, y_min, y_max])
        plt.savefig(log_dir + 'centroid_evolution_'+str(i)+'.pdf', bbox_inches='tight')

def centroid_heatmap(c_samples, log_dir, x_min, x_max, y_min, y_max, bins=10):
    for i in range(np.shape(np.array(c_samples))[1]):
        cs = np.array(c_samples)[:, i, :]
        cx, cy = cs[:, 0], cs[:, 1]
        import matplotlib.colors as mcolors
        plt.figure(figsize=(5,4))
        plt.hist2d(cx, cy, bins=bins, range = [ [x_min, x_max], [y_min,y_max]])
        plt.xlabel('$x$-coordinate')
        plt.ylabel('$y$-coordinate')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Number of samples')
        plt.savefig(log_dir + 'centroid_heat_'+str(i)+'.pdf', bbox_inches='tight')

def sample_autocov(k, m):
    num_samples, num_kernels, _ = m.shape
    est = 0
    for i in range(num_samples - k):  # for each sample
        for j in range(num_kernels):  # for each kernel in \nu
            est += np.dot(m[i + k, j, :], m[i, j, :])
    return est / num_samples

def plot_autocorr(c_samples, fname, lag_max):
    num = lag_max# min(100, len(c_samples))
    lags = np.linspace(1, lag_max, num, dtype=int)

    c_samples_np = np.array(c_samples)
    mean = np.mean(c_samples_np, axis=0)

    ac = lambda k: sample_autocov(k, c_samples_np - mean)
    acf = list(map(ac, lags)) / sample_autocov(0, c_samples_np - mean)

    plt.figure(figsize=(5,4))
    plt.plot(lags, acf, 'r.-')
    plt.xlabel('Lag')
    plt.xlim((1, lag_max))#len(c_samples)))
    plt.ylabel('Sample autocorrelation')
    plt.grid(linestyle='dotted')
    plt.savefig(fname + 'autocorrelation.pdf', bbox_inches='tight')

def fnl_histogram(fnls, fname, test_name, bins='auto'):
    plt.figure(figsize=(5,4))
    plt.hist(fnls, bins=bins, facecolor='green', alpha=0.75)
    plt.xlabel('Metamorphosis functional')
    plt.ylabel('Number of observed values')
    plt.grid(linestyle='dotted')

    # plot lines from
    import pickle
    po = open("lddmm.pickle", "rb")  # in src/!
    fnls_lddmm = pickle.load(po)
    po.close()

    po = open("mm.pickle", "rb")  # in src/!
    fnls_mm = pickle.load(po)
    po.close()

    #plt.axvline(x=sum(fnls_lddmm[test_name]), color='b', label='LDDMM')
    #plt.axvline(x=sum(fnls_mm[test_name]), color='r', label='Metamorphosis')
    plt.legend(loc='best')

    plt.savefig(fname + 'functional_histogram.pdf', bbox_inches='tight')

def plot_q(x0, xs, N, fname, nus=None, title=None):
    plt.figure(figsize=(5,4))
    plot_landmarks(x0, color='r', start_style='o--', label='$q_0$', markersize=9)
    plot_landmarks_traj(xs, N, lw=1)
    plot_landmarks(xs[-1], start_style='x:', label='$q_1$', markersize=15)
    if title:
        plt.title(title)
    if nus is not None:
        nx, ny = np.array(nus).T
        s = [12*2**4] * len(nus)
        plt.scatter(nx, ny, s=s, color='purple', alpha=.3, label='centroid')
    plt.legend(loc='best')
    plt.grid(linestyle='dotted')
    plt.xlabel('x')
    plt.xlabel('y')
    plt.savefig(fname + '.pdf', bbox_inches='tight')
    plt.close()

def plot_landmarks_traj(x, N, lw=.1):
    if len(x.shape) == 2:
        x = x.reshape((1,1,) + x.shape)
    if len(x.shape) == 3:
        x = x.reshape((1,) + x.shape)
    if len(x.shape) == 5:
        for i in range(x.shape[0]):
            plot_landmarks_traj(x[i], N, lw)
        return

    for i in range(N):
        plt.plot(x[:,0,i,0], x[:,0,i,1], 'k-', lw=lw)

def plot_landmarks(x, x0=None, lw=1., line_style='g--', markersize=5, color='b',
    start_style='x--', end_style='o-', label=None):
    if len(x.shape) == 2:
        x = x.reshape((1, 1,) + x.shape)
    if len(x.shape) == 3:
        x = x.reshape((1,) + x.shape)
    if len(x.shape) == 5:
        for i in range(x.shape[0]):
            plot_landmarks(x[i], x0=x0, lw=lw, line_style=line_style,
                start_style=start_style, end_style=end_style,
                markersize=markersize, color=color)
    if not x0 is None:
        x = np.concatenate((x0.reshape((1,)+x0.shape),x),axis=0)

    plt.plot(np.concatenate((x[0,0,:,0],[x[0,0,0,0],])),
        np.concatenate((x[0,0,:,1],[x[0,0,0,1],])), start_style, color=color,
        markersize=markersize, label=label)

    if x.shape[0] > 1:
        plt.plot(np.concatenate((x[-1,0,:,0],[x[-1,0,0,0],])),
            np.concatenate((x[-1,0,:,1],[x[-1,0,0,1],])), end_style,
            color=color, markersize=markersize)

    for i in range(x.shape[2]):
        plt.plot(x[:,0,i,0],x[:,0,i,1],line_style,lw=lw)

