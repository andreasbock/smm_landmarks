import theano
from theano import function
import theano.tensor as T
from termcolor import colored
from scipy.optimize import minimize, fmin_bfgs, fmin_cg
import numpy as np
import pickle

from lib import *

# landmark parameters
timesteps = 100

# mcmc parameters
maxiter = 5000  # shooting
beta = 0.2
q1_tolerance = 1e-01  # allow for slight mismatch owing to numerics

def run_mcmc(q0, q1, test_name, num_samples, num_nus=1, log_dir=None):
    log_freq = num_samples // min(10, num_samples)

    # kernel parameters
    sigma    = 0.7
    sigma_nu = 0.2

    num_landmarks, DIM = q0.shape
    N = theano.shared(num_landmarks)

    # kernel sigma
    SIGMA = theano.shared(np.array(sigma).astype(theano.config.floatX))
    # radius of the landmark
    SIGMA_NU = theano.shared(np.array(sigma_nu).astype(theano.config.floatX))
    # initialise '\nu' centroid(s)
    NU  = theano.shared(np.zeros([num_nus, DIM])).astype(theano.config.floatX)

    q1_theano = theano.shared(np.zeros([N.eval(), DIM]).astype(theano.config.floatX))
    q1_theano.set_value(q1.astype(theano.config.floatX))

    # timestepping
    Tend = 1 # final time
    n_steps = theano.shared(timesteps) # number of timesteps
    dt = Tend/n_steps # timesteps

    # general kernel for any sigma
    def Ksigma(q1, q2, sigma):
        r_sq = T.sqr(q1.dimshuffle(0, 'x' ,1) - q2.dimshuffle('x', 0, 1)).sum(2)

        return T.exp(- r_sq / (2.*sigma**2))

    # kernel function for the landmarks
    def Ker(q1,q2):
        return Ksigma(q1, q2, SIGMA)

    def nu(q):
        r_sq = T.sqr(q.dimshuffle(0, 'x', 1) - NU.dimshuffle('x', 0, 1)).sum(2).sum(1)
        return T.exp( - r_sq / (2. * SIGMA_NU**2))

    def met(q,p):
        return T.tensordot(nu(q),T.tensordot(p, p, [[1], [1]]).diagonal(), [[0], [0]])

    def H(q,p):
        return 0.5 * T.tensordot(p, T.tensordot(Ker(q, q), p, [[1], [0]]), [[0,1], [0,1]]) + met(q,p)

    # compile the previous functions
    pe = T.matrix('p')
    qe = T.matrix('q')
    qe1 = T.matrix('q')
    qe2 = T.matrix('q')
    sigma = T.scalar()

    print('Start compiling...')
    Kf = function([qe1, qe2], Ker(qe1, qe2))
    metf = function([qe, pe],met(qe, pe))
    Ksigmaf = function([qe1, qe2, sigma], Ksigma(qe1, qe2, sigma))
    nuf= function([qe], nu(qe))
    Hf = function([qe, pe],H(qe, pe))

    # compute gradients
    dq = lambda q, p: T.grad(H(q, p), p)
    dp = lambda q, p: -T.grad(H(q, p), q)
    dqf = function([qe, pe], dq(qe, pe))
    dpf = function([qe, pe], dp(qe, pe))

    # horizontal initial momentum
    p0 = 1./N.eval()*np.vstack((np.zeros(N.eval()), np.ones(N.eval()))).T
    p0 = np.array(int(N.get_value())*[[0.1, 0]])

    # create the initial condition array
    q0 = q0.astype(theano.config.floatX)
    p0 = p0.astype(theano.config.floatX)
    x0 = np.array([q0, p0]).astype(theano.config.floatX)

    def ode_f(x):
        dqt = dq(x[0], x[1])
        dpt = dp(x[0], x[1])
        return T.stack((dqt, dpt))

    def euler(x, dt):
        return x + dt * ode_f(x)

    x = T.tensor3('x')

    # create loop symbolic loop
    (cout, updates) = theano.scan(fn=euler,
                                  outputs_info=[x],
                                  non_sequences=[dt],
                                  n_steps=n_steps)
    # compile it
    simf = function(inputs=[x],
                    outputs=cout,
                    updates=updates)

    # create loss function and compile
    loss = 1./N*T.sum(T.sqr(cout[-1,0] - q1_theano))
    dloss = T.grad(loss,x)
    lossf = function(inputs=[x],
                     outputs=loss,
                     updates=updates)
    dlossf = function(inputs=[x],
                      outputs=[loss, dloss],
                      updates=updates)

    # do the shooting to find the initial momenta
    def shoot(q0, p0):
        def fopts(x):
            [y,gy] = dlossf(np.stack([q0,x.reshape([N.eval(), DIM])]).astype(theano.config.floatX),)
            return (y, gy[1].flatten().astype(np.double))

        return minimize(fopts, p0.flatten(), method='L-BFGS-B', jac=True,
            options={'disp': False, 'maxiter': maxiter})

    print('...compilation over!')
    def solve_mm(nus):
        nu_scaling = 1
        nu_vals = []
        for nu in nus:
            nx, ny = nu
            nu_vals.append([nx, ny])
        NU.set_value(nu_scaling * nu_vals)

        res = shoot(q0,p0)
        xs = simf(np.array([q0, res.x.reshape([N.eval(),
            DIM])]).astype(theano.config.floatX))

        h = []
        for i in range(np.shape(xs)[0]):
            h.append(Hf(xs[i,0],xs[i,1]))
        h = np.array(h).sum()*dt.eval()

        match_success = np.linalg.norm(xs[-1,0] - q1) < q1_tolerance
        return xs, h, res.success and match_success

    if log_dir is None:  # running locally
        log_dir = '../tex/mcmc_results/' + test_name
    log_dir += '/'

    import os
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # compute bounding box for modulus
    x_min  = min(np.min(q0[:, 0]), np.min(q1[:, 0]))
    y_min  = min(np.min(q0[:, 1]), np.min(q1[:, 1]))
    x_max = max(np.max(q0[:, 0]), np.max(q1[:, 0]))
    y_max = max(np.max(q0[:, 1]), np.max(q1[:, 1]))

    def periodic(vs, v_min, v_max):
        res = []
        for v in vs:
            if v < v_min:
                res.append(v_max + (v - v_min))
            elif v > v_max:
                res.append(v_min + (v - v_max))
            else:
                res.append(v)
        return res

    def propose_center(centers):
        c = np.sqrt(1. - beta**2) * centers + beta * np.random.normal(size=centers.shape)
        x, y = c[:, 0], c[:, 1]

        #while abs(x)>1 or abs(y)>1:
        #    c = np.sqrt(1. - beta**2) * centers + beta * np.random.normal(size=centers.shape)
        #    x, y = c[:, 0], c[:, 1]

        px, py = periodic(x, x_min, x_max), periodic(y, y_min, y_max)

        return np.dstack((px, py))[0]

    def acceptance_prob(h, h_prop):
        if shoot_success:
            return np.exp(h - h_prop)
        else:
            return -1

    # helper stuff for saving MCMC samples
    num_estimators = 10
    map_estimators_vals = np.array([1e8] * num_estimators)
    map_estimators = [None] * num_estimators

    print("Running MCMC with {} samples...".format(num_samples))

    # initial guess
    center_x = np.random.uniform(low=x_min, high=x_max, size=num_nus)
    center_y = np.random.uniform(low=y_min, high=y_max, size=num_nus)

    centers = np.dstack((center_x, center_y))[0]
    _, fnl, shoot_success = solve_mm(centers)

    fnls = [] # to store all the functional values
    c_samples = [centers]
    num_accepted = 0
    solver_failures = 0

    for i in range(num_samples - 1):
        msg = "Iteration {}".format(i + 1)

        # compute proposal
        center_prop = propose_center(centers)
        msg += "\n\t proposal   = {}".format(center_prop)

        xs_prop, fnl_prop, shoot_success = solve_mm(center_prop)
        # compute acceptance probability
        acc = acceptance_prob(fnl, fnl_prop)
        msg += "\n\t acc. prob. = {}".format(acc)
        if acc < 0:
            solver_failures += 1

        # test for acceptance
        if np.random.uniform() < acc:
            term_colour = 'green'
            num_accepted += 1

            # update
            centers = center_prop
            fnl = fnl_prop

            # save MAP estimators
            arg_max = map_estimators_vals.argmax()
            if fnl < map_estimators_vals[arg_max]:
                msg += "\n\t ! Saving MAP estimator"
                map_estimators_vals[arg_max] = fnl
                map_estimators[arg_max] = map_estimator(xs_prop, center_prop)

        else:
            term_colour = 'red'

        # log realisation for inspection
        if i % log_freq == 0 and term_colour == 'green':
            msg += "\n\t ! Saving sample!"
            plot_q(x0, xs_prop, num_landmarks, log_dir + 'sample_{}'.format(i),
                nus=centers)

        if acc>0:
            fnls.append(fnl)
            c_samples.append(centers)

        # flush message buffer
        msg += "\n\t functional = {}".format(fnl)
        msg += "\n\t # accepted = {}".format(num_accepted)
        print(colored(msg, term_colour))

    fh = open(log_dir + 'output.log','w')
    fh.write("timesteps: {}\n".format(timesteps)
            + "maxiter: {}\n".format(maxiter)
            + "num_samples: {}\n".format(num_samples)
            + "beta: {}\n".format(beta)
            + "q1_tol: {}\n".format(q1_tolerance)
            + "sigma: {}\n".format(SIGMA.eval())
            + "sigma_nu: {}\n\n".format(sigma_nu))
    fh.write("Number of accepted samples: {} => {}%\n".format(num_accepted,
        num_accepted/num_samples*100))
    fh.write("Number of solver failures: {} => {}%\n\n".format(solver_failures,
        solver_failures/num_samples*100))

    # save MAP estimators
    fh.write("MAP estimators functional evaluation:\n")
    for j, (me, val) in enumerate(zip(map_estimators, map_estimators_vals)):
        if me:
            # log and plot this MAP estimator
            fh.write("\t Functional = {} with center {}\n".format(val, me.centers))
            name = 'MAP_center_{}'.format(j)
            plot_q(x0, me.xs, num_landmarks, log_dir + name, nus=centers,title=str(val))

            # serialise too because we can
            po = open(log_dir + name + ".pickle", "wb")
            pickle.dump(me.xs, po)
            po.close()
    fh.close()

    # serialise results
    po = open(log_dir + "fnls.pickle", "wb")
    pickle.dump(fnls, po)
    po.close()

    po = open(log_dir + "c_samples.pickle", "wb")
    pickle.dump(c_samples, po)
    po.close()

    # plotting
    centroid_heatmap(c_samples, log_dir, x_min, x_max, y_min, y_max)
    centroid_plot(c_samples, log_dir, x_min, x_max, y_min, y_max)
    plot_autocorr(c_samples, log_dir,lag_max = len(c_samples))
    fnl_histogram(fnls, log_dir)
