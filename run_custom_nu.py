import theano
from theano import function
import theano.tensor as T
from termcolor import colored
from scipy.optimize import minimize, fmin_bfgs, fmin_cg
import numpy as np
import pickle

from lib import *

log_dir = '../tex/'

import os
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# landmark parameters
timesteps = 100

# mcmc parameters
maxiter = 7500 # shooting

def run_mm_custom_nu(q0, q1, test_name, nus, return_dict=True, plot=True):
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
    NU  = theano.shared(np.zeros([len(nus), DIM])).astype(theano.config.floatX)

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

    Kf = function([qe1, qe2], Ker(qe1, qe2))
    metf = function([qe, pe],met(qe, pe))
    Ksigmaf = function([qe1, qe2, sigma], Ksigma(qe1, qe2, sigma))
    nuf= function([qe], nu(qe))
    Hf = function([qe, pe], H(qe, pe))

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

    nu_vals = []
    for nu in nus:
        nx, ny = nu
        nu_vals.append([nx, ny])
    NU.set_value(nu_vals)

    res = shoot(q0, p0)
    xs = simf(np.array([q0, res.x.reshape([N.eval(),
        DIM])]).astype(theano.config.floatX))

    h = []
    for i in range(np.shape(xs)[0]):
        h.append(Hf(xs[i, 0], xs[i, 1]))
    h = np.array(h).sum()*dt.eval()

    if plot:
        plot_q(x0, xs, num_landmarks, log_dir + 'custom_nu_' + test_name, nus=nus)

    if return_dict:
        return {test_name: h}  # <--- should be (velocity_norm, |z|^2)!
    else:
        return h

if __name__ == "__main__":
    to_pickle = dict()
    to_pickle.update(run_mm_custom_nu(*criss_cross(num_landmarks=20), nus=[[0, 0]]))
    to_pickle.update(run_mm_custom_nu(*squeeze(num_landmarks=36),     nus=[[0, 0]]))
    to_pickle.update(run_mm_custom_nu(*pent_to_tri(num_landmarks=40), nus=[[-.5, .5]]))

    import pickle
    po = open("custom_nu.pickle", "wb")
    pickle.dump(to_pickle, po)
    po.close()
