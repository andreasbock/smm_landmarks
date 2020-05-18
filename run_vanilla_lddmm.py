import theano
from theano import pp, function
import theano.tensor as T
from termcolor import colored
from scipy.optimize import minimize,fmin_bfgs,fmin_cg
import numpy as np
import time
import pylab as plt
from lib import *

log_dir = '../tex/'

import os
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# initialize template and target, q0 and q1
def run_lddmm(q0, q1, test_name):
    # landmark parameters
    DIM = 2 # dimension of the image
    SIGMA = theano.shared(np.array(.7).astype(theano.config.floatX)) # radius of the landmark

    num_landmarks = q0.shape[0]
    N = theano.shared(num_landmarks) # number of landmarks
    q1_theano = theano.shared(np.zeros([N.eval(),DIM]).astype(theano.config.floatX))
    q1_theano.set_value(q1.astype(theano.config.floatX))

    # timestepping
    Tend = 1 # final time
    n_steps = theano.shared(200) # number of timesteps
    dt = Tend/n_steps # timesteps

    # general kernel for any sigma
    def Ksigma(q1, q2, sigma):
        r_sq = T.sqr(q1.dimshuffle(0, 'x', 1) - q2.dimshuffle('x', 0, 1)).sum(2)
        return T.exp( - r_sq / (2.*sigma**2) )

    # kernel function for the landmarks
    def Ker(q1, q2):
        return Ksigma(q1, q2, SIGMA)

    def H(q,p):
        return 0.5*T.tensordot(p, T.tensordot(Ker(q, q), p, [[1], [0]]), [[0, 1],
            [0, 1]])

    # compile the previous functions
    pe = T.matrix('p')
    qe = T.matrix('q')
    qe1 = T.matrix('q')
    qe2 = T.matrix('q')
    sigma = T.scalar()

    Kf = function([qe1, qe2], Ker(qe1, qe2))
    Ksigmaf = function([qe1, qe2, sigma], Ksigma(qe1, qe2, sigma))
    Hf = function([qe, pe], H(qe, pe))

    # compute gradients
    dq = lambda q,p: T.grad(H(q, p), p)
    dp = lambda q,p: -T.grad(H(q, p), q)
    dqf = function([qe, pe], dq(qe, pe))
    dpf = function([qe, pe], dp(qe, pe))

    # horizontal initial momentum
    p0 = 1./N.eval()*np.vstack((np.zeros(N.eval()), np.ones(N.eval()))).T
    p0 = np.array(int(N.get_value())*[[0.1, 0]])

    # create the initial condition array
    q0 = q0.astype(theano.config.floatX)
    p0 = p0.astype(theano.config.floatX)
    x0 = np.array([q0, p0]).astype(theano.config.floatX)

    # ode to solve (Hamiltonian system)
    def ode_f(x):

        dqt = dq(x[0], x[1])
        dpt = dp(x[0], x[1])

        return T.stack((dqt, dpt))

    # Forward Euler scheme
    def euler(x,dt):
        return x + dt*ode_f(x)

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
        maxiter = 5000
        def fopts(x):
            [y,gy] = dlossf(np.stack([q0,x.reshape([N.eval(), DIM])]).astype(theano.config.floatX),)
            return (y,gy[1].flatten().astype(np.double))

        res = minimize(fopts, p0.flatten(), method='L-BFGS-B', jac=True,
            options={'disp': False, 'maxiter': maxiter})

        return (res.x, res.fun)

    res = shoot(q0, p0)
    xs = simf(np.array([q0,res[0].reshape([N.eval(),DIM])]).astype(theano.config.floatX))
    h = []
    for i in range(np.shape(xs)[0]):
        h.append(Hf(xs[i,0], xs[i,1]))
    h = np.array(h).sum()*dt.eval()

    plot_q(x0, xs, num_landmarks, log_dir + 'lddmm_' + test_name)

    return {test_name: (h, float(res[1]))}

to_pickle = dict()
#to_pickle.update(run_lddmm(*pent_to_tri(num_landmarks=40)))
to_pickle.update(run_lddmm(*criss_cross(num_landmarks=20)))
to_pickle.update(run_lddmm(*squeeze(num_landmarks=36)))
#to_pickle.update(run_lddmm(*pringle(num_landmarks=16)))
#to_pickle.update(run_lddmm(*triangle_flip(num_landmarks=15)))

import pickle
po = open("lddmm.pickle", "wb")
pickle.dump(to_pickle, po)
po.close()
