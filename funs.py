from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import Normal, MultivariateNormalTriL
from tensorflow.contrib import slim


def normal_cell(hprev, zt, H):
    return tf.ones(H)

def ar1_cell(hprev, zt, name=None, reuse=False):
    return zt

def rnn_cell(hprev, zt, name=None, reuse=False):
    """basic RNN returning next hidden state at a specific timestep."""
    nin = zt.shape[-1].value
    nout = hprev.shape[-1].value
    with tf.variable_scope(name, default_name="rnn", values=[hprev, zt], reuse=reuse):
        wz = get_variable_wrap("kernel/input", [nin, nout], dtype=tf.float32, 
                               initializer=tf.random_normal_initializer(0, 0.01))
        wh = get_variable_wrap("kernel/hidden", [nout, nout],dtype=tf.float32, 
                               initializer=tf.random_normal_initializer(0, 0.01))
        bh = get_variable_wrap("bias", [nout], dtype=tf.float32, 
                               initializer=tf.random_normal_initializer(0, 0.01))
    
    return tf.tanh(ed.dot(hprev, wh) + ed.dot(zt, wz) + bh)
    
def lstm_cell(x, h, c, name=None, reuse=False):
    """LSTM returning hidden state and content cell at a specific timestep."""
    nin = x.shape[-1].value
    nout = h.shape[-1].value
    with tf.variable_scope(name, default_name="lstm", values=[x, h, c], reuse=reuse):
        wx = get_variable_wrap("kernel/input", [nin, nout * 4], dtype=tf.float32, 
                               initializer=tf.orthogonal_initializer(1.0))
        wh = get_variable_wrap("kernel/hidden", [nout, nout * 4],dtype=tf.float32,
                               initializer=tf.orthogonal_initializer(1.0))
        b = get_variable_wrap("bias", [nout * 4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

    z = ed.dot(x, wx) + ed.dot(h, wh) + b
    i, f, o, u = tf.split(z, 4, axis=0)
    i = tf.sigmoid(i)
    f = tf.sigmoid(f + 1.0)
    o = tf.sigmoid(o)
    u = tf.tanh(u)
    c = f * c + i * u
    h = o * tf.tanh(c)
    return h, c

def lstm_cell_1(x, h, c, name=None, reuse=False):
    """LSTM returning hidden state and content cell at a specific timestep."""
    nin = x.shape[-1].value
    nout = h.shape[-1].value
    with tf.variable_scope(name, default_name="lstm_1", values=[x, h, c], reuse=reuse):
        wx = get_variable_wrap("kernel/input", [nin, nout * 4], dtype=tf.float32, 
                               initializer=tf.orthogonal_initializer(1.0))
        wh = get_variable_wrap("kernel/hidden", [nout, nout * 4],dtype=tf.float32,
                               initializer=tf.orthogonal_initializer(1.0))
        b = get_variable_wrap("bias", [nout * 4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

    z = ed.dot(x, wx) + ed.dot(h, wh) + b
    i, f, o, u = tf.split(z, 4, axis=0)
    i = tf.sigmoid(i)
    f = tf.sigmoid(f + 1.0)
    o = tf.sigmoid(o)
    u = tf.tanh(u)
    c = f * c + i * u
    h = o * tf.tanh(c)
    return h, c

def lstm_cell_2(x, h, c, name=None, reuse=False):
    """LSTM returning hidden state and content cell at a specific timestep."""
    nin = x.shape[-1].value
    nout = h.shape[-1].value
    with tf.variable_scope(name, default_name="lstm_2", values=[x, h, c], reuse=reuse):
        wx = get_variable_wrap("kernel/input", [nin, nout * 4], dtype=tf.float32, 
                               initializer=tf.orthogonal_initializer(1.0))
        wh = get_variable_wrap("kernel/hidden", [nout, nout * 4],dtype=tf.float32,
                               initializer=tf.orthogonal_initializer(1.0))
        b = get_variable_wrap("bias", [nout * 4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

    z = ed.dot(x, wx) + ed.dot(h, wh) + b
    i, f, o, u = tf.split(z, 4, axis=0)
    i = tf.sigmoid(i)
    f = tf.sigmoid(f + 1.0)
    o = tf.sigmoid(o)
    u = tf.tanh(u)
    c = f * c + i * u
    h = o * tf.tanh(c)
    return h, c

def encode_z(hprev, L, name=None, reuse=False):
    # input: hprev should change to [#batch, dim] 
    #hprev = tf.expand_dims(hprev, 0)
    #hidden_dim = 15
    #with tf.variable_scope("prior"):
     #       prior = fc_act(hprev, hidden_dim, act=tf.nn.relu, name="fc_prior")
    #with tf.variable_scope("prior_mu"):
     #       prior_mu = fc_act(prior, L, name="fc_prior_mu")
    #with tf.variable_scope("prior_sigma"):
     #       prior_sigma = fc_act(prior, L, act=tf.nn.softplus, name="fc_prior_sigma")
    #zt = Normal(loc=tf.squeeze(prior_mu, 0), scale = tf.squeeze(prior_sigma, 0))
    
    #AR1 cell using difussion process: z_t = z_t-1 + eta
    #zt = Normal(hprev, 0.1)
   
    # NN for encoding ht -> mu_zt, sigma_zt
    H = hprev.shape[0]
    
    with tf.variable_scope(name, default_name="encode_z", reuse=reuse):
        Whz_mean = get_variable_wrap("Wmean", [H, L], dtype=tf.float32, 
                               initializer=tf.constant_initializer(0.0))
        bhz_mean = get_variable_wrap("bmean", [L], dtype=tf.float32, 
                               initializer=tf.constant_initializer(0.0))
        Whz_cov = get_variable_wrap("Wvar", [H, L], dtype=tf.float32, 
                               initializer=tf.constant_initializer(0.0))
        bhz_cov = get_variable_wrap("bvar", [L], dtype=tf.float32, 
                               initializer=tf.constant_initializer(0.0))
    
    #Whz_mean = tf.Variable(np.zeros([H, L]), dtype=tf.float32)
    #bhz_mean = tf.Variable(np.zeros(L), dtype=tf.float32)   
    #Whz_cov  = tf.Variable(np.zeros([H, L]), dtype=tf.float32) 
    #bhz_cov  = tf.Variable(np.zeros(L), dtype=tf.float32)
    
    zt = Normal(loc=ed.dot(hprev, Whz_mean) + bhz_mean, 
                scale=tf.nn.softplus(ed.dot(hprev, Whz_cov) + bhz_cov))
    return zt

def encode_z_ar1(hprev, L):
    H = hprev.shape[0]
    var = tf.Variable(np.ones([H]), dtype=tf.float32)
    zt = Normal(hprev, var)
    return zt

def get_variable_wrap(*args, **kwargs):
    try:
        return tf.get_variable(*args, **kwargs)
    except ValueError:
        tf.get_variable_scope().reuse_variables()
        return tf.get_variable(*args, **kwargs)

def fc_act(x, next_layer_size, act=None, name="fc"):
    nbatches = x.get_shape()[0]
    prev_layer_size = x.get_shape()[1]
    with tf.name_scope("fc"):
        w = get_variable_wrap("weights", [prev_layer_size, next_layer_size], 
                              dtype=tf.float, initializer=tf.random_normal_initializer())
        b = get_variable_wrap("bias", [next_layer_size], 
                              dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        o = tf.add(tf.matmul(x, w), b)
        if act: return act(o)
        else: return o

def neural_network(z, dim_out):
    """neural network model for mapping"""
    hidden_dim = 15
    net1 = slim.fully_connected(z, hidden_dim, activation_fn=None)
    net2 = slim.fully_connected(net1, dim_out, activation_fn=tf.tanh)
    return net2

def compute_optimal_rotation(L, L_true, scale=True):
    """Find a rotation matrix R such that F_inf.dot(R) ~= F_true"""
    from scipy.linalg import orthogonal_procrustes
    R = orthogonal_procrustes(L, L_true)[0]

    if scale:
        Lp = L.dot(R)
        s = (L_true*Lp).sum() / (Lp*Lp).sum()
        return R*s
    else:
        return R
    
def match_z(x,z):
    cp = np.corrcoef(x.T,z.T)[0,1]
    cn = np.corrcoef(-x.T,z.T)[0,1]
    if cp<cn:
        return -x
    else:
        return x

def dyn_lorenz(T, dt=0.01):

    stepCnt = T
    
    def lorenz(x, y, z, s=10, r=28, b=2.667):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return x_dot, y_dot, z_dot
    
    # Need one more for the initial values
    xs = np.empty((stepCnt + 1,))
    ys = np.empty((stepCnt + 1,))
    zs = np.empty((stepCnt + 1,))

    # Setting initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)

    # Stepping through "time".
    for i in range(stepCnt):
    # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    
    z = np.zeros((T, 3))
    z[:,0] = xs[:-1]
    z[:,1] = ys[:-1]
    z[:,2] = zs[:-1]
    return z
    
def dyn_sine(T, N, L):
    x = np.arange(T)
    z_true = np.zeros((N,L))
    z_true[:,0] = 0.5 * np.cos(2 * np.pi * x / 300 + np.pi)
    z_true[:,1] = 0.5 * np.cos(2 * np.pi * x / 25 + 1/3 * np.pi)
    z_true[:,2] = 0.5 * np.cos(2 * np.pi * x / 25 + 2/3 * np.pi)
    return z_true

def map_linear(z, L, D):
    Wz_true = np.random.normal(0, 1,[L,D])
    bz_true = np.random.normal(0, 1,[D])
    mu = np.dot(z, Wz_true)
    x = np.random.normal(mu, 0.1)
    return x

def map_sine(z, L, D):
    Wz_true = np.random.normal(0, 1,[L,D])
    bz_true = np.random.normal(0, 1,[D])
    mu = np.dot(z, Wz_true)
    x  = np.random.normal(np.sin(mu), 0.1)
    return x

def map_tanh(z, L, D):
    Wz_true = np.random.normal(0, 1,[L,D])
    bz_true = np.random.normal(0, 1,[D])
    mu = np.dot(z, Wz_true)
    x  = np.tanh(mu)
    return x

# define kernel 
def kernel_fun(X, X2=None, lengthscale=1.0, variance=1.0, name=None):
    from tensorflow.python.ops import control_flow_ops
    
    lengthscale = tf.convert_to_tensor(lengthscale)
    variance = tf.convert_to_tensor(variance)
    dependencies = [tf.assert_positive(lengthscale), tf.assert_positive(variance)]
    lengthscale = control_flow_ops.with_dependencies(dependencies, lengthscale)
    variance = control_flow_ops.with_dependencies(dependencies, variance)

    X = tf.convert_to_tensor(X)
    X = X / lengthscale
    Xs = tf.reduce_sum(tf.square(X), 1)
    if X2 is None:
        X2 = X
        X2s = Xs
    else:
        X2 = tf.convert_to_tensor(X2)
        X2 = X2 / lengthscale
        X2s = tf.reduce_sum(tf.square(X2), 1)

    r = tf.reshape(Xs, [-1, 1]) + tf.reshape(X2s, [1, -1]) - \
      2 * tf.matmul(X, X2, transpose_b=True)
    
    output = {
        'rbf': lambda r: variance * tf.exp(-r / 2),
        'matern32': lambda r: variance * (1. + np.sqrt(3.) * tf.sqrt(r + 1e-6)) * \
                                          tf.exp(-np.sqrt(3.) * tf.sqrt(r + 1e-6)),
        'matern52': lambda r: variance * (1. + np.sqrt(5.) * tf.sqrt(r + 1e-6) + \
                                          5./3. * (r + 1e-6)) * tf.exp(-np.sqrt(5.) * tf.sqrt(r + 1e-6)),
        'cosine': lambda r: variance * tf.cos(tf.sqrt(r + 1e-6))
    }[name](r)
    return output