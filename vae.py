"""

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials import mnist

A=B=40
x_dim=A*B
z_dim=2

eps=1e-9 # numerical stability


def orthogonal_initializer(scale = 1.1):
  ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
  '''
  print('Warning -- You have opted to use the orthogonal_initializer function')
  def _initializer(shape, dtype=tf.float32):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape) #this needs to be corrected to float32
    print('you have initialized one orthogonal matrix.')
    return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
  return _initializer


class NormalDistribution(object):
  """docstring for NormalDistribution"""
  def __init__(self, mu, sigma, logsigma):
    super(NormalDistribution, self).__init__()
    self.mu=mu
    self.sigma=sigma
    self.logsigma=logsigma
    
def linear(x,output_dim):
  #w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=tf.random_normal_initializer(mean=0.0, stddev=.01)) 
  w=tf.get_variable("w", [x.get_shape()[1], output_dim], initializer=orthogonal_initializer(1.1))
  b=tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
  return tf.matmul(x,w)+b

def ReLU(x,output_dim, scope):
  with tf.variable_scope(scope):
    return tf.nn.relu(linear(x,output_dim))

def encode(x):
  with tf.variable_scope("encoder"):
    for l in range(3):
      x=ReLU(x,150,"l"+str(l))
    return linear(x,4)
    #return tf.nn.relu(linear(x,z_dim))

def sampleNormal(mu,sigma):
  # note: sigma is diagonal standard deviation, not variance
  n01=tf.random_normal(mu.get_shape(), mean=0, stddev=1)
  return mu+sigma*n01

def sampleQ(h_enc):
  """
  Samples Zt ~ normrnd(mu,sigma) via reparameterization trick for normal dist
  mu is (batch,z_size)

  """
  with tf.variable_scope("sampleQ"):
    with tf.variable_scope("Q"):
      mu,log_sigma=tf.split(1,2,linear(h_enc,z_dim*2))
      sigma=tf.exp(log_sigma) # sigma_t, covariance of Q_phi
    return sampleNormal(mu,sigma), NormalDistribution(mu, log_sigma, sigma)

def decode(z):
  # with tf.variable_scope("decoder"):
  #   return tf.nn.relu(linear(z,x_dim))
  with tf.variable_scope("decoder"):
    for l in range(2):
      z=ReLU(z,200,"l"+str(l))
    return linear(z,x_dim)

def binary_crossentropy(t,o):
    return -(t*tf.log(o+eps) + (1.0-t)*tf.log(1.0-o+eps))

def recons_loss(x,x_recons):
  with tf.variable_scope("Lx"):
    return tf.reduce_sum(binary_crossentropy(x,x_recons),1) # sum across features

def latent_loss(Q):
  # KL distribution between distribution in latent space and some prior
  # (regularizer)
  with tf.variable_scope("Lz"):
    mu2=tf.square(Q.mu)
    sigma2=tf.square(Q.sigma)
    #return 0.5*tf.reduce_sum(1.+mu2+sigma2-2.*Q.logsigma,1) # sum across features
    # negative of the upper bound of posterior
    return -0.5*tf.reduce_sum(1+2*Q.logsigma-mu2-sigma2,1)

def sampleP_theta(h_dec):
  # sample x from bernoulli distribution with means p=W(h_dec)
  with tf.variable_scope("P_theta"):
    p=linear(h_dec,x_dim)
    return tf.sigmoid(p) # mean of bernoulli distribution

# BUILD NETWORK
batch_size=64
x=tf.placeholder(tf.float32, [batch_size, x_dim])
h_enc=encode(x) # encoded space
z,Q=sampleQ(h_enc) # z - latent space
#h_dec=decode(h_enc) # regular autoencoder
h_dec=decode(z) # decoded space
x_recons=sampleP_theta(h_dec) # original space

with tf.variable_scope("Loss"):
  L_x=recons_loss(x,x_recons)
  L_z=latent_loss(Q)
  loss=tf.reduce_mean(L_x)
  #loss=tf.reduce_mean(L_x+L_z) # average over minibatch -> single scalar

with tf.variable_scope("Optimizer"):
  learning_rate=1e-4
  optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.1, beta2=0.1) # beta2=0.1
  train_op=optimizer.minimize(loss)

saver = tf.train.Saver() # saves variables learned during training

# summaries
tf.scalar_summary("loss", loss)
tf.scalar_summary("L_x", tf.reduce_mean(L_x))
tf.scalar_summary("L_z", tf.reduce_mean(L_z))
all_summaries = tf.merge_all_summaries()


# TRAIN
init=tf.initialize_all_variables()
sess=tf.InteractiveSession()
sess.run(init)
# WRITER
writer = tf.train.SummaryWriter("/ltmp/vae", sess.graph_def)

# PLANE TASK
ckpt_file="vaemodel_plane.ckpt"
from plane_data2 import PlaneData
dataset=PlaneData("plane.npz","env0.png")
dataset.initialize()

# resume training
#saver.restore(sess, ckpt_file)

# # TRAIN
if True:
  train_iters=50000
  for i in range(int(train_iters)):
    (x_val,u_val,x_next_val)=dataset.sample(batch_size)
    #x_val=dataset.sample(batch_size)
    feed_dict={
      x:x_val
    }
    results=sess.run([loss,all_summaries,train_op],feed_dict)
    writer.add_summary(results[1], i) # write summary data to disk
    if i%1000==0:
      print("iter=%d : Loss: %f" % (i,results[0]))
  # save variables
  print("Model saved in file: %s" % saver.save(sess,ckpt_file))

if True:
  saver.restore(sess, ckpt_file)
  (x_val,u_val,x_next_val)=dataset.sample(batch_size)
  #x_val=dataset.sample(batch_size)
  xr=sess.run(x_recons,{x:x_val})
  fig,arr=plt.subplots(10,2)
  for i in range(10):
    arr[i,0].matshow(x_val[i,:].reshape((A,B)),cmap=plt.cm.gray, vmin=0, vmax=1)
    arr[i,1].matshow(xr[i,:].reshape((A,B)),cmap=plt.cm.gray, vmin=0, vmax=1)
  plt.show()

sess.close()