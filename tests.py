
"""
test functions like KLGaussian to make sure you implemented correctly
"""


import numpy as np

# ground truth implementation
from divergence import gau_kl

pm=np.array([1.,1.,1.],dtype=np.float32) # true
pv=np.array([0.1,0.3,0.5],dtype=np.float32) # diagonal covariance
qm=np.array([0.,0.,0.],dtype=np.float32)
qv=np.array([1.,1.,1.],dtype=np.float32)

KL,a,b,c= gau_kl(pm, pv, qm, qv) # assumes diagonal covariances...
print('KL : %f' % (KL))
print('trace term : %f' % (a))
print('difference of means : %f' % (b))
print('ratio of determinants : %f' % (c))

# my implementation

import tensorflow as tf
from e2c import NormalDistribution, KLGaussian
batch_size=1
z_dim=3

I=tf.identity(np.tile(np.eye(z_dim,dtype=np.float32),[batch_size, 1, 1])) # identity matrix (batch_size, z_dim, z_dim)
zero_z=tf.constant(0.,shape=[batch_size,z_dim])

pm=pm.reshape((batch_size,z_dim))
pv=pv.reshape((batch_size,z_dim))
qm=qm.reshape((batch_size,z_dim))
qv=qv.reshape((batch_size,z_dim))

pmu=tf.constant(pm,shape=[batch_size,z_dim])
psigma=tf.constant(np.sqrt(pv))
P=NormalDistribution(pmu,psigma,tf.log(psigma),zero_z,zero_z)

qmu=tf.constant(qm,shape=[batch_size,z_dim])
qsigma=tf.constant(np.sqrt(qv))
Q=NormalDistribution(qmu,qsigma,tf.log(qsigma),zero_z,zero_z)

# sigma0=tf.constant([0.1,0.3,0.5])
# P=NormalDistribution(one_z, sigma0, tf.log(sigma0), zero_z, zero_z)
# Pz=NormalDistribution(zero_z, one_z, zero_z, zero_z, zero_z)# prior on Q_phi = mean 0, unit variance => logsigma=0

KL,a,b,c=KLGaussian(P,Q,"tmp")

sess=tf.InteractiveSession()

results=sess.run([KL,a,b,c])
print('KL : %f' % (results[0]))
print('trace term : %f' % (results[1]))
print('difference of means : %f' % (results[2]))
print('ratio of determinants : %f' % (results[3]))

sess.close()



