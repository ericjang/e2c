#!/usr/bin/env python

import matplotlib.pyplot as plt

import numpy as np
from numpy.random import randint
import os
from dataset import DataSet

num_t=80 # number of trajectories (i.e. number of initial states)
T=1000 # length of each trajectory sequence
u_dim=2 # control (action) dimension
w,h=40,40
x_dim=w*h
rw=1 # robot half-width

def get_params():
  return x_dim,u_dim,T

class PlaneData(DataSet):
  def __init__(self, fname, env_file):
    super(PlaneData, self).__init__()
    self.cache=fname
    self.initialized=False
    self.im=plt.imread(env_file) # grayscale
    self.params=(x_dim,u_dim,T)

  def is_colliding(self,p):
    if np.any([p-rw<0, p+rw>=w]):
      return True
    # check robot body overlap with obstacle field
    return np.mean(self.im[p[0]-rw:p[0]+rw+1, p[1]-rw:p[1]+rw+1]) > 0.05

  def compute_traj(self, max_dist=1):
    # computes P,U data for single trajectory
    # all P,U share the same environment obstacles.png
    P=np.zeros((T,2),dtype=np.int) # r,c position
    U=np.zeros((T,u_dim),dtype=np.int)
    P[0,:]=[rw,randint(rw,w-rw)] # initial location
    for t in range(1,T):
      p=np.copy(P[t-1,:])
      # dr direction
      d=randint(-1,2) # direction
      nsteps=randint(max_dist+1)
      dr=d*nsteps # applied control
      for i in range(nsteps):
        p[0]+=d
        if self.is_colliding(p):
          p[0]-=d
          break
      # dc direction
      d=randint(-1,2) # direction
      nsteps=randint(max_dist+1)
      dc=d*nsteps # applied control
      for i in range(nsteps):
        p[1]+=d
        if self.is_colliding(p):
          p[1]-=d # step back
          break
      P[t,:]=p
      U[t,:]=[dr,dc]
    return P,U

  def initialize(self):
    if os.path.exists(self.cache):
      self.load()
    else:
      self.precompute()
    self.initialized=True

  def compute_data(self):
    # compute multiple trajectories
    P=np.zeros((num_t,T,2),dtype=np.int)
    U=np.zeros((num_t,T,u_dim),dtype=np.int)
    for i in range(num_t):
      P[i,:,:], U[i,:,:] = self.compute_traj(max_dist=2)
    return P,U

  def precompute(self):
    print("Precomputing P,U...")
    self.P, self.U = self.compute_data()

  def save(self):
    print("Saving P,U...")
    np.savez(self.cache, P=self.P, U=self.U)

  def load(self):
    print("Loading P,U from %s..." % (self.cache))
    D=np.load(self.cache)
    self.P, self.U = D['P'], D['U']

  def getXp(self,p):
    # return image X given true state p (position) of robot
    x=np.copy(self.im)
    x[p[0]-rw:p[0]+rw+1, p[1]-rw:p[1]+rw+1]=1. # robot is white on black background
    return x.flat

  def getX(self,i,t):
    # i=trajectory index, t=time step
    return self.getXp(self.P[i,t,:])

  def getXTraj(self,i):
    # i=traj index
    X=np.zeros((T,x_dim),dtype=np.float)
    for t in range(T):
      X[t,:]=self.getX(i,t)
    return X

  def sample(self, batch_size):
    """
    computes (x_t,u_t,x_{t+1}) pair
    returns tuple of 3 ndarrays with shape
    (batch,x_dim), (batch, u_dim), (batch, x_dim)
    """
    if not self.initialized:
      raise ValueError("Dataset not loaded - call PlaneData.initialize() first.")
    traj=randint(0,num_t,size=batch_size) # which trajectory
    tt=randint(0,T-1,size=batch_size) # time step t for each batch
    X0=np.zeros((batch_size,x_dim))
    U0=np.zeros((batch_size,u_dim),dtype=np.int)
    X1=np.zeros((batch_size,x_dim))
    for i in range(batch_size):
      t=tt[i]
      p=self.P[traj[i], t, :]
      X0[i,:]=self.getX(traj[i],t)
      X1[i,:]=self.getX(traj[i],t+1)
      U0[i,:]=self.U[traj[i], t, :]
    return (X0,U0,X1)

  def getPSpace(self):
    """
    Returns all possible positions of agent
    """
    ww=h-2*rw
    P=np.zeros((ww*ww,2)) # max possible positions
    i=0
    p=np.array([rw,rw]) # initial location
    for dr in range(ww):
      for dc in range(ww):
        if not self.is_colliding(p+np.array([dr,dc])):
          P[i,:]=p+np.array([dr,dc])
          i+=1
    return P[:i,:]

  def getXPs(self, Ps):
    X=np.zeros((Ps.shape[0],x_dim))
    for i in range(Ps.shape[0]):
      X[i,:]=self.getXp(Ps[i,:])
    return X

if __name__ == "__main__":
  import matplotlib.animation as animation
  p=PlaneData("plane2.npz","env1.png")
  p.initialize()
  p.save()
  im=p.im
  A,B=im.shape

  # show sample tuples
  if True:
    fig, aa = plt.subplots(1,2)
    x0,u0,x1=p.sample(2)
    m1=aa[0].matshow(x0[0,:].reshape(w,w), cmap=plt.cm.gray, vmin = 0., vmax = 1.)
    aa[0].set_title('x(t)')
    m2=aa[1].matshow(x1[0,:].reshape(w,w), cmap=plt.cm.gray, vmin = 0., vmax = 1.)
    aa[1].set_title('x(t+1), u=(%d,%d)' % (u0[0,0],u0[0,1]))
    fig.tight_layout()
    def updatemat2(t):
      x0,u0,x1=p.sample(2)
      m1.set_data(x0[0,:].reshape(w,w))
      m2.set_data(x1[0,:].reshape(w,w))
      return m1,m2

    anim=animation.FuncAnimation(fig, updatemat2, frames=100, interval=1000, blit=True, repeat=True)

    Writer = animation.writers['imagemagick'] # animation.writers.avail
    writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
    anim.save('sample_obs.gif', writer=writer)

  #show trajectory
  if True:
    fig, ax = plt.subplots()
    X=p.getXTraj(0)
    mat=ax.matshow(X[0,:].reshape((A,B)), cmap=plt.cm.gray, vmin = 0., vmax = 1.)
    def updatemat(t):
      mat.set_data(X[t,:].reshape((A,B)))
      return mat,
    anim = animation.FuncAnimation(fig, updatemat, frames=T-1, interval=30, blit=True, repeat=True)
    plt.show()
