print('1')
import os
import time
import numpy as np
import tensorflow as tf
import scipy.io as sio
from config_cect import argparser
import utils_revise as utils
import tensorflow.contrib.slim as slim
from models_100x100 import generator as gen
import tensorflow_probability as tfp
from numpy import linalg as LA
t1 = time.time()
test=1
print(test)
tfd= tfp.distributions

PARAMS = argparser()
print(test)
np.random.seed(PARAMS.seed_no)
tf.reset_default_graph()
#tf.compat.v1.reset_default_graph
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

plt.close('all')
plt.rcParams.update({'font.size': 22})

from sewar.full_ref import uqi

PARAMS = argparser()
np.random.seed(PARAMS.seed_no)
N = PARAMS.n_mcmc #np.size(mcmc_samps, 0)
burn = int(PARAMS.burn_mcmc*N)


sample_dir = './exps/mcmc/RetryEverything/patient_id{}/measurednoiseVar{}_likelihoodVar{}/phasevec{}_N{}_seed{}/'.format(PARAMS.patient_id, PARAMS.meas_var, PARAMS.like_var, PARAMS.phase_vec, N, PARAMS.seed_no)


mcmc_totalsamps = np.load(sample_dir + '/samples.npy')



#x=[32000, 64000, 96000, 128000, 160000,192000,224000,256000] added
x=[	64000,		128000,	192000,		256000,	320000,		384000,		448000,		512000,	576000,	640000,	704000,		768000,		832000,	 896000,		960000,		1024000]
#x=[10240000]

norm_xmap=np.zeros((len(x),4))
norm_xmean=np.zeros((len(x),4))
norm_std=np.zeros((len(x),4))
print(x)
x= [1024000]


iteration=0
for samplenumber in (x):   #added
  N=samplenumber//1  #added
  print(N)
  mcmc_samps=mcmc_totalsamps[0:N,:,:]  #added
  burn = int(PARAMS.burn_mcmc*N)
  #burn= 128000
  afterburning= mcmc_samps[burn:,:,:] #new
  eff_samps=np.squeeze(mcmc_samps[burn:,:,:])
  #eff_samps = np.squeeze(afterburning[::60000,:,:])
  
  n_eff = N-burn
  
  #n_eff=len(afterburning[::60000,:,:])
  
  print('n_eff is')
  print(n_eff)
  
  batch_size = PARAMS.batch_size
  n_iter = int(n_eff/batch_size)
  
  
  # histogram of first 25 components of posterior
  plt.figure(figsize=(20, 20))
  for ii in range(25):
      plt.subplot(5,5,ii+1)
      plt.hist(eff_samps[:, ii], 50, density=True);
      plt.xlabel('z_{}'.format(ii))
  plt.tight_layout()
  plt.savefig('./{}/hist_eff_samples25_{}'.format(sample_dir,N))   #fixed
  # trace plot of first 25 components of posterior
  plt.figure(figsize=(20, 20))
  for ii in range(25):
      plt.subplot(5,5,ii+1)
      plt.plot(mcmc_samps[:, 0, ii])
      plt.ylabel('z_{}'.format(ii))
  plt.tight_layout()
  plt.savefig('./{}/trace_eff_samples25_{}'.format(sample_dir,N))  #fixed
  #plt.show()
  
  '''data'''
  def preprocess_fn(img):
      data=sio.loadmat(img)
      axial=2*data['features_axial']-np.ones(data['features_axial'].shape)
      #coronal=2*data['features_coronal']-np.ones(data['features_axial'].shape)
      #sagittal=2*data['features_sagittal']-np.ones(data['features_axial'].shape)
      axial_test=axial[3000:,:,:,:]
      #coronal_train=coronal[0:2100,:,:,:]
      #sagittal_train=sagittal[0:2100,:,:,:]
      
      test_set = axial_test   #np.vstack((axial_train,coronal_train,sagittal_train))
      img = tf.to_float(test_set)
      return img
  
  
  img_paths="./Data/data_n421_c2_fix_norm.mat" 
  data_pool = utils.DiskImageData(img_paths, batch_size, shuffle=False, preprocess_fn=preprocess_fn) 
  print(len(data_pool))
  
  x_true1 = data_pool.batch()[0,:,:,:][PARAMS.patient_id]
  print(x_true1.shape)
  meas4d = np.load(sample_dir+'/meas4d.npy')
  mask_np = np.zeros((batch_size, PARAMS.img_h, PARAMS.img_w, PARAMS.img_c))
  mask_np[:,:,:,np.nonzero(PARAMS.phase_vec)] = 1.
  
  
  
  z_mean = np.mean(eff_samps, axis=0)
  # tflow graph 
  with tf.Graph().as_default() as g:
      z1 = tf.placeholder(tf.float32, shape=[batch_size, PARAMS.z_dim])         
      gen_out = gen(z1, reuse=tf.AUTO_REUSE, training=False)
      diff_img = gen_out - tf.constant(meas4d)
      visible_img = tf.boolean_mask(diff_img, mask_np)
      #gen_out_visible = tf.slice(gen_out, [0,0], [batch_size, 784])
      #diff_img_visible = gen_out_visible - tf.constant(y_hat2d)    
      
  
      variables_to_restore = slim.get_variables_to_restore()   
      saver = tf.train.Saver(variables_to_restore)
  
  
  with tf.Session(graph=g) as sess:
      saver.restore(sess, PARAMS.model_path)    
      loss = np.zeros((n_eff))
      x_mean = np.zeros((PARAMS.img_h, PARAMS.img_w, PARAMS.img_c))
      x2_mean = np.zeros((PARAMS.img_h, PARAMS.img_w, PARAMS.img_c))    
      for ii in range(n_iter):
          g_z, diff = sess.run([gen_out, visible_img], feed_dict={z1:eff_samps[ii*batch_size:(ii+1)*batch_size, :]})
          x_mean = x_mean + np.mean(g_z, axis=0)
          x2_mean = x2_mean + np.mean(g_z**2, axis=0)
          for kk in range(batch_size):
              loss[(ii*batch_size)+kk] = 0.5*np.linalg.norm(diff)**2 + 0.5*PARAMS.like_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2
  
      x_mean = x_mean/n_iter
      x2_mean = x2_mean/n_iter    
      std = np.sqrt(np.maximum((x2_mean - (x_mean)**2), 0))
      print("*********************")
      print(f"max var = {np.max(std)} min var = {np.min(std)}")
  
      map_ind = np.argmin(loss)
      x_map = sess.run(gen_out, feed_dict={z1: np.tile(np.expand_dims(eff_samps[map_ind,:], axis=0), (batch_size, 1))})  
      g_zmean = sess.run(gen_out, feed_dict={z1: np.tile(z_mean, (batch_size, 1))})
  
  
      rec_error = np.linalg.norm(x_map[0,:,:,:]-meas4d[0,:,:,:])/(PARAMS.img_h*PARAMS.img_w*PARAMS.img_c)
      #mask_ = np.zeros((PARAMS.img_h, PARAMS.img_w, PARAMS.img_c))
      #mask_[:,:,np.nonzero(PARAMS.phase_vec)] = 1.
      #mx = np.ma.masked_array(meas4d[0,:,:,:], mask=mask_) 
  
      print('1')
      # some stats
  
      def figure_helper(arr, title, lim):
          fig, axs = plt.subplots(2,2,figsize=(20,20))
          norms=np.zeros(4)
          i=0
          for ii in range(2):
              for jj in range(2):
                  if lim==1:
                      im = axs[ii,jj].imshow(arr[:,:,ii*2+jj], vmin=-1, vmax=1, cmap='coolwarm')
                      
                  elif lim==2:
                      im = axs[ii,jj].imshow(arr[:,:,ii*2+jj], vmin=np.min(arr), vmax=np.max(arr), cmap='inferno')
                      axs[ii,jj].set_title(f'std. dev = {np.mean(arr[:,:,ii*2+jj]):.6f}')   #we fixed std
                      norms[i]=LA.norm(arr[:,:,ii*2+jj])#added
                  else:
                      im = axs[ii,jj].imshow(arr[:,:,ii*2+jj], vmin=np.min(arr), vmax=np.max(arr), cmap='inferno')
                  i=i+1
                      #axs[ii,jj].set_title(r'$|arr|_{{L1}}/|xtrue|_{{L1}}$={:.5f} and $|arr|_{{L2}}/|xtrue|_{{L2}}$={:.5f}'.format(np.sum(np.abs(arr)[:,:,ii*2+jj])/(np.sum(np.abs(x_true[:,:,ii*2+jj]))), np.linalg.norm(arr[:,:,ii*2+jj])/(np.linalg.norm(x_true[:,:,ii*2+jj]))))
  
                  fig.colorbar(im, ax=axs[ii,jj])
          plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
          plt.savefig(sample_dir+'/{}.png'.format(title))
          return norms
      
      def figure_helper_withsubtitles(arr, title, lim):
          fig, axs = plt.subplots(2,2,figsize=(20,20))
          for ii in range(2):
              for jj in range(2):
                  if lim==1:
                      im = axs[ii,jj].imshow(arr[:,:,ii*2+jj], vmin=-1, vmax=1)
                  else:
                      im = axs[ii,jj].imshow(arr[:,:,ii*2+jj], vmin=np.min(arr), vmax=np.max(arr))
                  fig.colorbar(im, ax=axs[ii,jj])
          plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
          plt.savefig(sample_dir+'/{}.png'.format(title))
      
  
  normsofx_true=figure_helper(x_true1[:,:,:], 'x_true_{}'.format(N), 1)  #fixed
  normsOfmap=figure_helper(x_map[0,:,:,:], 'x_map_{}_coolwarm'.format(N), 1)  #fixed
 # normsOfmean=figure_helper(x_mean, 'x_mean_{}'.format(N), 1)
  #figure_helper(g_zmean[0,:,:,:], 'g(z_mean)_{}'.format(N), 1)   #fixed
 # figure_helper(x_map[0,:,:,:]-x_true1[:,:,:], 'diff_{}_{}'.format(rec_error,N), 100)   #fixed
 # normsOfstd=figure_helper(std, 'std(x)_{}'.format(N), 2)  #fixed
  #figure_helper(meas4d[0,:,:,:], 'y_{}'.format(N), 100)  #fixed
      
      
  #norm_xmap[iteration,:]=normsOfmap  #added
  #norm_xmean[iteration,:]=normsOfmean
  #norm_std[iteration,:]=normsOfstd
      
  
      #np.save(sample_dir+'/x_true.npy', x_true1)      
  np.save(sample_dir+'/x_std_{}.npy'.format(N), std)  #fixed
  np.save(sample_dir+'/x_mean_{}.npy'.format(N), x_mean)  #fixed
  np.save(sample_dir+'/x_map_{}.npy'.format(N), x_map[0,:,:,:])  #fixed
      #np.save(sample_dir+'/g_zmean_{}.npy'.format(N), g_zmean[0,:,:,:])  
      #np.save(sample_dir+'/mask_.npy', mask_[0,:,:,:])
  iteration=iteration+1

#np.save(sample_dir+'/norm_xmap.npy', norm_xmap)  #added
#np.save(sample_dir+'/norm_xmean.npy', norm_xmean)
#np.save(sample_dir+'/norm_std.npy', norm_std)

