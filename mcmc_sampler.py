import os
import time
import numpy as np
import tensorflow as tf
import scipy.io as sio
from config_cect import argparser
import utils_revise as utils
import tensorflow.contrib.slim as slim
from models_100x100 import generator as gen  #make sure consistency with models100x100 and noise case
import tensorflow_probability as tfp
t1 = time.time()
tfd= tfp.distributions
PARAMS = argparser()


np.random.seed(PARAMS.seed_no)
tf.reset_default_graph()

# parameters
N = PARAMS.n_mcmc
burn = int(PARAMS.burn_mcmc*N)

batch_size = PARAMS.batch_size

save_dir = './exps/mcmc/RetryEverythingwithoutNOISE/patient_id{}/measurednoiseVar{}_likelihoodVar{}/phasevec{}_N{}_seed{}/'.format(PARAMS.patient_id, PARAMS.meas_var, PARAMS.like_var, PARAMS.phase_vec, N, PARAMS.seed_no)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)    
    

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

"""
meas_path = './gaussian_noise/measurednoiseVar{}_seed{}'.format(PARAMS.meas_var, PARAMS.seed_no)
if not os.path.exists(meas_path):
    print('***** Measurement does not exist !! *****')
    print('** Generating one with meas_var={} **'.format(PARAMS.meas_var))
    os.makedirs(meas_path)
    img_dim = PARAMS.img_h*PARAMS.img_w*PARAMS.img_c
    noise_mat3d = np.random.multivariate_normal(mean=np.zeros((img_dim)), cov=np.eye(img_dim, img_dim)*PARAMS.noise_var, size=1).reshape(PARAMS.img_h, PARAMS.img_w, PARAMS.img_c)    
    #noise_mat3d = np.random.rand(PARAMS.img_h, PARAMS.img_w, PARAMS.img_c)-0.5
    np.save(meas_path+'/noise3d.npy', noise_mat3d)    
else:
    noise_mat3d = np.load(meas_path+'/noise3d.npy')
"""



noisy_mat3d = x_true1 #+ noise_mat3d
noisy_mat4d = np.tile(noisy_mat3d, (batch_size, 1, 1, 1)).astype(np.float32)
mask_np = np.zeros(noisy_mat4d.shape)
mask_np[:,:,:,np.nonzero(PARAMS.phase_vec)] = 1.
print(f'Shape of y_hat = {noisy_mat4d.shape}')
print(f'Shape of mask_np = {mask_np.shape}')
dim_visible = int(np.sum(mask_np))
np.save(save_dir+'/meas4d.npy', noisy_mat4d)
#R_mat = np.zeros((batch_size, 50, 50, 4), dtype=np.float32)
#R_mat[:,:,:,np.nonzero(PARAMS.phase_vec)] = 1.


# tflow graph for posterior computation
with tf.Graph().as_default() as g:
    def joint_log_prob(z):              
        gen_out = gen(z, reuse=tf.AUTO_REUSE, training=False)
        diff_img = gen_out - tf.constant(noisy_mat4d)
        visible_img =tf.boolean_mask(diff_img, mask_np)
        #gen_out_visible = tf.slice(gen_out, [0, 0], [1, 784])   
        #diff_img_visible = gen_out_visible - tf.constant(y_hat2d)      
        #diff_img = tf.reshape(tf.math.multiply(R_mat, gen_out-tf.constant(noisy_mat4d)), [dim_like])
        
        prior = tfd.MultivariateNormalDiag(loc=np.zeros(PARAMS.z_dim, dtype=np.float32), scale_diag=np.ones(PARAMS.z_dim, dtype=np.float32))
        like = tfd.MultivariateNormalDiag(loc=np.zeros(dim_visible, dtype=np.float32), scale_diag=np.sqrt(PARAMS.like_var)*np.ones(dim_visible, dtype=np.float32))
        
        return (prior.log_prob(z) + like.log_prob(visible_img))

                                          

    def unnormalized_posterior(z):
        return joint_log_prob(z) 
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                    tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=unnormalized_posterior, step_size=np.float32(1.), num_leapfrog_steps=3),
                   num_adaptation_steps=int(0.8*burn))
    

    #initial_state = tf.constant(np.zeros((PARAMS.batch_size, PARAMS.z_dim)).astype(np.float32))
    initial_state = tf.constant(np.random.normal(0,0.1,size=(PARAMS.batch_size, PARAMS.z_dim)).astype(np.float32))
    samples, [st_size, log_accept_ratio] = tfp.mcmc.sample_chain(
      num_results=N,
      num_burnin_steps=burn,
      current_state=initial_state,
      kernel=adaptive_hmc,
      trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size,
                             pkr.inner_results.log_accept_ratio])
    p_accept = tf.reduce_mean(tf.exp(tf.minimum(log_accept_ratio, 0.)))

     
    zz = tf.placeholder(tf.float32, shape=[N-burn, PARAMS.z_dim])    
    gen_out1 = gen(zz, reuse=tf.AUTO_REUSE, training=False)
    
   
    variables_to_restore = slim.get_variables_to_restore()   
    saver = tf.train.Saver(variables_to_restore)
    

with tf.Session(graph=g) as sess:    
    saver.restore(sess, PARAMS.model_path)   
     
    samples_ = sess.run(samples)
    np.save(save_dir+'/samples.npy', samples_)
    print(time.time() - t1)
    #print('HMC acceptance ratio = {}'.format(sess.run(p_accept)))
