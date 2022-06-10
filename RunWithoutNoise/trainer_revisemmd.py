from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import glob
import utils_revise as utils #change
import traceback
import numpy as np
import tensorflow as tf
import scipy.spatial
import matplotlib.pyplot as plt
import models_100x100 as models
import scipy.io as sio #add
from PIL import Image#add
from config_cect import argparser#add
plt.rcParams.update({'font.size':20})

""" param """
epoch = 895
batch_size = 64
lr = 0.0002
z_dim = 100
n_critic = 5
gpu_id = 1

tf.reset_default_graph()
PARAMS = argparser()
np.random.seed(PARAMS.seed_no)
''' data '''

def preprocess_fn(img):
    data=sio.loadmat(img)
    axial=2*data['features_axial']-np.ones(data['features_axial'].shape)
   
    axial_train=axial[0:3000,:,:,:]
    #coronal_train=coronal[0:2100,:,:,:]
    #sagittal_train=sagittal[0:2100,:,:,:]
    
    train_set = axial_train   #np.vstack((axial_train,coronal_train,sagittal_train))
    img = tf.to_float(train_set)
    return img


img_paths="./Data/data_n421_c2_fix_norm.mat" 
data_pool = utils.DiskImageData(img_paths, batch_size, preprocess_fn=preprocess_fn) 


""" graphs """
with tf.device('/gpu:%d' % gpu_id):
    ''' models '''
    generator = models.generator
    discriminator = models.discriminator_wgan_gp

    ''' graph '''
    # inputs
    real = tf.placeholder(tf.float32, shape=[None, 100, 100, 4]) #change dimension:shape=[None, 64, 64, 3]
    z = tf.placeholder(tf.float32, shape=[None, z_dim])
    #dummy = tf.Variable(name='dummy', trainable=False, initial_value=z)
    # generate
    fake = generator(z, reuse=False)

    # dicriminate
    r_logit, r_g0, r_g1, r_g2, r_g3, r_g4, r_g5, r_g6  = discriminator(real, reuse=False)
    f_logit, f_g0, f_g1, f_g2, f_g3, f_g4, f_g5, f_g6  = discriminator(fake)

    # losses
    def gradient_penalty(real, fake, f):
        def interpolate(a, b):
            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

        x = interpolate(real, fake)
        pred = f(x)
        gradients = tf.gradients(pred, x)[0]
        slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp

    wd = tf.reduce_mean(r_logit) - tf.reduce_mean(f_logit)
    gp = gradient_penalty(real, fake, discriminator)
    style_diff0 = tf.reduce_mean(tf.linalg.norm(r_g0 - f_g0, axis=(1,2))**2) 
    style_diff1 = tf.reduce_mean(tf.linalg.norm(r_g1 - f_g1, axis=(1,2))**2) 
    style_diff2 = tf.reduce_mean(tf.linalg.norm(r_g2 - f_g2, axis=(1,2))**2) 
    style_diff3 = tf.reduce_mean(tf.linalg.norm(r_g3 - f_g3, axis=(1,2))**2) 
    style_diff4 = tf.reduce_mean(tf.linalg.norm(r_g4 - f_g4, axis=(1,2))**2) 
    style_diff5 = tf.reduce_mean(tf.linalg.norm(r_g5 - f_g5, axis=(1,2))**2) 
    style_diff6 = tf.reduce_mean(tf.linalg.norm(r_g6 - f_g6, axis=(1,2))**2) 
    

    d_loss = -wd + gp * 10.0
    g_loss0 = -tf.reduce_mean(f_logit) #+ (10.0*(style_diff1+style_diff2+style_diff3))
    g_loss_style = 100*(style_diff1+style_diff2+style_diff3+style_diff4+style_diff5+style_diff6) 
    g_loss = g_loss0 + g_loss_style

    # otpims
    d_var = utils.trainable_variables('discriminator')
    g_var = utils.trainable_variables('generator')
    d_step = tf.train.AdamOptimizer(learning_rate=0.0004, beta1=0, beta2=0.9).minimize(d_loss, var_list=d_var)
    g_step = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0, beta2=0.9).minimize(g_loss, var_list=g_var)
    #d_step = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.9).minimize(d_loss, var_list=d_var)
    #g_step = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.9).minimize(g_loss, var_list=g_var)

    # summaries
    d_summary = utils.summary({wd: 'wd', gp: 'gp'})
    g_summary = utils.summary({g_loss: 'g_loss'})

    # sample
    f_sample = generator(z, training=False)




def mmd(x_true, x_fake):
    """
    computes MMD between two distributions using samples from these distributions (using Gaussian kernel).
    reference1: https://arxiv.org/pdf/1505.03906.pdf
    reference2: https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
    reference3: https://stats.stackexchange.com/questions/276497/maximum-mean-discrepancy-distance-distribution
    """
    x_true = np.reshape(x_true, [-1, 100*100*4])
    x_fake = np.reshape(x_fake, [-1, 100*100*4]) 
    x_true_dist = scipy.spatial.distance.pdist(x_true)     
    x_fake_dist = scipy.spatial.distance.pdist(x_fake)     
    x_true_fake = scipy.spatial.distance.cdist(x_true, x_fake)

    mmd_true_b = (2*np.sum(np.exp(-0.5*(x_true_dist**2))) + (x_true.shape[0]))/(x_true.shape[0]**2)
    mmd_fake_b = (2*np.sum(np.exp(-0.5*(x_fake_dist**2))) + (x_fake.shape[0]))/(x_fake.shape[0]**2)
    mmd_cross_b = np.sum(np.exp(-0.5*(x_true_fake**2)))/(x_fake.shape[0]*x_true.shape[0])

    mmd_true_u = (np.sum(np.exp(-0.5*(x_true_dist**2))))/(x_true.shape[0]*(x_true.shape[0]-1)) 
    mmd_fake_u = (np.sum(np.exp(-0.5*(x_fake_dist**2))))/(x_fake.shape[0]*(x_fake.shape[0]-1))
    mmd_cross_u = mmd_cross_b


    return mmd_true_b + mmd_fake_b - (2*mmd_cross_b), mmd_true_u + mmd_fake_u - (2*mmd_cross_u)




mmd_log = []
""" train """
''' init '''
# session
sess = utils.session()
# iteration counter
it_cnt, update_cnt = utils.counter()
# saver
saver = tf.train.Saver(max_to_keep=None)
# summary writer
summary_writer = tf.summary.FileWriter('./cect_summaries_00/cect_wgan_gp_00', sess.graph) #('./summaries/celeba_wgan_gp', sess.graph)

''' initialization '''
ckpt_dir = './cect_checkpoints/{}'.format(PARAMS.prefix)#'./checkpoints/celeba_wgan_gp'
utils.mkdir(ckpt_dir + '/')
if not utils.load_checkpoint(ckpt_dir, sess):
    sess.run(tf.global_variables_initializer())

''' train '''
try:
    z_ipt_sample = np.random.normal(size=[batch_size, z_dim])

    batch_epoch = len(data_pool) // (batch_size * n_critic)
    print(f'data pool length is {len(data_pool)}')
    print(f'batch size is {batch_size}')
    print(f'n_critic is {n_critic}')
    print(f'batch_epoch size is {batch_epoch}')

    max_it = epoch * batch_epoch
    loss_array=np.zeros((max_it,5))
    print(f'max_it is {max_it}')
    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        # train D
        for i in range(n_critic):
            # batch data
            real_ipt = data_pool.batch()
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: real_ipt, z: z_ipt})
        summary_writer.add_summary(d_summary_opt, it)

        # train G
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        real_ipt = data_pool.batch()
        #g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: z_ipt})
        g_summary_opt, _, g_0, g_style,d_0,wass,GradPen = sess.run([g_summary, g_step, g_loss0, g_loss_style,d_loss,wd,gp], feed_dict={real: real_ipt, z: z_ipt})
        summary_writer.add_summary(g_summary_opt, it)
        loss_array[it,0]=g_0
        loss_array[it,1]=g_style
        loss_array[it,2]=d_0
        loss_array[it,3]=wass
        loss_array[it,4]=GradPen      
        
        # display
        if it % 1 == 0:
            #print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))
            print("Epoch: (%3d) (%5d/%5d) g_loss0={:.3f} and g_style={:.3f}".format(g_0, g_style) % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % 800 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if (it + 1) % 800 ==0: 
            f_sample_opt = sess.run(f_sample, feed_dict={z: z_ipt_sample})
            mmd_b, mmd_u = mmd(real_ipt, f_sample_opt)
            mmd_log.append([epoch, mmd_b, mmd_u])

            save_dir = './cect_images_while_training/{}'.format(PARAMS.prefix) #'./sample_images_while_training/celeba_wgan_gp'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            for n in range(4):
                fig, axs = plt.subplots(8,8, figsize=(8,8))
                for ii in range(8):
                    for jj in range(8):
                        axs[ii,jj].imshow(f_sample_opt[ii*8+jj,:,:,n], cmap='inferno',vmin=-1, vmax=1)
                        axs[ii,jj].axis('off')
                plt.subplots_adjust(wspace=0, hspace=0)
                plt.savefig('{}/Epoch{}_({}of{})_phase{}'.format(save_dir, epoch, it_epoch, batch_epoch, n+1)) 


    
    mmd_np = np.array(mmd_log)
    plt.figure(figsize=(14,10))
    plt.semilogy(mmd_np[:,0], mmd_np[:,1], 'o-', linewidth=6, markersize=15)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel(r'$MMD^2_b$')
    plt.savefig(f"{save_dir}/mmd_biased.png")

    plt.figure(figsize=(16,10))
    plt.plot(mmd_np[:,0], mmd_np[:,2], 'o-', linewidth=6, markersize=15)
    plt.yscale('symlog')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel(r'$MMD^2_u$')
    plt.savefig(f"{save_dir}/mmd_unbiased.png")
    np.save(f"{save_dir}/mmd_np.npy", mmd_np)

    GeneratorL=loss_array[batch_epoch-1::batch_epoch,0]
    StyleL=loss_array[batch_epoch-1::batch_epoch,1]
    DiscL=loss_array[batch_epoch-1::batch_epoch,2]
    wassd=loss_array[batch_epoch-1::batch_epoch,3]
    GradPena=loss_array[batch_epoch-1::batch_epoch,4]
    
    save_dir = './cect_images_while_training/{}'.format(PARAMS.prefix)
    plt.figure()
    plt.plot(GeneratorL) 
    plt.legend(['Generator loss'])
    plt.draw()
    plt.savefig('{}/Generator Loss'.format(save_dir))
    
    plt.figure()
    plt.plot(StyleL)
    plt.legend(['Style loss'])
    plt.draw()
    plt.savefig('{}/Style Loss'.format(save_dir))
    
    plt.figure()
    plt.plot(DiscL)
    plt.legend(['Discriminator loss'])
    plt.draw()
    plt.savefig('{}/Discriminator Loss'.format(save_dir))
    
    np.save('loss_array100snoise',loss_array)
    
    plt.figure()
    plt.plot(wassd)
    plt.legend(['Wasserstein Distance'])
    plt.draw()
    plt.savefig('{}/WD'.format(save_dir))
    
    plt.figure()
    plt.plot(GradPena)
    plt.legend(['Gradient Penalty'])
    plt.draw()
    plt.savefig('{}/GP'.format(save_dir))
except Exception as e:
    traceback.print_exc()
finally:
    print(" [*] Close main session!")
    sess.close()
