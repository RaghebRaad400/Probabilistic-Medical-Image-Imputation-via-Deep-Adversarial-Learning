import argparse

def argparser():
    parser = argparse.ArgumentParser(description='Hyper-parameters')

    # == model parameters ==
    # GAN training 
    parser.add_argument('--train_batch_size', type=int, default=64, help='batch size for GAN training')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--z_dim', type=int, default=100)   
    parser.add_argument('--seed_no', type=int, default=1008)
    parser.add_argument('--img_h', type=int, default=100)
    parser.add_argument('--img_w', type=int, default=100)
    parser.add_argument('--img_c', type=int, default=4)
    parser.add_argument('--prefix', type=str, default='cect_wgan_gp', help='nickname for training')
    parser.add_argument('--model_path', type=str, default = './checkpoints/Epoch_(119)_(25of25).ckpt')    
    parser.add_argument('--save_freq', type=int, default=100, help='frequency of saving the model')
    parser.add_argument('--sample_freq', type=int, default=100, help='frequency of saving fake images')
    parser.add_argument('--log_freq', type=int, default=1)

    # measurement parameters for inverse problem
    parser.add_argument('--noise_var', required=True, type=float)
    parser.add_argument('--img_no', required=False, type=int, choices=range(202500,202599), help='image number of test image in test set; should be between 202500-202599 by default. If you want to change the images that are considered in the test set, youshould make relevant changes in data_celeba.py file') 
    parser.add_argument('--phase_vec', required=True, type=int, nargs=4, choices=(0,1))
    parser.add_argument('--patient_id', required=True, type=int)
    parser.add_argument('--phase', required=True, type=str, choices=('learning_prior', 'inference'))
    # parameters for mcmc 
    parser.add_argument('--n_mcmc', type=int, default=64000)    
    parser.add_argument('--burn_mcmc', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for posterior sampling')
    parser.add_argument('--meas_var', type=float, default=0)
    parser.add_argument('--like_var', type=float, default=0.1)
    
    # parameters for oed experiments
    parser.add_argument('--n_oed', type=int, default=64000)    
    parser.add_argument('--burn_oed', type=float, default=0.5)    
    parser.add_argument('--iter_no', type=int)
    parser.add_argument('--tile_size', type=int, default=16)
    
    # parameters for inpainting experiments
    parser.add_argument('--n_inpaint', type=int, default=64000)
    parser.add_argument('--burn_inpaint', type=float, default=0.5)
    parser.add_argument('--start_row', type=int)
    parser.add_argument('--end_row', type=int)
    parser.add_argument('--start_col', type=int)
    parser.add_argument('--end_col', type=int)
    
    config = parser.parse_args()
    return config
