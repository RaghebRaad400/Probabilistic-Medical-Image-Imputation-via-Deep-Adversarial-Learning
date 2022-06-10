# Probabilistic-Medical-Image-Imputation-via-Deep-Adversarial-Learning

You would need to use Tensorflow 1.14 for the code to run properly.
First Step: Loading the dataset which is in a 3368x100x100x4 format. Change the images range from [0,1] to [-1,1].
Batch file:
Use trainer_revisemmd.py for the learning_prior phase and mcmc_sampler.py for the inference monte carlo sampling. Mcmc_stats.py is just to calculate mean and standard deviation and statistical data.
The following is an example of an mcmc_stats run with patient 1 and a prediction of the first phase.
source activate tf-gpu1.14
python mcmc_stats.py \
        --n_mcmc 1024000 \
        --patient_id 1 \
        --noise_var 1.0 \
        --phase_vec 0 1 1 1    \
        --batch_size 1 \
        --phase inference \
        --seed_no 1 \
        --model_path './100snoisemmdnewagain/Epoch_(177)_(7of9).ckpt' \
        --meas_var 0.0 \
        --like_var 1.00 \
        > output2.out
