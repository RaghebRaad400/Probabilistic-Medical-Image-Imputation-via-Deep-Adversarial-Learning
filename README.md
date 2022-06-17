# Probabilistic-Medical-Image-Imputation-via-Deep-Adversarial-Learning

You would need to use Tensorflow 1.14 for the code to run properly. Tensorflow-probability 0.7 is required to be installed. The data could not be uploaded into the repository because it is larger than the GitHub limits. Our data file is 3368x100x100x4. The data needs to be scaled to [-1 1] range before running code.
The RunWithNoise folder represents the Enhanced GAN model while the RunWithoutNoise represents the Standard GAN model.

Batch file:

Use trainer_revisemmd.py for the learning_prior phase and mcmc_sampler.py for the inference monte carlo sampling. Mcmc_stats.py is just to calculate mean and standard deviation and statistical data.

The following is an example of an mcmc_stats run with patient 1 and a prediction of the first phase.

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

The config.py file mentions the meaning of each of the variables shown above. 
