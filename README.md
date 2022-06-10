# Probabilistic-Medical-Image-Imputation-via-Deep-Adversarial-Learning

You would need to use Tensorflow 1.14 for the code to run properly.
Batch file:
Use trainer_revisemmd.py for the learning_prior phase and mcmc_sampler.py for the inference monte carlo sampling. Mcmc_stats.py is just to calculate mean and standard deviation and statistical data.
Adding noise happens by uncommenting the the if statements in the generator function in models_100x100.py.
