# vande
Variational Autoencoding for Anomaly Detection

### setup
requires Python3 and TF 2.X

## Particle VAE

### train
```
main_train_particle_vae.py
```

parameters
- run_n ... experiment number
- beta ... beta coefficient for Kullback-Leibler divergence term
- loss ... 3D+KL loss or MSE+KL loss (from losses module)
- reco_loss ... 3D or MSE (from losses module)
- cartesian ... True/False: constituents coordinates (if False: cylindrical)

### predict
```
main_predict_particle_vae.py
```
parameters
- run_n ... experiment number
- cartesian ... True/False: constituents coordinates (if False: cylindrical)
