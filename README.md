# Learned Tensor RPCA 

This repository implements the original ScaledGD alogrithm presented in Fast and Provable Tensor Robust Principal Component Analysis via Scaled Gradient Descent \[2\], along with algorithms and experiments described in Deep Unfolded Tensor Robust PCA with Self-supervised Learning \[1\]. 
The `src` contains all the code while the `outputs` directory contains subdirectories that store outputs of the experiments.
Example outputs can be found in `outputs`.
More organization details in the `src` directory below. 
For details on citing this repository, please scroll to the bottom of this page.

### Tensor RPCA

`rpca.py` contains the standard tensor RPCA algorithm from \[2\]. 
This can be used out of the box, independent of learning.

### Model

Our deep unfolded RPCA model can be found in `model.py`.

### Helper functions

Some functions related to tensor generation and training are shared across multiple files. 
These are placed in `training_functions.py`.
Part of this file is adapted from \[3\].

### Synthetic Experiments

Synthetic experiments are split into 2 files: `synthetic_bayesian_train.py` for the baseline and `synthetic_train.py` for learned methods via our model.
Within each file, one can customize the scenario to run tests on.
For instance, one could manipulate the the ranks, tensor size, corruption sparsity, hyperparameters, etc.

### Background Subtraction

Code for background subtraction can be found in `background_separation_train.py`. 
Hyperparameters and processing can be adjusted within the file.
In order to run this file, the real videos dataset must be downloaded from http://backgroundmodelschallenge.eu


\[1\] Dong, H., Shah, M., Donegan, S., & Chi, Y. (2023). Deep Unfolded Tensor Robust PCA with Self-supervised Learning. International Conference on Acoustics, Speech, and Signal Processing (ICASSP).

\[2\] Dong, H., Tong, T., Ma, C., & Chi, Y. (2022). Fast and Provable Tensor Robust Principal Component Analysis via Scaled Gradient Descent. arXiv preprint arXiv:2206.09109.

\[3\] Cai, H., Liu, J., & Yin, W. (2021). Learned robust pca: A scalable deep unfolding approach for high-dimensional outlier detection. Advances in Neural Information Processing Systems, 34, 16977-16989.


# How to Cite

If you use or build upon this work, please cite the [paper](https://arxiv.org/abs/2206.09109) that the original ScaledGD tensor RPCA algorithm was introduced:

    @article{dong2022fast,
        title={Fast and provable tensor robust principal component analysis via scaled gradient descent},
        author={Dong, Harry and Tong, Tian and Ma, Cong and Chi, Yuejie},
        journal={arXiv preprint arXiv:2206.09109},
        year={2022}
    }

If you also use or build the learnable version of the algorithm, please also cite the following [paper](https://arxiv.org/abs/2212.11346):

    @article{dong2022deep,
        title={Deep Unfolded Tensor Robust PCA with Self-supervised Learning},
        author={Dong, Harry and Shah, Megna and Donegan, Sean and Chi, Yuejie},
        journal={arXiv preprint arXiv:2212.11346},
        year={2022}
    }

