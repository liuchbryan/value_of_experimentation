**Note: This code repository is no longer being maintained. Please refer to [this repository](https://github.com/liuchbryan/ranking_under_lower_uncertainty) instead for the more up-to-date (and more robust) version of the same work.**

This repository contains the code used in the paper [What is the value of Experimentation & Measurement?](https://ieeexplore.ieee.org/document/8970749), which appeared in the IEEE ICDM 2019 conference.

Requirements: `python>=3.6`, `numpy`, `scipy`, `pandas`, `matplotlib`

Experiments, case studies, and empirical extensions in the paper can be found in the following notebooks:

* Section V - Empirical verification of the theoretical value of expectations and variances: `src/theoretical_quantity_verification.ipynb`
* Section VI - Case studies (e-commerce companies & marketing companies): `src/value_gained_simulation.ipynb`
* Appendix B-A - Empirical calculation of the risk: `src/var_D_bound.ipynb`
* Appendix B-B - Valuation Under Independent t-Distributed Assumptions: `src/normal_t_comparison.ipynb`
* Appendix B-C - Partial Estimation / Measurement Noise Reduction: `src/partial_noise_reduction.ipynb`
