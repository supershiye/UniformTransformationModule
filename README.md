# Uniform Transformation: Refining Latent Representation in Variational Autoencoders
Ye Shi, C. S. George Lee
Accepted by CASE 2024
## Overview
The Uniform Transformation (UT) module is a novel approach introduced in my research to address irregular distributions in the latent space of Variational Autoencoders (VAEs). VAEs are a type of artificial intelligence model used to compress and then reconstruct data, creating compact representations of complex information. However, these models often face issues with how they summarize data, leading to irregularities that can negatively impact their performance.

The UT module aims to reshape these irregular summaries into more regular, uniform ones, thereby enhancing the models' ability to understand and generate data. It consists of three stages:

1. **Gaussian Kernel Density Estimation (G-KDE) Clustering**: This stage involves clustering the latent variables using a non-parametric algorithm that identifies the distribution patterns without assuming a fixed number of clusters beforehand.

2. **Gaussian Mixture (GM) Modeling**: After clustering, this stage constructs a probabilistic model that represents the latent variable distribution as a mixture of Gaussian distributions.

3. **Probability Integral Transform (PIT)**: The final stage applies a statistical technique to transform the irregularly distributed latent variables into a uniform distribution, ensuring better disentanglement and interpretability of the latent space.

By integrating the UT module into VAEs, my research demonstrated significant improvements in the disentanglement and interpretability of latent representations, which is crucial for various automation applications.

## Instructions
This repository contains the UT module code to replicate the experimental from the paper.
TBD
