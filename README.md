
# Mixture Density Model and EM Algorithm
This repository contains an example of **Mixture Models** and their application in data analysis using the **Expectation-Maximization (EM) Algorithm**. The mixture model is used to estimate the parameters of subpopulations in a population where the exact subpopulation of each data point is unknown.

## Overview

A **mixture model** arises when an observation \( Y \) is drawn from a population composed of distinct subpopulations, but it is unknown from which of these subpopulations \( Y \) is drawn. In this case, the mixture density of \( Y \) can be written as:

$$
f(y; \theta) = \sum_{r=1}^{p} \pi_r f_r(y; \theta)
$$

Where:
- $$\( \pi_r \)$$ is the probability that $$\( Y \)$$ comes from the $$\( r \)$$-th subpopulation, and $$\( \sum_{r=1}^{p} \pi_r = 1 \)$$.
- $$\( f_r(y; \theta) \)$$ is the conditional density of $$\( Y \)$$ given that it comes from the $$\( r \)$$-th subpopulation.
- $$\( \theta \)$$ represents all the unknown parameters of the model (including the $$\( \pi_r \)$$ and other parameters for each subpopulation's distribution).

## Goal

The goal is to use the **Expectation-Maximization (EM) Algorithm** to estimate:
- The parameters of the subpopulations.
- The probability that each data point belongs to each subpopulation.

This is done by iterating between two main steps:
1. **Expectation Step (E-Step)**: Calculate the expected log-likelihood of the parameters given the current estimates and the data.
2. **Maximization Step (M-Step)**: Maximize the expected log-likelihood to update the parameters.

## Example: Galaxy Velocities

### Problem Context

The example involves the analysis of **velocities** of galaxies in the **Corona Borealis** region. These galaxies are suspected to belong to different "superclusters," and the goal is to estimate how many clusters (subpopulations) there are in the data.

The velocities are thought to follow a **multimodal distribution** if these galaxies form distinct clusters. If no clusters exist, the distribution should be unimodal.

### Process

1. **Data**: The velocities of 82 galaxies are measured.
2. **Mixture Model**: A mixture of normal distributions is assumed, where each subpopulation (or cluster) is modeled by a normal distribution.
3. **Expectation-Maximization (EM) Algorithm**: This is applied to estimate the number of subpopulations by fitting mixtures of normal distributions to the data.

### Results

The algorithm iterates to find the best fit for different values of \( p \) (number of components in the mixture). The results suggest the best fit occurs when \( p = 3 \) or \( p = 4 \), indicating that there are **three or four clusters** in the galaxy velocities.

## EM Algorithm Steps

### E-Step (Expectation Step)

For each data point $$\( y_j \)$$, compute the **weight** $$\( w_r(y_j; \theta) \)$$ that indicates how likely it is that $$\( y_j \)$$ came from the $$\( r \)-th$$ subpopulation:

$$
P(U = r | Y = y_j; \theta) = \frac{\pi_r f_r(y_j; \theta)}{\sum_{s=1}^{p} \pi_s f_s(y_j; \theta)}
$$

Where $$\( w_r(y_j; \theta) \)$$ is the probability that the $$\( j \)-th$$ data point belongs to the $$\( r \)-th $$ subpopulation.

### M-Step (Maximization Step)

Update the parameters based on the current weights $$\( w_r(y_j; \theta) \)$$:
1. Update the mixing probabilities $$\( \pi_r \)$$ by averaging the weights for each component.
2. Update the parameters of the normal distributions (e.g., means $$\( \mu_r \)$$ and variances $$\( \sigma_r^2 \)$$)using the weighted data points.

### Iteration

Repeat the **E-step** and **M-step** iteratively until convergence, where the parameter estimates stop changing or change minimally.

## Model Selection and Fit

The algorithm can be run for different values of \( p \), and the log-likelihood values are compared to determine the best fit. The log-likelihood for each fit is maximized during the EM algorithm iterations. The **best fit** corresponds to the model with the highest log-likelihood.

For the galaxy velocity data, the EM algorithm fits mixtures with different numbers of components and yields the following log-likelihood values:

- For $$\( p = 1 \)$$: -240.42
- For $$\( p = 3 \)$$: -203.48
- For $$\( p = 4 \)$$: -202.52
- For $$\( p = 5 \)$$: -192.42

The \( p = 3 \) and \( p = 4 \) mixtures are considered the best fits, with both showing evidence of clustering in the data.

## Usage

To replicate the analysis:
1. Apply the **EM algorithm** to your data, choosing appropriate initial values for the mixing probabilities $$\( \pi_r \)$$, means $$\( \mu_r \)$$, and variances $$\( \sigma_r^2 \)$$.
2. Iterate between the **E-step** and **M-step** to update the parameters.
3. Choose the number of components \( p \) based on the log-likelihood or other model selection criteria.

## Conclusion

This example demonstrates how the **EM algorithm** can be used to estimate the number of clusters in a mixture model. The application to galaxy velocity data illustrates how the algorithm can identify clusters and provide insights into the underlying structure of the data.
