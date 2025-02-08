
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns 
from scipy.stats import gaussian_kde
np.random.seed(89) 
#this data_set can be found in Davinson statistical models tab 5.10
galaxy_velocities = np.array([9172, 9350, 9483, 9558, 9775, 10227, 10406, 16084, 16170, 18419, 18552, 18600, 
                              18927, 19052, 19070, 19330, 19343, 19349, 19440, 19473, 19529, 19541, 19547, 
                              19663, 19846, 19856, 19863, 19914, 19918, 19973, 19989, 20166, 20175, 20179,
                              20196, 20215, 20221, 20415, 20629, 20795, 20821, 20846, 20875, 20986, 21137, 
                              21492, 21701, 21814, 21921, 21960, 22185, 22209, 22242, 22249, 22314, 22374, 
                              22495, 22746, 22747, 22888, 22914, 23206, 23241, 23263, 23484, 23538, 23542, 
                              23666, 23706, 23711, 24129, 24285, 24289, 24366, 24717, 24990, 25633, 26960,
                              26995,32065,32789, 34279 ])
                             
                             
# Plotting the histogram and the density curve using seaborn
plt.figure(figsize=(10,6))

# Plot histogram
sns.histplot(galaxy_velocities, kde=True, color='blue', stat='density', bins=15)

plt.title("Galaxy Velocities Density", fontsize=16)
plt.xlabel("Velocity (km/s)", fontsize=12)
plt.ylabel("Density", fontsize=12)

# Show plot
plt.show()



def EM(X,n_components = 3, tolerance = 1e-1000):

    n_data = len(X)

    # Initialize means, variances, and weights

    means = np.random.choice(galaxy_velocities, n_components)
    variances = np.full(n_components, np.var(X))
    weights = np.ones(n_components) / n_components

    # EM Algorithm
   
    log_likelihoods = []

    for iteration in range(1000):
        # E-step: Calculate responsibilities (probabilities)
        responsibilities = np.zeros((n_data, n_components))
        
        for i in range(n_components):
            responsibilities[:, i] = weights[i] * norm.pdf(X, means[i], np.sqrt(variances[i]))
        
        # Normalize the responsibilities
        responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]
        
        # M-step: Update parameters
        # Update means
        for i in range(n_components):
            means[i] = np.sum(responsibilities[:, i] * galaxy_velocities) / np.sum(responsibilities[:, i])
        
        # Update variances
        for i in range(n_components):
            variances[i] = np.sum(responsibilities[:, i] * (galaxy_velocities - means[i])**2) / np.sum(responsibilities[:, i])
        
        # Update weights
        for i in range(n_components):
            weights[i] = np.sum(responsibilities[:, i]) / n_data
        
        # Log-likelihood
        log_likelihood = np.sum(np.log(np.sum(responsibilities * weights, axis=1)))
        log_likelihoods.append(log_likelihood)
        
        # Check for convergence (if log-likelihood change is below threshold)
        if iteration > 0 and np.abs(log_likelihood - log_likelihoods[-2]) < tolerance:
            print(f"Converged at iteration {iteration}")
            break
    return weights, means, variances , log_likelihoods


weights, means, variances,log_likelihoods_score = EM(galaxy_velocities)

plt.figure(figsize=(10,6))

# Plot histogram
plt.plot(log_likelihoods_score)
plt.title("log_likelihoods_score", fontsize=16)
plt.xlabel("iteration", fontsize=12)
plt.ylabel("value", fontsize=12)

# Show plot
plt.show()

# Calcolare la densità usando KDE
kde = gaussian_kde(galaxy_velocities, bw_method='silverman')  # Banda scelta con il metodo di Silverman
x_vals = np.linspace(galaxy_velocities.min(), galaxy_velocities.max(), 500)
densità_start = kde(x_vals)
densità = []
for r in range(len(weights)):
    # Calcola la densità per ciascun componente
    densità_componente = norm.pdf(x_vals, means[r], np.sqrt(variances[r]))
    densità.append(densità_componente)
# Grafico
plt.figure(figsize=(8, 5))
plt.plot(x_vals, densità_start*5, label="start density")
for r in range(len(weights)):
    plt.plot(x_vals, densità[r], label= f"Z {r+1}")
plt.title("Densità stimata delle velocità delle galassie")
plt.xlabel("Velocità delle galassie")
plt.ylabel("Densità")
plt.legend()
plt.show()