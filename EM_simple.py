import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns 

# Impostiamo il seed per la riproducibilità
np.random.seed(42)

# Parametri delle due gaussiane (reali)
mu1, sigma1 = 2, 1.0  # Media e deviazione standard della prima gaussiana
mu2, sigma2 = 7, 1.5  # Media e deviazione standard della seconda gaussiana

# Generiamo dati da due gaussiane
n_samples = 300
X1 = np.random.normal(mu1, sigma1, n_samples // 2)
X2 = np.random.normal(mu2, sigma2, n_samples // 2)
#Uniamo i dati in un unico array
X = np.concatenate([X1, X2]).reshape(-1, 1)

sns.displot(X, kde=True)
# Funzione di stima del parametro della mistura usando EM
def EM_gaussian_mixture(X, n_components=2, max_iter=1000, tol=1e-100):
    # Inizializzazione
    n_samples = X.shape[0]
    weights = np.ones(n_components) / n_components  # Pesi iniziali uniformi
    means = np.random.choice(X.flatten(), n_components)  # Medie iniziali casuali
    covariances = np.array([np.var(X) for _ in range(n_components)])  # Varianze iniziali (uguali)
    
    log_likelihood_old = 0
    
    for iteration in range(max_iter):
        # ========================
        # E-step: calcolare le probabilità di appartenenza (responsabilità)
        # ========================
        responsibilities = np.zeros((n_samples, n_components))
        
        for k in range(n_components):
            pdf_k = norm.pdf(X, loc=means[k], scale=np.sqrt(covariances[k]))
            responsibilities[:, k] = weights[k] * pdf_k.flatten()
        
        # Normalizzazione per ottenere le probabilità
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # ========================
        # M-step: aggiornamento dei parametri
        # ========================
        N_k = responsibilities.sum(axis=0)  # Somma delle responsabilità per ogni componente
        
        # Aggiornamento dei pesi
        weights = N_k / n_samples
        
        # Aggiornamento delle medie
        means = (responsibilities.T @ X) / N_k[:, np.newaxis]
        
        # Aggiornamento delle varianze
        # Calcolare la differenza tra ogni dato e la media di ciascuna componente (broadcasting)
        diff = X - means.T  # Questo porta 'means' alla forma (1, 2) per il broadcasting, quindi la differenza ha forma (300, 2)

        # Aggiornamento delle varianze (moltiplicazione delle responsabilità per la differenza al quadrato)
        covariances = (responsibilities * diff ** 2).sum(axis=0) /  N_k
        
        # Calcolo della log-verosimiglianza per il controllo della convergenza
        log_likelihood_new = np.sum(np.log(responsibilities.sum(axis=1)))
        if np.abs(log_likelihood_new - log_likelihood_old) < tol:
            break
        log_likelihood_old = log_likelihood_new

    return weights, means, covariances

# Applicazione del metodo EM
weights, means, covariances = EM_gaussian_mixture(X, n_components=2)

# Stampa dei parametri stimati
print("Pesi stimati (phi):", weights)
print("Medie stimate (mu):", means)
print("Varianze stimate (sigma^2):", covariances)

# Grafico delle densità
x_vals = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
#densità_stimata = np.zeros_like(x_vals)

densità_mistura = norm.pdf(x_vals, loc=mu1, scale=np.sqrt(sigma1)) +  norm.pdf(x_vals, loc=mu2, scale=np.sqrt(sigma2))

#for k in range(2):
    #densità_stimata += weights[k] * norm.pdf(x_vals, loc=means[k], scale=np.sqrt(covariances[k]))

# Densità delle due gaussiane originali per confronto
densità_1_stimata = norm.pdf(x_vals, loc=means[0], scale=np.sqrt(covariances[0]))
densità_2_stimata = norm.pdf(x_vals, loc=means[1], scale=np.sqrt(covariances[1]))

# Grafico della distribuzione stimata
plt.figure(figsize=(8, 5))
plt.plot(x_vals, densità_mistura, color='red', label="Densità")
plt.plot(x_vals, densità_1_stimata, color='blue', label="Densità1_EM")
plt.plot(x_vals, densità_2_stimata, color='green', label="Densità2_EM")
plt.title("Stima della mistura di gaussiane univariate tramite EM")
plt.xlabel("X")
plt.ylabel("Densità")
plt.legend()
plt.show()



