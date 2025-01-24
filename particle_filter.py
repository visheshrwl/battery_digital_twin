# particle_filter.py
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def power_law(x, log10_a, b):
    """Capacity fade model using power law."""
    return 1 - ((10**log10_a) * x**b)

def create_gaussian_particles(mean, std, N):
    """Initialize particles with Gaussian distribution."""
    particles = np.empty((N, len(mean)))
    for i in range(len(mean)):
        particles[:, i] = mean[i] + (np.random.randn(N) * std[i])
    return particles

def predict(particles, std):
    """Propagate particles through the process model."""
    for i in range(particles.shape[1]):
        particles[:, i] += np.random.randn(particles.shape[0]) * std[i]
    return particles

def update(particles, weights, measurement_noise_std, capacity, cycle):
    """Update particle weights based on the measurement."""
    capacity_pred = power_law(cycle, particles[:,0], particles[:,1])
    pdf_vals = stats.norm(capacity_pred, measurement_noise_std).pdf(capacity)
    weights *= pdf_vals
    weights += 1.e-300  # Avoid zeros
    weights /= np.sum(weights, axis=0)
    return weights

def estimate(particles, weights):
    """Estimate mean and standard deviation of parameters."""
    means = np.average(particles, weights=weights, axis=0)
    stds = np.sqrt(np.average((particles - means)**2, weights=weights, axis=0))
    return means, stds

def resample(particles, weights):
    """Resample particles based on weights."""
    resampled_particles = np.empty_like(particles)
    for i in range(particles.shape[1]):
        indices = np.random.choice(np.arange(particles.shape[0]), size=particles.shape[0], p=weights[:,i])
        resampled_particles[:,i] = particles[indices, i]
    weights = np.ones_like(weights) / particles.shape[0]
    return resampled_particles, weights

def PF(K_data, Q_data, K_fpt, K_eol, idx_fpt, eol_threshold,
       log10_a_initial, b_initial, measurement_noise_std, process_noise_std,
       n_particles, n_param_samples=500, plot_each_cycle=False, plot_final_RUL=False):
    """
    Particle Filter implementation for capacity prediction.
    """
    cycles = np.arange(K_fpt, K_eol + 1, 1, dtype=int).reshape(-1)
    capacities = Q_data[idx_fpt:]
    eol_cycle = K_eol
    
    # Initialize particles and weights
    particles = create_gaussian_particles((log10_a_initial, b_initial),
                                          (process_noise_std[0], process_noise_std[1]),
                                          n_particles)
    weights = np.ones_like(particles) / n_particles
    
    log10_a_pred = []
    b_pred = []
    eol_pred = []
    rul_pred = []
    rul_pred_lb = []
    rul_pred_ub = []
    
    for cycle, capacity in zip(cycles, capacities):
        # Step 1: Predict
        particles = predict(particles, process_noise_std)
        
        # Step 2: Update
        weights = update(particles, weights, measurement_noise_std, capacity, cycle)
        
        # Step 3: Estimate
        estimated_means, estimated_stds = estimate(particles, weights)
        log10_a_pred.append(estimated_means[0])
        b_pred.append(estimated_means[1])
        
        # Step 4: Resample
        particles, weights = resample(particles, weights)
        
        # EOL and RUL Prediction
        log10_a_samples = np.random.choice(particles[:,0], n_param_samples, p=weights[:,0])
        b_samples = np.random.choice(particles[:,1], n_param_samples, p=weights[:,1])
        
        simulated_cycles = np.arange(cycle, 10000, 1)
        cap_trajectory_simulation = power_law(simulated_cycles[:, np.newaxis], log10_a_samples, b_samples)
        
        eol_samples = np.where(cap_trajectory_simulation <= eol_threshold, simulated_cycles[:, np.newaxis], 100000).min(axis=0)
        rul_samples = eol_samples - cycle
        
        eol_pred.append(np.median(eol_samples))
        rul_pred.append(np.median(rul_samples))
        rul_pred_lb.append(np.percentile(rul_samples, 5))
        rul_pred_ub.append(np.percentile(rul_samples, 95))
        
        # Plotting (if enabled)
        if plot_each_cycle:
            plt.figure(figsize=(4,3.5), dpi=100)
            plt.plot(K_data, Q_data, color='k', label='Measured Capacity')
            plt.hlines(eol_threshold, 0, 2500, colors='r', linestyles='--')
            plt.vlines(cycle, 0, capacity, colors='g', linestyles=':', label='Current Cycle')
            plt.plot(simulated_cycles, np.median(cap_trajectory_simulation, axis=1), 'b-.', label='PF Prediction')
            plt.xlabel('Cycle Number')
            plt.ylabel('Discharge Capacity (Ah)')
            plt.legend()
            plt.show()
    
    results_dic = {
        'eol_pred': np.array(eol_pred),
        'rul_pred': np.array(rul_pred),
        'true_rul': np.arange(eol_cycle, 0, -1)[idx_fpt:],
        'log10_a_pred': np.array(log10_a_pred),
        'b_pred': np.array(b_pred),
        'cycles': cycles,
        'capacities': capacities,
        'rul_pred_lb': np.array(rul_pred_lb),
        'rul_pred_ub': np.array(rul_pred_ub)
    }
    
    if plot_final_RUL:
        plt.figure(figsize=(4,3.5), dpi=100)
        plt.plot(np.arange(2, eol_cycle, 1), np.arange(eol_cycle, 2, -1), label='True RUL', color='k')
        plt.plot(cycles, rul_pred, label='PF RUL Prediction', color='b')
        plt.fill_between(cycles, rul_pred_lb, rul_pred_ub, color='b', alpha=0.3, label='95% Confidence Interval')
        plt.xlabel('Cycle Number')
        plt.ylabel('RUL')
        plt.legend()
        plt.show()

    if plot_each_cycle:
        plt.figure(figsize=(4,3.5), dpi=100)
        plt.plot(K_data, Q_data, color='k', label='Measured Capacity')
        plt.hlines(eol_threshold, 0, 2500, colors='r', linestyles='--')
        plt.vlines(cycle, 0, capacity, colors='g', linestyles=':', label='Current Cycle')
        plt.plot(simulated_cycles, np.median(cap_trajectory_simulation, axis=1), 'b-.', label='PF Prediction')
        plt.xlabel('Cycle Number')
        plt.ylabel('Discharge Capacity (Ah)')
        plt.legend()

    
    return results_dic
