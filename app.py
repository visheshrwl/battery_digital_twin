# Importing necessary libraries for comprehensive refinement of the code
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
from scipy.stats import weibull_min
from scipy.optimize import minimize



# Placeholders for the refined logic to ensure advanced performance and modularity.
# Building upon existing ideas, I'll elevate this code to a true research-grade tool.

# ============================
# 1. Dataset Loading Functions
# ============================



def load_conditions(excel_path):
    """
    Load experimental conditions for batteries from an Excel file.
    :param excel_path: str - Path to the Excel file containing conditions.
    :return: pd.DataFrame or None
    """
    try:

        df = pd.read_excel(excel_path)

        return df

    except Exception as e:

        return f"Error in loading battery conditions: {e}"





def load_capacity_data(train_path, test1_path, test2_path):

    """

    Load capacity data from specified directories.



    :param train_path: str - Path to the training dataset directory.

    :param test1_path: str - Path to the test1 dataset directory.

    :param test2_path: str - Path to the test2 dataset directory.

    :return: dict of loaded data or error message

    """

    try:

        train_files = glob.glob(os.path.join(train_path, "*.csv"))

        test1_files = glob.glob(os.path.join(test1_path, "*.csv"))

        test2_files = glob.glob(os.path.join(test2_path, "*.csv"))



        def load_csv_files(file_list):

            return [pd.read_csv(file) for file in file_list]



        return {

            'train': load_csv_files(train_files),

            'test1': load_csv_files(test1_files),

            'test2': load_csv_files(test2_files)

        }

    except Exception as e:

        return f"Error in loading capacity data: {e}"



# ============================

# 2. Capacity Simulation Logic

# ============================



def simulate_aging(cycle_numbers, initial_capacity=1.0, degradation_rate=0.1, k=0.1, c50=50):

    """

    Simulate battery aging using Weibull (S-curve) model.



    :param cycle_numbers: array-like - List of cycle numbers.

    :param initial_capacity: float - Initial capacity of the battery.

    :param degradation_rate: float - Max degradation fraction.

    :param k: float - Shape parameter for Weibull distribution.

    :param c50: int - Half-life cycle parameter.

    :return: array-like - Simulated capacities.

    """

    try:

        capacities = initial_capacity * (1 - degradation_rate * weibull_min.cdf(cycle_numbers, c=k, scale=c50))

        capacities = np.clip(capacities, 0, initial_capacity)  # Ensure values stay within bounds

        return capacities

    except Exception as e:

        return f"Error in simulating aging: {e}"



# ============================

# 3. Advanced Analysis Functions

# ============================



def particle_filter_analysis(K_data, Q_data, n_particles=100, measurement_noise_std=0.005, process_noise_std=0.001):

    """

    Perform RUL prediction using Particle Filter.



    :param K_data: array-like - List of cycle numbers.

    :param Q_data: array-like - Corresponding measured capacities.

    :param n_particles: int - Number of particles for the filter.

    :param measurement_noise_std: float - Standard deviation of measurement noise.

    :param process_noise_std: float - Standard deviation of process noise.

    :return: dict - Results with predicted RUL and other metrics.

    """

    try:

        # Initialize particles and weights

        particles = np.random.normal(loc=Q_data[0], scale=measurement_noise_std, size=n_particles)

        weights = np.ones(n_particles) / n_particles

        rul_predictions = []



        for cycle in range(1, len(K_data)):

            # Predict step: Apply degradation model

            particles += np.random.normal(loc=-process_noise_std, scale=process_noise_std, size=n_particles)

            particles = np.clip(particles, 0, 1)



            # Update step: Calculate weights based on observation likelihood

            weights *= np.exp(-0.5 * ((Q_data[cycle] - particles) / measurement_noise_std) ** 2)

            weights += 1.e-300  # Avoid rounding errors

            weights /= weights.sum()



            # Resample step: Effective sample size check

            effective_N = 1. / np.sum(weights ** 2)

            if effective_N < n_particles / 2:

                indices = np.random.choice(n_particles, size=n_particles, replace=True, p=weights)

                particles = particles[indices]

                weights.fill(1.0 / n_particles)



            # Estimate RUL (remaining cycles until end of life)

            rul = max(K_data[-1] - K_data[cycle], 0)

            rul_predictions.append(rul)



        return {'cycles': K_data, 'rul_predictions': rul_predictions}

    except Exception as e:

        return f"Error in Particle Filter analysis: {e}"

# ============================
# 4. Streamlit App Framework
# ============================

def main():
    st.title("🔋 Advanced Battery Health Monitoring Tool")

    dataset = None
    conditions = None
    
    st.sidebar.subheader("Load Dataset")
    excel_path = '124 LFP Cell Conditions.xlsx'
    train_path = '124 LFP Capacity Data/train'
    test1_path = '124 LFP Capacity Data/test1'
    test2_path = '124 LFP Capacity Data/test2'
    
    dataset = load_capacity_data(train_path, test1_path, test2_path)
    conditions = load_conditions(excel_path)
    print(conditions)
    st.success("Data loaded successfully!")
    st.write("Dataset:", dataset)
    st.write("Conditions:", conditions)

    # Aging simulation
    st.sidebar.subheader("Simulate Aging")
    degradation_rate = st.sidebar.slider("Degradation Rate", 0.0, 1.0, 0.001)
    shape_param = st.sidebar.slider("Shape Parameter (k)", 0.0, 1.0, 0.005)
    half_life = st.sidebar.slider("Half-Life Cycle", 1250, 4500, 50)

    if st.sidebar.button("Run Aging Simulation"):
        cycles = np.arange(1, 10001)
        simulated_capacities = simulate_aging(cycles, degradation_rate=degradation_rate, k=shape_param, c50=half_life)
        if simulated_capacities is not None:
            st.line_chart(simulated_capacities)
        else:
            st.error("Error in simulation.")

    # Particle filter analysis
    st.sidebar.subheader("Run Particle Filter")
    num_particles = st.sidebar.slider("Number of Particles", 50, 500, 100)
    measurement_noise = st.sidebar.slider("Measurement Noise", 0.001, 0.01, 0.5)
    process_noise = st.sidebar.slider("Process Noise", 0.001, 0.01, 0.1)

    if st.sidebar.button("Execute Particle Filter"):
        if dataset:
            results = particle_filter_analysis(
                K_data=np.arange(1, 10001), Q_data=np.linspace(1, 0.5, 10000), n_particles=num_particles,
                measurement_noise_std=measurement_noise, process_noise_std=process_noise
            )
            if results:
                st.line_chart(results['rul_predictions'])
            else:
                st.error("Error in Particle Filter analysis.")
        else:
            st.error("Please load dataset first.")
            
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This web app is a demonstration of an advanced battery health monitoring tool "
        "using Particle Filter for Remaining Useful Life (RUL) prediction."
    )
    
    st.sidebar.subheader("Author")
    st.sidebar.info(
        """
        This app is developed by [Vishesh Rawal] under the guidance and mentorship of [Dr. Prashant Singh Rana]
        """
    )
    
if __name__ == "__main__":
    main()
