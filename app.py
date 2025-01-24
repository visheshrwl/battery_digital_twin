# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import your custom modules
from data_processing import load_conditions, load_capacity_data, extrapolate_capacity, create_capacity_dataset, load_processed_data
from particle_filter import PF
from optimization import calculate_total_discharge_Ah, calculate_mean_discharge_time, total_discharge_ah_utility, mean_dchg_time_utility, mean_chg_time_utility, optimize_replacement_time
from utils import plot_capacity_fade, plot_utility_functions, plot_single_cell_optimization

# Page configuration
st.set_page_config(
    page_title="Battery Capacity Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Battery Capacity Fade Prediction and Optimization")

# Sidebar for user inputs
st.sidebar.header("Controls")

# File paths (adjust as necessary)
excel_path = '124 LFP Cell Conditions.xlsx'
train_path = '124 LFP Capacity Data/train'
test1_path = '124 LFP Capacity Data/test1'
test2_path = '124 LFP Capacity Data/test2'
processed_data_path = 'capacity_dataset_124_lfp.pkl'

# Button to create/load dataset
if 'dataset_loaded' not in st.session_state:
    st.session_state['dataset_loaded'] = False

if not st.session_state['dataset_loaded']:
    if st.sidebar.button("Create and Load Dataset"):
        with st.spinner("Processing data..."):
            my_dic = create_capacity_dataset(train_path, test1_path, test2_path, excel_path, processed_data_path)
            st.session_state['dataset'] = my_dic
            st.session_state['dataset_loaded'] = True
            st.success("Dataset created and loaded successfully!")
else:
    st.sidebar.success("Dataset already loaded.")

# Load data from pickle
@st.cache_data
def load_data():
    return load_processed_data(processed_data_path)

if st.session_state['dataset_loaded']:
    dataset = st.session_state['dataset']
    all_cycles = dataset['x_train'] + dataset['x_test1'] + dataset['x_test2']
    all_capacities = dataset['y_train'] + dataset['y_test1'] + dataset['y_test2']
    all_c1 = list(dataset['train_conds']['c1'].to_numpy()) + list(dataset['test1_conds']['c1'].to_numpy()) + list(dataset['test2_conds']['c1'].to_numpy())
    all_percent = list(dataset['train_conds']['percent'].to_numpy()) + list(dataset['test1_conds']['percent'].to_numpy()) + list(dataset['test2_conds']['percent'].to_numpy())
    all_c2 = list(dataset['train_conds']['c2'].to_numpy()) + list(dataset['test1_conds']['c2'].to_numpy()) + list(dataset['test2_conds']['c2'].to_numpy())

# Sidebar for user inputs
st.sidebar.header("Battery Selection")

# Select Cell
cell_idx = st.sidebar.selectbox(
    'Select Cell Index',
    options=range(len(all_cycles)),
    format_func=lambda x: f"Cell {x+1}"
)

# Toggles
plot_each_cycle = st.sidebar.checkbox("Plot Each Cycle Prediction", value=False)
plot_final_RUL = st.sidebar.checkbox("Plot Final RUL", value=True)

# Optimization Parameters
st.sidebar.header("Optimization Parameters")
ah_weight = st.sidebar.slider("Weight for Total Ah", 0.0, 1.0, 0.5)
mtbc_weight = st.sidebar.slider("Weight for Mean Time Between Charges", 0.0, 1.0, 0.5)


# Main Area
st.header(f"Selected Cell: {cell_idx + 1}")

# Display Capacity Data
st.subheader("Capacity Data")
st.write("### Cycle Data", all_cycles[cell_idx])
st.write("### Capacity Data", all_capacities[cell_idx])

# Button to run Particle Filter
if st.sidebar.button("Run Particle Filter"):
    with st.spinner("Running Particle Filter..."):
        # Define necessary parameters
        cap_threshold = 0.985
        eol_threshold = 0.50
        log10_a_initial = -15.77
        b_initial = 5.45
        measurement_noise_std = 0.005
        process_noise_std = (0.05, 0.05)
        n_particles = 200
        
        # Determine train/test split (assuming test_ids start from index 41)
        test_ids = range(41, len(all_cycles))
        if cell_idx in test_ids:
            # Retrieve test cell data
            Q_test = all_capacities[cell_idx]
            K_test = all_cycles[cell_idx]
            idx_fpt = np.where(Q_test <= cap_threshold)[0][0]
            K_fpt = K_test[idx_fpt]
            K_eol = K_test[np.where(Q_test <= eol_threshold)[0][0]] if np.any(Q_test <= eol_threshold) else K_test[-1]
        else:
            st.error("Selected cell is not in the test set.")
            st.stop()
        
        # Run Particle Filter
        results_dic = PF(
            K_data=K_test,
            Q_data=Q_test,
            K_fpt=K_fpt,
            K_eol=K_eol,
            idx_fpt=idx_fpt,
            eol_threshold=eol_threshold,
            log10_a_initial=log10_a_initial,
            b_initial=b_initial,
            measurement_noise_std=measurement_noise_std,
            process_noise_std=process_noise_std,
            n_particles=n_particles,
            plot_each_cycle=plot_each_cycle,
            plot_final_RUL=plot_final_RUL
        )
        
        st.session_state['results_dic'] = results_dic
        st.session_state['K_test'] = K_test
        st.session_state['Q_test'] = Q_test
        
        st.success("Particle Filter executed successfully!")
        
        # Display Capacity Fade Plot
        st.subheader("Capacity Fade Curves")
        fig1, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(K_test, Q_test, label='Measured Capacity')
        ax1.hlines(eol_threshold, 0, 2500, colors='r', linestyles='--', label='EOL Threshold')
        ax1.set_xlim([0, 2500])
        ax1.set_ylim([0.4, 1.15])
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Discharge Capacity (Ah)')
        ax1.legend()
        st.pyplot(fig1)
        
        # Display RUL Prediction Plot
        if plot_final_RUL:
            st.subheader("RUL Prediction")
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.plot(results_dic['cycles'], results_dic['rul_pred'], label='PF RUL Prediction', color='b')
            ax2.fill_between(results_dic['cycles'], results_dic['rul_pred_lb'], results_dic['rul_pred_ub'], color='b', alpha=0.3, label='95% Confidence Interval')
            ax2.plot(np.arange(2, K_eol, 1), np.arange(K_eol, 2, -1), label='True RUL', color='k')
            ax2.set_xlabel('Cycle Number')
            ax2.set_ylabel('RUL')
            ax2.legend()
            st.pyplot(fig2)


# Optimization Section
    if 'results_dic' in st.session_state:
        if st.sidebar.button("Optimize Replacement Time"):
            with st.spinner("Optimizing Replacement Time..."):
                results_dic = st.session_state['results_dic']
                Q_data = st.session_state['Q_test']
                K_data = st.session_state['K_test']
                
                weights = np.array([ah_weight, mtbc_weight])
                optimal_cycle, function_values = optimize_replacement_time(
                    results_dic=results_dic,
                    Q_data=Q_data,
                    K_data=K_data,
                    weights=weights,
                    ah_bounds=(300,1000),
                    mtbc_bounds=(0.210,0.250),
                    eol_threshold=0.50
                )
                
                st.write(f"**Optimal Replacement Cycle:** {int(optimal_cycle)}")
                
                # Plot Optimization Results
                utility_cycles = results_dic['cycles']
                
                plt.figure(figsize=(6,4))
                plt.plot(utility_cycles, function_values, label='Utility')
                plt.axvline(optimal_cycle, color='g', linestyle='--', label='Optimal Replacement')
                plt.xlabel('Cycle Number')
                plt.ylabel('Utility')
                plt.legend()
                st.pyplot(plt)
    else:
        st.warning("Please run the Particle Filter before performing optimization.")