# optimization.py
import numpy as np
import matplotlib.pyplot as plt

def calculate_total_discharge_Ah(cycle_1, cycle_2, K_data, Q_data):
    """Calculate the total discharge Ah between two cycles."""
    cycle_1 = int(cycle_1)
    cycle_2 = int(cycle_2)
    cycle_idxs = np.arange(cycle_1 - 2, cycle_2 - 2, 1, dtype=int)
    cumulative_Ah = np.sum(Q_data[cycle_idxs])
    return cumulative_Ah

def calculate_mean_discharge_time(cycle_1, cycle_2, K_data, Q_data):
    """Calculate the mean discharge time between two cycles."""
    cycle_1 = int(cycle_1)
    cycle_2 = int(cycle_2)
    cycle_idxs = np.arange(cycle_1 - 2, cycle_2 - 2, 1, dtype=int)
    mean_recharge_interval = np.mean(Q_data[cycle_idxs] / 4.4)  # 4C discharge
    return mean_recharge_interval

def total_discharge_ah_utility(Ah, lower_bound, upper_bound):
    """Utility function for total discharge Ah."""
    Ah = np.array(Ah).reshape(-1)
    utility = np.where(Ah < lower_bound, 0,
                       np.where(Ah > upper_bound, 1,
                                (Ah - lower_bound) / (upper_bound - lower_bound)))
    return utility.reshape(-1,1)

def mean_dchg_time_utility(mean_hours, lower_bound, upper_bound):
    """Utility function for mean discharge time."""
    mean_hours = np.array(mean_hours).reshape(-1)
    utility = np.where(mean_hours < lower_bound, 0,
                       np.where(mean_hours > upper_bound, 1,
                                (mean_hours - lower_bound) / (upper_bound - lower_bound)))
    return utility.reshape(-1,1)

def mean_chg_time_utility(mean_hours, lower_bound, upper_bound):
    """Utility function for mean charge time."""
    mean_hours = np.array(mean_hours).reshape(-1)
    utility = np.where(mean_hours < lower_bound, 1,
                       np.where(mean_hours > upper_bound, 0,
                                (-1 / (upper_bound - lower_bound)) * mean_hours + (upper_bound / (upper_bound - lower_bound))))
    return utility.reshape(-1,1)

def optimize_replacement_time(results_dic, Q_data, K_data, weights, 
                              ah_bounds=(300,1000), mtbc_bounds=(0.210,0.250),
                              eol_threshold=0.50):
    """
    Optimize the replacement cycle based on utility functions.
    """
    cycles = results_dic['cycles']
    rul_pred = results_dic['rul_pred']
    rul_pred_lb = results_dic['rul_pred_lb']
    rul_pred_ub = results_dic['rul_pred_ub']
    
    # Define utility weights
    ah_weight, mtbc_weight = weights
    
    # Define optimization criteria
    function_values = []
    utility_cycles = cycles  # Can be customized if needed
    
    for x in utility_cycles:
        Ah = calculate_total_discharge_Ah(2, x, K_data, Q_data)
        Ah_utility = total_discharge_ah_utility(Ah, ah_bounds[0], ah_bounds[1])
        
        dchg_time = calculate_mean_discharge_time(2, x, K_data, Q_data)
        dchg_time_utility = mean_dchg_time_utility(dchg_time, mtbc_bounds[0], mtbc_bounds[1])
        
        first_life_utility = (ah_weight * Ah_utility) + (mtbc_weight * dchg_time_utility)
        function_values.append(first_life_utility)
    
    function_values = np.array(function_values).flatten()
    idx_max = np.argmax(function_values)
    cycle_max = utility_cycles[idx_max]
    
    return cycle_max, function_values
