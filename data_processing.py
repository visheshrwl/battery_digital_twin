# data_processing.py
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def load_conditions(file_path='124 LFP Cell Conditions.xlsx'):
    """Load and preprocess cell conditions from an Excel file."""
    conditions = pd.read_excel(file_path)
    train_conds = conditions[conditions['Dataset'] == 'Train'].reset_index(drop=True)
    test1_conds = conditions[conditions['Dataset'] == 'Prim. Test'].drop(index=42).reset_index(drop=True)
    test2_conds = conditions[conditions['Dataset'] == 'Sec. test'].reset_index(drop=True)
    return train_conds, test1_conds, test2_conds

def load_capacity_data(train_path='124 LFP Capacity Data/train',
                      test1_path='124 LFP Capacity Data/test1',
                      test2_path='124 LFP Capacity Data/test2',
                      num_train=41, num_test1=42, num_test2=40):
    """Load capacity data for training and testing datasets."""
    def load_data_subset(subset_path, num_cells):
        x, y = [], []
        for i in range(1, num_cells + 1):
            data = np.loadtxt(f'{subset_path}/cell{i}.csv', delimiter=',')
            x.append(data[:,0])
            y.append(data[:,1])
        return x, y

    x_train, y_train = load_data_subset(train_path, num_train)
    x_test1, y_test1 = load_data_subset(test1_path, num_test1)
    x_test2, y_test2 = load_data_subset(test2_path, num_test2)
    return x_train, y_train, x_test1, y_test1, x_test2, y_test2

def extrapolate_capacity(all_x, all_y, extrap_len=3000, last_n=30):
    """Extrapolate capacity fade curves using a linear model."""
    def linear_model(x, a, b):
        return a*x + b

    extrapolated_cap_data_all = []
    extrapolated_cycle_data_all = []
    for i in range(len(all_x)):
        data = all_y[i]
        x = np.arange(1, len(data)+1, 1)
        x = x[-last_n:]
        y = data[-last_n:]
        
        parameters, _ = curve_fit(linear_model, x, y)
        a, b = parameters
        next_x = np.arange(x[-1] + 1, x[-1] + extrap_len + 1, 1)
        next_y = linear_model(next_x, a, b)
        
        ext_data = np.concatenate((data, next_y))
        extrapolated_cap_data_all.append(ext_data)
        extrapolated_cycle_data_all.append(np.arange(2, len(ext_data)+2, 1))
    
    return extrapolated_cap_data_all, extrapolated_cycle_data_all

def save_processed_data(file_path, data_dict):
    """Save processed data using pickle."""
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)

def load_processed_data(file_path):
    """Load processed data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def create_capacity_dataset(train_path='124 LFP Capacity Data/train',
                           test1_path='124 LFP Capacity Data/test1',
                           test2_path='124 LFP Capacity Data/test2',
                           excel_path='124 LFP Cell Conditions.xlsx',
                           output_path='capacity_dataset_124_lfp.pkl'):
    """Create and save the processed capacity dataset."""
    train_conds, test1_conds, test2_conds = load_conditions(excel_path)
    x_train, y_train, x_test1, y_test1, x_test2, y_test2 = load_capacity_data(train_path, test1_path, test2_path)
    
    all_x = x_train + x_test1 + x_test2
    all_y = y_train + y_test1 + y_test2
    
    extrapolated_cap_data_all, extrapolated_cycle_data_all = extrapolate_capacity(all_x, all_y)
    
    # Regroup the data
    x_train_ext = extrapolated_cycle_data_all[0:41]
    x_test1_ext = extrapolated_cycle_data_all[41:41+42]
    x_test2_ext = extrapolated_cycle_data_all[41+42:41+42+40]
    
    y_train_ext = extrapolated_cap_data_all[0:41]
    y_test1_ext = extrapolated_cap_data_all[41:41+42]
    y_test2_ext = extrapolated_cap_data_all[41+42:41+42+40]
    
    # Create the save dictionary
    my_dic = {
        'x_train': x_train_ext,
        'x_test1': x_test1_ext,
        'x_test2': x_test2_ext,
        'y_train': y_train_ext,
        'y_test1': y_test1_ext,
        'y_test2': y_test2_ext,
        'train_conds': train_conds,
        'test1_conds': test1_conds,
        'test2_conds': test2_conds,
    }
    
    save_processed_data(output_path, my_dic)
    
    return my_dic
