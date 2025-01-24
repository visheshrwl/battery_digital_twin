# utils.py
import numpy as np
import matplotlib.pyplot as plt

def plot_capacity_fade(all_cycles, all_capacities, lifetimes, sort_idxs, color_list, save_path='Figures/all_124_capacity_fade_curves.pdf'):
    """Plot capacity fade curves for all cells."""
    plt.figure(num=1, dpi=400, figsize=(4,3.5))
    plt.rcParams["font.family"] = "Times New Roman"
    fontsize=12
    for c, idx in enumerate(sort_idxs):
        plt.plot(all_cycles[idx], all_capacities[idx], color=color_list[c], linewidth=1)
    plt.hlines(0.88,0,2500, colors='k', linestyles='-.')
    plt.xlim([0,2500])
    plt.ylim([0.4, 1.15])
    plt.xlabel('Cycle Number', fontsize=fontsize)
    plt.ylabel('Discharge Capacity (Ah)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(save_path, bbox_inches = "tight")
    plt.show()

def plot_utility_functions(ah_utility_values, mtbc_utility_values, ah_bounds, dchg_time_bounds, colors, linestyles, save_paths=['Figures/Ah_utility.pdf', 'Figures/MTBC_utility.pdf']):
    """Plot utility functions for total Ah and mean discharge time."""
    fontsize=14
    # Total Ah utility
    plt.figure(dpi=400, figsize=(4,3.5))
    plt.plot(np.linspace(3, 2000, 1000), ah_utility_values, color=colors[0], linestyle=linestyles[0], linewidth=2)
    plt.xlabel('Total Ah Throughput', fontsize=fontsize)
    plt.ylabel('Utility', fontsize=fontsize)
    plt.savefig(save_paths[0], bbox_inches = "tight")
    plt.show()
    
    # MTBC utility
    plt.figure(dpi=400, figsize=(4,3.5))
    plt.plot(np.linspace(0.15, 0.3, 1000), mtbc_utility_values, color=colors[1], linestyle=linestyles[1], linewidth=2)
    plt.xlabel('Mean Time Between Charges', fontsize=fontsize)
    plt.ylabel('Utility', fontsize=fontsize)
    plt.savefig(save_paths[1], bbox_inches = "tight")
    plt.show()

def plot_single_cell_optimization(K_data, Q_data, complete_original_cycles, combined_pf_capacity, cycle_max, K_eol, online_idx, cell_idx, test_str='Primary Test Cell', save_path='Figures/utility_panel_cell.pdf'):
    """Plot optimization results for a single cell."""
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(4,7), dpi=400)
    fig.tight_layout(pad=2.75)
    fontsize = 14
    aa = 0
    start = K_eol  # Adjust if needed
    ax[aa].plot(K_data, Q_data, color='k', label='Measured Capacity', linewidth=1.75)
    ax[aa].plot(complete_original_cycles[start:], combined_pf_capacity[start:] / combined_pf_capacity[0], color='#2E86C1', label='PF Projection', linestyle='--', linewidth=1.75)
    ax[aa].vlines(cycle_max, -1, 2, color='#E74C3C', label='Optimal Retirement', linestyle='-.', linewidth=1.75)
    ax[aa].vlines(complete_original_cycles[online_idx], -10, 12, color='#707B7C', label='Current Cycle', linewidth=1.75, linestyle=':')
    ax[aa].set_xlabel('Cycle Number', fontsize=fontsize)
    ax[aa].set_ylabel('Normalized Discharge Capacity', fontsize=fontsize)
    ax[aa].tick_params(axis='both', labelsize=fontsize-1)
    ax[aa].set_ylim([0, 1.05])
    ax[aa].set_xlim([np.min(complete_original_cycles), K_eol])
    ax[aa].grid()
    ax[aa].set_title(f'{test_str} {cell_idx + 1}', fontsize=fontsize)
    
    leg = ax[aa].legend(fontsize=fontsize - 2, loc="upper center", ncol=2, frameon=True, framealpha=1, edgecolor='k', fancybox=False, 
                        bbox_to_anchor=(0.5,1.35))
    for line in leg.get_lines():
        line.set_linewidth(1.5)
    
    aa = 1
    colors = ['#8E44AD','#2ECC71','#3498DB']
    linestyles = ['-.','--','-']
    ax[aa].plot(cycles, 0.5*function_values[:,1], color=colors[0], label='Total Ah', linestyle=linestyles[0])
    ax[aa].plot(cycles, 0.5*function_values[:,2], color=colors[1], label='MTBC', linestyle=linestyles[1])
    ax[aa].plot(cycles, function_values[:,0], color=colors[2], label='Overall', linestyle=linestyles[2])
    ax[aa].vlines(complete_original_cycles[online_idx], -10, 12, color='#707B7C', label='Current Cycle', linewidth=1.25, linestyle=':')
    ax[aa].vlines(cycle_max, -1, 2, color='#E74C3C', label='Optimal Retirement', linestyle='-.', linewidth=1.75)
    ax[aa].plot(cycle_max, function_values[idx_max,0], color='#E74C3C', marker='o', markersize=6)
    ax[aa].set_xlabel('Cycle Number', fontsize=fontsize)
    ax[aa].set_ylabel('Utility', fontsize=fontsize)
    ax[aa].tick_params(axis='both', labelsize=fontsize-1)
    ax[aa].set_ylim([0, 1.05])
    ax[aa].grid()
    ax[aa].set_xlim([np.min(complete_original_cycles), K_eol])
    leg = ax[aa].legend(fontsize=fontsize - 2, loc="upper center", ncol=2, frameon=True, framealpha=1, edgecolor='k', fancybox=False,
                        bbox_to_anchor=(0.5,-0.18))
    for line in leg.get_lines():
        line.set_linewidth(1.5)
    plt.savefig(save_path, bbox_inches = "tight")
    plt.show()
