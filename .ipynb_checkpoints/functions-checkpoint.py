import pandas as pd
import uproot 
import awkward as ak

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

from scipy.optimize import curve_fit

def load_rectangular_data(run_list, directory):
    data_rect_array=pd.DataFrame() #it will contain the columns with rectangular arrays
    for run in (run_list):
        df = pd.DataFrame() 

        try:
            if run<10000:
                file_path = f"{directory}/reco_run0{run}_3D.root"
                
            else:
                file_path = f"{directory}/reco_run{run}_3D.root"
            print(f"loading from {file_path}")   
            file=uproot.open(file_path)
            tree=file["Events"]
            
            branches = tree.arrays(library="np")
            
            
            rectangular_columns = [i for i in tree.keys() if i in ['run', 'event', 'pedestal_run', 'cmos_integral', 'cmos_mean','cmos_rms', 
                                                      't_DBSCAN','t_variables','lp_len','t_pedsub','t_saturation','t_zerosup','t_xycut',
                                                      't_rebin','t_medianfilter','t_noisered','nSc', 'nRedpix','Lime_pressure', 'Atm_pressure', 
                                                      'Lime_temperature', 'Atm_temperature', 'Humidity', 'Mixture_Density']]
                
            for column in rectangular_columns:
                df[column]=ak.ravel(ak.Array(branches[column])).tolist()
    
            data_rect_array= data_rect_array.append(df, ignore_index=True)
    
        except:
            print('File of run ', run, 'not found in cloud.')
        
    return data_rect_array
#------------------------------------------------------------------------
def load_jagged_data(run_list, directory):
    
    data_jagged_array=pd.DataFrame() #jagged arrays
    for run in (run_list):
        df = pd.DataFrame() 
        try:
            if run<10000:
                file_path = f"{directory}/reco_run0{run}_3D.root"  
            else:
                file_path = f"{directory}/reco_run{run}_3D.root"
                
            file=uproot.open(file_path)
            tree=file["Events"]
            branches = tree.arrays(library="np")
        
        
            # The cited columns have rectangular arrays, so we have to discard them because the other columns in the dataset have jagged arrays.
             # 'redpix_ix', redpix_iy, redpix_iz and sc_redpixIdx are jagged arrays but had to be excluded 
            jagged_columns = [i for i in tree.keys() if i not in ['run', 'event', 'pedestal_run', 'cmos_integral', 'cmos_mean','cmos_rms', 
                                                  't_DBSCAN','t_variables','lp_len','t_pedsub','t_saturation','t_zerosup','t_xycut',
                                                  't_rebin','t_medianfilter','t_noisered','nSc', 'nRedpix', 'redpix_ix','redpix_iy','redpix_iz','sc_redpixIdx', 
                                                  'Lime_pressure', 'Atm_pressure', 'Lime_temperature', 'Atm_temperature', 'Humidity','Mixture_Density']]
        
            for column in jagged_columns:
                df[column]=ak.ravel(ak.Array(branches[column])).tolist()        
        
            df['Run'] = run
            
            data_jagged_array= data_jagged_array.append(df, ignore_index=True)
    
        except:
            print('File of run ', run, 'not found in cloud.')
            
    return data_jagged_array
#------------------------------------------------------------------------

def fit_gaussian_to_histogram_absolute(data, n_bins, x_limit, y_limit, label, title, value,path_to_save_figure):
    """
    Fits a Gaussian distribution to an absolute value histogram of the input data.

    Parameters:
        data (array-like): Input data (array of floats).
        n_bins (int): Number of bins for the histogram.
        x_limit (float): Upper limit for x-axis.
        label (str): Label for the histogram.

    Returns:
        tuple: Fitted parameters (A, mu, sigma) of the Gaussian and their errors.
    """
    # Create the histogram with absolute counts
    hist, bin_edges = np.histogram(data, bins=n_bins, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers
    bin_width = bin_edges[1] - bin_edges[0]  # Calculate bin width

    # Define the Gaussian function for absolute counts
    def gaussian(x, A, mu, sigma):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

    # Initial guesses for the parameters
    max_height = np.max(hist)  # Amplitude (maximum bin count)
    mu_guess = bin_centers[np.argmax(hist)]  # Centroid (bin with maximum count)
    sigma_guess = (bin_edges[-1] - bin_edges[0]) / 4  # Rough estimate for sigma
    
    # Enforce bounds: A > 0, sigma > 0
    bounds = ([0, -np.inf, 0], [np.inf, np.inf, np.inf])

    # Fit the Gaussian to the histogram
    initial_guess = [max_height, mu_guess, sigma_guess]
    params, covariance = curve_fit(gaussian, bin_centers, hist, p0=initial_guess, bounds=bounds)
    A_fit, mu_fit, sigma_fit = params

    # Extract uncertainties (errors) from the covariance matrix
    errors = np.sqrt(np.diag(covariance))
    A_err, mu_err, sigma_err = errors

    # Plot the histogram and the fitted Gaussian
    plt.bar(bin_centers, hist, width=bin_width, label=label, alpha=0.6)
    x_fit = np.linspace(bin_edges[0], bin_edges[-1], 500)
    y_fit = gaussian(x_fit, A_fit, mu_fit, sigma_fit)
    plt.plot(x_fit, y_fit, 'r-', label='Fitted Gaussian')

    # Display the fit parameters on the plot
    plt.text(0.80, 0.25, f"A = {A_fit:.2f}\nμ = {mu_fit:.2f}\nσ = {sigma_fit:.2f}",
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.xlim(0, x_limit)
    plt.ylim(0, y_limit)
    plt.xlabel(value)
    plt.ylabel('Counts')  # Changed from 'Density' to 'Counts'
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(path_to_save_figure)
    plt.show()

    # Return the fitted parameters
    return params, errors

