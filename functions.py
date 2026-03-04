import pandas as pd
import uproot 
import awkward as ak

import numpy as np
import os
import copy
import re

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

from scipy.optimize import curve_fit
RECT_COLS = [
    'run','event','pedestal_run', 
    'cmos_integral','cmos_mean','cmos_rms',
    #'t_DBSCAN','t_variables','lp_len','t_pedsub',
    't_saturation','t_zerosup',
    #'t_xycut','t_rebin','t_medianfilter','t_noisered',
    'nSc','nRedpix',
    'Lime_pressure','Atm_pressure',
    'Lime_temperature','Atm_temperature',
    'Humidity',#'Mixture_Density'
]

def load_rectangular_data_fast(run_list, directory, extra_meta=None, verbose=True):
    rows = []
    for run in run_list:
        try:
            file_path = ( f"{directory}/reco_run0{run}_3D.root" if run < 10000 else f"{directory}/reco_run{run}_3D.root")
            #print(f"loading from {file_path}")   
            tree = uproot.open(file_path)["Events"]
            arr = tree.arrays(RECT_COLS, library="ak")
            df = pd.DataFrame({c: ak.to_numpy(arr[c]) for c in RECT_COLS})

            # meta/labels
            df["Run"] = run
            if extra_meta:
                for k, v in extra_meta.items():
                    df[k] = v

            rows.append(df)

        except Exception as e:
            if verbose:
                print(f"[WARN] run {run} não carregou: {e}")

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    
    
    

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
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_and_plot_histogram(ax, data, gain_label, color, FIT_LIMITS):
    lims   = FIT_LIMITS[gain_label]
    data   = data[(data > lims['x_min']) & (data < lims['x_max'])]

    hist, bin_edges = np.histogram(data, bins=lims['n_bins'])
    bin_centers     = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width       = bin_edges[1] - bin_edges[0]

    # Gaussian fit
    p0 = [hist.max(), bin_centers[hist.argmax()], (lims['x_max']-lims['x_min'])/6]
    params, cov = curve_fit(gaussian, bin_centers, hist, p0=p0,
                            bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]))
    A_fit, mu_fit, sigma_fit = params
    errs = np.sqrt(np.diag(cov))

    # Energy resolution (FWHM / mean)
    fwhm       = 2.355 * sigma_fit
    resolution = (fwhm / mu_fit) * 100  # in %

    # Plot
    ax.bar(bin_centers, hist, width=bin_width, alpha=0.5, color=color, label=gain_label)
    x_fit = np.linspace(lims['x_min'], lims['x_max'], 500)
    ax.plot(x_fit, gaussian(x_fit, *params), color=color, linewidth=2, label='Gaussian fit')
    ax.axvline(mu_fit, color=color, linestyle='--', linewidth=1)
    ax.set_xlabel(sc_variable)
    ax.set_ylabel('Counts')
    ax.set_title(f'{gain_label}\nμ={mu_fit:.1f}±{errs[1]:.1f}  σ={sigma_fit:.1f}±{errs[2]:.1f}  R={resolution:.1f}%')
    ax.legend()

    return params, errs, resolution
#------------------------------------------------------------------------
def saturation_metrics(df):
    if df.empty:
        return {}

    sat = df["t_saturation"].fillna(0).to_numpy() > 0
    out = {
        "n_events": int(len(df)),
        "n_saturated": int(sat.sum()),
        "frac_saturated": float(sat.mean()),
        "cmos_integral_p50": float(np.nanpercentile(df["cmos_integral"], 50)),
        "cmos_integral_p95": float(np.nanpercentile(df["cmos_integral"], 95)),
        "cmos_integral_p99": float(np.nanpercentile(df["cmos_integral"], 99)),
    }
    if sat.any():
        s = df.loc[sat, "cmos_integral"].to_numpy()
        out.update({
            "sat_cmos_integral_p50": float(np.nanpercentile(s, 50)),
            "sat_cmos_integral_p95": float(np.nanpercentile(s, 95)),
            "sat_cmos_integral_p99": float(np.nanpercentile(s, 99)),
        })
    if "nRedpix" in df.columns:
        out["nRedpix_p95"] = float(np.nanpercentile(df["nRedpix"], 95))
    return out


def saturation_report(df_all, by=("dataset_id",)):
    rows = []
    for keys, g in df_all.groupby(list(by)):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        row = {by[i]: keys[i] for i in range(len(by))}
        row.update(saturation_metrics(g))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("frac_saturated", ascending=False)
#------------------------------------------------------------------------

def hist(x, bins):
    h, e = np.histogram(x, bins=bins)
    c = 0.5*(e[:-1] + e[1:])
    return c, h

def find_peak_center(x, bins, search_range=None):
    c, h = hist(x, bins)
    m = np.isfinite(c)
    if search_range is not None:
        m &= (c >= search_range[0]) & (c <= search_range[1])
    if not np.any(m):
        return np.nan
    return float(c[m][np.argmax(h[m])])

def normalize_to_peak_area(x, bins, peak_center, half_window):
    c, h = hist(x, bins)
    m = (c >= peak_center-half_window) & (c <= peak_center+half_window)
    area = float(h[m].sum())
    scale = 1.0/area if area > 0 else np.nan
    return c, h*scale, scale

def overlay_hg_lg(df_hg, df_lg, bins=200, peak_half_window=80, title="Fe HG vs LG"):
    # para comparação de espectro, remove saturados (senão distorce cauda)
    x_hg = df_hg[df_hg["t_saturation"].fillna(0)==0]["cmos_integral"].to_numpy()
    x_lg = df_lg[df_lg["t_saturation"].fillna(0)==0]["cmos_integral"].to_numpy()

    # encontra pico automaticamente
    p_hg = find_peak_center(x_hg, bins)
    p_lg = find_peak_center(x_lg, bins)

    c_hg, h_hg, _ = normalize_to_peak_area(x_hg, bins, p_hg, peak_half_window)
    c_lg, h_lg, _ = normalize_to_peak_area(x_lg, bins, p_lg, peak_half_window)

    plt.figure(figsize=(10,4))
    plt.step(c_hg, h_hg, where="mid", label=f"HG peak@{p_hg:.1f}")
    plt.step(c_lg, h_lg, where="mid", label=f"LG peak@{p_lg:.1f}")
    plt.yscale("log")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.xlabel("cmos_integral (a.u.)")
    plt.ylabel("Counts (norm. to Fe peak area)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------
def normalize_bkg_to_calib(bkg_x, calib_x, bins=200, sideband=(0, 300)):
    c_b, h_b = hist(bkg_x, bins)
    c_c, h_c = hist(calib_x, bins)

    sb = (c_b >= sideband[0]) & (c_b <= sideband[1])
    ab = float(h_b[sb].sum())
    ac = float(h_c[sb].sum())

    scale = (ac/ab) if ab > 0 else np.nan
    return c_b, h_b*scale, scale
#------------------------------------------------------------------------------
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
def parse_description(desc: str):
    """
    Retorna (kind, gain, step) a partir de run_description.
    kind: "Fe" | "BKG" | "Other"
    gain: "HG" | "LG"
    step: 1..5 ou None
    """
    d = (desc or "").strip()

    # gain
    gain = "LG" if re.search(r"low\s*gain|LOW\s*Gain|Low\s*Gain", d) else "HG"

    # kind
    if re.search(r"\bBKG\b|Background", d, re.IGNORECASE):
        kind = "BKG"
    elif re.search(r"\bFe\b|55Fe|Fe Calibration|Daily Calibration", d, re.IGNORECASE):
        kind = "Fe"
    else:
        kind = "Other"

    # step
    m = re.search(r"step\s*([1-5])", d, re.IGNORECASE)
    step = int(m.group(1)) if m else None

    return kind, gain, step


def build_datasets_from_logbook(
    logbook_df,
    run_range,                 # (start, stop)
    steps_of_interest,         # lista do teu configs.py
    selected_gem_voltage=None  # ex: 400
):
    """
    Filtra Run4 + steps_of_interest + (opcional) GEM1_V e cria dict:
      dataset_id -> {runs:[...], meta:{...}}
    """
    start, stop = run_range

    mask = (
        (logbook_df["source_type"] == 1) &
        (logbook_df["run"] >= start) &
        (logbook_df["run"] <= stop) &
        (logbook_df["run_description"].isin(steps_of_interest))
    )
    if selected_gem_voltage is not None and "GEM1_V" in logbook_df.columns:
        mask &= (logbook_df["GEM1_V"] == selected_gem_voltage)

    df = logbook_df.loc[mask].copy()

    # parse labels
    parsed = df["run_description"].apply(parse_description)
    df["kind"] = parsed.apply(lambda x: x[0])
    df["gain"] = parsed.apply(lambda x: x[1])
    df["step"] = parsed.apply(lambda x: x[2])

    # aqui é onde entram as tuas “combinações de VGEM e drift field”
    # usa as colunas que existirem no logbook (ex: GEM1_V, DRIFT_V, etc.)
    group_cols = ["kind","gain","step"]
    for c in ["GEM1_V", "DRIFT_V", "DRIFT_FIELD", "VGEM", "drift_field"]:
        if c in df.columns:
            group_cols.append(c)

    datasets = {}
    for key, g in df.groupby(group_cols):
        # dataset_id legível
        if not isinstance(key, tuple):
            key = (key,)
        key_dict = dict(zip(group_cols, key))

        dataset_id = f"{key_dict.get('kind','X')}_step{key_dict.get('step','NA')}_{key_dict.get('gain','X')}"
        if "GEM1_V" in key_dict:
            dataset_id += f"_VGEM{key_dict['GEM1_V']}"
        if "DRIFT_V" in key_dict:
            dataset_id += f"_DRIFTV{key_dict['DRIFT_V']}"
        if "DRIFT_FIELD" in key_dict:
            dataset_id += f"_DRIFTF{key_dict['DRIFT_FIELD']}"

        datasets[dataset_id] = {
            "runs": sorted(g["run"].unique().tolist()),
            "meta": key_dict
        }

    return datasets, df
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

