import numpy as np  # Importing NumPy for numerical operations
import pandas as pd  # Importing Pandas for data manipulation and analysis
import matplotlib.pyplot as plt
import geostatspy.GSLIB as GSLIB                       # GSLIB utilies, visualization and wrapper
import geostatspy.geostats as geostats                 # variogram calculations  
import numpy as np
from scipy.stats import linregress
from scipy.integrate import quad

def min_max_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def calculate_overlap_percentage(data1, data2, bins=50):
    # Calculate histograms
    hist1, bins = np.histogram(data1, bins=bins, range=(np.min(data1), np.max(data1)), density=True)
    hist2, _ = np.histogram(data2, bins=bins, range=(np.min(data2), np.max(data2)), density=True)
    
    # Calculate overlap area
    overlap = np.sum(np.minimum(hist1, hist2) * np.diff(bins))
    
    # Calculate area under data1's histogram for normalization
    data1_area = np.sum(hist1 * np.diff(bins))
    
    # Normalize overlap by data1's area and convert to percentage
    overlap_percentage = (overlap / data1_area) * 100
    return overlap_percentage


# Function to load and process data based on file extension
def load_and_process(filename):
    if filename.endswith('.npz'):
        data = np.load(filename)
        images = [data[key] for key in data]
        return np.stack(images).squeeze().std(axis=0).mean(axis=0)
    else:
        return np.load(filename).squeeze().std(axis=0).mean(axis=0)
    
    
# Function to load and process data
def load_and_process_data(GenAI_path, TI_path):
    df_GenAI = pd.read_excel(GenAI_path, skiprows=2)
    df_TI = pd.read_excel(TI_path, skiprows=2)
    
    df_GenAI.columns = ['Time_Day', 'Date', 'Recovery_Factor']
    df_TI.columns = ['Time_Day', 'Date', 'Recovery_Factor']
    
    for df_temp in [df_GenAI, df_TI]:
        df_temp['Time_Day'] = pd.to_numeric(df_temp['Time_Day'], errors='coerce')
        df_temp['Recovery_Factor'] = pd.to_numeric(df_temp['Recovery_Factor'], errors='coerce')
        df_temp.dropna(inplace=True)
    
    # Custom functions for the 25th and 75th percentiles
    def percentile_25(x):
        return np.percentile(x, 25)
    
    def percentile_75(x):
        return np.percentile(x, 75)
    
    # Updated aggregation with 25th and 75th percentiles
    grouped_GenAI = df_GenAI.groupby('Time_Day')['Recovery_Factor'].agg([percentile_25, percentile_75, np.mean])
    grouped_TI = df_TI.groupby('Time_Day')['Recovery_Factor'].agg([percentile_25, percentile_75, np.mean])
    
    return grouped_GenAI, grouped_TI


# Function to calculate variograms for given data
def calculate_variograms(data):
    vario_y_all = []
    for img in data:
        nlagy, varioy, nppy = geostats.gam(
            img, tmin=-9999, tmax=9999, xsiz=10, ysiz=10,
            ixd=0, iyd=1, nlag=31, isill=1.0
        )
        vario_y_all.append(varioy)
    vario_y_all = np.array(vario_y_all)
    median_y = np.mean(vario_y_all, axis=0)
    iqr_y = np.percentile(vario_y_all, [25, 75], axis=0)
    return nlagy, median_y, iqr_y,vario_y_all


def plot_with_percentiles(ax, x, data, label, color, marker, linestyle,color2):
    mean = np.mean(data, axis=1)
    lower = np.percentile(data, 25, axis=1)
    upper = np.percentile(data, 75, axis=1)

    ax.plot(x, mean, color=color2, marker=marker, linestyle=linestyle,linewidth=2)
    ax.fill_between(x, lower, upper, color=color, alpha=0.5, label=label)

def calculate_overlap_area(sample1, sample2):
    quantiles1 = np.percentile(sample1, np.linspace(0, 100, 100))
    quantiles2 = np.percentile(sample2, np.linspace(0, 100, 100))
    
    slopes, intercepts = [], []
    for i in range(len(quantiles1) - 1):
        slope, intercept, _, _, _ = linregress([quantiles1[i], quantiles1[i+1]], [quantiles2[i], quantiles2[i+1]])
        slopes.append(slope)
        intercepts.append(intercept)
    
    def segment_linear_fit(x, index):
        return slopes[index] * x + intercepts[index]
    
    total_absolute_area = sum(
        quad(lambda x: abs(segment_linear_fit(x, i) - x), quantiles1[i], quantiles1[i+1])[0]
        for i in range(len(slopes))
    )
    
    return total_absolute_area


def calculate_overlaps(reals,fakes,fake_re_normalize = True, func = calculate_overlap_percentage):
    """
    real: (B,C,H,W)
    fake: (B,C,H,W)
    """
    if fake_re_normalize:
        fakes = (fakes - fakes.mean()) / fakes.std() * reals.std() + reals.mean()

    # Normalize the datasets
    sample1 = min_max_normalize(reals)
    sample2 = min_max_normalize(fakes)

    # Flatten the combined tensor from data1
    sample1_flattened = sample1.flatten()

    # Initialize a list to hold overlap percentages for the current num
    current_overlaps = []

    for img in sample2:
        # Flatten the image
        sample2_flattened = img.flatten()

        # Calculate the overlap percentage and append it to the current list        
        overlap_pct = func(sample1_flattened, sample2_flattened)

        current_overlaps.append(overlap_pct)
    return current_overlaps

def calculate_variogram_overlap(reals, fakes, Range):
    """
    reals: (B,H,W)
    fakes: (B,H,W)
    *~150 samples are enough
    """
    nlag_new, median_new, iqr_new,Vari1 = calculate_variograms(reals)
    nlag, median_epoch, iqr_epoch,Vari2 = calculate_variograms(fakes)
    
    lower_bound = np.maximum(iqr_epoch[0, :], iqr_new[0, :])
    upper_bound = np.minimum(iqr_epoch[1, :], iqr_new[1, :])
    
    overlap = np.where(upper_bound >= lower_bound, True, False)
    
    Oc = abs((lower_bound * overlap-upper_bound * overlap)[:int(Range/10)]).mean()
    Tot = abs((iqr_new[0, :]-iqr_new[1, :])[:int(Range/10)]).mean()
    
    return Oc/Tot*100