#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectrometer File Loader
Created on Fri Jul  8 15:04:03 2022
@author: rtirawat

Peak Fitting
modified by: agaulding
modified by: jstarger 8/28/23

For more information regarding the development and use of this code, please refer to the following publication.
Journal: Cell Reports Physical Science, 2025
Title: "Formation Trajectories of Solution-Processed Perovskite Thin Films from Mixed Solvents"
Authors: Jesse L. Starger, Amy E. Louks, Kelly Schutt, E. Ashley Gaulding, Robert W. Epps, Rosemary C. Bramante, 
Ross A. Kerner, Kai Zhu, Joseph J. Berry, Nicolas J. Alvarez, Richard A. Cairncross, and Axel F. Palmstrom

For support, please contact Jesse L. Starger (js4959 [at] drexel [dot] edu)
For questions regarding the original publication, see:
Lead contact: Axel F. Palmstrom (axel [dot] palmstrom [at] nrel [dot] gov)
Additional Correspondence: Nicolas J. Alvarez (alvarez [at] drexel [dot] edu)
"""


# In[1]:



import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# Below is a list of "standard" module imports, not all are used.



import matplotlib.pyplot as plt

pd.plotting.register_matplotlib_converters()
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (3.25, 3.25)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from matplotlib.ticker import FormatStrFormatter
from matplotlib import pyplot as plt




def select_fldr(title=None):
    root = tk.Tk()
    root.withdraw()
    fld_selected = filedialog.askdirectory(title=title)
    
    if fld_selected == '':
        quit()
    
    return fld_selected



def select_files(title=None):
    root = tk.Tk()
    root.withdraw()
    files_selected = filedialog.askopenfilenames(title=title)
    
    if files_selected == '':
        quit()
    
    return files_selected


def select_file():
    root = tk.Tk()
    root.withdraw()
    files_selected = filedialog.askopenfilename()
    
    if files_selected == '':
        quit()
    
    return files_selected


def file_loader(path=None):
    
    if path is None:
        path = select_file()
    
    # get abscissa (UPDATE TO APPROPRIATE DIRECTORY WHERE SAVED ON GIVEN COMPUTER)
    x_pth = os.getcwd()+'\spectrometer_abcissa.csv'
    x_vals = pd.read_csv(x_pth, header=None, names=['wavelength'])
    col_names = ['zeros'] + list(x_vals.wavelength)
    
    # read in data
    data = pd.read_csv(path, header=None, sep='\t', skiprows=15, # 15 rows in header
                       index_col=0, names=col_names)
    # drop zeros
    data.drop(columns=['zeros'], inplace=True)
    
    # change index type to timestamp
    data.index = pd.to_datetime(data.index)
    
    # get file name and first characters identifying file "AEL_#"
    
    return data,path


# In[2]:


# Load desired specra .txt file
# NOTE: Sometimes the window pops up in the background.

df,file_name = file_loader()

# Split path to get file name.
# Get timestamp

name_str = file_name.split('/')[-1].split('__')[0].split('_')
date = df.index[0].strftime("%Y%m%d") # get date from timestamp in df
export_name = name_str[0]
for i in range(len(name_str)-1):
    export_name = export_name + '_' + name_str[i+1]
    
export_name


# In[3]:


df


# In[4]:


# Only use if want to select specific time range.
# Will need to run all curves first to see plots with timestamp to know appropriate range.

# import datetime # Format - hour, min, sec, millisec
# df = df.between_time(datetime.time(17,52,11,0), datetime.time(17,52,22,0),include_start=True,include_end=True)


# In[5]:


"""
Reflectance curves processing code to determine film thickness.

E. Ashley Gaulding

Outputs:

- thickness vs. time (.csv)
- thickness vs. time (.png) plot with linear fit and rate
"""


# In[6]:


# Select range over which to analyze interference peaks.
# 450 - 900 nm generally cuts out worst end noise of spectrometer.

start_range = 500
end_range = 900

df2 = df.loc[:, lambda x: x.columns >= start_range].copy()
df3 = df2.loc[:, lambda x: x.columns <= end_range].copy()

# In[7]:


# Plot raw data for all curves for range specified above. Used as a checkpoint, but not necessary.

import matplotlib.pyplot as plt

x = np.array(df3.columns.values)
j = len(df3.index)

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

for i in range(j):
    y = df3.iloc[i]
    ax.plot(x, y)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Relectance %')
plt.show()


# In[8]:


"""
Function for detemining film thickness from number of reflectance interference peaks in a given wavelength range.
m : # of peaks
w1 : starting (shorter) wavelength (nm)
w2 : ending (longer) wavelength (nm)
n : refractive index
ang : incident angle
"""

def film_thickness(m, w1, w2, n, ang):
    v1 = 1/(w1*10**-7) # converts from wavelength in nm to wavenumber in cm^-1
    v2 = 1/(w2*10**-7)
    d = (v1 - v2)

    return((m-1)/(2*d*(np.sqrt(n**2-np.sin(ang)**2)))*10**4)




# In[9]:


"""
Modified 2023/08/10
Data processing code. This can take several minutes to run if the files are large (~10MB)

Will need to adjust variable 'n' below based on refractive index of film.
And variable 'ang' if incident angle changes.
These are under the comment "Variables for film thickness calculation" in the 'for' loop

Data is saved as .csv
"""

import scipy
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import detrend
from scipy.signal import savgol_filter
from scipy.signal import normalize
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

x = np.array(df3.columns.values) # create array of wavelength values
length = len(df3.index) # number of curves (dataframe rows)
num_peaks_list = []
film_thickness_list = []
seconds_list = []
seconds_list_0 = []
break_points_list = [] # for detrend function. Breaks data up into sections to apply linear least squares fit over that range for detrend.
num_cols = len(df3.columns)

for i in range(length): # iterate through all curves (datafrane rows) 
    
    y = df3.iloc[i] # raw data
    
    y0 = normalize(y-(np.min(y)+0.001), np.max(y)-np.min(y)) # normalize between 0 to 1. # 0.001 factor added b/c get coeff. error if too close to zero.
    y0 = y0[0] - 0.5 # center middle of waveform at zero 
    
    # create a 3-pole lowpass filter at 0.01x Nyquist frequency
    b, a = butter(3, 0.01, 'lowpass') # create Butterworth lowpass filter to eliminate high frequency noise
        # First parameter is # of poles, 2nd is the critical freq (Nyquist freq for digital system)
        # 0.01 default critical frequency, 0.02-0.03 if more sensitivity is needed
    y1 = filtfilt(b, a, y0) # apply the filter, forwards and backwards, which eliminates phase shift after filtering.
    
    # Cutoff ends of filtered data as the ends sometimes misbehave.
    y2 = y1[50:len(y1)-50]
    x2 = x[50:len(x)-50]
    
    # Checkpoint plots
    
    # plt.plot(x, y, color='black') # raw data
    # plt.title(str(df.index[i]), loc = 'left') # Timestamp
    # plt.show()
    
    # plt.plot(x, y0, color='blue') # after normalization
    # plt.show()
    
    # plt.plot(x, y0, color='blue') # after normalization
    # plt.plot(x, y1, color='red') # after lowpass filter
    # plt.plot(x2, y2, color='black') # after normalization
    # plt.show()
 
    #Find peaks
    peaks = find_peaks(y2, height = -0.2, prominence = 0.03, distance = 10) # empirically tweaked...but hopefully shoudn't need to adjust in the future since data is normalized. height is absolute height of peaks to be counted
    height = peaks[1]['peak_heights'] # list of the heights of the peaks
    peak_pos = x2[peaks[0]] # list of the peaks positions

    # Plot of final processed curve along with counted peaks
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(x2,y2)
    ax.scatter(peak_pos, height, color = 'r', s = 15, marker = 'D', label = 'Maxima')
    ax.set_title(str(df.index[i]), loc = 'left') # Timestamp
    ax.legend()
    ax.grid()
    plt.show()
    
    # Variables for film thickness calculation
    m = len(peaks[0])
    if m < 3: # Skips data with less than 3 peaks as it's too few peaks to calculate the thickness accurately.
        m = np.nan
        continue
    n = 1.4305 # Enter appropriate value for refractive index here.
    w1 = min(peak_pos)
    w2 = max(peak_pos)
    ang = 0 # incident angle. Zero is normal to surface.
    # n: for isopropanol = 1.3776, DMF = 1.4305, methoxyethanol = 1.4024 at 20 °C/D, NMP = 1.47, glass = 1.52
    # ITO = 1.9 (https://refractiveindex.info/?shelf=other&book=In2O3-SnO2&page=Moerland)
        # https://materion.com/resource-center/product-data-and-related-literature/inorganic-chemicals/oxides/ito-tin-doped-indium-oxide-for-optical-coating
        # https://www.pvlighthouse.com.au/refractive-index-library
    # FTO = 1.9
        # https://www.ijsrp.org/research-paper-0818/ijsrp-p8060.pdf
    
    num_peaks_list.append(m)
    thickness = film_thickness(m, w1, w2, n, ang)
    film_thickness_list.append(thickness)
    
    seconds = df3.index[i].minute*60 + df3.index[i].second + df3.index[i].microsecond*10**-6 # Convert index timestamp to seconds
    seconds_list_0.append(seconds)

seconds_list = np.array(seconds_list_0)-min(seconds_list_0) # set first curve to t=0sec

df_plot = pd.DataFrame({'Time (s)': seconds_list, 'Film Thickness (um)': film_thickness_list})
df_plot_2 = pd.DataFrame({'Time (s)': seconds_list_0, 'Film Thickness (um)': film_thickness_list})
df_plot.to_csv(export_name + '_thickness_vs_time.csv', index=False)


# In[10]:


df_plot_2


# In[11]:


# Plot before offsetting t = 0 to be first good data point. 
# This is just to get an idea of how far into data collection the good data used for the calculations is.

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)

fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
df_plot_2.plot(x='Time (s)', y='Film Thickness (um)', ax=ax, kind='scatter', color='blue')
ax.legend(loc="upper right", frameon=False)
ax.yaxis.set_major_locator(MultipleLocator(0.25))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Film Thickness (um)')
plt.ylim(0,5)
plt.show()

fig_name = date + '_' + name_str[0]
# fig.savefig(export_name +'_pre_outlier_'+'thickness_vs_time.png', transparent=True, bbox_inches='tight')


# In[12]:


# Define equation for use in curve_fit, in this case a linear function
def func(x, a, b):
    return a*x +b


# In[13]:


# Linear fit of thickness vs. time data
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

x_time = np.array(df_plot['Time (s)'])
y_thick = np.array(df_plot['Film Thickness (um)'])

# fit of curve
popt, pcov = curve_fit(func, x_time, y_thick)

# calculate R squared value
y_temp = func(x_time, *popt)
r2 = r2_score(y_thick, y_temp)

print('slope, y-intercept:')
print(popt)
print('R^2 value:')
print(r2)
plt.plot(np.array(df_plot['Time (s)']), func(np.array(df_plot['Time (s)']), *popt), 'b--')


# In[14]:


# Output linear fit equation

eq_clean = 'y = ' + str(round(popt[0], 3)) + 'x' + ' + ' + str(round(popt[1], 2))
print(eq_clean)


# In[15]:


# Final plot of Thickness vs. Time data


fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
df_plot.plot(x='Time (s)', y='Film Thickness (um)', ax=ax, kind='scatter', color='blue')
ax.plot(np.array(df_plot['Time (s)']), func(np.array(df_plot['Time (s)']), *popt), 'b--', label=str(abs(round(popt[0], 3)*1000))+' nm/sec')
ax.legend(loc="upper right", frameon=False)
ax.yaxis.set_major_locator(MultipleLocator(0.25))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Film Thickness (um)')
plt.ylim(0,10)
plt.show()

# fig_name = date + '_' + name_str[0]
fig.savefig(export_name+'_thickness_vs_time.png', transparent=True, bbox_inches='tight')


# In[ ]:




