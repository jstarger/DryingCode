#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 2024
@author: jstarger

Film thickness and time shift from confocal probe

1. Film thickness data as csv from confocal probe
2. Curve fit drying data to multi solvent drying model with known beta
3. Iterate h0 and tn until height does not change


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

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# %% Read data and get filename

file_name = 'mapbi4_protocol_confocalDT-IFC2421_2024-08-20_16-04-41_811_time0-125values.csv'
data = pd.read_csv(file_name, header=0, skiprows=1, skipfooter=0)

raw_data = 'mapbi4_protocol_confocalDT-IFC2421_2024-08-20_16-04-41.811.csv'
rawdata = pd.read_csv(raw_data, encoding='Windows-1252', sep=';', header=0, skiprows=10, skipfooter=0)
rawdata = rawdata.to_numpy()

# %% Split path to get file name

name_str = file_name.split('/')[-1].split('__')[0].split('_')
export_name = name_str[0]
for i in range(len(name_str)-2):
    export_name = export_name + '_' + name_str[i+1]
    
drying_data = data.to_numpy()
offset = 1 #[um] offset thickness based on known/est average dry film thickness
drying_data[:,1] = drying_data[:,1] + offset
dryingdata2 = np.copy(drying_data)
tadj_dryingdata = np.copy(drying_data)


# %% define function for evaporation model

def f(t, y, c):
    phia = y[0]
    phib = y[1]
    phic = 1-y[0]-y[1]
    h = y[2]
    
    alphaa = c[0,0]
    Va = c[0,1]
    alphab = c[1,0]
    Vb = c[1,1]
    alphac = c[2,0]
    Vc = c[2,1]
    
    phiVa = phia/Va
    phiVb = phib/Vb
    phiVc = phic/Vc
    
    phiVtot = phiVa + phiVb + phiVc
    
    xa = phiVa/phiVtot
    xb = phiVb/phiVtot
    xc = phiVc/phiVtot
    
    dhdt = -(alphaa*xa + alphab*xb + alphac*xc)
    
    dydt = [((phia/h)*(-dhdt))-((1/h)*alphaa*xa),\
            ((phib/h)*(-dhdt))-((1/h)*alphab*xb),\
            dhdt]
    return dydt


#%% Function for curve fit
def func(datatime, h0, tn):
    
    #Define solvent properties
    R = 8.3145e-5 #[m^3 bar/K/mol]
    T = 30+273.15 #[K]
    beta = 0.007 #[m/s] Must be determined based on pure solvent experiments
    volfrac1 = 1 #starting volume fraction of solvent 1
    volfracsolid = 0.134 # volume fraction of solid component
    
    #Define solvent 'a' - DMF
    Mwa = 73.09e-3 #[kg/mol]
    rhoa = 944 #[kg/m3]
    Va = Mwa/rhoa #[m3/mol]
    
    # Antoine (https://webbook.nist.gov/cgi/cbook.cgi?Name=dmf&Units=SI)
    aA = 3.93068
    aB = 1337.716
    aC = -82.648
    pva = 10**(aA-(aB/(T+aC))) #[bar]
    
    # alpha - expansion coefficient
    alphaa = pva*Va/R/T
    
    nA = 1.4305
    
    #Define solvent 'b' - DMSO
    Mwb = 78.13e-3 #[kg/mol]
    rhob = 1100.4 #[kg/m3]
    Vb = Mwb/rhob #[m3/mol]
    
    # Antoine (https://webbook.nist.gov/cgi/inchi/InChI%3D1S/C2H6OS/c1-4(2)3/h1-2H3)
    # Valid 293 to 323
    # Less than 10% rel difference up to 415 K
    # See DMSO pvap comparison excel sheet
    
    bA = 5.23039
    bB = 2239.161
    bC = -29.215
    pvb = 10**(bA-(bB/(T+bC))) #[bar]
    
    # alpha - expansion coefficient
    alphab = pvb*Vb/R/T
    
    nB = 1.4793
    
    #Define solid component properties 'c'
    alphac = 0 # Nonvolatile
    rhoc = 6.6e-3 #[mol/cm3]
    Vc = 1/(rhoc*(10**6)) #[mol/m3] 
    
    nC = 2.5 #Estimated from approximations
    # https://doi.org/10.1007/s10854-018-0340-2
    # https://refractiveindex.info/?shelf=other&book=CsPbI3&page=Aftenieva
    # https://doi.org/10.1088/1361-648X/aa6e6c
    # https://doi.org/10.1016/j.optmat.2023.113558
    
    #Define time spans, initial values, and constants
    
    tc = (h0*10**-6)/beta #[s] time constant to dimensionalize time
    
    # tspan = np.linspace(0,datatime[-1]/tc,len(datatime)) #choose based on constants
    datatime[0] = tn
    tspan = datatime/tc
    
    yinit = [volfrac1*(1-volfracsolid),\
             (1-volfrac1)*(1-volfracsolid),\
                 1] #[phia0, phib0, h0]
    
    #Define film components of interest    
    c = np.array([[alphaa, Va],[alphab, Vb],[alphac, Vc]])
    
    
    #Solve differential equation    
    #Define time spans, initial values, and constants    
    sol = solve_ivp(lambda t, y: f(t, y, c),\
                    [tspan[0], tspan[-1]], yinit, t_eval = tspan,\
                        rtol = 1e-9)
        
    #Define outputs
    h = sol.y[2,:]*h0
    phi1 = sol.y[0,:]
    phi2 = sol.y[1,:]
    phi3 = 1 - phi1 - phi2
    n = nA*phi1 + nB*phi2 + nC*phi3
    
    return h, n, T, sol, beta, volfracsolid


#%% wrapper function to return only h(t) for curve_fit
def hfunc(datatime, h0, tn):
    return func(datatime, h0, tn)[0]



#%% Time Shift

h0 = np.array([drying_data[0,1]]) #[um] units
tn = np.array([0]) #initial guess
parameters, covariance = curve_fit(hfunc, drying_data[:,0], drying_data[:,1], p0 = [h0[-1], tn[-1]])

h0 = np.append(h0, parameters[0])
tn = np.append(tn, parameters[1])

#iterate to find n adjusted beta
h0_error = abs((h0[-1]-h0[-2])/h0[-1])

while h0_error > 0.001:
    parameters, covariance = curve_fit(hfunc, tadj_dryingdata[:,0], tadj_dryingdata[:,1], p0 = [h0[-1], tn[-1]])
    
    h0 = np.append(h0, parameters[0])
    tn = np.append(tn, parameters[1])
    
    tadj_dryingdata[0,:] = [tn[-1], h0[-1]] #set initial point to fit h0
    
    #iterate to find t adjusted start time
    h0_error = abs((h0[-1]-h0[-2])/h0[-1])


#%% Figures

plt.figure()
fig1 = plt.figure(figsize=(12, 8))


ax1 = plt.subplot(231)
ax1.text(-0.2, 1.15, 'A', transform=ax1.transAxes, size=20)
t = np.array((rawdata[0:,1]-rawdata[0,1])/1000) #time in [s] (divide by 1000 from ms)
dist01 = np.array(rawdata[0:,4]*1000) #distance in um
ax1.scatter(t,dist01,s=1)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Absolute Distance From Probe [um]')
ax1.set_title('Raw Confocal Probe Data')


ax2 = plt.subplot(232)
ax2.text(-0.2, 1.15, 'B', transform=ax2.transAxes, size=20)
ax2.scatter(dryingdata2[:,0], dryingdata2[:,1],s=1)
ax2.set_ylim([0,12])
ax2.set_xlim([-30, 130])
ax2.axvline(0,color='black',linestyle=':')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel(r'$\Delta$Thickness (um)')
ax2.set_title("Invert data with known" "\n" r"height offset, $\delta=1$ $\mu$m")


ax3 = plt.subplot(233)
ax3.text(-0.2, 1.15, 'C', transform=ax3.transAxes, size=20)
ax3.scatter(dryingdata2[:,0], dryingdata2[:,1],s=1)
ax3.scatter(tn[0],h0[0],s=50)
ax3.plot(tadj_dryingdata[:,0], func(tadj_dryingdata[:,0],h0[0],tn[0])[0],'r--')
ax3.set_ylim([0,12])
ax3.set_xlim([-30, 130])
ax3.axvline(0,color='black',linestyle=':')
ax3.set_xlabel(r'$\Delta$Time (s)')
ax3.set_ylabel(r'$\Delta$Thickness (um)')
ax3.set_title(r"Apply model based on known" "\n" r"initial ink composition and $\beta$")

ax4 = plt.subplot(234)
ax4.text(-0.2, 1.15, 'D', transform=ax4.transAxes, size=20)
ax4.scatter(dryingdata2[:,0], dryingdata2[:,1],s=1)
ax4.scatter(parameters[1],parameters[0],s=50)
ax4.plot(tadj_dryingdata[:,0], func(tadj_dryingdata[:,0],parameters[0], parameters[1])[0],'r--')
ax4.set_ylim([0,12])
ax4.set_xlim([-30, 130])
ax4.axvline(0,color='black',linestyle=':')
ax4.set_xlabel(r'$\Delta$Time (s)')
ax4.set_ylabel(r'$\Delta$Thickness (um)')
ax4.set_title(r"Adjust $h_0$ and $t_0$ based on" "\n" r"least squares fit until $\Delta h_0<0.001$")


ax5 = plt.subplot(235)
ax5.text(-0.2, 1.15, 'E', transform=ax5.transAxes, size=20)
ax5.scatter(tadj_dryingdata[1:,0]-tn[-1], tadj_dryingdata[1:,1],s=1)
ax5.plot(tadj_dryingdata[:,0]-parameters[1], func(tadj_dryingdata[:,0],parameters[0], parameters[1])[0],'r--')
ax5.scatter(0,parameters[0],s=50)
ax5.set_ylim([0,12])
ax5.set_xlim([-30, 130])
ax5.axvline(0,color='black',linestyle=':')
ax5.set_xlabel(r'$\Delta$Time (s)')
ax5.set_ylabel(r'$\Delta$Thickness (um)')
ax5.set_title(r"Replot all data with new $h_0$ and $\Delta t$")


ax6 = inset_axes(ax5, width="40%", height ="50%")
sol = func(tadj_dryingdata[:,0],parameters[0],parameters[1])[3]
ax6.plot(tadj_dryingdata[:,0],sol.y[0,0:], '-.k', label='DMF')
# ax6.plot(tadj,sol.y[1,0:], '--k', label='DMSO')
ax6.plot(tadj_dryingdata[:,0],1-sol.y[0,0:]-sol.y[1,0:], '-k', label=r'Solid')
ax6.set_xlabel('t [s]')
ax6.set_ylabel(r'$\phi$')
# ax6.legend(loc="lower right")



fig1.tight_layout()
plt.show()


#%% save plot
# fig1.savefig('Confocaltimeshift.pdf')
