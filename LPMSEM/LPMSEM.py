#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday, April 14, 2025
@author: js4959

Lumped-Parameter Mixed-Solvent Evaporation Model

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
from scipy.integrate import solve_ivp
import pickle

#%% Define pickle save function for exporting results to process map files    
def save(filename, *args):
    # Get global dictionary
    glob = globals()
    d = {}
    for v in args:
        # Copy over desired values
        d[v] = glob[v]
    with open(filename, 'wb') as f:
        # Put them in the file 
        pickle.dump(d, f)

# %% Define derivative function
def f(t, y, c):
    phia = y[0]
    phib = y[1]
    phic = 1-y[0]-y[1]
    h = y[2]
    
    #unpack 'A'
    Va = c[0][3]
    alphaa = c[0][5]

    #unpack 'B'   
    Vb = c[1][3]
    alphab = c[1][5]
    
    #unpack 'C'
    Vc = c[2][0]
    alphac = 0    
    
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


# %% Start for loop for solvent volume fraction and and solvent composition
# Add desired conditions to svf_array and sol1_array
# If more than two solvents are desired, derivative function and sol1_array need to be adjusted

solutions = []

svf_array = [0.2] #[-] solid volume fraction (solute concentration)
sol1_array = [0.2, 0.5, 0.8] #[-] volume fraction of component 1 in liquid (solvent) phase

for volfrac1 in sol1_array:
    for volfracsolid in svf_array:
    
        # %% Define film and coating properties
        #Constants
        R = 8.3145e-5 #[m^3 bar/K/mol]
        T = 313.15 #[K] Temperature (isothermal)
        
        #Mass and heat transfer coefficients
        beta = 3e-3 #[m/s] 1e-3 slow convection, 1e-2 fast convection
 
        #Coating Parameters
        h0 = 20e-6 #[m] initial film height
        tc = h0/beta #[s] time constant to dimensionalize time

       
        #Heat transfer model properties
        #Can also use to pass parameters through 'sol' file
        evap_prop = [R, beta, h0, tc, T]

    
        # %% Define solvent 'a' - DMF
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
        
        aprop = [aA, aB, aC, Va, rhoa, alphaa]
        
        # %% Define solvent 'b' - DMSO
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
        
        bprop = [bA, bB, bC, Vb, rhob, alphab]
        
        # %% Define solid component properties 'c', typically perovskite
        alphac = 0 # Nonvolatile
        Mwc = 620 #[g/mol] MAPbI3
        rhoc = 6.6e-3 #[mol/cm3]
        Vc = 1/(rhoc*(10**6)) #[mol/m3] 
        
        cprop = [Vc, rhoc, alphac]
        
        
        
        # %% Define time spans, initial values, and constants
        tspan = np.linspace(0,1e6,10000000) #choose based on constants
            
        yinit = [volfrac1*(1-volfracsolid),\
                 (1-volfrac1)*(1-volfracsolid),\
                     1] #[phia0, phib0, h0, T]
        
        # %% Enter the values corresponding to the solvents of interest and constants    
        c = [aprop, bprop, cprop, evap_prop]
        
        
        # %% Define event to terminate solution in reasonable time
        # Specify fraction of solvent to evaporate
        evappc = 0.999 #evaporation 99.9% of solvent
        evapfrac = (1-volfracsolid)*(1-evappc) + volfracsolid
        
        
        def event(t, y):
            return y[2]-evapfrac
        
        event.terminal = True
        
        # %% Solve
                
        sol = solve_ivp(lambda t, y: f(t, y, c), 
                        [tspan[0], tspan[-1]], yinit, t_eval = tspan,\
                            events = [event], rtol = 1e-9)
        sol.T = T
        sol.volfrac1 = volfrac1
        sol.volfracsolid = volfracsolid
        sol.evap_prop = evap_prop
    
        solutions.append(sol)

        # %% Save solutions to file   
        # Uncomment this section to plot process paths
        filename = str('solid_%i_DMF_%i_%i_K' %(sol.volfracsolid*100,sol.volfrac1*100, sol.T))
        save(filename, 'sol')

#%% Plot Figures

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(5, 6))
outer = gridspec.GridSpec(np.size(svf_array), np.size(sol1_array), wspace=0.4, hspace=0.2)


comb_tot = np.size(sol1_array)*np.size(svf_array)
for i in range(comb_tot):
    inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=outer[i], wspace=0.1, hspace=0.3)

    for j in range(2):
        ax = plt.Subplot(fig, inner[j])
        sol = solutions[i]
        tc = sol.evap_prop[3]
        
        if j == 0:
            ax.plot(sol.t*tc,sol.y[0,0:], '-.k', label='DMF')
            ax.plot(sol.t*tc,sol.y[1,0:], '--k', label='DMSO')
            ax.plot(sol.t*tc,1-sol.y[0,0:]-sol.y[1,0:], '-k', label=r'Solid')
            # ax.plot(sol.t*tc,sol.y[0,0:]/sol.y[1,0:], ':k', label=r'Solvent Ratio')
            ax.set_ylabel(r'vol fraction [$\phi$]')
            ax.set_xlabel('Time [s]')
            ax.set_ylim([0, 1])
            ax.set_xlim([0,sol.t[-1]*tc])
            # ax.set_xticklabels([])
            ax.set_title('%.2f:%.2f (DMF:DMSO), %i K' %(sol.volfrac1, 1-sol.volfrac1, sol.T))
            ax.legend(loc='upper left')

        
        if j == 1:
            ax.plot(sol.t*tc, sol.y[2,0:], '-k')
            ax.set_ylabel(r'Film height [$h/h_0$]')
            ax.set_xlabel('Time [s]')
            ax.set_ylim([0, 1])
            ax.set_xlim([0,sol.t[-1]*tc])
            ax.axhline(y=sol.volfracsolid,xmin=tspan[0],xmax=tspan[-1]*tc,\
            c='black',ls='--', label = 'Solid Film')
                
            
        fig.add_subplot(ax)

fig.show()
