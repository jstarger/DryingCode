# -*- coding: utf-8 -*-
"""
Created on Monday, April 14, 2025
@author: js4959

Process Map Plots

For more information regarding the development and use of this model, please refer to the following publication.
Journal: Cell Reports Physical Science, 2025
Title: "Formation Trajectories of Solution-Processed Perovskite Thin Films from Mixed Solvents"
Authors: Jesse L. Starger, Amy E. Louks, Kelly Schutt, E. Ashley Gaulding, Robert W. Epps, Rosemary C. Bramante, 
Ross A. Kerner, Kai Zhu, Joseph J. Berry, Nicolas J. Alvarez, Richard A. Cairncross, and Axel F. Palmstrom


For support, please contact Jesse L. Starger (js4959 [at] drexel [dot] edu)
For questions regarding the original publication, see:
Lead contact: Axel F. Palmstrom (axel [dot] palmstrom [at] nrel [dot] gov)
Additional Correspondence: Nicolas J. Alvarez (alvarez [at] drexel [dot] edu)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
import matplotlib.patches as patches

plt.style.use('tableau-colorblind10')

from matplotlib.ticker import FormatStrFormatter

plt.rc('font', size=8, weight='light')
plt.rc('axes', linewidth=1)
plt.rcParams["font.family"] = "Arial"


# %% Upload pickle solutions from LPMSEM

import pickle

def load(filename):
    # Get global dictionary
    glob = globals()
    with open(filename, 'rb') as f:
        for k, v in pickle.load(f).items():
            # Set each global variable to the value from the file
            glob[k] = v
  
# Load each individual solution pickle file
# Add more solutions as needed

load('solid_20_DMF_20_313_K')
sol1 = sol

load('solid_20_DMF_50_313_K')
sol2 = sol

load('solid_20_DMF_80_313_K')
sol3 = sol


# %% Process Map

fig1 = plt.figure(constrained_layout=False, figsize=([8,4.5]))
gs = fig1.add_gridspec(2,4)

lbl1 = str('%.2f:%.2f (DMF:DMSO)' %(sol1.volfrac1, 1-sol1.volfrac1))
lbl2 = str('%.2f:%.2f (DMF:DMSO)' %(sol2.volfrac1, 1-sol2.volfrac1))
lbl3 = str('%.2f:%.2f (DMF:DMSO)' %(sol3.volfrac1, 1-sol3.volfrac1))

ax1 = fig1.add_subplot(gs[0:,2:])
ax1.text(-0.1, 1.05, 'E', transform=ax1.transAxes, size=10)

ax1.plot(1-sol1.y[0,0:]-sol1.y[1,0:], abs(sol1.y[1,0:]/(sol1.y[1,0:]+sol1.y[0,0:])), '-', linewidth=1, label=lbl1)
ax1.plot(1-sol2.y[0,0:]-sol2.y[1,0:], abs(sol2.y[1,0:]/(sol2.y[1,0:]+sol2.y[0,0:])), '-', linewidth=1, label=lbl2)
ax1.plot(1-sol3.y[0,0:]-sol3.y[1,0:], abs(sol3.y[1,0:]/(sol3.y[1,0:]+sol3.y[0,0:])), '-', linewidth=1, label=lbl3)


ax1.set_ylabel(r'%DMSO in Remaining Solvent')
ax1.set_xlabel(r'Solute Volume Fraction ($\phi_{solute}$)')
ax1.set_ylim([-0.05, 1.05])
ax1.set_xlim([0, 1])
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

# Approximate solubilities (Cite Petrov et al.)
# Petrov, A. A.; Ordinartsev, A. A.; Fateev, S. A.; Goodilin, E. A.; Tarasov, A. B. Solubility of Hybrid Halide Perovskites in DMF and DMSO. Molecules 2021, 26 (24), 7541. https://doi.org/10.3390/molecules26247541.
plt.axvline(x = 0.6, color = 'k', linestyle = '--', label='MAPbI3 in DMSO, 373 K')

#Add gray shading for supersaturation
rect = patches.Rectangle((0.6, 0), 0.4, 1, edgecolor='none', facecolor='lightgrey')
ax1.add_patch(rect)
ax1.text(0.56, 0.25, 'Saturation', fontsize=8, rotation=90)
ax1.legend(loc='lower right')

#%% Additional plots

ax5 = fig1.add_subplot(gs[0,0])
ax5.text(-0.25, 1.15, 'A', transform=ax5.transAxes, size=10)
ax5.plot(sol1.t*sol1.evap_prop[3], 1-sol1.y[0,0:]-sol1.y[1,0:], '-', label=lbl1)
ax5.plot(sol2.t*sol2.evap_prop[3], 1-sol2.y[0,0:]-sol2.y[1,0:], '-', label=lbl2)
ax5.plot(sol3.t*sol3.evap_prop[3], 1-sol3.y[0,0:]-sol3.y[1,0:], '-', label=lbl3)
ax5.set_ylabel('Solute Volume\n'+r'Fraction ($\phi_{solute}$)')
ax5.set_xlabel('Time [s]')
ax5.legend(fontsize=6, loc='lower right')

ax4 = fig1.add_subplot(gs[0,1])
ax4.text(-0.25, 1.15, 'B', transform=ax4.transAxes, size=10)
ax4.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax4.plot(sol1.t*sol1.evap_prop[3], sol1.y[1,0:]/(sol1.y[0,0:]+sol1.y[1,0:]), '-', label=lbl1)
ax4.plot(sol2.t*sol2.evap_prop[3], sol2.y[1,0:]/(sol2.y[0,0:]+sol2.y[1,0:]), '-', label=lbl2)
ax4.plot(sol3.t*sol3.evap_prop[3], sol3.y[1,0:]/(sol3.y[0,0:]+sol3.y[1,0:]), '-', label=lbl3)
ax4.set_ylabel('%DMSO in\n'+'Remaining Solvent')
ax4.set_xlabel('Time [s]')

ax2 = fig1.add_subplot(gs[1,0])
ax2.text(-0.25, 1.15, 'C', transform=ax2.transAxes, size=10)
ax2.plot(sol1.t*sol1.evap_prop[3], 1-sol1.y[0,0:]-sol1.y[1,0:], '-', label=lbl1)
ax2.plot(sol2.t*sol2.evap_prop[3], 1-sol2.y[0,0:]-sol2.y[1,0:], '-', label=lbl2)
ax2.plot(sol3.t*sol3.evap_prop[3], 1-sol3.y[0,0:]-sol3.y[1,0:], '-', label=lbl3)
ax2.set_ylabel('Solute Volume\n'+r'Fraction ($\phi_{solute}$)')
ax2.set_xlabel(r'Dimensionless Time ($\hat t$)')

ax3 = fig1.add_subplot(gs[1,1])
ax3.text(-0.25, 1.15, 'D', transform=ax3.transAxes, size=10)
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax3.plot(sol1.t, sol1.y[1,0:]/(sol1.y[0,0:]+sol1.y[1,0:]), '-', label=lbl1)
ax3.plot(sol2.t, sol2.y[1,0:]/(sol2.y[0,0:]+sol2.y[1,0:]), '-', label=lbl2)
ax3.plot(sol3.t, sol3.y[1,0:]/(sol3.y[0,0:]+sol3.y[1,0:]), '-', label=lbl3)
ax3.set_ylabel('%DMSO in\n'+'Remaining Solvent')
ax3.set_xlabel(r'Dimensionless Time ($\hat t$)')

fig1.tight_layout()
plt.show()


# %% Save figure as pdf

# fig1.savefig('Process_map_v4.pdf') 
