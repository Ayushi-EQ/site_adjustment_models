# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 08:59:51 2024

@author: ati47
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

Features = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\9. Model development\Correlation\Features.csv"))
Features['Z1'][Features['Z1'] == 0] = 1
save_dir = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Journal papers\Paper 1\Figures\Figure Outputs")
# save_dir = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 2\1. Datasets\Categorization figures\NZGMDB")
#%% Bar graphs
desired_geomorph_order = ['Hill', 'Valley', 'Basin Edge', 'Basin']
desired_basin_order = ['Non-Basin', 'Unmodeled Basin', 'Type 1 Basin', 'Type 3 Basin', 'Type 4 Basin']
geomorph_counts = Features['Geomorphology'].value_counts().reindex(desired_geomorph_order)
basin_counts = Features['Basin Type'].value_counts().reindex(desired_basin_order)
fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True,sharey=True)
geomorph_categories = geomorph_counts.index.tolist()
geomorph_values = geomorph_counts.values
bars1 = ax[0].bar(geomorph_categories, geomorph_values, color=['#800000','#9B59B6','#33FFBD','#FF5733'])
ax[0].set_title('Number of Sites by Geomorphology', fontsize=16)
ax[0].set_xlabel('Geomorphology', fontsize=14)
ax[0].set_ylabel('Number of Sites', fontsize=14)
for bar in bars1:
    height = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}',
               ha='center', va='bottom', fontsize=12)
basin_categories = basin_counts.index.tolist()
basin_values = basin_counts.values
bars2 = ax[1].bar(basin_categories, basin_values, color=['#800000', '#FF5733', '#33FFBD', '#1F618D', '#F4D03F'])
ax[1].set_title('Number of Sites by Basin Type', fontsize=16)
ax[1].set_xlabel('Basin Type', fontsize=14)
# ax[1].set_ylabel('Number of Sites', fontsize=14)
for bar in bars2:
    height = bar.get_height()
    ax[1].text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}',
               ha='center', va='bottom', fontsize=12)
plt.savefig(os.path.join(save_dir,"GeomBasinTypecats.png"),dpi=300,bbox_inches='tight')
#%% Distribution of all obtained parameters
fig,ax = plt.subplots(2,2,figsize=(13.37,8.97),constrained_layout=True)
Vs30Q1      = Features['Vs30'][Features['QVs30']=='Q1']
Vs30Q2      = Features['Vs30'][Features['QVs30']=='Q2']
Vs30Q3      = Features['Vs30'][Features['QVs30']=='Q3']
# logbinsVs30 = np.linspace(min(Features['Vs30']),max(Features['Vs30']),15)
logbinsVs30 = np.arange(100, max(Features['Vs30']) + 100, 100)  # Ensuring even 100 m/s bins
alpha=0.7
colors= ['g', 'y', 'r']

ax[0][0].hist([Vs30Q1,Vs30Q2,Vs30Q3],bins=logbinsVs30,stacked=True,label=['Q1','Q2','Q3'],color=colors,alpha=alpha,edgecolor='k')
ax[0][0].set_xlabel('$V_{S30}$ (m/s)',fontsize=20)
ax[0][0].set_ylabel('Frequency',fontsize=20)
ax[0][0].legend(loc='best',fontsize=20,frameon=False)
ax[0][0].text(-0.15,0.95,'(a)',transform=ax[0][0].transAxes, size=20)
T0Q1      = Features['T0'][Features['QT0']=='Q1']
T0Q2      = Features['T0'][Features['QT0']=='Q2']
T0Q3      = Features['T0'][Features['QT0']=='Q3']
# logbinsT0 = np.logspace(np.log10(min(Features['T0'])),np.log10(max(Features['T0'])),15)
d = np.log10(max(Features['T0'])) - np.log10(min(Features['T0']))
logbinsT0 = np.logspace(np.log10(min(Features['T0'])), np.log10(max(Features['T0'])), num=3*int(np.ceil(d)))
ax[0][1].hist([T0Q1,T0Q2,T0Q3],bins=logbinsT0,stacked=True,label=['Q1','Q2','Q3'],color=colors,alpha=alpha,edgecolor='k')
ax[0][1].set_xlabel('$T_{0}$ (s)',fontsize=20)
ax[0][1].set_ylabel('Frequency',fontsize=20)
ax[0][1].set_xscale('log')
x_values = [0.04,0.1,0.2,0.5,1,2,3,4,5,6,7]
ax[0][1].set_xticks(x_values)
ax[0][1].set_xticklabels([str(val) for val in x_values])
ax[0][1].xaxis.set_minor_locator(mticker.NullLocator())
ax[0][1].text(-0.12,0.95,'(b)',transform=ax[0][1].transAxes, size=20)
Z1Q1      = Features['Z1'][Features['QZ1']=='Q1']
Z1Q2      = Features['Z1'][Features['QZ1']=='Q2']
Z1Q3      = Features['Z1'][Features['QZ1']=='Q3']
d = np.log10(max(Features['Z1'])) - np.log10(min(Features['Z1']))
logbinsZ1 = np.logspace(np.log10(min(Features['Z1'])),np.log10(max(Features['Z1'])),num=3*int(np.ceil(d)))
ax[1][0].hist([Z1Q1,Z1Q2,Z1Q3],bins=logbinsZ1,stacked=True,label=['Q1','Q2','Q3'],color=colors,alpha=alpha,edgecolor='k')
ax[1][0].set_xlabel('$Z_{1.0}$ (m)',fontsize=20)
ax[1][0].set_xscale('log')
x_values = [1,10,50,100,300,1000]
ax[1][0].set_xticks(x_values)
ax[1][0].set_xticklabels([str(val) for val in x_values])
ax[1][0].set_ylabel('Frequency',fontsize=20)
ax[1][0].xaxis.set_minor_locator(mticker.NullLocator())
ax[1][0].text(-0.15,0.95,'(c)',transform=ax[1][0].transAxes, size=20)
Z25Q1      = Features['Z25'][Features['QZ25']=='Q1']
Z25Q2      = Features['Z25'][Features['QZ25']=='Q2']
Z25Q3      = Features['Z25'][Features['QZ25']=='Q3']
d = np.log10(max(Features['Z25'])) - np.log10(min(Features['Z25']))
logbinsZ25 = np.logspace(np.log10(min(Features['Z25'])),np.log10(max(Features['Z25'])),num=3*int(np.ceil(d)))
ax[1][1].hist([Z25Q1,Z25Q2,Z25Q3],bins=logbinsZ25,stacked=True,label=['Q1','Q2','Q3'],color=colors,alpha=alpha,edgecolor='k')
ax[1][1].set_xlabel('$Z_{2.5}$ (m)',fontsize=20)
ax[1][1].set_xscale('log')
ax[1][1].text(-0.12,0.95,'(d)',transform=ax[1][1].transAxes, size=20)
x_values = [100,200,500,1000,5000,10000]
ax[1][1].set_xticks(x_values)
ax[1][1].set_xticklabels([str(val) for val in x_values])
ax[1][1].set_ylabel('Frequency',fontsize=20)
ax[1][1].xaxis.set_minor_locator(mticker.NullLocator())
for i in range(0,2):
    for j in range(0,2):
        ax[i][j].tick_params(labelsize=16)
# plt.savefig(os.path.join(save_dir,"Inputparams.png"),dpi=300,bbox_inches='tight')
plt.savefig(os.path.join(save_dir,"Inputparams.pdf"),bbox_inches='tight')
#%% Distribution of all obtained parameters (with geomorphology)
fig,ax = plt.subplots(2,2,figsize=(13.37,8.97),constrained_layout=True)
Vs30Basin       = Features['Vs30'][Features['Geomorphology']=='Basin']
Vs30Basinedge   = Features['Vs30'][Features['Geomorphology']=='Basin Edge']
Vs30Valley      = Features['Vs30'][Features['Geomorphology']=='Valley']
Vs30Hill        = Features['Vs30'][Features['Geomorphology']=='Hill']
# colors = ['b','#006400','#4b0082','#800000']
colors = ['#1E90FF', '#66C266', '#9370DB', '#993333']
# logbinsVs30 = np.linspace(min(Features['Vs30']),max(Features['Vs30']),15)
alpha=1
logbinsVs30 = np.arange(100, max(Features['Vs30']) + 100, 100)  # Ensuring even 100 m/s bins
ax[0][0].hist([Vs30Basin,Vs30Basinedge,Vs30Valley,Vs30Hill],bins=logbinsVs30,stacked=True,label=['Basin','Basin Edge','Valley','Hill'],color=colors,alpha=alpha,edgecolor='k')
ax[0][0].set_xlabel('$V_{S30}$ (m/s)',fontsize=20)
ax[0][0].set_ylabel('Frequency',fontsize=20)
ax[0][0].legend(loc='best',fontsize=20,frameon=False)
ax[0][0].xaxis.set_minor_locator(mticker.NullLocator())
ax[0][0].text(-0.15,0.95,'(a)',transform=ax[0][0].transAxes, size=20)
T0Basin       = Features['T0'][Features['Geomorphology']=='Basin']
T0Basinedge   = Features['T0'][Features['Geomorphology']=='Basin Edge']
T0Valley      = Features['T0'][Features['Geomorphology']=='Valley']
T0Hill        = Features['T0'][Features['Geomorphology']=='Hill']
d = np.log10(max(Features['T0'])) - np.log10(min(Features['T0']))
logbinsT0 = np.logspace(np.log10(min(Features['T0'])),np.log10(max(Features['T0'])),num=3*int(np.ceil(d)))
ax[0][1].hist([T0Basin,T0Basinedge,T0Valley,T0Hill],bins=logbinsT0,stacked=True,label=['Basin','Basin Edge','Valley','Hill'],color=colors,alpha=alpha,edgecolor='k')
ax[0][1].set_xlabel('$T_{0}$ (s)',fontsize=20)
ax[0][1].set_ylabel('Frequency',fontsize=20)
ax[0][1].set_xscale('log')
x_values = [0.04,0.1,0.2,0.5,1,2,3,4,5,6,7]
ax[0][1].set_xticks(x_values)
ax[0][1].set_xticklabels([str(val) for val in x_values])
ax[0][1].xaxis.set_minor_locator(mticker.NullLocator())
ax[0][1].text(-0.12,0.95,'(b)',transform=ax[0][1].transAxes, size=20)
Z1Basin       = Features['Z1'][Features['Geomorphology']=='Basin']
Z1Basinedge   = Features['Z1'][Features['Geomorphology']=='Basin Edge']
Z1Valley      = Features['Z1'][Features['Geomorphology']=='Valley']
Z1Hill        = Features['Z1'][Features['Geomorphology']=='Hill']
d = np.log10(max(Features['Z1'])) - np.log10(min(Features['Z1']))
logbinsZ1 = np.logspace(np.log10(min(Features['Z1'])),np.log10(max(Features['Z1'])),num=3*int(np.ceil(d)))
ax[1][0].hist([Z1Basin,Z1Basinedge,Z1Valley,Z1Hill],bins=logbinsZ1,stacked=True,label=['Basin','Basin Edge','Valley','Hill'],color=colors,alpha=alpha,edgecolor='k')
ax[1][0].set_xlabel('$Z_{1.0}$ (m)',fontsize=20)
ax[1][0].set_xscale('log')
x_values = [1,10,50,100,300,1000]
ax[1][0].set_xticks(x_values)
ax[1][0].set_xticklabels([str(val) for val in x_values])
ax[1][0].set_ylabel('Frequency',fontsize=20)
ax[1][0].xaxis.set_minor_locator(mticker.NullLocator())
ax[1][0].text(-0.15,0.95,'(c)',transform=ax[1][0].transAxes, size=20)
Z25Basin       = Features['Z25'][Features['Geomorphology']=='Basin']
Z25Basinedge   = Features['Z25'][Features['Geomorphology']=='Basin Edge']
Z25Valley      = Features['Z25'][Features['Geomorphology']=='Valley']
Z25Hill        = Features['Z25'][Features['Geomorphology']=='Hill']
d = np.log10(max(Features['Z25'])) - np.log10(min(Features['Z25']))
logbinsZ25 = np.logspace(np.log10(min(Features['Z25'])),np.log10(max(Features['Z25'])),num=3*int(np.ceil(d)))
ax[1][1].hist([Z25Basin,Z25Basinedge,Z25Valley,Z25Hill],bins=logbinsZ25,stacked=True,label=['Basin','Basin Edge','Valley','Hill'],color=colors,alpha=alpha,edgecolor='k')
ax[1][1].set_xlabel('$Z_{2.5}$ (m)',fontsize=20)
ax[1][1].set_xscale('log')
ax[1][1].text(-0.12,0.95,'(d)',transform=ax[1][1].transAxes, size=20)
x_values = [100,200,500,1000,5000,10000]
ax[1][1].set_xticks(x_values)
ax[1][1].set_xticklabels([str(val) for val in x_values])
ax[1][1].set_ylabel('Frequency',fontsize=20)
ax[1][1].xaxis.set_minor_locator(mticker.NullLocator())
for i in range(0,2):
    for j in range(0,2):
        ax[i][j].tick_params(labelsize=16)
# plt.savefig(os.path.join(save_dir,"InputparamsGeom.png"),dpi=300,bbox_inches='tight')
plt.savefig(os.path.join(save_dir,"InputparamsGeom.pdf"),bbox_inches='tight')
#%% Distribution of all obtained parameters (with basin categories)
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

# Vs30 histogram using Basin Type
Vs30NonBasin   = Features['Vs30'][Features['Basin Type'] == 'Non-Basin']
Vs30Unmodeled  = Features['Vs30'][Features['Basin Type'] == 'Unmodeled Basin']
Vs30Type1      = Features['Vs30'][Features['Basin Type'] == 'Type 1 Basin']
Vs30Type2      = Features['Vs30'][Features['Basin Type'] == 'Type 2 Basin']
Vs30Type3      = Features['Vs30'][Features['Basin Type'] == 'Type 3 Basin']
Vs30Type4      = Features['Vs30'][Features['Basin Type'] == 'Type 4 Basin']

colors = ['#800000', '#FF5733', '#33FFBD', '#9B59B6', '#1F618D', '#F4D03F']

logbinsVs30 = np.arange(100, max(Features['Vs30']) + 100, 100)
ax[0][0].hist([Vs30NonBasin, Vs30Unmodeled, Vs30Type1, Vs30Type2, Vs30Type3, Vs30Type4],
              bins=logbinsVs30, stacked=True,
              label=['Non-Basin', 'Unmodeled Basin', 'Type 1 Basin', 'Type 2 Basin', 'Type 3 Basin', 'Type 4 Basin'],
              color=colors, edgecolor='k')
ax[0][0].set_xlabel('$V_{S30}$ (m/s)', fontsize=16)
ax[0][0].set_ylabel('Frequency', fontsize=16)
ax[0][0].legend(loc='best', fontsize=14, frameon=False)
ax[0][0].xaxis.set_minor_locator(mticker.NullLocator())
ax[0][0].text(-0.12, 0.95, '(a)', transform=ax[0][0].transAxes, size=14)

# T0 histogram using Basin Type
T0NonBasin   = Features['T0'][Features['Basin Type'] == 'Non-Basin']
T0Unmodeled  = Features['T0'][Features['Basin Type'] == 'Unmodeled Basin']
T0Type1      = Features['T0'][Features['Basin Type'] == 'Type 1 Basin']
T0Type2      = Features['T0'][Features['Basin Type'] == 'Type 2 Basin']
T0Type3      = Features['T0'][Features['Basin Type'] == 'Type 3 Basin']
T0Type4      = Features['T0'][Features['Basin Type'] == 'Type 4 Basin']

d = np.log10(max(Features['T0'])) - np.log10(min(Features['T0']))
logbinsT0 = np.logspace(np.log10(min(Features['T0'])), np.log10(max(Features['T0'])),
                        num=3 * int(np.ceil(d)))
ax[0][1].hist([T0NonBasin, T0Unmodeled, T0Type1, T0Type2, T0Type3, T0Type4],
              bins=logbinsT0, stacked=True,
              label=['Non-Basin', 'Unmodeled Basin', 'Type 1 Basin', 'Type 2 Basin', 'Type 3 Basin', 'Type 4 Basin'],
              color=colors, edgecolor='k')
ax[0][1].set_xlabel('$T_{0}$ (s)', fontsize=16)
ax[0][1].set_ylabel('Frequency', fontsize=16)
ax[0][1].set_xscale('log')
x_values = [0.04, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7]
ax[0][1].set_xticks(x_values)
ax[0][1].set_xticklabels([str(val) for val in x_values])
ax[0][1].xaxis.set_minor_locator(mticker.NullLocator())
ax[0][1].text(-0.12, 0.95, '(b)', transform=ax[0][1].transAxes, size=14)

# Z1 histogram using Basin Type
Z1NonBasin   = Features['Z1'][Features['Basin Type'] == 'Non-Basin']
Z1Unmodeled  = Features['Z1'][Features['Basin Type'] == 'Unmodeled Basin']
Z1Type1      = Features['Z1'][Features['Basin Type'] == 'Type 1 Basin']
Z1Type2      = Features['Z1'][Features['Basin Type'] == 'Type 2 Basin']
Z1Type3      = Features['Z1'][Features['Basin Type'] == 'Type 3 Basin']
Z1Type4      = Features['Z1'][Features['Basin Type'] == 'Type 4 Basin']

d = np.log10(max(Features['Z1'])) - np.log10(min(Features['Z1']))
logbinsZ1 = np.logspace(np.log10(min(Features['Z1'])), np.log10(max(Features['Z1'])),
                        num=3 * int(np.ceil(d)))
ax[1][0].hist([Z1NonBasin, Z1Unmodeled, Z1Type1, Z1Type2, Z1Type3, Z1Type4],
              bins=logbinsZ1, stacked=True,
              label=['Non-Basin', 'Unmodeled Basin', 'Type 1 Basin', 'Type 2 Basin', 'Type 3 Basin', 'Type 4 Basin'],
              color=colors, edgecolor='k')
ax[1][0].set_xlabel('$Z_{1.0}$ (m)', fontsize=16)
ax[1][0].set_xscale('log')
x_values = [0.01, 1, 10, 50, 100, 300, 1000]
ax[1][0].set_xticks(x_values)
ax[1][0].set_xticklabels([str(val) for val in x_values])
ax[1][0].set_ylabel('Frequency', fontsize=16)
ax[1][0].xaxis.set_minor_locator(mticker.NullLocator())
ax[1][0].text(-0.12, 0.95, '(c)', transform=ax[1][0].transAxes, size=14)

# Z2.5 histogram using Basin Type
Z25NonBasin   = Features['Z25'][Features['Basin Type'] == 'Non-Basin']
Z25Unmodeled  = Features['Z25'][Features['Basin Type'] == 'Unmodeled Basin']
Z25Type1      = Features['Z25'][Features['Basin Type'] == 'Type 1 Basin']
Z25Type2      = Features['Z25'][Features['Basin Type'] == 'Type 2 Basin']
Z25Type3      = Features['Z25'][Features['Basin Type'] == 'Type 3 Basin']
Z25Type4      = Features['Z25'][Features['Basin Type'] == 'Type 4 Basin']

d = np.log10(max(Features['Z25'])) - np.log10(min(Features['Z25']))
logbinsZ25 = np.logspace(np.log10(min(Features['Z25'])), np.log10(max(Features['Z25'])),
                         num=3 * int(np.ceil(d)))
ax[1][1].hist([Z25NonBasin, Z25Unmodeled, Z25Type1, Z25Type2, Z25Type3, Z25Type4],
              bins=logbinsZ25, stacked=True,
              label=['Non-Basin', 'Unmodeled Basin', 'Type 1 Basin', 'Type 2 Basin', 'Type 3 Basin', 'Type 4 Basin'],
              color=colors, edgecolor='k')
ax[1][1].set_xlabel('$Z_{2.5}$ (m)', fontsize=16)
ax[1][1].set_xscale('log')
x_values = [100, 200, 500, 1000, 5000, 10000]
ax[1][1].set_xticks(x_values)
ax[1][1].set_xticklabels([str(val) for val in x_values])
ax[1][1].set_ylabel('Frequency', fontsize=16)
ax[1][1].xaxis.set_minor_locator(mticker.NullLocator())
ax[1][1].text(-0.12, 0.95, '(d)', transform=ax[1][1].transAxes, size=14)
plt.savefig(os.path.join(save_dir,"InputparamsBasinType.png"),dpi=300,bbox_inches='tight')
# plt.savefig(os.path.join(save_dir,"InputparamsGeom.pdf"),bbox_inches='tight')
#%% Calculated parameters (Hx, topographic slope, roughness)
fig,ax = plt.subplots(1,3,figsize=(14.92,3.88),constrained_layout=True)
H1250Basin          = Features['H1250'][Features['Geomorphology']=='Basin']
H1250Basinedge      = Features['H1250'][Features['Geomorphology']=='Basin Edge']
H1250Valley         = Features['H1250'][Features['Geomorphology']=='Valley']
H1250Hill           = Features['H1250'][Features['Geomorphology']=='Hill']
r1250Basin          = Features['r1250'][Features['Geomorphology']=='Basin']
r1250Basinedge      = Features['r1250'][Features['Geomorphology']=='Basin Edge']
r1250Valley         = Features['r1250'][Features['Geomorphology']=='Valley']
r1250Hill           = Features['r1250'][Features['Geomorphology']=='Hill']
SlopeBasin          = Features['Slope'][Features['Geomorphology']=='Basin']
SlopeBasinedge      = Features['Slope'][Features['Geomorphology']=='Basin Edge']
SlopeValley         = Features['Slope'][Features['Geomorphology']=='Valley']
SlopeHill           = Features['Slope'][Features['Geomorphology']=='Hill']
logbinsH1250 = np.linspace(min(Features['H1250']),max(Features['H1250']),15)
alpha=alpha
ax[0].hist([H1250Basin,H1250Basinedge,H1250Valley,H1250Hill],logbinsH1250,stacked=True,label=['Basin','Basin Edge','Valley','Hill'],color=colors,alpha=alpha,edgecolor='k')
ax[0].set_xlabel('$H_{1250}$ (m)',fontsize=20)
ax[0].set_ylabel('Frequency',fontsize=20)
ax[0].set_xlim(-100,100)

ax[0].text(-0.22,0.97,'(a)',transform=ax[0].transAxes, size=20)
logbinsr1250 = np.linspace(min(Features['r1250']),max(Features['r1250']),15)
ax[1].hist([r1250Basin,r1250Basinedge,r1250Valley,r1250Hill],logbinsr1250,stacked=True,label=['Basin','Basin Edge','Valley','Hill'],color=colors,alpha=alpha,edgecolor='k')
ax[1].set_xlabel('$r_{1250}$ (m)',fontsize=20)
ax[1].set_ylabel('Frequency',fontsize=20)
ax[1].text(-0.12,0.97,'(b)',transform=ax[1].transAxes, size=20)
ax[1].legend(loc='best',fontsize=18,frameon=False)
logbinsSlope = np.linspace(min(Features['Slope']),max(Features['Slope']),15)
ax[2].hist([SlopeBasin,SlopeBasinedge,SlopeValley,SlopeHill],logbinsSlope,stacked=True,label=['Basin','Basin Edge','Valley','Hill'],color=colors,alpha=alpha,edgecolor='k')
ax[2].set_xlabel('$S$ (m/m)',fontsize=20)
ax[2].set_ylabel('Frequency',fontsize=20)
ax[2].text(-0.12,0.97,'(c)',transform=ax[2].transAxes, size=20)
for i in range(0,3):
    ax[i].tick_params(labelsize=16)
plt.savefig(os.path.join(save_dir,"Calcparams.pdf"),bbox_inches='tight')