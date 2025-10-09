# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:17:19 2023

@author: ati47
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import mplcursors as mpl
import os.path
import addcopyfighandler

# plt.style.use('classic')
plt.rcParams["font.family"] = "Times New Roman"


def load_res_pSA_from_csv(im_data_ffp: Path):
    """Loads the residual PSA values from a csv file
    (I.e. Data from Ayushi)
    """
    im_df = pd.read_csv(im_data_ffp, index_col=0).T

    # Only interested in pSA
    pSA_columns = np.asarray([cur_col for cur_col in im_df.columns if cur_col.startswith("pSA")])
    im_df = im_df.loc[:, pSA_columns]

    # Get the pSA period values
    pSA_period_values = [float(cur_c.rsplit("_", maxsplit=1)[-1]) for cur_c in pSA_columns]
    im_df.columns = pSA_period_values

    # Ensure columns were already sorted
    assert np.all(np.sort(pSA_period_values) == pSA_period_values)

    # Drop any with nan
    im_df = im_df.dropna()

    return im_df
def hover_format(sel):
    index = sel.index
    if index is not None:
        name = OldVsNew.loc[index, 'stat_name']
    sel.annotation.set(text=name ,ha='center',va='center')
    
def load_res_pSA_from_csvFS(im_data_ffp: Path):
    """Loads the residual PSA values from a csv file
    (I.e. Data from Ayushi)
    """
    im_df = pd.read_csv(im_data_ffp, index_col=0).T

    # Only interested in pSA
    pSA_columns = np.asarray([cur_col for cur_col in im_df.columns if cur_col.startswith("X")])
    im_df = im_df.loc[:, pSA_columns]

    # Get the pSA period values
    pSA_period_values = [float(cur_c.rsplit("X", maxsplit=1)[-1]) for cur_c in pSA_columns]
    im_df.columns = pSA_period_values

    # Ensure columns were already sorted
    assert np.all(np.sort(pSA_period_values) == pSA_period_values)

    # Drop any with nan
    im_df = im_df.dropna()

    return im_df
    
OlddS2SPath = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\superceded\1. Initial Input\R - Self calculation\PJsreStationBiased_sim.csv")
NewdS2SPath = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\New Results\Residuals\PJsreStationBiased_sim.csv")
OldStats    = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\superceded\1. Initial Input\stations.csv"),index_col='stat_id')
NewStats    = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\stations.csv"),index_col='stat_id')
OlddS2S     = load_res_pSA_from_csv(OlddS2SPath)
NewdS2S     = load_res_pSA_from_csv(NewdS2SPath)
NewS2SerrPa = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\New Results\Residuals\PJSreStationerror_sim.csv")
NewS2Serr   = load_res_pSA_from_csv(NewS2SerrPa)
OldS2SerrPa = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\superceded\1. Initial Input\R - Self calculation\PJSreStationerror_sim.csv")
OldS2Serr   = load_res_pSA_from_csv(OldS2SerrPa)
Old = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\superceded\1. Initial Input\R - Self calculation\PJsvarCompsBiased_sim.csv"))
New = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\New Results\Residuals\PJsvarCompsBiased_sim.csv"))
T   = Old.iloc[6:,1]
OldbiasnSA = Old.iloc[0:6,2]
Oldbias = Old.iloc[6:,2]
NewbiasnSA = New.iloc[0:6,2]
Newbias = New.iloc[6:,2]
Oldsigma = Old.iloc[6:,6]
Newsigma = New.iloc[6:,6]
OldsigmanSA = Old.iloc[0:6,6]
NewsigmanSA = New.iloc[0:6,6]
Oldtau = Old.iloc[6:,3]
Newtau = New.iloc[6:,3]
OldtaunSA = Old.iloc[0:6,3]
NewtaunSA = New.iloc[0:6,3]
OldphiS2S = Old.iloc[6:,4]
NewphiS2S = New.iloc[6:,4]
OldphiS2SnSA = Old.iloc[0:6,4]
NewphiS2SnSA = New.iloc[0:6,4]
OldphiSS = Old.iloc[6:,5]
NewphiSS = New.iloc[6:,5]
OldphiSSnSA = Old.iloc[0:6,5]
NewphiSSnSA = New.iloc[0:6,5]
Oldbias.index = OlddS2S.T.index
Newbias.index = NewdS2S.T.index
aOlddS2S = OlddS2S.T.add(Oldbias,axis=0)
aOlddS2S = aOlddS2S.T
aNewdS2S = NewdS2S.T.add(Newbias,axis=0)
aNewdS2S = aNewdS2S.T

Old = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\superceded\1. Initial Input\stations.csv"))
New = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\stations.csv"))
Sim = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simstations.csv"))
OldVs30 = Old[['stat_name','vs30']]
OldVs30 = OldVs30.rename(columns={'vs30':'Old Vs30'})
NewVs30 = New[['stat_name','vs30','Geom']]
NewVs30 = NewVs30.rename(columns={'vs30':'New Vs30'})
Simstats = Sim[['Station Code']]
Simstats = Simstats.rename(columns={'Station Code':'stat_name'})
Simnames = Simstats.stat_name.tolist()

label        = None
clicked_line = None
clicked_line_color = None
def onpick(event):
    global label, clicked_line, clicked_line_color
    if not isinstance(event.artist, plt.Line2D):
        return
    if label is not None:
        label.remove()
    if clicked_line is not None:
        clicked_line.set(color=clicked_line_color, zorder=2, linewidth=1)
    clicked_line = event.artist
    clicked_line_color = clicked_line.get_color()
    clicked_line.set(color='maroon', zorder=10, linewidth=5)
    plt.gcf().canvas.draw()
    
def on_add(sel):
    label = sel.artist.get_label()
    sel.annotation.set(text=label, ha='center', va='center')
#%% Comparison between Old and New Vs30
OldVsNew = pd.merge(Simstats,OldVs30,on='stat_name')
OldVsNew = pd.merge(OldVsNew,NewVs30,on='stat_name')
cat_col  = {'Basin':'r','Basin-edge':'green','Valley':'purple','Hill':'cyan'}
fig,axes = plt.subplots(figsize=(10.99,8.14),constrained_layout=True)
scatter = plt.scatter(OldVsNew['New Vs30'],OldVsNew['Old Vs30'],c='k',label=OldVsNew['stat_name'])
plt.xlabel('Wotherspoon et al. Vs30 (m/s)',size=20)
plt.ylabel('Foster et al. 2019 Vs30 (m/s)',size=20)
plt.axline((100,100),(1500,1500),c='k',linestyle='-')
plt.axline((80,0),slope=1.0,c='k',linestyle='--')
plt.axline((0,80),slope=1.0,c='k',linestyle='--')
plt.xlim([100,1600])
plt.ylim([100,1600])
mpl.cursor(scatter,multiple=True).connect("add", hover_format)
# fig.savefig('Old Vs New Vs30.png')
#%%
fig1,axes1 = plt.subplots(figsize=(10.99,8.14),constrained_layout=True)
scatter = axes1.scatter(OldVsNew['New Vs30'],OldVsNew['Old Vs30'],c='w')
mpl.cursor(scatter,multiple=True).connect("add", hover_format)
plotted_categories = set()
for category,color in cat_col.items():
    category_data = OldVsNew[OldVsNew['Geom']==category]
    axes1.scatter(category_data['New Vs30'],category_data['Old Vs30'],c=color,label=category if category not in plotted_categories else '')

axes1.legend(fontsize=25,loc = 'best',frameon=False)
plt.xlabel('Wotherspoon et al. Vs30 (m/s)',size=20)
plt.ylabel('Foster et al. 2019 Vs30 (m/s)',size=20)
plt.axline((100,100),(1500,1500),c='k',linestyle='-')
plt.axline((80,0),slope=1.0,c='k',linestyle='--')
plt.axline((0,80),slope=1.0,c='k',linestyle='--')
plt.xlim([100,1600])
plt.ylim([100,1600])
fig1.savefig('Old Vs New Vs30 Geom.png')

#%% Comparison between Old and New bias, standard deviation, dS2S - Main Paper
fig1 = plt.figure(figsize=(13.29,4.65))
x = [0, 1, 2, 3, 4, 5]
x_values = ["PGA", "PGV", "CAV", "AI", "$D_{s575}$", "$D_{s595}$"]

gs1 = fig1.add_gridspec(
    nrows=1, ncols=2, width_ratios=[3, 1],
    left=0.07, right=0.46, top=0.97, bottom=0.15,
    wspace=0.05, hspace=0.25
)
ax1 = fig1.add_subplot(gs1[0,0])
ax1.semilogx(T,Oldbias,'r',linewidth=3,label = 'Lee et al. (2022)')
ax1.semilogx(T,Newbias,'b',linewidth=3,label = 'This study')
ax1.axhline(0,c='k',linestyle='--')
# ax1.axvline(1,c='k',linestyle='--')
ax1.set_ylim([-0.6,0.6])
ax1.set_xlim([0.01,10])
ax1.set_ylabel('Model prediction bias, a',size=20)
ax1.set_xlabel('Vibration period, T (s)',size=20)
leg = ax1.legend(fontsize=18,loc='upper right')
leg.get_frame().set_edgecolor('k')
ax1.text(0.011,-0.46,'Overprediction',size=20,fontstyle='italic')
ax1.text(0.011,0.22,'Underprediction',size=20,fontstyle='italic')
ax1.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax1.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax1.tick_params(axis='x', pad=5)
ax2 = fig1.add_subplot(gs1[0,1])
ax2.set_xticks(x, x_values, size=14)
ax2.scatter(x_values, OldbiasnSA, s=100, c='r', marker='o')
ax2.scatter(x_values, NewbiasnSA, s=100, c='b', marker='o')
ax2.plot([-1.0, 10.0], [0, 0], color='k', linestyle='--')
ax2.set_xlim([-1.0, 6.0])
ax2.set_xticklabels(x_values, rotation=90)
ax2.set_ylim([-0.6, 0.6])
ax2.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax2.tick_params(axis='x', pad=5)
ax2.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.4,alpha=0.5)
ax2.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.4,alpha=0.5)
ax2.yaxis.tick_right()


gs2 = fig1.add_gridspec(
    nrows=1, ncols=2, width_ratios=[3, 1],
    left=0.58, right=0.97, top=0.96, bottom=0.15,
    wspace=0.05, hspace=0.25
)
ax7 = fig1.add_subplot(gs2[0,0])
ax7.semilogx(T,OldphiS2S,'r',linewidth=3,label = 'Lee et al. (2022)')
ax7.semilogx(T,NewphiS2S,'b',linewidth=3,label = 'This study')
ax7.set_ylim([0,0.8])
ax7.set_xlim([0.01,10])
# ax7.axvline(1,c='k',linestyle='--')
ax7.set_ylabel('Site-to-site standard deviation, $\phi_{S2S}$',size=20)
ax7.set_xlabel('Vibration period, T (s)',size=20)
ax7.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax7.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4,alpha=0.5)
ax7.tick_params(axis='x', pad=5)
ax8 = fig1.add_subplot(gs2[0,1])
ax8.set_xticks(x, x_values, size=14)
ax8.scatter(x_values, OldphiS2SnSA, s=100, c='r', marker='o')
ax8.scatter(x_values, NewphiS2SnSA, s=100, c='b', marker='o')
ax8.set_xlim([-1.0, 6.0])
ax8.set_xticklabels(x_values, rotation=90)
ax8.set_ylim([0, 0.8])
ax8.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax8.tick_params(axis='x', pad=5)
ax8.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.4,alpha=0.5)
ax8.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.4,alpha=0.5)
ax8.yaxis.set_label_position("right")
ax8.yaxis.tick_right()

fig1.text(0.008, 0.95, '(a)', fontsize=20, fontweight='bold') 
fig1.text(0.50, 0.95, '(b)', fontsize=20, fontweight='bold')


fig1.savefig('Old Vs New bias & stdMP.pdf')

#%% Comparison between Old and New bias, standard deviation, dS2S - ESupp
fig1 = plt.figure(figsize=(19.2, 9.83))
x = [0, 1, 2, 3, 4, 5]
x_values = ["PGA", "PGV", "CAV", "AI", "$D_{s575}$", "$D_{s595}$"]
gs1 = fig1.add_gridspec(nrows=2, ncols=2, width_ratios=[3,1], left=0.05, right=0.30,top=0.98, bottom=0.08,wspace=0.05,hspace=0.25)
ax1 = fig1.add_subplot(gs1[0,0])
ax1.semilogx(T,Oldbias,'r',linewidth=3,label = 'Lee et al. (2022)')
ax1.semilogx(T,Newbias,'b',linewidth=3,label = 'This study')
ax1.axhline(0,c='k',linestyle='--')
# ax1.axvline(1,c='k',linestyle='--')
ax1.set_ylim([-0.6,0.6])
ax1.set_xlim([0.01,10])
ax1.set_ylabel('Model prediction bias, a',size=20)
ax1.set_xlabel('Vibration period, T (s)',size=20)
leg = ax1.legend(fontsize=18,loc='upper right')
leg.get_frame().set_edgecolor('k')
ax1.text(0.011,-0.46,'Overprediction',size=20,fontstyle='italic')
ax1.text(0.011,0.22,'Underprediction',size=20,fontstyle='italic')
ax1.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax1.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
ax2 = fig1.add_subplot(gs1[0,1])
ax2.set_xticks(x, x_values, size=14)
ax2.scatter(x_values, OldbiasnSA, s=100, c='r', marker='o')
ax2.scatter(x_values, NewbiasnSA, s=100, c='b', marker='o')
ax2.plot([-1.0, 10.0], [0, 0], color='k', linestyle='--')
ax2.set_xlim([-1.0, 6.0])
ax2.set_xticklabels(x_values, rotation=90)
ax2.set_ylim([-0.6, 0.6])
ax2.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax2.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.4)
ax2.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.4)
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()

gs2 = fig1.add_gridspec(nrows=2, ncols=2, width_ratios=[3,1], left=0.39, right=0.64, top=0.98, bottom=0.08, wspace=0.05,hspace=0.25)
ax3 = fig1.add_subplot(gs2[0,0])
ax3.semilogx(T,Oldsigma,'r',linewidth=3,label = 'Lee et al. (2022)')
ax3.semilogx(T,Newsigma,'b',linewidth=3,label = 'This study')
ax3.set_ylim([0,1])
ax3.set_xlim([0.01,10])
# ax3.axvline(1,c='k',linestyle='--')
ax3.set_ylabel('Total standard deviation, $\sigma$',size=20)
ax3.set_xlabel('Vibration period, T (s)',size=20)
ax3.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax3.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4) 
ax4 = fig1.add_subplot(gs2[0,1])
ax4.set_xticks(x, x_values, size=14)
ax4.scatter(x_values, OldsigmanSA, s=100, c='r', marker='o')
ax4.scatter(x_values, NewsigmanSA, s=100, c='b', marker='o')
ax4.set_xlim([-1.0, 6.0])
ax4.set_xticklabels(x_values, rotation=90)
ax4.set_ylim([0, 1])
ax4.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax4.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.4)
ax4.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.4)
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()

ax5 = fig1.add_subplot(gs1[1,0], sharex=ax1)
ax5.semilogx(T,Oldtau,'r',linewidth=3,label = 'Lee et al. (2022)')
ax5.semilogx(T,Newtau,'b',linewidth=3,label = 'This study')
ax5.set_ylim([0,0.8])
ax5.set_xlim([0.01,10])
# ax5.axvline(1,c='k',linestyle='--')
ax5.set_ylabel(r'Between-event standard deviation, $\tau$',size=20)
ax5.set_xlabel('Vibration period, T (s)',size=20)
ax5.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax5.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
fig1.text(0.001, 0.98, '(a)', fontsize=20, fontweight='bold') 
fig1.text(0.34, 0.98, '(b)', fontsize=20, fontweight='bold')
fig1.text(0.001, 0.50, '(c)', fontsize=20, fontweight='bold')
fig1.text(0.34, 0.50, '(d)', fontsize=20, fontweight='bold')
fig1.text(0.68, 0.50, '(e)', fontsize=20, fontweight='bold')
leg = ax5.legend(fontsize=18,loc='lower left')
leg.get_frame().set_edgecolor('k')
ax6 = fig1.add_subplot(gs1[1,1])
ax6.set_xticks(x, x_values, size=14)
ax6.scatter(x_values, OldtaunSA, s=100, c='r', marker='o')
ax6.scatter(x_values, NewtaunSA, s=100, c='b', marker='o')
ax6.set_xlim([-1.0, 6.0])
ax6.set_xticklabels(x_values, rotation=90)
ax6.set_ylim([0, 0.8])
ax6.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax6.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.4)
ax6.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.4)
ax6.yaxis.set_label_position("right")
ax6.yaxis.tick_right()

ax7 = fig1.add_subplot(gs2[1,0], sharex=ax3)
ax7.semilogx(T,OldphiS2S,'r',linewidth=3,label = 'Lee et al. (2022)')
ax7.semilogx(T,NewphiS2S,'b',linewidth=3,label = 'This study')
ax7.set_ylim([0,0.8])
ax7.set_xlim([0.01,10])
# ax7.axvline(1,c='k',linestyle='--')
ax7.set_ylabel('Site-to-site standard deviation, $\phi_{S2S}$',size=20)
ax7.set_xlabel('Vibration period, T (s)',size=20)
ax7.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax7.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
ax8 = fig1.add_subplot(gs2[1,1])
ax8.set_xticks(x, x_values, size=14)
ax8.scatter(x_values, OldphiS2SnSA, s=100, c='r', marker='o')
ax8.scatter(x_values, NewphiS2SnSA, s=100, c='b', marker='o')
ax8.set_xlim([-1.0, 6.0])
ax8.set_xticklabels(x_values, rotation=90)
ax8.set_ylim([0, 0.8])
ax8.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax8.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.4)
ax8.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.4)
ax8.yaxis.set_label_position("right")
ax8.yaxis.tick_right()

gs3 = fig1.add_gridspec(nrows=2, ncols=2, width_ratios=[3,1], left=0.73, right=0.96,top=0.98, bottom=0.08, wspace=0.05,hspace=0.25)
ax9 = fig1.add_subplot(gs3[1,0])
ax9.semilogx(T,OldphiSS,'r',linewidth=3,label = 'Lee et al. (2022)')
ax9.semilogx(T,NewphiSS,'b',linewidth=3,label = 'This study')
ax9.set_ylim([0,0.8])
ax9.set_xlim([0.01,10])
# ax9.axvline(1,c='k',linestyle='--')
ax9.set_ylabel('Single station standard deviation, $\phi_{SS}$',size=20)
ax9.set_xlabel('Vibration period, T (s)',size=20)
ax9.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax9.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
ax10 = fig1.add_subplot(gs3[1,1])
ax10.set_xticks(x, x_values, size=14)
ax10.scatter(x_values, OldphiSSnSA, s=100, c='r', marker='o')
ax10.scatter(x_values, NewphiSSnSA, s=100, c='b', marker='o')
ax10.set_xlim([-1.0, 6.0])
ax10.set_xticklabels(x_values, rotation=90)
ax10.set_ylim([0, 0.8])
ax10.tick_params(labelsize=16,direction='in', axis='both', which='both')
ax10.grid(color='k', linestyle=(0, (5, 10)), which='major', linewidth=0.4)
ax10.grid(color='k', linestyle=(0, (5, 10)), which='minor', linewidth=0.4)
ax10.yaxis.set_label_position("right")
ax10.yaxis.tick_right()


fig1.savefig('Old Vs New bias & stdES.pdf')


#%% Old Vs New dS2S on the same figure
for station_id in Simnames:
    # station_id  = 'ADCS'
    OldVs30 = OldStats[OldStats['stat_name']==station_id]['vs30'].values[0]
    NewVs30 = NewStats[NewStats['stat_name']==station_id]['vs30'].values[0]
    QVs30   = NewStats[NewStats['stat_name']==station_id]['QVs30'].values[0]
    T       = OlddS2S.columns
    oldind  = 'Station_' + str(OldStats.index[OldStats['stat_name']==station_id].values[0])
    newind  = 'Station_' + str(NewStats.index[NewStats['stat_name']==station_id].values[0])
    fig,ax  = plt.subplots(figsize=(8.91,6.99),constrained_layout=True)
    plt.semilogx(T,OlddS2S.loc[oldind],'r',label='Lee et al. (2022)',linewidth=1.5)
    plt.semilogx(T,NewdS2S.loc[newind],'b',label='This study',linewidth=1.5)
    plt.fill_between(T, np.subtract(NewdS2S.loc[newind],(1.0 * NewS2Serr.loc[newind])), np.add(NewdS2S.loc[newind],(1.0 * NewS2Serr.loc[newind])),facecolor=[0.8, 0.8, 1.0], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
    plt.fill_between(T, np.subtract(OlddS2S.loc[oldind],(1.0 * OldS2Serr.loc[oldind])), np.add(OlddS2S.loc[oldind],(1.0 * OldS2Serr.loc[oldind])),facecolor=[1, 0.8, 0.8], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
    plt.title(f"{station_id}, OldVs30 = {OldVs30}, NewVs30 = {NewVs30} m/s ({QVs30})",fontsize=18,fontweight='bold')
    plt.xlabel('Vibration Period, T (s)', size=18)
    plt.ylabel('${\it \delta S2S_s}$', size=18)
    plt.xlim([0.01,10])
    plt.ylim([-1.5,1.5])
    plt.axhline(0, color='gray', linewidth=1.5,linestyle='--')
    plt.axvline(1,c='gray',linestyle='--')
    plt.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    plt.tick_params(labelsize=16,direction='in', axis='both', which='both')
    leg = plt.legend(fontsize=18,loc='lower right')
    leg.get_frame().set_edgecolor('k')
    plt.text(min(plt.xlim())+0.005, max(plt.ylim())*0.90, 'Underprediction', size=16, fontweight='bold')
    plt.text(min(plt.xlim())+0.005, min(plt.ylim())*0.90, 'Overprediction', size=16, fontweight='bold')
    plt.savefig(os.path.join('GeomorphologyOldVsNew','%s.png' %station_id), dpi=300)
    plt.close(fig)
#%% Old Vs New a + dS2S on the same figure
for station_id in Simnames:
    # station_id  = 'CRLZ'
    OldVs30 = OldStats[OldStats['stat_name']==station_id]['vs30'].values[0]
    NewVs30 = NewStats[NewStats['stat_name']==station_id]['vs30'].values[0]
    QVs30   = NewStats[NewStats['stat_name']==station_id]['QVs30'].values[0]
    T       = aOlddS2S.columns
    oldind  = 'Station_' + str(OldStats.index[OldStats['stat_name']==station_id].values[0])
    newind  = 'Station_' + str(NewStats.index[NewStats['stat_name']==station_id].values[0])
    fig,ax  = plt.subplots(figsize=(8.91,6.99),constrained_layout=True)
    plt.semilogx(T,aOlddS2S.loc[oldind],'r',label='Lee et al. (2022)',linewidth=1.5)
    plt.semilogx(T,aNewdS2S.loc[newind],'b',label='This study',linewidth=1.5)
    plt.fill_between(T, np.subtract(aNewdS2S.loc[newind],(1.0 * NewS2Serr.loc[newind])), np.add(aNewdS2S.loc[newind],(1.0 * NewS2Serr.loc[newind])),facecolor=[0.8, 0.8, 1.0], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
    plt.fill_between(T, np.subtract(aOlddS2S.loc[oldind],(1.0 * OldS2Serr.loc[oldind])), np.add(aOlddS2S.loc[oldind],(1.0 * OldS2Serr.loc[oldind])),facecolor=[1, 0.8, 0.8], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
    plt.title(f"{station_id}, OldVs30 = {OldVs30}, NewVs30 = {NewVs30} m/s ({QVs30})",fontsize=18,fontweight='bold')
    plt.xlabel('Vibration Period, T (s)', size=18)
    plt.ylabel('${ a +  \delta S2S_s}$', size=18)
    plt.xlim([0.01,10])
    plt.ylim([-1.5,1.5])
    plt.axhline(0, color='gray', linewidth=1.5,linestyle='--')
    plt.axvline(1,c='gray',linestyle='--')
    plt.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    plt.tick_params(labelsize=16,direction='in', axis='both', which='both')
    leg = plt.legend(fontsize=18,loc='lower right')
    leg.get_frame().set_edgecolor('k')
    plt.text(min(plt.xlim())+0.005, max(plt.ylim())*0.90, 'Underprediction', size=16, fontweight='bold')
    plt.text(min(plt.xlim())+0.005, min(plt.ylim())*0.90, 'Overprediction', size=16, fontweight='bold')
    plt.savefig(os.path.join('Bias added','%s.png' %station_id), dpi=300)
    plt.close(fig)
#%% Station subcategory analysis
df_categories = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\3. Vs30 sensitivity\Station_subcategory.csv"))
df_categories = df_categories.sort_values(by='stat_name',ascending=True)
OldStats['Oldstat_id'] = ''
NewStats['Newstat_id']=  ''
for i in range(1,383):
    OldStats['Oldstat_id'][i] = 'Station_'+ str(OldStats.index.values[i-1])
    
for j in range(1,385):
    NewStats['Newstat_id'][j] = 'Station_'+ str(NewStats.index.values[j-1])
    
df_categories = pd.merge(df_categories,OldStats[['stat_name','Oldstat_id']],on='stat_name')
df_categories = pd.merge(df_categories,NewStats[['stat_name','Newstat_id']],on='stat_name')
grouped_data  = df_categories.groupby('Substation category')
oldstatids    = df_categories['Oldstat_id'].unique()
newstatids    = df_categories['Newstat_id'].unique()
filtaOlddS2S  = aOlddS2S[aOlddS2S.index.isin(oldstatids)]
filtaOlderrdS2S = OldS2Serr[OldS2Serr.index.isin(oldstatids)]
filtaNewdS2S  = aNewdS2S[aNewdS2S.index.isin(newstatids)]
filtaNewerrdS2S = NewS2Serr[NewS2Serr.index.isin(newstatids)]
cluster_labels= np.array(df_categories['Substation category'])
fig, axes = plt.subplots(2, 2, figsize=(9.89, 7.92),constrained_layout = True)
grouped_data = filtaNewdS2S.groupby(cluster_labels)
cluster_mean = filtaNewdS2S.groupby(cluster_labels).mean()
cluster_std  = filtaNewdS2S.groupby(cluster_labels).std()
oldclustermn = filtaOlddS2S.groupby(cluster_labels).mean()
oldclustersd = filtaOlddS2S.groupby(cluster_labels).std()
i =0 
axes[0][0].text(0.45,-1.4,'Overprediction',size=20,fontstyle='italic')
axes[0][0].text(0.42,0.8,'Underprediction',size=20,fontstyle='italic')

for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
    ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color='gray',alpha=0.4)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'b',linewidth=3)
    ax.fill_between(cluster_data.columns,np.subtract(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),np.add(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),facecolor='gray', edgecolor='None', alpha=0.4)
    ax.semilogx(oldclustermn.loc[cluster_id].index, oldclustermn.loc[cluster_id].values,'r',linewidth=3)
    # ax.fill_between(oldclustermn.loc[cluster_id].index,np.subtract(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),np.add(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),facecolor=[1,0.8,0.8], edgecolor='None', alpha=0.6)
    ax.semilogx([],[],label='Lee et al. (2022)',color='r',linewidth=3)
    ax.semilogx([],[],label='This study',color='b',linewidth=3)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='k',linestyle='--')
    if i % 2 == 0:
        ax.set_ylabel('${ a +  \delta S2S_s}$', size=20)
    ax.text(0.011, 1.28, f"{cluster_id} (N = {len(cluster_data)})", size=17, fontweight='bold')
    ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    i = i+1
leg = axes[0][0].legend(fontsize = 14,loc='best')
leg.get_frame().set_edgecolor('k')
for ax in axes[1]:
    ax.set_xlabel('Vibration Period, T (s)', size=20)
plt.savefig('OldVsNew subcategories.pdf')
# plt.savefig('OldVsNew subcategories.png',dpi=300)
#%% Standard deviation of the subcategories comparison
grouped_data_dS2S = filtaNewdS2S.groupby(cluster_labels)
grouped_data_errdS2S = filtaNewerrdS2S.groupby(cluster_labels)
cluster_std = {}
for label, group_errdS2S in grouped_data_errdS2S:
    group_dS2S = grouped_data_dS2S.get_group(label)
    std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
    cluster_std[label]=std_cluster
cluster_std = pd.DataFrame(cluster_std)
cluster_std = cluster_std.transpose()
grouped_data_olddS2S = filtaOlddS2S.groupby(cluster_labels)
grouped_data_olderrdS2S = filtaOlderrdS2S.groupby(cluster_labels)
cluster_oldstd = {}
for label, group_errdS2S in grouped_data_olderrdS2S:
    group_dS2S = grouped_data_olddS2S.get_group(label)
    std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
    cluster_oldstd[label]=std_cluster
cluster_oldstd = pd.DataFrame(cluster_oldstd)
cluster_oldstd = cluster_oldstd.transpose()
fig1,axes1 = plt.subplots(figsize=(11.76,8.24),constrained_layout=True)
colors = ['r','b','g','purple']
axes1.semilogx(T,OldphiS2S,'orange',linewidth=3,label='All sites - Lee et al. (2022)')
axes1.semilogx(T,NewphiS2S,'k',linewidth=3,label='All sites - This study')
for i in range(len(cluster_std)):
    # axes1.semilogx(cluster_oldstd.columns,cluster_oldstd.iloc[i,:],color=colors[i],linestyle='--')
    axes1.semilogx(cluster_std.columns,cluster_std.iloc[i,:],color=colors[i],linestyle='-',label=cluster_std.index[i])

axes1.set_xlabel('Vibration Period, T (s)', size=20)
axes1.set_ylabel('Standard deviation of the subcategories', size=20)
axes1.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
axes1.tick_params(labelsize=16,direction='in', axis='both', which='both')
axes1.set_xlim([0.01, 10])
axes1.legend(fontsize=14)
fig1.savefig('Standard deviation subcategories comparison.png',dpi=300)
#%% Geomorphic subcategory analysis
df_categories = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\3. Vs30 sensitivity\Station_subcategory.csv"))
df_categories = df_categories.sort_values(by='stat_name',ascending=True)
OldStats['Oldstat_id'] = ''
NewStats['Newstat_id']=  ''
for i in range(1,383):
    OldStats['Oldstat_id'][i] = 'Station_'+ str(OldStats.index.values[i-1])
    
for j in range(1,385):
    NewStats['Newstat_id'][j] = 'Station_'+ str(NewStats.index.values[j-1])
    
df_categories = pd.merge(df_categories,OldStats[['stat_name','Oldstat_id']],on='stat_name')
df_categories = pd.merge(df_categories,NewStats[['stat_name','Newstat_id']],on='stat_name')
grouped_data  = df_categories.groupby('Geomorphology')
oldstatids    = df_categories['Oldstat_id'].unique()
newstatids    = df_categories['Newstat_id'].unique()
filtaOlddS2S  = aOlddS2S[aOlddS2S.index.isin(oldstatids)]
filtaNewdS2S  = aNewdS2S[aNewdS2S.index.isin(newstatids)]
cluster_labels= np.array(df_categories['Geomorphology'])
fig, axes = plt.subplots(2, 2, figsize=(9.89, 7.92),constrained_layout = True)
grouped_data = filtaNewdS2S.groupby(cluster_labels)
cluster_mean = filtaNewdS2S.groupby(cluster_labels).mean()
cluster_std  = filtaNewdS2S.groupby(cluster_labels).std()
oldclustermn = filtaOlddS2S.groupby(cluster_labels).mean()
oldclustersd = filtaOlddS2S.groupby(cluster_labels).std()
i =0 
for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
    ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color='gray')
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'k',linewidth=3)
    ax.fill_between(cluster_data.columns,np.subtract(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),np.add(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),facecolor='gray', edgecolor='None', linestyle='dashed', linewidth=1.5, alpha=0.8)
    ax.semilogx(oldclustermn.loc[cluster_id].index, oldclustermn.loc[cluster_id].values,color = 'orange',linewidth=3)
    # ax.fill_between(oldclustermn.loc[cluster_id].index,np.subtract(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),np.add(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),facecolor=[1,0.8,0.8], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
    ax.semilogx([],[],label='Lee et al. (2022)',color='orange',linewidth=3)
    ax.semilogx([],[],label='This study',color='k',linewidth=3)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='maroon',linestyle='--')
    if i % 2 == 0:
        ax.set_ylabel('${ a +  \delta S2S_s}$', size=20)
    ax.text(0.011,1.28,f"{cluster_id}",size =20,fontweight='bold')
    # ax.text(1,1.3,f"Count = {len(cluster_data)}")
    ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    i = i+1
leg = axes[0][0].legend(fontsize = 14,loc='lower left')
leg.get_frame().set_edgecolor('k')
for ax in axes[1]:
    ax.set_xlabel('Vibration Period, T (s)', size=20)
plt.savefig('Geomorphic station subcategories.png',dpi=300)
#%% Geomorphic categories analysis
df_categories = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\3. Vs30 sensitivity\Station_geom.csv"))
df_categories = df_categories.sort_values(by='stat_name',ascending=True)
OldStats['Oldstat_id'] = ''
NewStats['Newstat_id']=  ''
for i in range(1,383):
    OldStats['Oldstat_id'][i] = 'Station_'+ str(OldStats.index.values[i-1])
    
for j in range(1,385):
    NewStats['Newstat_id'][j] = 'Station_'+ str(NewStats.index.values[j-1])
    
df_categories = pd.merge(df_categories,OldStats[['stat_name','Oldstat_id']],on='stat_name')
df_categories = pd.merge(df_categories,NewStats[['stat_name','Newstat_id']],on='stat_name')
grouped_data  = df_categories.groupby('Geomorphology')
oldstatids    = df_categories['Oldstat_id'].unique()
newstatids    = df_categories['Newstat_id'].unique()
filtaOlddS2S  = aOlddS2S[aOlddS2S.index.isin(oldstatids)]
filtaNewdS2S  = aNewdS2S[aNewdS2S.index.isin(newstatids)]
filtaOlderrdS2S = OldS2Serr[OldS2Serr.index.isin(oldstatids)]
filtaNewerrdS2S = NewS2Serr[NewS2Serr.index.isin(newstatids)]
cluster_labels= np.array(df_categories['Geomorphology'])
fig, axes = plt.subplots(2, 2, figsize=(9.89, 7.92),constrained_layout = True)
grouped_data = filtaNewdS2S.groupby(cluster_labels)
desired_order  = ['Basin','Basin-edge','Valley','Hill']
grouped_data = sorted(grouped_data, key=lambda x: desired_order.index(x[0]))
cluster_mean = filtaNewdS2S.groupby(cluster_labels).mean()
cluster_std  = filtaNewdS2S.groupby(cluster_labels).std()
oldclustermn = filtaOlddS2S.groupby(cluster_labels).mean()
oldclustersd = filtaOlddS2S.groupby(cluster_labels).std()

i =0 
axes[0][0].text(0.011,-1.3,'Overprediction',size=20,fontstyle='italic')
axes[0][0].text(0.011,1.0,'Underprediction',size=20,fontstyle='italic')
for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
    ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color='gray',alpha=0.4)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'b',linewidth=3)
    ax.fill_between(cluster_data.columns,np.subtract(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),np.add(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),facecolor='gray', edgecolor='None', linestyle='dashed', linewidth=1.5, alpha=0.4)
    ax.semilogx(oldclustermn.loc[cluster_id].index, oldclustermn.loc[cluster_id].values,'r',linewidth=3)
    # ax.fill_between(oldclustermn.loc[cluster_id].index,np.subtract(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),np.add(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),facecolor=[1,0.8,0.8], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
    ax.semilogx([],[],label='Lee et al. (2022)',color='r',linewidth=3)
    ax.semilogx([],[],label='This study',color='b',linewidth=3)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='k',linestyle='--')
    if i % 2 == 0:
        ax.set_ylabel('${ a +  \delta S2S_s}$', size=20)
    ax.text(0.011, 1.28, f"{cluster_id} (N = {len(cluster_data)})", size=17, fontweight='bold')
    ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    i = i+1
leg = axes[0][0].legend(fontsize = 14,loc='lower right')
leg.get_frame().set_edgecolor('k')
for ax in axes[1]:
    ax.set_xlabel('Vibration Period, T (s)', size=20)
plt.savefig('Geomorphic station categories.pdf')
#%% Geomorphic categories analysis - Osaka
df_categories = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\3. Vs30 sensitivity\Station_geom.csv"))
df_categories = df_categories.sort_values(by='stat_name',ascending=True)
OldStats['Oldstat_id'] = ''
NewStats['Newstat_id']=  ''
for i in range(1,383):
    OldStats['Oldstat_id'][i] = 'Station_'+ str(OldStats.index.values[i-1])
    
for j in range(1,385):
    NewStats['Newstat_id'][j] = 'Station_'+ str(NewStats.index.values[j-1])
    
df_categories = pd.merge(df_categories,OldStats[['stat_name','Oldstat_id']],on='stat_name')
df_categories = pd.merge(df_categories,NewStats[['stat_name','Newstat_id']],on='stat_name')
grouped_data  = df_categories.groupby('Geomorphology')
oldstatids    = df_categories['Oldstat_id'].unique()
newstatids    = df_categories['Newstat_id'].unique()
filtaOlddS2S  = aOlddS2S[aOlddS2S.index.isin(oldstatids)]
filtaNewdS2S  = aNewdS2S[aNewdS2S.index.isin(newstatids)]
filtaOlderrdS2S = OldS2Serr[OldS2Serr.index.isin(oldstatids)]
filtaNewerrdS2S = NewS2Serr[NewS2Serr.index.isin(newstatids)]
cluster_labels= np.array(df_categories['Geomorphology'])
fig, axes = plt.subplots(2, 2, figsize=(9.89, 7.92),constrained_layout = True)
desired_order = ["Basin","Basin-edge","Valley","Hill"]
grouped_data = NewdS2S.groupby(cluster_labels)
sorted_grouped_data = [grouped_data.get_group(cluster_id) for cluster_id in desired_order]
cluster_mean = filtaNewdS2S.groupby(cluster_labels).mean()
cluster_std  = filtaNewdS2S.groupby(cluster_labels).std()
oldclustermn = filtaOlddS2S.groupby(cluster_labels).mean()
oldclustersd = filtaOlddS2S.groupby(cluster_labels).std()
i =0 
colors = ['red','orange','blue','green']
for cluster_id, cluster_data, ax in zip(desired_order,sorted_grouped_data, axes.flatten()):
    ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color=colors[i])
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'k',linewidth=3)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1)-cluster_data.values.T.std(axis=1),'k--',linewidth=2)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1)+cluster_data.values.T.std(axis=1),'k--',linewidth=2)
    # ax.fill_between(cluster_data.columns,np.subtract(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),np.add(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),facecolor=colors[i], edgecolor='None', linestyle='dashed', linewidth=1.5, alpha=0.4)
    # ax.semilogx(oldclustermn.loc[cluster_id].index, oldclustermn.loc[cluster_id].values,'orange',linewidth=3)
    # ax.fill_between(oldclustermn.loc[cluster_id].index,np.subtract(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),np.add(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),facecolor=[1,0.8,0.8], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
    # ax.semilogx([],[],label='Lee et al. (2022)',color='orange',linewidth=3)
    # ax.semilogx([],[],label='This study',color='k',linewidth=3)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='maroon',linestyle='--')
    if i % 2 == 0:
        ax.set_ylabel('${\delta S2S_s}$', size=20)
    ax.text(0.011,1.28,f"{cluster_id}",size =20,fontweight='bold')
    # ax.text(1,1.3,f"Count = {len(cluster_data)}")
    ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    i = i+1
# leg = axes[0][0].legend(fontsize = 14,loc='lower left')
# leg.get_frame().set_edgecolor('k')
for ax in axes[1]:
    ax.set_xlabel('Vibration Period, T (s)', size=20)
# plt.savefig('Geomorphic station categories.png',dpi=300)
#%% Standard deviation of the geomorphic categories comparison
grouped_data_dS2S = NewdS2S.groupby(cluster_labels)
grouped_data_errdS2S = NewS2Serr.groupby(cluster_labels)
cluster_std = {}
for label, group_errdS2S in grouped_data_errdS2S: 
    group_dS2S = grouped_data_dS2S.get_group(label)
    group_errdS2Shill = group_errdS2S
    std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
    cluster_std[label]=std_cluster
cluster_std = pd.DataFrame(cluster_std)
cluster_std = cluster_std.transpose()
# grouped_data_olddS2S = filtaOlddS2S.groupby(cluster_labels)
# grouped_data_olderrdS2S = filtaOlderrdS2S.groupby(cluster_labels)
# cluster_oldstd = {}
# for label, group_errdS2S in grouped_data_olderrdS2S:
#     group_dS2S = grouped_data_olddS2S.get_group(label)
#     std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
#     cluster_oldstd[label]=std_cluster
# cluster_oldstd = pd.DataFrame(cluster_oldstd)
# cluster_oldstd = cluster_oldstd.transpose()
fig1,axes1 = plt.subplots(figsize=(11.76,8.24),constrained_layout=True)
colors = ['red','lime','purple','cyan']
axes1.semilogx(T,OldphiS2S,'orange',linewidth=3,label='All sites - Lee et al. (2022)')
axes1.semilogx(T,NewphiS2S,'k',linewidth=3,label='All sites - This study')
# desired_order = ['Basin','Basin-edge','Valley','Hill']
# cluster_std   = cluster_std.reindex(desired_order)
# for i in range(len(cluster_std)):
#     # axes1.semilogx(oldclustersd.columns,oldclustersd.iloc[i,:],color=colors[i],linestyle='--')
#     axes1.semilogx(cluster_std.columns,cluster_std.iloc[i,:],color=colors[i],linestyle='-',label=cluster_std.index[i],linewidth=1)

axes1.set_xlabel('Vibration Period, T (s)', size=25)
axes1.set_ylabel('Total standard deviation', size=25)
axes1.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
axes1.tick_params(labelsize=20,direction='in', axis='both', which='both')
axes1.set_xlim([0.01, 10])
axes1.set_ylim([0.0, 0.65])
leg = axes1.legend(fontsize=14)
leg.get_frame().set_edgecolor('k')
# fig1.savefig('Standard deviation geomorphic categories comparison.png',dpi=300)
#%% Bias standard deviation comparison for Q1, Q2, Q3 Vs30 sites
df_categories = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\3. Vs30 sensitivity\Station_geom.csv"))
df_categories = df_categories.sort_values(by='stat_name',ascending=True)
OldStats['Oldstat_id'] = ''
NewStats['Newstat_id']=  ''
for i in range(1,383):
    OldStats['Oldstat_id'][i] = 'Station_'+ str(OldStats.index.values[i-1])
    
for j in range(1,385):
    NewStats['Newstat_id'][j] = 'Station_'+ str(NewStats.index.values[j-1])
    
df_categories = df_categories.dropna()
df_categories = pd.merge(df_categories,OldStats[['stat_name','Oldstat_id']],on='stat_name')
df_categories = pd.merge(df_categories,NewStats[['stat_name','Newstat_id']],on='stat_name')
grouped_data  = df_categories.groupby('QVs30')
oldstatids    = df_categories['Oldstat_id'].unique()
newstatids    = df_categories['Newstat_id'].unique()
filtaOlddS2S  = aOlddS2S[aOlddS2S.index.isin(oldstatids)]
filtaNewdS2S  = aNewdS2S[aNewdS2S.index.isin(newstatids)]
filtaOlderrdS2S = OldS2Serr[OldS2Serr.index.isin(oldstatids)]
filtaNewerrdS2S = NewS2Serr[NewS2Serr.index.isin(newstatids)]
cluster_labels= np.array(df_categories['QVs30'])
fig, axes = plt.subplots(2, 2, figsize=(9.89, 7.92),constrained_layout = True)
grouped_data = filtaNewdS2S.groupby(cluster_labels)
cluster_mean = filtaNewdS2S.groupby(cluster_labels).mean()
cluster_std  = filtaNewdS2S.groupby(cluster_labels).std()
oldclustermn = filtaOlddS2S.groupby(cluster_labels).mean()
oldclustersd = filtaOlddS2S.groupby(cluster_labels).std()
grouped_data_dS2S = filtaNewdS2S.groupby(cluster_labels)
grouped_data_errdS2S = filtaNewerrdS2S.groupby(cluster_labels)
cluster_std = {}
for label, group_errdS2S in grouped_data_errdS2S:
    group_dS2S = grouped_data_dS2S.get_group(label)
    std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
    cluster_std[label]=std_cluster
cluster_std = pd.DataFrame(cluster_std)
cluster_std = cluster_std.transpose()
grouped_data_olddS2S = filtaOlddS2S.groupby(cluster_labels)
grouped_data_olderrdS2S = filtaOlderrdS2S.groupby(cluster_labels)
cluster_oldstd = {}
for label, group_errdS2S in grouped_data_olderrdS2S:
    group_dS2S = grouped_data_olddS2S.get_group(label)
    std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
    cluster_oldstd[label]=std_cluster
cluster_oldstd = pd.DataFrame(cluster_oldstd)
cluster_oldstd = cluster_oldstd.transpose()
i =0 


for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
    l = []
    for index in cluster_data.index:
        stat_name = NewStats[NewStats['Newstat_id'] == index]['stat_name'].values[0]
        if len(stat_name) > 0:
            l.append(stat_name)
        else:
            l.append(None)
    ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color='gray',label=l, picker=True,pickradius=5)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'k',linewidth=3)
    ax.fill_between(cluster_data.columns,np.subtract(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),np.add(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),facecolor='gray', edgecolor='None', linestyle='dashed', linewidth=1.5, alpha=0.8)
    ax.semilogx(oldclustermn.loc[cluster_id].index, oldclustermn.loc[cluster_id].values,color = 'orange',linewidth=3)
    # ax.fill_between(oldclustermn.loc[cluster_id].index,np.subtract(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),np.add(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),facecolor=[1,0.8,0.8], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
    ax.semilogx([],[],label='Lee et al. (2022)',color='orange',linewidth=3)
    ax.semilogx([],[],label='This study',color='k',linewidth=3)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='maroon',linestyle='--')
    if i % 2 == 0:
        ax.set_ylabel('${ a +  \delta S2S_s}$', size=20)
    ax.text(0.011,1.28,f"{cluster_id}",size =20,fontweight='bold')
    ax.text(1,1.3,f"Count = {len(cluster_data)}")
    ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    i = i+1
# leg = axes[0][0].legend(fontsize = 14,loc='lower left')
# leg.get_frame().set_edgecolor('k')


for ax in axes[1]:
    ax.set_xlabel('Vibration Period, T (s)', size=20)
    
for i in range(len(grouped_data),2*2):
    fig.delaxes(axes.flatten()[i]) 
    
fig.canvas.mpl_connect("pick_event", onpick)
cursor       = mpl.cursor(multiple=True)
cursor.connect("add",on_add)

fig.savefig('Vs30 Quality bias comparison.png',dpi=300)
fig1,axes1 = plt.subplots(figsize=(11.76,8.24),constrained_layout=True)
colors = ['g','b','r']
axes1.semilogx(T,OldphiS2S,'orange',linewidth=3,label='All sites - Lee et al. (2022)')
axes1.semilogx(T,NewphiS2S,'k',linewidth=3,label='All sites - This study')
for i in range(len(cluster_std)):
    # axes1.semilogx(oldclustersd.columns,oldclustersd.iloc[i,:],color=colors[i],linestyle='--')
    axes1.semilogx(cluster_std.columns,cluster_std.iloc[i,:],color=colors[i],linestyle='-',label=cluster_std.index[i])

axes1.set_xlabel('Vibration Period, T (s)', size=20)
axes1.set_ylabel('Standard deviation of Vs30 categories', size=20)
axes1.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
axes1.tick_params(labelsize=16,direction='in', axis='both', which='both')
axes1.set_xlim([0.01, 10])
leg = axes1.legend(fontsize=14)
leg.get_frame().set_edgecolor('k')
fig1.savefig('Standard deviation Vs30 categories comparison - This study.png',dpi=300)
#%% Bias standard deviation comparison for Type 1, Type 2, Type 4 sites, Unmodelled, Non-Basin sites
df_categories = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\3. Vs30 sensitivity\Station_geom.csv"))
df_categories = df_categories.sort_values(by='stat_name',ascending=True)
OldStats['Oldstat_id'] = ''
NewStats['Newstat_id']=  ''
for i in range(1,383):
    OldStats['Oldstat_id'][i] = 'Station_'+ str(OldStats.index.values[i-1])
    
for j in range(1,385):
    NewStats['Newstat_id'][j] = 'Station_'+ str(NewStats.index.values[j-1])
    
df_categories = pd.merge(df_categories,OldStats[['stat_name','Oldstat_id']],on='stat_name')
df_categories = pd.merge(df_categories,NewStats[['stat_name','Newstat_id']],on='stat_name')
grouped_data  = df_categories.groupby('Basin Type')
oldstatids    = df_categories['Oldstat_id'].unique()
newstatids    = df_categories['Newstat_id'].unique()
filtaOlddS2S  = aOlddS2S[aOlddS2S.index.isin(oldstatids)]
filtaNewdS2S  = aNewdS2S[aNewdS2S.index.isin(newstatids)]
filtaOlderrdS2S = OldS2Serr[OldS2Serr.index.isin(oldstatids)]
filtaNewerrdS2S = NewS2Serr[NewS2Serr.index.isin(newstatids)]
cluster_labels= np.array(df_categories['Basin Type'])
fig, axes = plt.subplots(2, 3, figsize=(17.875, 9.3),constrained_layout = True)
grouped_data = filtaNewdS2S.groupby(cluster_labels)
cluster_mean = filtaNewdS2S.groupby(cluster_labels).mean()
cluster_std  = filtaNewdS2S.groupby(cluster_labels).std()
oldclustermn = filtaOlddS2S.groupby(cluster_labels).mean()
oldclustersd = filtaOlddS2S.groupby(cluster_labels).std()
grouped_data_dS2S = filtaNewdS2S.groupby(cluster_labels)
grouped_data_errdS2S = filtaNewerrdS2S.groupby(cluster_labels)
cluster_std = {}
for label, group_errdS2S in grouped_data_errdS2S:
    group_dS2S = grouped_data_dS2S.get_group(label)
    std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
    cluster_std[label]=std_cluster
cluster_std = pd.DataFrame(cluster_std)
cluster_std = cluster_std.transpose()
grouped_data_olddS2S = filtaOlddS2S.groupby(cluster_labels)
grouped_data_olderrdS2S = filtaOlderrdS2S.groupby(cluster_labels)
cluster_oldstd = {}
for label, group_errdS2S in grouped_data_olderrdS2S:
    group_dS2S = grouped_data_olddS2S.get_group(label)
    std_cluster = np.sqrt(np.mean(group_dS2S**2, axis=0) + np.mean(group_errdS2S**2, axis=0))
    cluster_oldstd[label]=std_cluster
cluster_oldstd = pd.DataFrame(cluster_oldstd)
cluster_oldstd = cluster_oldstd.transpose()
i =0 


for (cluster_id, cluster_data), ax in zip(grouped_data, axes.flatten()):
    l = []
    for index in cluster_data.index:
        stat_name = NewStats[NewStats['Newstat_id'] == index]['stat_name'].values[0]
        if len(stat_name) > 0:
            l.append(stat_name)
        else:
            l.append(None)
    ax.semilogx(cluster_data.columns, cluster_data.values.T,linewidth=1, color='gray',label=l, picker=True,pickradius=5)
    ax.semilogx(cluster_data.columns, cluster_data.values.T.mean(axis=1),'k',linewidth=3)
    ax.fill_between(cluster_data.columns,np.subtract(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),np.add(cluster_data.values.T.mean(axis=1),cluster_data.values.T.std(axis=1)),facecolor='gray', edgecolor='None', linestyle='dashed', linewidth=1.5, alpha=0.8)
    ax.semilogx(oldclustermn.loc[cluster_id].index, oldclustermn.loc[cluster_id].values,color = 'orange',linewidth=3)
    # ax.fill_between(oldclustermn.loc[cluster_id].index,np.subtract(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),np.add(oldclustermn.loc[cluster_id].values,oldclustersd.loc[cluster_id].values),facecolor=[1,0.8,0.8], edgecolor=None, linestyle='dashed', linewidth=1.5, alpha=0.6)
    ax.semilogx([],[],label='Lee et al. (2022)',color='orange',linewidth=3)
    ax.semilogx([],[],label='This study',color='k',linewidth=3)
    ax.set_xlim([0.01, 10])
    ax.set_ylim([-1.5, 1.5])
    ax.axhline(0,color='maroon',linestyle='--')
    if i % 3 == 0:
        ax.set_ylabel('${ a +  \delta S2S_s}$', size=20)
    ax.text(0.011,1.28,f"{cluster_id}",size =20,fontweight='bold')
    ax.text(1,1.3,f"Count = {len(cluster_data)}")
    ax.tick_params(labelsize=16,direction='in', axis='both', which='both')
    ax.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
    i = i+1
# leg = axes[0][0].legend(fontsize = 14,loc='lower left')
# leg.get_frame().set_edgecolor('k')


for ax in axes[1]:
    ax.set_xlabel('Vibration Period, T (s)', size=20)
    
for i in range(len(grouped_data),2*3):
    fig.delaxes(axes.flatten()[i]) 
    
fig.canvas.mpl_connect("pick_event", onpick)
cursor       = mpl.cursor(multiple=True)
cursor.connect("add",on_add)

fig.savefig('Basin Quality bias comparison.png',dpi=300)
fig1,axes1 = plt.subplots(figsize=(11.76,8.24),constrained_layout=True)
colors = ['cyan','b','g','purple','red']
axes1.semilogx(T,OldphiS2S,'orange',linewidth=3,label='All sites - Lee et al. (2022)')
axes1.semilogx(T,NewphiS2S,'k',linewidth=3,label='All sites - This study')
for i in range(len(cluster_std)):
    # axes1.semilogx(oldclustersd.columns,oldclustersd.iloc[i,:],color=colors[i],linestyle='--')
    axes1.semilogx(cluster_std.columns,cluster_std.iloc[i,:],color=colors[i],linestyle='-',label=cluster_std.index[i])

axes1.set_xlabel('Vibration Period, T (s)', size=20)
axes1.set_ylabel('Standard deviation of basin quality categories', size=20)
axes1.grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
axes1.tick_params(labelsize=16,direction='in', axis='both', which='both')
axes1.set_xlim([0.01, 10])
leg = axes1.legend(fontsize=14)
leg.get_frame().set_edgecolor('k')
fig1.savefig('Standard deviation Basin categories comparison.png',dpi=300)
#%% FS and RS on same graph
FSOlddS2SPath = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\superceded\1. Initial Input\FAS Hanning\PJsreStationBiased_sim.csv")
FSNewdS2SPath = Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\FAS Hanning\PJsreStationBiased_sim.csv")
FSOldStats    = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\superceded\1. Initial Input\stations.csv"),index_col='stat_id')
FSNewStats    = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\stations.csv"),index_col='stat_id')
FSOlddS2S     = load_res_pSA_from_csvFS(FSOlddS2SPath)
FSNewdS2S     = load_res_pSA_from_csvFS(FSNewdS2SPath)
FSOld = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\superceded\1. Initial Input\FAS\PJsvarCompsBiased_sim.csv"))
FSNew = pd.read_csv(Path(r"C:\Users\ati47\OneDrive - University of Canterbury\Desktop\PhD\10. Research\Objective 1\1. Residuals\1. Calculations\Simulation residuals\FAS\PJsvarCompsBiased_sim.csv"))
FST   = FSOld.iloc[0:,1]
f     = 1/FST
FSOldbias = FSOld.iloc[0:,2]
FSNewbias = FSNew.iloc[0:,2]
FSOldsigma = FSOld.iloc[0:,6]
FSNewsigma = FSNew.iloc[0:,6]
FSOldtau = FSOld.iloc[0:,3]
FSNewtau = FSNew.iloc[0:,3]
FSOldphiS2S = FSOld.iloc[0:,4]
FSNewphiS2S = FSNew.iloc[0:,4]
FSOldphiSS = FSOld.iloc[0:,5]
FSNewphiSS = FSNew.iloc[0:,5]
FSOldbias.index = FSOlddS2S.T.index
FSNewbias.index = FSNewdS2S.T.index
FSaOlddS2S = FSOlddS2S.T.add(Oldbias,axis=0)
FSaOlddS2S = FSaOlddS2S.T
FSaNewdS2S = FSNewdS2S.T.add(Newbias,axis=0)
FSaNewdS2S = FSaNewdS2S.T
fig1,ax1 = plt.subplots(2,3,figsize=(19.2,9.83),constrained_layout=True)
ax1[0,0].semilogx(T,Oldbias,'r',label = 'Lee et al. (2022) RS')
ax1[0,0].semilogx(T,Newbias,'b',label = 'This study RS')
ax1[0,0].semilogx(f,FSOldbias,'r--',label = 'Lee et al. (2022) FS')
ax1[0,0].semilogx(f,FSNewbias,'b--',label = 'This study FS')
ax1[0,0].axhline(0,c='gray',linestyle='--')
ax1[0,0].axvline(1,c='gray',linestyle='--')
ax1[0,0].set_ylim([-1,2.5])
ax1[0,0].set_xlim([0.01,10])
ax1[0,0].set_ylabel('Model prediction bias, a',size=20)
ax1[0,0].set_xlabel('Vibration period, T (s)',size=20)
leg = ax1[0,0].legend(fontsize=18,loc='upper right')
leg.get_frame().set_edgecolor('k')
ax1[0,0].text(0.11,-0.8,'Overprediction',size=20)
ax1[0,0].text(0.11,0.8,'Underprediction',size=20)
ax1[0,0].tick_params(labelsize=16,direction='in', axis='both', which='both')
ax1[0,0].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)

ax1[0,1].semilogx(T,Oldsigma,'r',label = 'Lee et al. (2022) RS')
ax1[0,1].semilogx(T,Newsigma,'b',label = 'This study RS')
ax1[0,1].semilogx(f,FSOldsigma,'r--',label = 'Lee et al. (2022) FS')
ax1[0,1].semilogx(f,FSNewsigma,'b--',label = 'This study FS')
ax1[0,1].set_ylim([0,1.5])
ax1[0,1].set_xlim([0.01,10])
ax1[0,1].axvline(1,c='gray',linestyle='--')
ax1[0,1].set_ylabel('Total standard deviation, $\sigma$',size=20)
ax1[0,1].set_xlabel('Vibration period, T (s)',size=20)
ax1[0,1].tick_params(labelsize=16,direction='in', axis='both', which='both')
ax1[0,1].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)

ax1[1,0].semilogx(T,Oldtau,'r',label = 'Lee et al. (2022) RS')
ax1[1,0].semilogx(T,Newtau,'b',label = 'This study RS')
ax1[1,0].semilogx(f,FSOldtau,'r--',label = 'Lee et al. (2022) FS')
ax1[1,0].semilogx(f,FSNewtau,'b--',label = 'This study FS')
ax1[1,0].set_ylim([0,0.5])
ax1[1,0].set_xlim([0.01,10])
ax1[1,0].axvline(1,c='gray',linestyle='--')
ax1[1,0].set_ylabel(r'Between-event standard deviation, $\tau$',size=20)
ax1[1,0].set_xlabel('Vibration period, T (s)',size=20)
ax1[1,0].tick_params(labelsize=16,direction='in', axis='both', which='both')
ax1[1,0].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)
leg = ax1[1,0].legend(fontsize=18,loc='lower left')
leg.get_frame().set_edgecolor('k')

ax1[1,1].semilogx(T,OldphiS2S,'r',label = 'Lee et al. (2022) RS')
ax1[1,1].semilogx(T,NewphiS2S,'b',label = 'This study RS')
ax1[1,1].semilogx(f,FSOldphiS2S,'r--',label = 'Lee et al. (2022) FS')
ax1[1,1].semilogx(f,FSNewphiS2S,'b--',label = 'This study FS')
ax1[1,1].set_ylim([0,1.2])
ax1[1,1].set_xlim([0.01,10])
ax1[1,1].axvline(1,c='gray',linestyle='--')
ax1[1,1].set_ylabel('Site-to-site standard deviation, $\phi_{S2S}$',size=20)
ax1[1,1].set_xlabel('Vibration period, T (s)',size=20)
ax1[1,1].tick_params(labelsize=16,direction='in', axis='both', which='both')
ax1[1,1].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)

ax1[1,2].semilogx(T,OldphiSS,'r',label = 'Lee et al. (2022) RS')
ax1[1,2].semilogx(T,NewphiSS,'b',label = 'This study RS')
ax1[1,2].semilogx(f,FSOldphiSS,'r--',label = 'Lee et al. (2022) FS')
ax1[1,2].semilogx(f,FSNewphiSS,'b--',label = 'This study FS')
ax1[1,2].set_ylim([0,1.2])
ax1[1,2].set_xlim([0.01,10])
ax1[1,2].axvline(1,c='gray',linestyle='--')
ax1[1,2].set_ylabel('Single station standard deviation, $\phi_{SS}$',size=20)
ax1[1,2].set_xlabel('Vibration period, T (s)',size=20)
ax1[1,2].tick_params(labelsize=16,direction='in', axis='both', which='both')
ax1[1,2].grid(color='gray', linestyle='dashed', which='both', linewidth=0.4)

fig1.delaxes(ax1[0,2])
fig1.savefig('Old Vs New bias & std RS & FS Hanning.png')
#%%