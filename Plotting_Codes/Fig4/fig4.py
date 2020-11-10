#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:38:25 2020

@author: mingrenshen
"""

import pandas as pd
import matplotlib.pyplot as plt

# Font for figure
font_axis_publish = {
        'color':  'black',
        'weight': 'bold',
        'size': 20,
        }
labelsize = 15
markSizeSetter = 9

fignow = plt.figure()
ax = plt.gca()

dat_GT = pd.read_excel("ground_truth_density.xlsx")
dat_Jack = pd.read_excel("Jack_density_data.xlsx")
dat_ML = pd.read_excel("YOLO.xlsx")


# multiple line plot

## Plot ML Results
plt.plot( dat_ML['dose(dpa)'], dat_ML['Proposed Corrected Density']/(10**16), marker='.', markersize=0.5, color='blue', linewidth=2, label ='ML Analysis Data')

## Plot Jack's Results
plt.errorbar(dat_Jack["dose(dpa)"], dat_Jack["total loop density per cc"]/(10**16), xerr=0.0, yerr=dat_Jack["error"]/(10**16), markersize=markSizeSetter,label='Haley et al. Data',color='orange',capsize=5, linewidth=3)

## Plot GT Results
plt.plot( dat_GT['dose(dpa)'], dat_GT['Proposed Corrected Density']/(10**16), marker='o', markersize=markSizeSetter, color='red', linewidth=3,label='Ground Truth Data')

plt.legend(prop={'weight': 'bold', 'size': 12}, loc="upper left")

#plt.plot(dat['IoU'], dat['median'], color='r')
#plt.fill_between(dat['dose(dpa)'], dat['75th_percentile'], dat['25_percentile'], color='gray', alpha=0.2)

# Figure information
plt.ylabel('Loop density($ \\times 10^{16}$) [$cm^{-3}$]', fontdict=font_axis_publish)

#plt.xlim(0,1000)
plt.xlabel('Dose (dpa)', fontdict=font_axis_publish)

for ticklabel in (ax.get_xticklabels()):
    ticklabel.set_fontsize(labelsize)
    ticklabel.set_fontweight('bold')

for ticklabel in (ax.get_yticklabels()):
    ticklabel.set_fontsize(labelsize)
    ticklabel.set_fontweight('bold')

#plt.show()
fignow.tight_layout()
fignow.savefig("fig4.png",format = "png",dpi=300, bbox_inches='tight', pad_inches=0.2)