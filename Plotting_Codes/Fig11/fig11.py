#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:38:25 2019

@author: mingrenshen
"""

import pandas as pd
import matplotlib.pyplot as plt

# Font for figure
font_axis_publish = {
        'color':  'black',
        'weight': 'bold',
        'size': 35,
        }
labelsize = 20

fignow = plt.figure(figsize=(12,8))
ax = plt.gca()

dat = pd.read_csv("trackpyDC.csv")

plt.hist(dat['DC'], bins = 50)

# plt.plot(dat['dose(dpa)'], dat['median'], color='b')
# plt.fill_between(dat['dose(dpa)'], dat['75th_percentile'], dat['25_percentile'], color='gray', alpha=0.2)

# Figure information
plt.ylabel('Defect Counts', fontdict=font_axis_publish)

#plt.xlim(0,1000)
plt.xlabel('Diffusion Coefficient', fontdict=font_axis_publish)

for ticklabel in (ax.get_xticklabels()):
    ticklabel.set_fontsize(labelsize)
    ticklabel.set_fontweight('bold')

for ticklabel in (ax.get_yticklabels()):
    ticklabel.set_fontsize(labelsize)
    ticklabel.set_fontweight('bold')

#plt.show()

fignow.savefig("Fig11.png",format = "png",dpi=300, bbox_inches='tight', pad_inches=0.2)