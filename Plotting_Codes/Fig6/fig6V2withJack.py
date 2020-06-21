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
lineWidthSetter = 2
markSizeSetter = 12

fignow = plt.figure(figsize=(12,8))
ax = plt.gca()

dat = pd.read_excel("cropdataset_1.xlsx")
plt.plot(dat['dose(dpa)'], dat['median'], color='b')
plt.fill_between(dat['dose(dpa)'], dat['75th_percentile'], dat['25_percentile'], color='gray', alpha=0.2)

# Read in Jack's Data and GT Data
dat_human = pd.read_excel("humanResultsSummary.xlsx")
# GT
plt.plot(dat_human['dose(dpa)'], dat_human['Median_GT'], marker = 'o', markersize = markSizeSetter, color='r',linewidth = lineWidthSetter,label ="GT Median")
plt.plot(dat_human['dose(dpa)'], dat_human['Quat25_GT'], marker = 'D', markersize = markSizeSetter, color='r',linewidth = lineWidthSetter,label ="GT Q1")
plt.plot(dat_human['dose(dpa)'], dat_human['Quat75_GT'], marker = '^',markersize = markSizeSetter, color='r',linewidth = lineWidthSetter,label ="GT Q3")
# Jack
plt.plot(dat_human['dose(dpa)'], dat_human['Median_Jack'],marker = 'o', markersize = markSizeSetter, linestyle = ':', color='g',linewidth = lineWidthSetter,label ="Jack Median")
plt.plot(dat_human['dose(dpa)'], dat_human['Quat25_Jack'],marker = 'D', markersize = markSizeSetter, linestyle = ':', color='g',linewidth = lineWidthSetter,label ="Jack Q1")
plt.plot(dat_human['dose(dpa)'], dat_human['Quat75_Jack'], marker = '^', markersize = markSizeSetter, linestyle = ':', color='g',linewidth = lineWidthSetter,label ="Jack Q3")

# Figure information
plt.ylabel('Median Size (nm)', fontdict=font_axis_publish)

#plt.xlim(0,1000)
plt.xlabel('Dose (dpa)', fontdict=font_axis_publish)

plt.legend(prop={'weight': 'bold', 'size': 15})

for ticklabel in (ax.get_xticklabels()):
    ticklabel.set_fontsize(labelsize)
    ticklabel.set_fontweight('bold')

for ticklabel in (ax.get_yticklabels()):
    ticklabel.set_fontsize(labelsize)
    ticklabel.set_fontweight('bold')

#plt.show()


fignow.savefig("Fig6.png",format = "png",dpi=300, bbox_inches='tight', pad_inches=0.2)