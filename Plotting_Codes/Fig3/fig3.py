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

fignow = plt.figure()
ax = plt.gca()

dat = pd.read_excel("YOLO-PrecisionRecall.xlsx")
markSizeSetter = 9
# multiple line plot
plt.plot( 'IoU', 'Precison', data=dat, marker='o', markersize=markSizeSetter, color='red', linewidth=2)
plt.plot( 'IoU', 'Recall', data=dat, marker='^', markersize=markSizeSetter, color='green', linewidth=2)
plt.plot( 'IoU', 'F1', data=dat, marker='H', markersize=markSizeSetter,color='blue', linewidth=2)
plt.legend(prop={'weight': 'bold', 'size': 15})

#plt.plot(dat['IoU'], dat['median'], color='r')
#plt.fill_between(dat['dose(dpa)'], dat['75th_percentile'], dat['25_percentile'], color='gray', alpha=0.2)

# Figure information
plt.ylabel('Performance', fontdict=font_axis_publish)

#plt.xlim(0,1000)
plt.xlabel('IoU', fontdict=font_axis_publish)

for ticklabel in (ax.get_xticklabels()):
    ticklabel.set_fontsize(labelsize)
    ticklabel.set_fontweight('bold')

for ticklabel in (ax.get_yticklabels()):
    ticklabel.set_fontsize(labelsize)
    ticklabel.set_fontweight('bold')

#plt.show()
fignow.tight_layout()
fignow.savefig("fig3.png",format = "png",dpi=300, bbox_inches='tight', pad_inches=0.2)