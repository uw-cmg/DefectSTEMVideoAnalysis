from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import pims
import trackpy as tp
print("6/13 9：46")
font_axis_publish = {
        'color':  'black',
        'weight': 'bold',
        'size': 30,
        }
font_title_publish = {
        'color':  'black',
        'weight': 'bold',
        'size': 15,
        }

#sr=30 #search range
#mem=45 #memory

plt.ion
# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')
listofImage=[]
for i in range (1176):
    listofImage.append('./frame_crop/Frame'+str(i)+'.jpg')
aa = pims.ImageSequence(listofImage, as_grey=True)
# f = tp.locate(aa[0], 33, invert=True)
f=tp.locate(aa[0], 15, invert=True)

sr=10
minmass=1200
mem=45
f = tp.batch(aa[:], 11, minmass=1200, invert=True)
t = tp.link_df(f,10, memory=45)

#fig, ax = plt.subplots(figsize=(20,12))
plt.hist(f['mass'], bins=20)
#ax.set(xlabel='mass', ylabel='count')
plt.savefig("mass_distribution.png")

t_Filtered = tp.filter_stubs(t, 30) #filter out Empheremeral trajectories
t.to_csv("./trackpyResult/t_unfiltered_"+str(sr)+"_"+str(mem)+"minmass"+minmass+".csv")
t_Filtered.to_csv("./trackpyResult/t_filtered_"+str(sr)+"_"+str(mem)+"minmass"+minmass+".csv")

# compare filtered and unfiltered
print('Before:', t['particle'].nunique())
print('After:', t_Filtered['particle'].nunique())

fig, ax = plt.subplots(figsize=(20,12))
fig=tp.plot_traj(t_Filtered, fontsize=30)
t2 = t_Filtered
fig=plt.figure()
fig=tp.annotate(t2[t2['frame'] == 0], aa[0]);
fig.figure.savefig("./PureTrackpy/t2Annotation_"+str(sr)+"_"+str(mem)+"minmass"+minmass+".jpg")
d = tp.compute_drift(t2)
fig=plt.figure()
fig=d.plot()
tm = tp.subtract_drift(t_Filtered.copy(), d)
fig.figure.savefig("./PureTrackpy/comDrift_"+str(sr)+"_"+str(mem)+"minmass"+minmass+".jpg")

fig, ax = plt.subplots(figsize=(20,12))
fig=tp.plot_traj(tm)
ax.set_xlabel("X (pixel)", fontdict=font_axis_publish)
ax.set_ylabel("Y (pixel)", fontdict=font_axis_publish)
ax.tick_params('both',labelsize=25)
ax2 = ax.twiny()
ax2.set_xlabel("X (nm)", fontdict=font_axis_publish)
ax2.set_xlim(0,416)
ax2.tick_params('both',labelsize=25)
ax3=ax.twinx()
ax3.set_ylabel("Y (nm)",fontdict=font_axis_publish)
ax3.set_ylim(0,264)
ax3.tick_params('both',labelsize=25)
ax3.invert_yaxis()
#fig.set_title("Trackpy Trajectory with Computing Drift",pad=20, fontdict=font_title_publish)
ax.autoscale(enable=True)
ax2.autoscale(enable=True)
ax3.autoscale(enable=True)

for ticklabel in (ax.get_xticklabels()):
    ticklabel.set_fontsize(25)
    ticklabel.set_fontweight('bold')
    
for ticklabel in (ax.get_yticklabels()):
    ticklabel.set_fontsize(25)
    ticklabel.set_fontweight('bold')

for ticklabel in (ax.get_xticklabels()):
    ticklabel.set_fontsize(25)
    ticklabel.set_fontweight('bold')
    
for ticklabel in (ax.get_yticklabels()):
    ticklabel.set_fontsize(25)
    ticklabel.set_fontweight('bold')
    
for ticklabel in (ax2.get_xticklabels()):
    ticklabel.set_fontsize(25)
    ticklabel.set_fontweight('bold')
    
for ticklabel in (ax2.get_yticklabels()):
    ticklabel.set_fontsize(25)
    ticklabel.set_fontweight('bold')
    
for ticklabel in (ax3.get_xticklabels()):
    ticklabel.set_fontsize(25)
    ticklabel.set_fontweight('bold')
    
for ticklabel in (ax3.get_yticklabels()):
    ticklabel.set_fontsize(25)
    ticklabel.set_fontweight('bold')

fig.figure.savefig("./PureTrackpy/traj_"+str(sr)+"_"+str(mem)+"minmass"+minmass+".jpg")
fig.figure.savefig("./PureTrackpy/compu_"+str(sr)+"_"+str(mem)+"minmass"+minmass+".png", format="png", dpi=300, bbox_inches="tight", pad_inches=0.2)
