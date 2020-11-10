#!/usr/bin/env python3.6
from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import pims
import trackpy as tp
plt.ion
# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 6))
mpl.rc('image', cmap='gray')
aa = pims.ImageSequence('./frame_crop/*.jpg')
f= pd.read_csv("./trackpyResult/force_allFrame.csv")
fig=plt.figure()
fig, ax = plt.subplots()
t = tp.link_df(f, 5, memory=3)
t1 = tp.filter_stubs(t, 50)

im = tp.imsd(t, 3.7e-4, 6.8777,max_lagtime=1000)
im.to_csv('./trackpyResult/lagtime1000.csv')

t.to_csv("./trackpy/t_withsize.csv")
t1.to_csv("./trackpy/t1_withsize.csv")
# Compare the number of particles in the unfiltered and filtered data.
print('Before:', t['particle'].nunique())
print('After:', t1['particle'].nunique())
#fig=plt.figure()
#fig=tp.mass_size(t1.groupby('particle').mean()); # convenience function -- just plots size vs. mass
#fig.figure.savefig("./trackpyResult/particle.jpg")
fig=plt.figure()
fig=tp.plot_traj(t1)
fig.figure.savefig("./trackpyResult/trajectoryI_withsize.jpg")
t2 = t1
fig=plt.figure()
fig=tp.annotate(t2[t2['frame'] == 0], aa[0]);
fig.figure.savefig("./trackpyResult/t2Annotation_withsize.jpg")
d = tp.compute_drift(t2)
fig=plt.figure()
fig=d.plot()
tm = tp.subtract_drift(t1.copy(), d)
fig.figure.savefig("./trackpyResult/comDrift_withsize.jpg")
fig=plt.figure()
fig=tp.plot_traj(tm)
fig.figure.savefig("./trackpyResult/traj_withsize.jpg")
