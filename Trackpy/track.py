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
aa = pims.ImageSequence('./frame_crop/*.jpg', as_grey=True)
# f = tp.locate(aa[0], 33, invert=True)
f= pd.read_csv("./trackpyResult/forceCSV_allFrame.csv")
fig=plt.figure()
#fig=tp.annotate(f, aa[0])
#fig.figure.savefig("./trackpyResult/trackpyAnnotation.jpg")
#f = tp.batch(aa[:], 11, minmass=200, invert=True);
#f = tp.batch(aa[:], 11, invert=True);
fig, ax = plt.subplots()
t = tp.link_df(f, 5, memory=3)
t1 = tp.filter_stubs(t, 50)

msd3D = tp.emsd(t1, mpp=3.7e-4, fps=6.877)
ax = msd3D.plot(style='o', label='MSD in 3D')
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
ax.set(xlim=(0, 16), ylim=(0, 20))
ax.legend(loc='upper left')
pass
slope = np.linalg.lstsq(msd3D.index[:, np.newaxis], msd3D)[0][0]
ax = msd3D.plot(style='o', label='MSD in 3D')
ax.plot(np.arange(20), slope * np.arange(20), label='linear fit')
ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [$\mu$m$^2$]', xlabel='lag time $t$')
ax.set(xlim=(0, 16), ylim=(0, 20))
ax.legend(loc='upper left')
print(r'The diffusion constant is {0:.2f} μm²/s'.format(slope / 6))

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
fig.figure.savefig("./trackpyResult/trajectoryI.jpg")
t2 = t1
fig=plt.figure()
fig=tp.annotate(t2[t2['frame'] == 0], aa[0]);
fig.figure.savefig("./trackpyResult/t2Annotation.jpg")
d = tp.compute_drift(t2)
fig=plt.figure()
fig=d.plot()
tm = tp.subtract_drift(t1.copy(), d)
fig.figure.savefig("./trackpyResult/comDrift.jpg")
fig=plt.figure()
fig=tp.plot_traj(tm)
fig.figure.savefig("./trackpyResult/traj.jpg")
