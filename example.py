#This code generates some simulated data and applies the GSBS state segmentation method to that data.
#you can find code for more realistic simulations in https://github.com/lgeerligs/State-segmentation-GSBS

import numpy as np
from statesegmentation import GSBS
import matplotlib.pyplot as plt

#generate timeseries data with distinct states

nvox = 100
nstates = 10
statelen = 20
noise_sd = 1
ntime = nstates * statelen
state_means = np.random.randn(nvox, nstates)

state_labels = np.zeros(ntime, dtype=int)
start_TR=0
for i in range(0,nstates):
    state_labels[start_TR:(start_TR+statelen)]=i
    start_TR=start_TR+statelen

evData = np.zeros([ntime, nvox])
for t in range(0,ntime):
    evData[t, :] = state_means[:, state_labels[t]]

evData = evData + noise_sd * np.random.randn(ntime,nvox)

# apply state segmentation
states = GSBS(x=evData, kmax=200)
states.fit()

#plot results
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
ax1.imshow(np.corrcoef(evData))
ax1.set_xlabel('time')
ax1.set_ylabel('time')
ax2.plot(states.states)
ax2.set_ylabel('state')
ax2.set_xlabel('time')
plt.show()

