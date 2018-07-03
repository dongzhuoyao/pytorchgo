# Credit: Josh Hemann

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


n_groups = 4

old = (61.58, 63.74, 64.96, 64.98)

new = (65.88, 65.52, 65.10, 65.32)

all = (63.73, 64.63, 65.03, 65.15)


#fig = plt.figure(figsize=(16.18 / 1.2, 10 / 1.2))
fig, ax = plt.subplots(figsize=(16.18 / 2, 10 / 2))
index = np.arange(n_groups)
bar_width = 0.25

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, old, bar_width,
                alpha=opacity, color='b',
                yerr=0, error_kw=error_config,
                label='old')

rects2 = ax.bar(index + bar_width, new, bar_width,
                alpha=opacity, color='r',
                yerr=0, error_kw=error_config,
                label='new')


rects3 = ax.bar(index + bar_width * 2, all, bar_width,
                alpha=opacity, color='g',
                yerr=0, error_kw=error_config,
                label='all')

ax.set_xlabel('Tempterature Setting')
ax.set_ylabel('Validation mIoU')
#ax.set_title('Distillation Temperature(T) Ablation Study')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(('T=1', 'T=2', 'T=4', 'T=8'))
ax.set_ylim(ymin=61)
ax.set_ylim(ymax=66)
ax.legend()

fig.tight_layout()
fig.savefig('temperature_ablation.png')

#plt.show()
