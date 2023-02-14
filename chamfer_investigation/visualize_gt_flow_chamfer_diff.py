# Load pickle file 'merged_dicts.pkl'
import pickle
import math
with open('merged_dicts.pkl', 'rb') as f:
    merged_dicts = pickle.load(f)



import matplotlib.pyplot as plt
import numpy as np
# Make a plot of the endpoint error vs. k for fast speed


def make_plot(merged_dicts, get_list_fn, title):
    xs = merged_dicts.keys()
    ys_median = []
    ys_lower = []
    ys_upper = []

    for k in sorted(merged_dicts.keys()):
        lst = get_list_fn(merged_dicts[k])
        arr = np.array(sorted(lst))
        len(arr) // 2
        ys_median.append(arr[len(arr) // 2])
        ys_lower.append(arr[int(len(arr) * 0.1)])
        ys_upper.append(arr[int(len(arr) * 0.9)])

    ys_median = np.array(ys_median)
    ys_lower = np.array(ys_lower)
    ys_upper = np.array(ys_upper)
    plt.plot(xs, ys_median, label='median')
    plt.fill_between(xs, ys_lower, ys_upper, alpha=0.5, label='std')

    plt.xlabel('k')
    plt.ylabel('Endpoint error (m)')
    plt.title(f'EPE {title}')
    

plt.subplot(2, 1, 1)
make_plot(merged_dicts, lambda x: x['speed_error']['fast'], "fast speed")
plt.subplot(2, 1, 2)
make_plot(merged_dicts, lambda x: x['speed_error']['slow'], "slow speed")
plt.tight_layout()
plt.show()

cls_ids = sorted(merged_dicts[1]['class_error'].keys())
for cls_id_idx, cls_id in enumerate(cls_ids):
    plt.subplot(math.ceil(len(cls_ids) / 4), 4, cls_id_idx + 1)
    make_plot(merged_dicts, lambda x: x['class_error'][cls_id], f"Class {cls_id}")

# Make tight layout with no overlapping subplots and no edge padding.
plt.tight_layout()
plt.show()
