import numpy as np
from simple_tsne import tsne, momentum_func
from sklearn.datasets import load_digits


import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use("seaborn-whitegrid")
rcParams["font.size"] = 18
rcParams["figure.figsize"] = (12, 8)


digits, digit_class = load_digits(return_X_y=True)
rand_idx = np.random.choice(np.arange(digits.shape[0]), size=500, replace=False)
data = digits[rand_idx, :].copy()
classes = digit_class[rand_idx]

low_dim = tsne(data, 2, 30, 500, 100, momentum_func, pbar=True, random_state=42)

scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], cmap="tab10", c=classes)
plt.legend(*scatter.legend_elements(), fancybox=True, bbox_to_anchor=(1.05, 1))
plt.show()
