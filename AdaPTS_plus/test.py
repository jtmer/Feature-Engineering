import torch
import numpy as np

x = np.linspace(-2, 2, num=200)
y = x**2
patch_size = 1
x_p = x.reshape(4//patch_size, -1)
y_p = y.reshape(4//patch_size, -1)

y_p_mean = np.mean(y_p, axis=-1, keepdims=True)
y_p_std = np.std(y_p, axis=-1, keepdims=True)
y_p = (y_p-y_p_mean)/y_p_std
y_norm = y_p.reshape(-1)

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 4))
plt.plot(x,y_norm)
plt.savefig('test.png')
plt.close()