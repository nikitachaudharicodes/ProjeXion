import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys

path = sys.argv[1]
data = np.loadtxt(path, delimiter=",", dtype=str)
header = data[0].astype(str)
data = data[1:].astype(float)

fig, ax = plt.subplots(1, 1)
epochs = np.arange(len(data))
ax.plot(epochs, data[:,0], label='Train', color='blue', linestyle='-')
ax.plot(epochs, data[:,1], label='Validation', color='orange', linestyle='--')
ax.set_title('Train and Validation Loss by Epoch', fontdict={'weight': 'bold', 'size': 16})
ax.set_xlabel('Epochs', fontdict={'weight': 'bold', 'size': 12})
ax.set_ylabel('Loss', fontdict={'weight': 'bold', 'size': 12})
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.legend()
fig.savefig('Losses.jpg')
plt.show()