import numpy as np, matplotlib.pyplot as plt
d = np.load("processed_rerec/test.npz")
X, y = d["X"], d["y"]
print(X.shape, y.mean())
fig, ax = plt.subplots(2, 4, figsize=(14, 6))
for i in range(4):
    ax[0, i].imshow(X[y == 0][i].astype(float), origin="lower", aspect="auto"); ax[0, i].set_title("real")
    ax[1, i].imshow(X[y == 1][i].astype(float), origin="lower", aspect="auto"); ax[1, i].set_title("fake")
plt.show()