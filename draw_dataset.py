# Code source: Ga?l Varoquaux
# License: BSD 3 clause

import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

from dataset import load_dataset


class Bunch(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def trans_complex_to_real(ndarry_data):
    new_ndarry_data = []
    for row in ndarry_data:
        new_row = []
        for item in row:
            new_row.append(item.real)
        new_ndarry_data.append(new_row)
    return np.array(new_ndarry_data)


def process_dataset_iris(pkl_dataset_dir):
    iris = {}
    dataset_load = load_dataset(pkl_dataset_dir)
    data = dataset_load['data']
    data = trans_complex_to_real(data)
    target = dataset_load['singers_label']
    target_names = dataset_load['dict_singer_label']
    new_target_names = ['aerosmith', 'beatles', 'creedence_clearwater_revival', 'cure', 'dave_matthews_band',
                        'depeche_mode', 'fleetwood_mac', 'garth_brooks',
                        'green_day', 'led_zeppelin', 'madonna', 'metallica', 'prince', 'queen', 'radiohead', 'roxette',
                        'steely_dan',
                        'suzanne_vega', 'tori_amos', 'u2']
    # print target_names
    author = 'zhang xu-long'
    return Bunch(data=data, target=target, target_names=new_target_names, author=author)


np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
# iris = datasets.load_iris()
iris = process_dataset_iris('dataset.pkl')
X = iris.data
y = iris.target

fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

for name, label in [('aerosmith', 0), ('beatles', 1),
                    ('creedence_clearwater_revival', 2), ('cure', 3), ('dave_matthews_band', 4), ('depeche_mode', 5),
                    ('fleetwood_mac', 6), ('garth_brooks', 7), ('green_day', 8), ('led_zeppelin', 9), ('madonna', 10),
                    ('metallica', 11), ('prince', 12), ('queen', 13), ('radiohead', 14), ('roxette', 15),
                    ('steely_dan', 16), ('suzanne_vega', 17), ('tori_amos', 18), ('u2', 19)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)

x_surf = [X[:, 0].min(), X[:, 0].max(),
          X[:, 0].min(), X[:, 0].max()]
y_surf = [X[:, 0].max(), X[:, 0].max(),
          X[:, 0].min(), X[:, 0].min()]
x_surf = np.array(x_surf)
y_surf = np.array(y_surf)
v0 = pca.transform(pca.components_[0])
v0 /= v0[-1]
v1 = pca.transform(pca.components_[1])
v1 /= v1[-1]

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()
