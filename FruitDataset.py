#!/usr/bin/python
##!/usr/bin/env python
# -*-coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import sns as sns
import seaborn as sns
from matplotlib import cm

fruits = pd.read_table('fruit_data_with_colors.txt')
fruits.head()
print(fruits.shape)

print(fruits['fruit_name'].unique())

print(fruits.groupby('fruit_name').size())


# The data is pretty balanced except mandarin. We will just have to go with it.
sns.countplot(fruits['fruit_name'], label="Count")
plt.show()

# Box plot for each numeric variable will give us a clearer idea of the distribution of the input variables:
fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9),
                                        title='Box Plot for each input variable')
plt.savefig('fruits_box')
plt.show()

# It looks like perhaps color score has a near Gaussian distribution.
fruits.drop('fruit_label' ,axis=1).hist(bins=30, figsize=(9,9))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('fruits_hist')
plt.show()



# Some pairs of attributes are correlated (mass and width). This suggests a high correlation and a predictable relationship.

feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']

cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('fruits_scatter_matrix')