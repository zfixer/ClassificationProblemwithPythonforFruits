#!/usr/bin/python
##!/usr/bin/env python
# -*-coding:utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
fruits = pd.read_table('fruit_data_with_colors.txt')
fruits.head()
print(fruits.shape)


print(fruits['fruit_name'].unique())

print(fruits.groupby('fruit_name').size())



sns.countplot(fruits['fruit_name'],label="Count")
plt.show()