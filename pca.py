import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd


iris=load_iris()

data=iris.data
feature_names=iris.feature_names
y=iris.target
print(y)
df=pd.DataFrame(data,columns=feature_names)
df["kume"]=y
x=data
pca=PCA()
pca.fit(x)
x_pca=pca.transform(x)
print(x_pca)

df["p1"]=x_pca[:,0]
df["p2"]=x_pca[:,1]

color=["red","pink","purple"]

for each in range(3):
    plt.scatter(df.p1[df.kume==each],df.p2[df.kume==each],color=color[each],label=iris.target_names[each])

plt.show()