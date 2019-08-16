import pandas as pd

from kmeans import KMeans

df = pd.read_csv("http://cs.joensuu.fi/sipu/datasets/s4.txt", header=None, delimiter=r"\s+")
data = df.iloc[:,0:2].values

kmeans = KMeans(n_clusters=16).fit(data)
import matplotlib.pyplot as plt
plt.scatter(data[:,0],data[:,1], c=kmeans.labels_, cmap='rainbow')
plt.show()