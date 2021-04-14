from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import fowlkes_mallows_score as FM_score

# 运行次数
iter_times = 5

iris = datasets.load_iris()
X = iris.data
y = iris.target

model_KM = KMeans(n_clusters=3)
model_EM = GMM(n_components=3)
model_SC = SpectralClustering(n_clusters=3)

sum_KM = 0
sum_EM = 0
sum_SC = 0
# 迭代计算多次，取平均值
for i in range(iter_times):
    model_KM.fit(X)
    model_EM.fit(X)
    model_SC.fit(X)

    KM_y = model_KM.predict(X)
    EM_y = model_EM.predict(X)
    SC_y = model_SC.fit_predict(X)

    sum_KM += FM_score(KM_y, y)
    sum_EM += FM_score(EM_y, y)
    sum_SC += FM_score(SC_y, y)

print('Score of KMeans is {}'.format(sum_KM/iter_times))
print('Score of EM is {}'.format(sum_EM/iter_times))
print('Score of SpectralClustering is {}'.format(sum_SC/iter_times))
# 可视化结果
x_axis = X[:, 0]
y_axis = X[:, 2]

# 原结果
plt.scatter(x_axis, y_axis, c=y)
plt.title("Original")
plt.show()
plt.close()

# KMeans聚类结果
plt.scatter(x_axis, y_axis, c=KM_y)
plt.title("K-means")
plt.show()
plt.close()

# EM聚类结果
plt.scatter(x_axis, y_axis, c=EM_y)
plt.title("EM")
plt.show()
plt.close()

# 谱聚类结果
plt.scatter(x_axis, y_axis, c=SC_y)
plt.title("Spectral Clustering")
plt.show()
plt.close()
