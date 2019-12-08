import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist



class SpectralCluster:

	def __init__(self, n_clusters, method='ng'):
		self.n_clusters_ = n_clusters
		self.labels_ = None
		self.method_ = method

	# works for higher dimensional data
	def __ng_fit(self, data):
		sig_sq = 5**2
		A = np.exp(-(squareform(pdist(data))**2)/2*sig_sq)


		D = np.diag((A.sum(axis=1))**(-0.5))
		L = D.dot(A).dot(D)


		vals, vecs = np.linalg.eigh(L)

		vals = vals[np.argsort(vals)][::-1]
		vecs = vecs[:, np.argsort(vals)]

		kmeans = KMeans(n_clusters = self.n_clusters_)

		eigmat = vecs[:, :self.n_clusters_]

		norms = np.sqrt((eigmat*eigmat).sum(axis=1))

		eigmat = eigmat/norms[:, None]

		kmeans.fit(eigmat)

		self.labels_ = kmeans.labels_


	# faster but only for two dimensional data
	def __knn_fit(self, data, k):
		A = kneighbors_graph(data, k).toarray()
		D = np.diag(A.sum(axis=1))
		L = D-A

		vals, vecs = np.linalg.eig(L)

		vecs = vecs[:, np.argsort(vals)]
		vals = vals[np.argsort(vals)]

		if self.n_clusters_ == 2:
			self.labels_ = vecs[:, 1] > 0
			return

		kmeans = KMeans(n_clusters = self.n_clusters_)

		eigmat = np.real(vecs[:, :self.n_clusters_])

		kmeans.fit(eigmat)

		self.labels_ = kmeans.labels_



	def fit(self, data):
		self.data_ = data

		if self.n_clusters_ == 1:
			self.labels_ = np.zeros(len(data))
			return

		if self.method_ == 'ng':
			self.__ng_fit(data)
		
		elif self.method_[0:3] == 'knn':
			self.__knn_fit(data, int(method[3:]))

			

	def plot(self):
		sns.scatterplot(x=self.data_[:, 0], y=self.data_[:,1], hue=self.labels_, legend=False)
		plt.show()





#X, labels = make_circles(n_samples=500, noise=0.1, factor=0.01)
X, labels = make_blobs(n_samples=2000, n_features=2, centers=4, cluster_std=0.6)

sc = SpectralCluster(4)
sc.fit(X)
sc.plot()


sc2 = SpectralCluster(4, 'knn8')
sc.fit(X)
sc.plot()