# 2 Exercises for Unsupervised Machine Learning
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn import preprocessing

# 1 Principal Component Analysis
#1a) reading dataset

df = pd.read_csv("data/olympics.csv", index_col = "id")
#looking at the output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#print(df)
df.describe()
df.drop(columns="score")
# I drop the score because it is already covered by the other variables and this way we can reduce dimensionality

#1b scaling the data
scaler = StandardScaler()
scaler.fit(df)
df_scaled = pd.DataFrame(scaler.transform(df))
#scaled = pd.DataFrame(df_scaled)
df_scaled.var()

#1c fitting PCA model
pca = PCA(random_state=42).fit(df_scaled)
sns.heatmap(pca.components_, xticklabels=df.columns, cmap="viridis", annot=True)

# first component load most prominently: 110

components=pd.DataFrame(pca.components_)
print(components)

var = pd.DataFrame(pca.explained_variance_ratio_, columns=["Explained Variance"])
var.index.name ="Principal Component"
var["Cum. Expl. Var."] = var ["Explained Variance"].cumsum()
var.plot(kind ="bar")

#1d) how many components needed? - 6 components needed


#2 Clustering
#2a) loading dataset
iris = load_iris()
X = iris['data']
y = iris['target']

#2b) scaling data
X_scaled = preprocessing.scale(X)

#2c) fitting kmeans, agglomerative, and DBSCAN model
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
#print(kmeans.labels_)
df = pd.DataFrame(X, columns=iris["feature_names"])
df['kmeans'] = kmeans.labels_
df.groupby('kmeans').mean()

agg = AgglomerativeClustering(n_clusters=3, affinity="euclid",
                              linkage="complete")
agg.fit(X_scaled)
df['agg'] = agg.labels_

dbscan = DBSCAN(min_samples=2, eps=1, metric="euclidean")
dbscan.fit(X_scaled)
df['dbscan'] = dbscan.labels_
df['dbscan'].value_counts()

#print(df)

#2d) adding variables

# Everything that DBSCAN cannot cluster is labeled as noise, so points that are far away from every cluster
# but also points that lie between clusters and do not 'belong' to one.
# DBSCAN however has the highest silhouette score

print(f"DBSCAN Silhouette Score: {silhouette_score(X_scaled, dbscan.labels_)}")
print(f"K-Means Silhouette Score: {silhouette_score(X_scaled, kmeans.labels_)}")
print(f"Agglomerative Silhouette Score: {silhouette_score(X_scaled, agg.labels_)}")

X = pd.DataFrame(iris['data'], columns=iris['feature_names'])

df[['sepal_width', 'petal_length']] = X[['sepal width (cm)', 'petal length (cm)']]
df['dbscan'] = df['dbscan'].replace([-1], ['Noise'])

df_merge = pd.melt(df, id_vars=['sepal_width', 'petal_length'], var_name='Clusters')

df_merge = df_merge.rename(columns={'value': 'Cluster Assignment'})


#2g) plotting scatter plot
plt = sns.FacetGrid(df_merge, col="Clusters")
plt = plt.map_dataframe(sns.scatterplot, x="sepal_width", y="petal_length")
plt = plt.add_legend()

plt.savefig('output/cluster_petal.pdf')

