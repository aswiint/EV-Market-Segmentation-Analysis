import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.mosaicplot import mosaic
from yellowbrick.features import PCA as PCAVisualizer

# EXPLORING DATA

df = pd.read_csv("Vehicle_Config.csv")
pd.set_option("display.max_columns", None)
df["Model"] = df["Brand"].str.cat(df["Model"], sep=" ")
df = df.drop("Brand", axis=1)
df = df[df["FastCharge_KmH"] != "-"]
df["FastCharge_KmH"] = pd.to_numeric(df["FastCharge_KmH"])
df["RapidCharge"] = df["RapidCharge"].map({"Yes": 1, "No": 0})
print("\nDataFrame Columns: ", df.columns)
print("\nDataFrame Shape: ", df.shape)
print("\nDataFrame Head: \n", df.head)

frame = df.drop(columns=["Model", "PowerTrain", "PlugType", "BodyStyle", "Vehicle_Segment", "Price_Lakhs"])
print("\nDataFrame Mean: \n", frame.mean())

# FEATURE SCALING

scaler = StandardScaler()
scaler.fit(frame)

# PRINCIPAL COMPONENTS

pca = PCA()
pca.fit(frame)
summary_df = pd.DataFrame(
    {"Standard deviation": pca.explained_variance_, "Proportion of Variance": pca.explained_variance_ratio_,
     "Cumulative Proportion": np.cumsum(pca.explained_variance_ratio_)},
    index=range(1, len(pca.explained_variance_) + 1))
print("\nImportance of Components: \n", summary_df)
rotation_matrix = pd.DataFrame(pca.components_, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"])
print("\nRotation matrix: \n", rotation_matrix)

# FACTOR-CLUSTER ANALYSIS (PERCEPTUAL MAP)

ev_pca = PCA(n_components=2)
ev_pca.fit(frame)
ev_proj = ev_pca.fit_transform(frame)
kmeans = KMeans(n_clusters=5)
ev_clusters = kmeans.fit_predict(frame)
visualizer = PCAVisualizer(scale=True, proj_features=True)
visualizer.fit_transform(frame)
visualizer.show()


# EXTRACTING SEGMENTS USING k-MEANS

# SCREE PLOT

np.random.seed(1234)
sse = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(frame)
    sse.append(kmeans.inertia_)
plt.figure(figsize=(12, 8))
plt.plot(range(2, 9), sse, marker="o")
plt.xlabel("Number of Segments")
plt.ylabel("Sum of Squared Distances")
plt.title("Scree Plot")
plt.show()

# GLOBAL STABILITY k-MEANS SEGMENTATION SOLUTION

adj_rand_scores = []
nboot = 100
for _ in range(nboot):
    idx = np.random.choice(len(frame), size=len(frame), replace=True)
    bootstrap_sample = frame.iloc[idx]
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(bootstrap_sample)
        labels = kmeans.labels_
        adj_rand_scores.append({"k": k, "adj_rand_score": adjusted_rand_score(df["Vehicle_Segment"], labels)})

adj_rand_df = pd.DataFrame(adj_rand_scores)
palette = sns.color_palette("husl", 7)
plt.figure(figsize=(12, 8))
sns.boxplot(x="k", y="adj_rand_score", data=adj_rand_df, palette=palette, medianprops=dict(linewidth=2.5))
plt.xlabel("Number of Segments")
plt.ylabel("Adjusted Rand Index")
plt.title("Global Stability of k-means Segmentation Solution")
plt.show()

# GORGE PLOT OF 4 SEGMENT k-MEANS SOLUTION

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.hist([s["adj_rand_score"] for s in adj_rand_scores if s["k"] == i + 2], bins=20, color="skyblue",
             edgecolor="black", alpha=0.7)
    plt.xlabel("Similarity")
    plt.ylabel("Percentage of Total")
    plt.title(f"""Cluster-{i + 2}""")
plt.suptitle("Gorge Plot of Four Segment k-means Solution")
plt.tight_layout()
plt.show()


# PROFILING SEGMENTS

# HIERARCHICAL CLUSTER ANALYSIS

Z = hierarchy.linkage(frame.T, method="average", metric="euclidean")
order = hierarchy.leaves_list(Z)
MD_k4 = np.random.rand(4, frame.shape[1])
sorted_data = MD_k4[:, order]
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
colors = ["gold", "orange", "lightgreen", "skyblue"]
total_sum = 0
for i in range(4):
    total_sum = total_sum+sum(sorted_data[i])

for i in range(4):
    row = i // 2
    col = i % 2
    cluster_sum = sum(sorted_data[i])
    cluster_percentage = (cluster_sum/total_sum)*100
    for j in sorted_data[i]:
        j_index = np.where(sorted_data[i] == j)[0]
        element_percentage = (j/cluster_sum)*100
        element_overall_percentage = (j/total_sum)*100
        marker_color = colors[i] if element_overall_percentage > 4 else "lightgrey"
        axs[row, col].barh(width=element_overall_percentage, y=frame.columns[j_index], color=marker_color)
        axs[row, col].plot(element_percentage, j_index, 'ro')
        axs[row, col].set_title("Cluster "+str(i+1)+": "+str(cluster_percentage.round(2))+"%")
plt.suptitle("Segment Profile Plot for Four-Segment Solution")
plt.tight_layout()
plt.show()

# SEGMENT SEPARATION PLOT USING PCA

pca = PCA(n_components=2)
principal_components = pca.fit_transform(frame)
ev_clusters = KMeans(n_clusters=5).fit_predict(frame)
visualizer = PCAVisualizer(scale=True, proj_features=True, classes=np.unique(ev_clusters), alpha=0.7)
visualizer.fit_transform(frame, ev_clusters)
annotations = [f"Cluster {i}" for i in range(5)]
visualizer.show(classes=np.unique(ev_clusters))
centers = []
for cluster in np.unique(ev_clusters):
    mask = (ev_clusters == cluster)
    center = np.mean(principal_components[mask], axis=0)
    centers.append(center)
for i, center in enumerate(centers):
    visualizer.ax.annotate(str(i), center, color="black", weight="bold")
visualizer.finalize()
visualizer.show()


# DESCRIBING SEGMENTS

ev_clusters = KMeans(n_clusters=5).fit_predict(frame)

mosaic_data = pd.crosstab(index=ev_clusters + 1, columns=df["PlugType"])
props = lambda key: {'color': 'skyblue' if '1' in key else ('yellowgreen' if '2' in key else ('salmon' if '3' in key
                                                                                              else ('orange' if '4' in
                                                                                                                key else
                                                                                                    'blueviolet')))}
mosaic(mosaic_data.stack(), title="Shaded Mosaic Plot for Clusters and PlugType Cross-Tab", axes_label=True,
       labelizer=lambda key: "", properties=props)
plt.show()

mosaic_data = pd.crosstab(index=ev_clusters+1, columns=df["BodyStyle"])
props = lambda key: {'color': 'skyblue' if '1' in key else ('yellowgreen' if '2' in key else ('salmon' if '3' in key
                                                                                              else ('orange' if '4' in
                                                                                                                key else
                                                                                                    'blueviolet')))}
mosaic(mosaic_data.stack(), title="Shaded Mosaic Plot for Clusters and BodyStyle Cross-Tab", axes_label=True,
       labelizer=lambda key: "", properties=props)
plt.show()

mosaic_data = pd.crosstab(index=ev_clusters+1, columns=df["Vehicle_Segment"])
props = lambda key: {'color': 'skyblue' if '1' in key else ('yellowgreen' if '2' in key else ('salmon' if '3' in key
                                                                                              else ('orange' if '4' in
                                                                                                                key else
                                                                                                    'blueviolet')))}
mosaic(mosaic_data.stack(), title="Shaded Mosaic Plot for Clusters and Vehicle_Segment Cross-Tab", axes_label=True,
       labelizer=lambda key: "", properties=props)
plt.show()

box_data = pd.DataFrame({"Price_Lakhs": df["Price_Lakhs"], "Cluster": ev_clusters})
plt.figure(figsize=(10, 6))
plt.boxplot([box_data[box_data["Cluster"] == i]["Price_Lakhs"] for i in range(5)], labels=["1", "2", "3", "4", "5"],
            patch_artist=True, notch=True, showmeans=True)
plt.xlabel("Cluster")
plt.ylabel("Price_Lakhs")
plt.title("Parallel Box and Whisker Plot of Price_Lakhs by Cluster")
plt.grid(axis='y')
plt.show()


# SELECTING THE TARGET SEGMENTS

k4 = ev_clusters = KMeans(n_clusters=5).fit_predict(frame)
df["Vehicle_Segment"] = df["Vehicle_Segment"].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "N": 7, "S": 8})
df["BodyStyle"] = df["BodyStyle"].map({"Sedan": 1, "Hatchback": 2, "Liftback": 3, "SUV": 4, "Pickup": 5, "MPV": 6,
                                       "Cabrio": 7, "SPV": 8, "Station": 9})
price = df.groupby(k4)["Price_Lakhs"].mean()
segment = df.groupby(k4)["Vehicle_Segment"].mean()
bodystyle = df.groupby(k4)["BodyStyle"].apply(lambda x: (x == 2).mean())
plt.figure(figsize=(10, 6))
plt.scatter(price, segment, s=1000*bodystyle, alpha=0.5)
plt.xlabel("Price_Lakhs")
plt.ylabel("Vehicle_Segment")
plt.title("Segment Evaluation Plot")
for i in range(5):
    plt.text(price[i], segment[i], i+1)
plt.show()
