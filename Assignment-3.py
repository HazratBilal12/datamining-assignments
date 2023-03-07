import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import itertools
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from PIL import Image

file = open(r'D:\CNG514_Spring_2021\Assignments_spring22\CNG514-Assignment-3_spring22\CNG514-Assignment-3-data'
            r'\Apple_sequence_dataset.txt')

my_list = []
lines = file.readlines()

for line in lines:
    data = eval(line)
    data = [segm[0] for segm in data]
    # print(data)

    my_list.append(data)
    dataset1 = my_list

a = TransactionEncoder()
a_data = a.fit(dataset1).transform(dataset1)
df = pd.DataFrame(a_data,columns=a.columns_)
df = df.replace(False,0)
# print(df)

# %% Frequent Itemsets and Rules Mining

df_FI = apriori(df, min_support = 0.75, use_colnames = True, verbose = 1)
# print(df_FI)

df_ar = association_rules(df_FI, metric = "confidence", min_threshold = 0.75)
# print(df_ar)

# %% Part 2:

for i in range(32, 39):
    if i == 38:
        continue

    file = r'D:\CNG514_Spring_2021\Assignments_spring22\CNG514-Assignment-3_spring22\CNG514-Assignment-3-data' \
           r'\Apple_fixation_dataset\P-{}.txt'.format(i)

    dataset = pd.read_csv(file, sep="\t")

    xi = np.array(dataset.iloc[:, [3, 4]].values)
    xi.shape
    xi = pd.DataFrame(xi)
    xi.columns = ['P1', 'P2']
    # print(xi)

# %% Elbow method for Estimating the Epsilon(E) value

    # neighbors = NearestNeighbors(n_neighbors=10)
    # neighbors_fit = neighbors.fit(xi)
    # distances, indices = neighbors_fit.kneighbors(xi)
    #
    # distances = np.sort(distances, axis=0)
    # distances = distances[:, 1]
    # plt.plot(distances)
    # plt.show()

# %% Computing DBSCAN
    db = DBSCAN(eps=70, min_samples=3).fit(xi)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(xi, labels))

    img = Image.open(r'D:\CNG514_Spring_2021\Assignments_spring22'
                     r'\CNG514-Assignment-3_spring22\CNG514-Assignment-3-data\APPLE.png')
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Add scatter point on the image.
    px, py = xi['P1'], xi['P2']
    colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod',
              'lightcyan', 'navy']
    vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
    ax.scatter(px, py, c=vectorizer(labels), zorder=1)
    plt.imshow(img, zorder=0)
    plt.show()


