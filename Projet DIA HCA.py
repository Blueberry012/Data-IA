import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
df=pd.read_csv("cleaned_data.csv")
#%%
df2020 = df[(df["TIME_PERIOD"]==2015)]
df2020=df2020.drop('TIME_PERIOD',axis=1)
#%% Heatmap afin de voir quelles variables sont intéréssantes à étudier entre-elles

correlation_matrix = df2020.drop('Pays', axis=1).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title("Matrice de corrélation")
plt.show()
# Essayons la même analyse que le K-Means avec le HCA afin de comparer ces deux méthodes
#%% Colums : - Total des voies d'accès de communication pour 100 habitants
#            - Total des abonnements au téléphone cellulaire mobile pour 100 habitants
#   La data est directement pris cru car on a aucune valeurs abérantes ou de valeurs NaN

dfhca=df2020[["Pays","Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]]

#%% Normalisons ensuite les données pour éviter des résultats biaisés ou inefficaces
#Si on a une grande echelle de donnée, les données doivent être normalisés (X-Xmin/Xmax-Xmin)
#Si on a ecart type (std) très grand, on doit standardiser (X-Moyenne(X) / S)
#   Utilisons la normalisation Min-Max pour que les données soient dans la même plage

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dfhca[["Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]]=scaler.fit_transform(dfhca[["Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]])
#%% Linkage & HCA, on reste sur 3 clusters comme K-Means pour comparer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

linkage_matrix = linkage(dfhca[["Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]], method='ward')
HCA = AgglomerativeClustering(n_clusters=3,metric='euclidean',linkage='ward')

#%% Dendogramme
dendrogram(linkage_matrix, labels=dfhca.index)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
plt.axhline(y=1,color='red', linestyle='-')
plt.show()

Labels = HCA.fit_predict(dfhca[["Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]])
plt.scatter(dfhca["Total des voies d'accès de communication pour 100 habitants"],dfhca["Total des abonnements au téléphone cellulaire mobile pour 100 habitants"],c=Labels)
plt.show()

dfhca['Cluster'] = Labels

#On voit que le résultat est très proche des K-Means.
#==> Avec ces deux méthodes de clustering on retrouve les même résultats
#==> On peut ainsi exploiter ces données. (analyse le contenu de dfhca ou dfkmean => ce sont les mêmes)