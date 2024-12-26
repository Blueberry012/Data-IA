import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# En effectuant un k-means la dessus, on pourra classer les pays selon leur niveau de développement réseau,
# que ce soit au niveau des infrastructures ou de l'utilisation par la population.

#%%
df=pd.read_csv("cleaned_data.csv")
#%%
df2020 = df[(df["TIME_PERIOD"]==2000)]
df2020=df2020.drop('TIME_PERIOD',axis=1)
#%% Heatmap afin de voir quelles variables sont intéréssantes à étudier entre-elles
correlation_matrix = df2020.drop('Pays', axis=1).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title("Matrice de corrélation")
plt.show()
#%% Colums : - Total des voies d'accès de communication pour 100 habitants
#            - Total des abonnements au téléphone cellulaire mobile pour 100 habitants
#   La data est directement pris cru car on a aucune valeurs abérantes ou de valeurs NaN
from sklearn.cluster import KMeans

X = df2020["Total des voies d'accès de communication pour 100 habitants"] 
Y = df2020["Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]

dfkmean=df2020[["Pays","Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]]

#%% Normalisons ensuite les données pour éviter des résultats biaisés ou inefficaces
#   Utilisons la normalisation Min-Max pour que les données soient dans la même plage
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dfkmean[["Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]]=scaler.fit_transform(dfkmean[["Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]])

#%% Methode elbow pour voir le nombre de clusters à créer 
W = []
for i in range(1, 15):
    km = KMeans(n_clusters = i, init = 'k-means++')
    km.fit(dfkmean[["Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]])
    W.append(km.inertia_)
plt.plot(range(1, 15), W)
plt.grid(True)
plt.xlabel('Number of clusters k')
plt.ylabel('Sum of squared distance')
plt.show()
#%% On retient ainsi que le elbow se situe a peu près au nombre de 3 clusters. On choisira donc 3 clusters
#   Leur signification sera ainsi:
#   C3: Pays peu développés en infrastructure réseau et en accessibilité
#   C2: Pays développés en infrastructure réseau et en accessibilité
#   C1: Pays très bien développés en infrastructure réseau et en accessibilité

#%% On analyse maintenant via k-means avec 3 clusters
from sklearn.cluster import KMeans

# Ajoute une colonne Cluster pour savoir a quel cluster le pays appartient

km = KMeans(n_clusters = 3, init="k-means++")
dfkmean['Cluster']= km.fit_predict(dfkmean[["Total des voies d'accès de communication pour 100 habitants","Total des abonnements au téléphone cellulaire mobile pour 100 habitants"]])

#%% Affichage

plt.scatter(dfkmean["Total des voies d'accès de communication pour 100 habitants"],dfkmean["Total des abonnements au téléphone cellulaire mobile pour 100 habitants"],c=dfkmean['Cluster'])

# Tracer les points avec une couleur par cluster
plt.scatter(dfkmean["Total des voies d'accès de communication pour 100 habitants"], 
            dfkmean["Total des abonnements au téléphone cellulaire mobile pour 100 habitants"], 
            c=dfkmean['Cluster'], cmap='viridis', marker='o')

# Tracer les centres des clusters
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=200, c='red', marker='X', label="Centres des clusters")

# Ajouter la légende de couleur
plt.colorbar(scatter, label='Cluster')

plt.show()

#FIN KMEANS (REVERIFIER VITE FAIT LE GRAPHE)








