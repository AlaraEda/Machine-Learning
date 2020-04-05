"""
Startercode bij Lesbrief: Machine Learning, CMTPRG01-9

Deze code is geschreven in Python3

Benodigde libraries:
- NumPy
- SciPy
- matplotlib
- sklearn

"""
from machinelearningdata import Machine_Learning_Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn import tree

""" helper functie om data uit de json te halen en om te zetten naar numpy array voor sklearn"""
def extract_from_json_as_np_array(key, json_data):
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])

    return np.array(data_as_array)


STUDENTNUMMER = "0931871"

# If/-statement
assert STUDENTNUMMER != "1234567", "Verander 1234567 in je eigen studentnummer"

# Console.log
print("STARTER CODE")

# Een data-object die jouw data van de server op haald
data = Machine_Learning_Data(STUDENTNUMMER)


# UNSUPERVISED LEARNING

# haal clustering data op
kmeans_training = data.clustering_training()

# extract de x waarden, In X zitten alle waaarden 
X = extract_from_json_as_np_array("x", kmeans_training)                                                 # In X zitten alle x en y waarden. 

# slice kolommen voor plotten (let op, dit is de y voor de y-as, niet te verwarren met een y van de data)
x = X[...,0]                                                                                            #Alle X-coordinaten
y = X[...,1]                                                                                            #Alle Y-coordinaten


# teken de punten uit je data
for i in range(len(x)):
    plt.plot(x[i], y[i], 'k')                                                                           # k = zwart, r = rood
    plt.title('k means centroids Alara')
    
plt.axis([min(x), max(x), min(y), max(y)])                                                              # De assen zetten.


# TODO: print deze punten uit en omcirkel de mogelijke clusters
circle1 = plt.Circle((28,48), 30, color="r", fill=False)
circle2 = plt.Circle((60,80), 30, color="r", fill=False)
circle3 = plt.Circle((87,14), 30, color="r", fill=False)
circle4 = plt.Circle((3,1), 30, color="r", fill=False)
circle5 = plt.Circle((14,96), 30, color="r", fill=False)

getFigure = plt.gcf()                                                                                   #Krijg het figuur
ax=getFigure.gca()                                                                                      #Krijg de polaire assen binnen van het figuur, 30

# Voeg de circles toe
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(circle3)
ax.add_artist(circle4)
ax.add_artist(circle5)

# TODO: ontdek de clusters mbv kmeans en teken een plot met kleurtjes
km = KMeans(n_clusters=5).fit(X)                                                                        #5 Klusters met de punten van X
centers = km.cluster_centers_                                                                           #Coordination of cluster center
group_dots = km.labels_                                                                                 #Predict cluster index voor elke cluster

plt.scatter(x, y, c=group_dots, s=8)                                                                    #De punten van X kleuren.
plt.scatter(centers[...,0], centers[...,1], marker="x", c='red')                                        #De center's

plt.show()                                                                                              #Plot alles



# SUPERVISED LEARNING

# haal data op voor classificatie
classification_training = data.classification_training()

# extract de data x = array met waarden, y = classificatie 0 of 1
X = extract_from_json_as_np_array("x", classification_training)

# dit zijn de werkelijke waarden, daarom kan je die gebruiken om te trainen
Y = extract_from_json_as_np_array("y", classification_training)


# TODO: leer de classificaties

# TODO: voorspel na het trainen de Y-waarden (je gebruikt hiervoor dus niet meer de
#       echte Y-waarden, maar enkel de X en je getrainde classifier) en noem deze
#       bijvoordeeld Y_predict

# TODO: vergelijk Y_predict met de echte Y om te zien hoe goed je getraind hebt


# haal data op om te testen
classification_test = data.classification_test()
# testen doen we 'geblinddoekt' je krijgt nu de Y's niet
X_test = extract_from_json_as_np_array("x", classification_test)

# TODO: voorspel na nog een keer de Y-waarden, en plaats die in de variabele Z
#       je kunt nu zelf niet testen hoe goed je het gedaan hebt omdat je nu
#       geen echte Y-waarden gekregen hebt.
#       onderstaande code stuurt je voorspelling naar de server, die je dan
#       vertelt hoeveel er goed waren.

Z = np.zeros(100) # dit is een gok dat alles 0 is... kan je zelf voorspellen hoeveel procent er goed is?

# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(Z.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
print("Classificatie accuratie (test): " + str(classification_test))

