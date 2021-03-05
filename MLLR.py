import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import plot_confusion_matrix as pcm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

plt.style.use("fivethirtyeight")

USAhousing = pd.read_csv('kc_house_data.csv')
# Features
house= USAhousing[USAhousing["grade"] >=6]
house= house[house["grade"] <=10]
x = house[['bedrooms', 'bathrooms', 'price', 'sqft_living', 'sqft_lot', 'yr_built', 'floors', 'sqft_basement']]
y = house["grade"]
print(f"8 Features, 5 classes, {len(x)} Entries")

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0, test_size=0.25)
print("Using 25% for Testing. 75% for Training.")

cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(xtrain, c=ytrain, marker='o', s=40, hist_kwds={'bins':14})
plt.show

fig=plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(xtrain['bedrooms'],xtrain['price'],xtrain['sqft_living'], c = ytrain, s=100)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_zlabel('Sq. Footage')
plt.show()


x = scale(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0, test_size=0.25)
lr = LogisticRegression(random_state=0, multi_class='auto', max_iter=10000)
lr.fit(xtrain,ytrain)
print(lr.score(xtest,ytest))
pcm(lr, xtest, ytest, normalize='true')
plt.show()




#KNN
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0, test_size=0.25)

neighbors = np.arange(1,50)
train_acc = np.empty(len(neighbors))
test_acc = np.empty(len(neighbors))

#run through testing for each k number of neighbors
for i,k in enumerate(neighbors):
    #training
    #setting the metric to minkowski and p = 1 set the algorithm used to manhattan
    knn = KNeighborsClassifier(n_neighbors=k,metric='minkowski',p=1, weights='distance')
    knn.fit(xtrain, ytrain)
    #Save accuracy for both training and testing
    train_acc[i] = knn.score(xtrain, ytrain)
    test_acc[i] = knn.score(xtest, ytest)
knn = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p=1, weights='distance')
knn.fit(xtrain, ytrain)

pcm(knn, xtest, ytest, normalize='true')
#Show Relevant plots based on requirements
plt.figure()
plt.plot(neighbors,test_acc,label="Testing Dataset Accuracy")
plt.legend()
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.show()
