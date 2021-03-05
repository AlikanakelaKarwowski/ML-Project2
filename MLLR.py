import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from matplotlib import cm
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import plot_confusion_matrix as pcm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from scipy import stats

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
#max, min, mean, median, mode, standard deviation
for i in range (6,11):
    j = x[house['grade'] == i]

    print(i)
    print(j.max())
    print(j.min())
    print(j.mean())
    print(np.std(j))

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



#temp
# Create classifier object: Create a linear SVM classifier
# C: Regularization parameter. Default C=1

lsvc = LinearSVC(C=100, random_state=0, tol=1e-4)
lsvc.fit(xtrain, ytrain)
print(f"Linear SVM Training set score: {100*lsvc.score(xtrain, ytrain):.2f}%")
print(f"Linear SVM Test set score: {100*lsvc.score(xtest, ytest):.2f}%")
#
lsvc.predict(xtest)
print(lsvc.coef_)
print(lsvc.intercept_)

pcm(lsvc, xtest, ytest, normalize='true')
plt.show()

# Create classifier object: Create a nonlinear SVM classifier
# kernel, default="rbf" = radial basis function
# if poly, default degree = 3

svc = SVC(degree=2, kernel='poly', random_state=0, gamma='auto')
svc.fit(xtrain, ytrain)
print(f"SVM Poly Training set score: {100*svc.score(xtrain, ytrain):.2f}%")
print(f"SVM Poly Test set score: {100*svc.score(xtest, ytest):.2f}%")

pcm(svc, xtest, ytest, normalize='true')
plt.show()

# Create classifier object: Create a nonlinear SVM classifier
# kernel, default="rbf" = radial basis function

svc = SVC(C=10, gamma='auto', random_state=100)
svc.fit(xtrain, ytrain)
print(f"SVM Gaussian Training set score: {100*svc.score(xtrain, ytrain):.2f}%")
print(f"SVM Gaussian Test set score: {100*svc.score(xtest, ytest):.2f}%")

pcm(svc, xtest, ytest, normalize='true')
plt.show()
# SVM with linear kernel

svc = SVC(C=10, degree=1, kernel='poly')
svc.fit(xtrain, ytrain)
print(f"SVM Gaussian Training set score: {100*svc.score(xtrain, ytrain):.2f}%")
print(f"SVM Gaussian Test set score: {100*svc.score(xtest, ytest):.2f}%")

pcm(svc, xtest, ytest, normalize='true')
plt.show()
