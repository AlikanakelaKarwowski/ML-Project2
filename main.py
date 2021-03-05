import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
###############################
# fruit dataset
###############################
houses = pd.read_csv('kc_house_data.csv')
houses.head()

classes =houses['grade'].mean()

X = houses[['price','bedrooms']]

y = houses['grade']


# plotting the data
plt.subplot(2, 1, 1)
plt.scatter(X['price'], y, marker='o', color='blue', s=12)
plt.xlabel('price')
plt.ylabel('grade')

plt.subplot(2, 1, 2)
plt.scatter(X['bedrooms'], y, marker='o', color='blue', s=12)
plt.xlabel('bedrooms')
plt.ylabel('grade')
plt.show()

#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



# Create a linear model : Linear regression (aka ordinary least squares)
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"lr.coef_: {lr.coef_}")
print(f"sum lr.coef_^2: {sum(lr.coef_*lr.coef_)}")
print(f"lr.intercept_: {lr.intercept_}")

# Estimate the accuracy of the classifier on future data, using the test data
# score = 1-relative score
# R^2(y, hat{y}) = 1 - {sum_{i=1}^{n} (y_i - hat{y}_i)^2}/{sum_{i=1}^{n} (y_i - bar{y})^2}
##########################################################################################
print(f"Training set score: {lr.score(X_train, y_train):.2f}")
print(f"Test set score: {lr.score(X_test, y_test):.2f}")

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation: default 5-fold cross validation cv=5
predicted = cross_val_predict(lr, X, y, cv=10)

# plotting the data
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.scatter(y, predicted, edgecolors=(0, 0, 0))
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax1.set_xlabel('Measured')
ax1.set_ylabel('Predicted')

# Leave one out: Provides train/test indices to split data in train/test sets. Each
# sample is used once as a test set (singleton) while the remaining samples form the training set.
# n= the number of samples
loo = LeaveOneOut()
loo.get_n_splits(X)

lr = LinearRegression()
predicted = []
measured = []
for train_index, test_index in loo.split(X):
    print(f"TRAIN: {train_index} TEST:{test_index}")
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    lr.fit(X_train, y_train)
    predicted.append(lr.predict(X_test)[0])
    measured.append(y_test[0])

ax2.scatter(measured, predicted, edgecolors=(0, 0, 0))
ax2.plot([min(measured), max(measured)], [min(measured), max(measured)], 'k--', lw=4)
ax2.set_xlabel('Measured')
ax2.set_ylabel('Predicted')
plt.show()

###############################################################
# Ridge regression --- a more stable model
# In ridge regression, the coefficients (w) are chosen not only so that they predict well on the training
# data, but also to fit an additional constraint. We also want the magnitude of coefficients
# to be as small as possible; in other words, all entries of w should be close to zero.
# This constraint is an example of what is called regularization. Regularization means explicitly
# restricting a model to avoid overfitting.
# minimizing ||y - Xw||^2_2 + alpha * ||w||^2_2
# Note: the smaller alpha = the less restriction.
###############################################################
ridge = Ridge(alpha=10).fit(X_train, y_train)
print(f"ridge.coef_: {ridge.coef_}")
print(f"sum ridge.coef_^2: {sum(ridge.coef_*ridge.coef_)}")
print(f"ridge.intercept_: {ridge.intercept_}")
print(f"Training set score: {ridge.score(X_train, y_train):.2f}")
print(f"Test set score: {ridge.score(X_test, y_test):.2f}")

###############################################################
# Lasso regression --- a more stable model
# In lasso regression, the coefficients (w) are chosen not only so that they predict well on the training
# data, but also to fit an additional constraint. We also want the magnitude of coefficients
# to be as small as possible; in other words, all entries of w should be close to zero.
# This constraint is an example of what is called regularization. Regularization means explicitly
# restricting a model to avoid overfitting.
# minimizing ||y - Xw||_2 + alpha * ||w||_2
# Note: the smaller alpha = the less restriction.
###############################################################
lasso = Lasso(alpha=0.1).fit(X_train, y_train)

print(f"lasso.coef_: {lasso.coef_}")
print(f"sum lasso.coef_^2: {sum(lasso.coef_*lasso.coef_)}")
print(f"lasso.intercept_: {lasso.intercept_}")
print(f"Training set score: {lasso.score(X_train, y_train):.2f}")
print(f"Test set score: {lasso.score(X_test, y_test):.2f}")

#====== SVM

# create a mapping from fruit label value to fruit name to make results easier to interpret
X = houses[['price','bedrooms','bathrooms','sqft_living','sqft_lot','sqft_above','sqft_basement','yr_built']]
y = houses['grade']

from sklearn.model_selection import train_test_split
#random_state: set seed for random# generator
#test_size: default 25% testing, 75% training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# partition the data into two classes
y_train_1 = y_train == 1  # mandarin in True class, others in False class
y_test_1 = y_test == 1   # mandarin in True class, others in False class
y_train = 2 - y_train_1  # mandarin = 1; others =2
y_test = 2 - y_test_1

seeData = True
if seeData:
    # plotting a scatter matrix
    
    cmap = cm.get_cmap('gnuplot')
    scatter = scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

    # plotting a 3D scatter plot
       # must keep
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(X_train['price'], X_train['bedrooms'], X_train['sqft_living'], c = y_train, marker = 'o', s=100)
    ax.set_xlabel('price')
    ax.set_ylabel('bedrooms')
    ax.set_zlabel('sqft_living')
    plt.show()


# Create classifier object: Create a linear SVM classifier
# C: Regularization parameter. Default C=1

lsvc = LinearSVC(C=100, random_state=10, tol=1e-4)
lsvc.fit(X_train, y_train)
print(f"Linear SVM Training set score: {100*lsvc.score(X_train, y_train):.2f}%")
print(f"Linear SVM Test set score: {100*lsvc.score(X_test, y_test):.2f}%")
#
lsvc.predict(X_test)
print(lsvc.coef_)
print(lsvc.intercept_)

# Create classifier object: Create a nonlinear SVM classifier
# kernel, default="rbf" = radial basis function
# if poly, default degree = 3

svc = SVC(degree=2, kernel='poly', random_state=1, gamma='auto')
svc.fit(X_train, y_train)
print(f"SVM Poly Training set score: {100*svc.score(X_train, y_train):.2f}%")
print(f"SVM Poly Test set score: {100*svc.score(X_test, y_test):.2f}%")

# Create classifier object: Create a nonlinear SVM classifier
# kernel, default="rbf" = radial basis function

svc = SVC(C=10, gamma='auto', random_state=100)
svc.fit(X_train, y_train)
print(f"SVM Gaussian Training set score: {100*svc.score(X_train, y_train):.2f}%")
print(f"SVM Gaussian Test set score: {100*svc.score(X_test, y_test):.2f}%")

# SVM for multiple classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=2)

# SVM with linear kernel

svc = SVC(C=10, degree=1, kernel='poly')
svc.fit(X_train, y_train)
print(f"SVM Gaussian Training set score: {100*svc.score(X_train, y_train):.2f}%")
print(f"SVM Gaussian Test set score: {100*svc.score(X_test, y_test):.2f}%")

# kNN

knn = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform')
knn.fit(X_train, y_train)
print(f"kNN Training set score: {100*knn.score(X_train, y_train):.2f}%")
print(f"kNN Test set score: {100*knn.score(X_test, y_test):.2f}%")

