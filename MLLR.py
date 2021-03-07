import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from pandas.plotting import scatter_matrix
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import plot_confusion_matrix as pcm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.svm import SVC, LinearSVC
from multiprocessing import Process
plt.style.use("fivethirtyeight")

def find_stats(x):
    #max, min, mean, median, mode, standard deviation
    for i in range (6,11):
        j = x[house['grade'] == i]
        print(f"\nStatistics for Grade: {i}")
        print(f'\nMax:\n{round(j.max(),3)}')
        print(f'\nMin:\n{round(j.min(),3)}')
        print(f'\nMean:\n{round(j.mean(),3)}')
        print(f'\nSTD:\n{round(np.std(j),3)}')

    print(f'\nMedian:\nHARDCODED VALUE FROM EXCEL')
    print(f'\nMode:\nHARDCODED VALUE FROM EXCEL')

def show_data(x, y):
    # Show Graphical Representations of our Data
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

def log_reg(x, y, show_pcm=False):
    x = scale(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0, test_size=0.25)

    lr = LogisticRegression(random_state=0, multi_class='auto', max_iter=10000)
    lr.fit(xtrain,ytrain)
    print(f"Logistic Regression Score: {round(lr.score(xtest,ytest),3)}")
    if show_pcm:
        pcm(lr, xtest, ytest, normalize='true')
        plt.show()

def k_nearest(x, y, show_pcm=False):
    x = scale(x)
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
    print(f"KNN Score: {round(knn.score(xtest,ytest),3)}")
    if show_pcm:
        pcm(knn, xtest, ytest, normalize='true')
        #Show Relevant plots based on requirements
        plt.figure()
        plt.plot(neighbors,test_acc,label="Testing Dataset Accuracy KNN")
        plt.legend()
        plt.xlabel("n_neighbors")
        plt.ylabel("Accuracy")
        plt.show()

def linear_SVC(x, y, show_pcm=False):
    x = scale(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0, test_size=0.25)

    lsvc = LinearSVC(C=100, random_state=0, tol=1e-5)
    lsvc.fit(xtrain, ytrain)
    print(f"Linear SVM Training set score: {100*lsvc.score(xtrain, ytrain):.2f}%")
    print(f"Linear SVM Test set score: {100*lsvc.score(xtest, ytest):.2f}%")
    
    lsvc.predict(xtest)
    print(f"Linear SVC Coeffienct: {lsvc.coef_}")
    print(f"Linear SVC Intercept: {lsvc.intercept_}")
    if show_pcm:
        pcm(lsvc, xtest, ytest, normalize='true')
        plt.show()

def svc_poly(x, y, show_pcm=False):
    x = scale(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0, test_size=0.25)

    svc = SVC(degree=2, kernel='poly', random_state=0, gamma='auto')
    svc.fit(xtrain, ytrain)
    print(f"SVM Poly Training set score: {100*svc.score(xtrain, ytrain):.2f}%")
    print(f"SVM Poly Test set score: {100*svc.score(xtest, ytest):.2f}%")
    
    if show_pcm:
        pcm(svc, xtest, ytest, normalize='true')
        plt.show()

def svc_rbf(x, y, show_pcm=False):
    x = scale(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0, test_size=0.25)

    svc = SVC(C=10, gamma='auto', random_state=0)
    svc.fit(xtrain, ytrain)
    print(f"SVM Gaussian Training set score: {100*svc.score(xtrain, ytrain):.2f}%")
    print(f"SVM Gaussian Test set score: {100*svc.score(xtest, ytest):.2f}%")

    if show_pcm:
        pcm(svc, xtest, ytest, normalize='true')
        plt.show()

def svc_linear_kernel(x, y, show_pcm=False):
    x = scale(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state=0, test_size=0.25)

    svc = SVC(C=10, degree=1, kernel='poly')
    svc.fit(xtrain, ytrain)
    print(f"SVM Gaussian Training set score: {100*svc.score(xtrain, ytrain):.2f}%")
    print(f"SVM Gaussian Test set score: {100*svc.score(xtest, ytest):.2f}%")

    if show_pcm:
        pcm(svc, xtest, ytest, normalize='true')
        plt.show()


if __name__ == "__main__": 
    USAhousing = pd.read_csv('kc_house_data.csv')
    
    # Features
    # Slicing our data set to more desireable values
    house= USAhousing[USAhousing["grade"] >=6]
    house= house[house["grade"] <=10]
    # Set x and y variables for testing
    x = house[['bedrooms', 'bathrooms', 'price', 'sqft_living', 'sqft_lot', 'yr_built', 'floors', 'sqft_basement']]
    y = house["grade"]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # MULTIPROCESSING IS GOING ON, IF YOU HAVE A SLOW COMPUTER LIMIT IT TO 2 or 3 Processes AT MOST #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    #Change the last arg from False to True to enable graphs 
    p1 = Process(target=log_reg, args=(x,y,False))
    p2 = Process(target=k_nearest, args=(x,y,False))
    p3 = Process(target=linear_SVC, args=(x,y,False,))
    p4 = Process(target=svc_poly, args=(x,y,False,))
    p5 = Process(target=svc_rbf, args=(x,y,False,))
    p6 = Process(target=svc_linear_kernel, args=(x,y,False,))
    
    #Start the Processes
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()

    #Join the Processes once they finish
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    
    show_data(x,y)
    find_stats(x)
