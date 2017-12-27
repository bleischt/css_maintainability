import os, sys

import CodeSmellParser

import numpy as np
import matplotlib.pyplot as plt
import graphviz

from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn import preprocessing 
from sklearn import metrics
from sklearn.model_selection import train_test_split

np.random.seed(42)


def checkArgs():
    if len(sys.argv) != 2:
        print('Incorrect args. Usage: python3 {} <sites_directory>'.format(sys.argv[0]))
        exit()
    return sys.argv[1]

def printSmellCounts(sitesDir):
    framesToSmells = loadCodeSmells(sitesDir)
    for framework,smells in framesToSmells.items():
        print(framework, ':', len(smells))
#loads code smells from each directory of sites located at sitesDir
#returns a dict of framework mapped to code smells, where code smells is a dict of 
#site matched to the smell values for that site
#example: {'drupal': {'google.com' : [smell1, smell2, ...], 'yahoo.com' : [smell1, ...]}, ... }
def loadCodeSmells(sitesDir):
    siteDirs = [name for name in os.listdir(sitesDir) if os.path.isdir('{}/{}'.format(sitesDir, name))]
    absSiteDir = os.path.abspath(sitesDir)
    framesToSmells = dict()
    for siteDir in siteDirs:
        smells = CodeSmellParser.readCodeSmells('{}/{}'.format(absSiteDir, siteDir), removeConnectionRefused=False)
        if len(smells) > 0:
            framesToSmells[siteDir] = smells
    return framesToSmells

#return a numpy array of unlabled code smell arrays
def getUnlabeledDataArray(framesToSmells):
    sitesToSmells = list(framesToSmells.values())
    smellDics = []
    for smell in sitesToSmells:
        smellDics.extend(list(smell.values()))
    smells = [list(smell.values()) for smell in smellDics]
    print(smells)
    return np.array(smells)

def getLabeledData(framesToSmells):
    features = []
    labels = []
    for framework,smells in framesToSmells.items():
        for smell in smells.values():
            features.append(list(smell.values()))
            labels.append(framework)
    return np.array(features),np.array(labels)

def scaleData(data):
    return preprocessing.scale(data)

def visualize_kmeans(data, numClusters, title):
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=numClusters)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def run_kmeans(framesToSmells, numClusters):
    data = getUnlabeledDataArray(framesToSmells)
    data_scaled = scaleData(data)
    kmeans = KMeans(n_clusters=numClusters)
    kmeans.fit(data_scaled)
    print('labels:', kmeans.labels_)
    print('cluster centers:', kmeans.cluster_centers_)

def visualize_tree(clf):
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("smells")

def run_decision_tree(framesToSmells):
    features,labels = getLabeledData(framesToSmells)
    features_scaled = scaleData(features)
    features_train,features_test,labels_train,labels_test = train_test_split(
            features_scaled, labels, test_size=0.33)
    clf = tree.DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    print(clf.score(features_test, labels_test))
    visualize_tree(clf)


def main():
    sitesDir = checkArgs()
    framesToSmells = loadCodeSmells(sitesDir)
    #run_kmeans(data_scaled, 3)
    run_decision_tree(framesToSmells)

main()

#sitesDir = checkArgs()
#printSmellCounts(sitesDir)
