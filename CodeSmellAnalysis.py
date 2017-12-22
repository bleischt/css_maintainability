import os, sys
import CodeSmellParser
import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing 

np.random.seed(42)


def checkArgs():
    if len(sys.argv) != 2:
        print('Incorrect args. Usage: python3 {} <sites_directory>'.format(sys.argv[0]))
        exit()
    return sys.argv[1]

#loads code smells from each directory of sites located at sitesDir
#returns a dict of framework mapped to code smells, where code smells is a dict of 
#site matched to the smell values for that site
#example: {'drupal': {'google.com' : [smell1, smell2, ...], 'yahoo.com' : [smell1, ...]}, ... }
def loadCodeSmells(sitesDir):
    siteDirs = [name for name in os.listdir(sitesDir) if os.path.isdir('{}/{}'.format(sitesDir, name))]
    absSiteDir = os.path.abspath(sitesDir)
    framesToSmells = dict()
    for siteDir in siteDirs:
        smells = CodeSmellParser.readCodeSmells('{}/{}'.format(absSiteDir, siteDir))
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

def scaleData(data):
    return preprocessing.scale(data)

def run_kmeans(data, numClusters):
    data_scaled = scaleData(data)
    kmeans = KMeans(n_clusters=numClusters)
    kmeans.fit(data_scaled)
    print('labels:', kmeans.labels_)
    print('cluster centers:', kmeans.cluster_centers_)

def run_decision_tree(data):
    pass

def main():
    sitesDir = checkArgs()
    framesToSmells = loadCodeSmells(sitesDir)
    data = getUnlabeledDataArray(framesToSmells)
    data_scaled = scaleData(data)
    run_kmeans(data_scaled, 3)

main()

#django_smells_dict = CodeSmellParser.readCodeSmells('sites/django_sites')
#django_smells = np.array([list(dic.values()) for site,dic in django_smells_dict.items()])
#wordpress_smells_dict = CodeSmellParser.readCodeSmells('sites/wordpress_sites')
#wordpress_smells = np.array([list(dic.values()) for site,dic in wordpress_smells_dict.items()])
#drupal_smells_dict = CodeSmellParser.readCodeSmells('sites/drupal_sites')
#drupal_smells = np.array([list(dic.values()) for site,dic in drupal_smells_dict.items()])
#smells = np.concatenate((django_smells, wordpress_smells, drupal_smells))
#print(smells)


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


def run():
    checkArgs()
