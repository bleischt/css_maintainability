import os, sys

from collections import defaultdict
import math

import CodeSmellParser

import numpy as np
import matplotlib.pyplot as plt
import graphviz
from prettytable import PrettyTable

from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.linear_model import LogisticRegression
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
            print(siteDir + ':', len(smells))
    return framesToSmells

#returns a numpy array of unlabled code smell feature sets
#feature order is preserved over each feature set, sorted lexicographically
#example: [[1, 4, 2], [3, 8, 5], ...], where 1 & 3 are values for the same feature
def getUnlabeledData(framesToSmells):
    return getFeaturesAndCorrespondingLabels_(framesToSmells)[0]

#returns a tuple of a feature array and label array
#each is a numpy array, each feature set corresponds by ordinally to each label
#feature order is preserved over each feature set, sorted lexicographically
#example: ([[1, 4, 2], [3, 8, 5], ...], [framework1, framework2, ...])
def getLabeledData(framesToSmells):
    return getFeaturesAndCorrespondingLabels_(framesToSmells)
    
def getFeaturesAndCorrespondingLabels_(framesToSmells):
    features = []
    labels = []
    for framework,smells in framesToSmells.items():
        for smell in smells.values():
            smellTuples = sorted(list(smell.items())) #sort dict items to ensure same order
            features.append([value for smell,value in smellTuples])
            labels.append(framework)
    return np.array(features),np.array(labels)

def outputLatexTable(framesToSmells):
    smell_frame_avg = calcFrameworkMeansAndDeviation(framesToSmells)
    frameworks = list(list(smell_frame_avg.values())[0].keys())
    rows = []
    for smell,frameToAvg in smell_frame_avg.items():
        row = [smell]
        for frame in frameworks:
            avgs = frameToAvg[frame]
            row.extend([str(round(avgs['mean'], 2)), str(round(avgs['standard_deviation'], 2))])
        rows.append(row)

    column_structure = ['|c' for i in range(len(frameworks) * 2 + 1)]
    column_structure[-1] += '|'
    table = "\\begin{figure}\n\\begin{center}\n\\begin{tabular}{ " 
    table += ''.join(column_structure) + " }\n\\cline{2-" + str(len(frameworks) * 2 + 1) + "}\n"
    table += "\multicolumn{1}{c}{} & "
    table += ' & '.join(["\multicolumn{2}{|c|}{" + frame + "}" for frame in frameworks])
    table += ' \\\\\n\\hline\nCode Smell & '
    table += ' & '.join(['mean & std-dev' for frame in frameworks])
    table += ' \\\\\n\\hline\n'
    table += '\\\\\n'.join([' & '.join(row) for row in rows]) + ' \\\\\n\\hline\n'
    table += '\\end{tabular}\n\\end{center}\n\\end{figure}\n'

    return table

def meanTable(framesToSmells):
    smell_frame_avg = calcFrameworkMeansAndDeviation(framesToSmells)
    frameworks = list(list(smell_frame_avg.values())[0].keys())
    subColHeaders = ['smellName']
    smellToAvg = defaultdict(list)
    for frame in frameworks:
        subColHeaders.append(frame + '-mean')
        subColHeaders.append(frame + '-std-dev')

    table = PrettyTable(subColHeaders)
    for smell,frameToAvg in smell_frame_avg.items():
        row = [smell]
        for frame in frameworks:
            avgs = frameToAvg[frame]
            row.extend([str(round(avgs['mean'], 2)), str(round(avgs['standard_deviation'], 2))])
        table.add_row(row) 
    print(table)

def calcFrameworkMeansAndDeviation(framesToSmells):
    smell_frame_values = aggregateSmells(framesToSmells)
    smell_frame_avg = defaultdict(lambda:defaultdict(dict))
    for smell,framesToValues in smell_frame_values.items():
        for frame,valueList in framesToValues.items():
            array = np.array([value for value in list(valueList)])
            smell_frame_avg[smell][frame] = {'mean': array.mean(), 'standard_deviation': array.std()} 
    return smell_frame_avg

#transform structure of code smell data from framework->website->>smellName->>>smellValue
#to smellName->framework->>list(smellValue)
def aggregateSmells(framesToSmells):
    smell_frame_values = defaultdict(lambda: defaultdict(list))
    for frame,smells in framesToSmells.items():    
        for smellList in smells.values():         
                for name, value in smellList.items():
                    smell_frame_values[name][frame].append(value)
    return smell_frame_values 

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

def run_kmeans(framesToSmells, numClusters, visualization=False):
    data = getUnlabeledData(framesToSmells)
    data_scaled = scaleData(data)
    kmeans = KMeans(n_clusters=numClusters)
    kmeans.fit(data_scaled)
    print('labels:', kmeans.labels_)
    print('cluster centers:', kmeans.cluster_centers_)
    if visualization:
        visualize_kmeans(data_scaled, numClusters, 'Code Smell Cluster')


def run_logistic_regression(framesToSmells):
    smellNames = sorted(list(list(framesToSmells.values())[0].values())[0].keys())
    features,labels = getLabeledData(framesToSmells)
    features_scaled = scaleData(features)
    features_train,features_test,labels_train,labels_test = train_test_split(
            features_scaled, labels, test_size=0.33)
    logreg = LogisticRegression()
    #logreg = LogisticRegression(multi_class='multinomial', solver='saga')
    logreg.fit(features_train, labels_train)
    coefs = logreg.coef_
    for classIndex,target in enumerate(logreg.classes_):
        print('class:', target)
        for smellIndex,smell in enumerate(smellNames):
            print('\t{}: {}'.format(smell, round(math.exp(coefs[classIndex][smellIndex]), 4)))
    print('intercept:', logreg.intercept_)
    print('score:', logreg.score(features_test, labels_test))

def visualize_tree(clf):
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("smells")

def run_decision_tree(framesToSmells, visualization=False):
    features,labels = getLabeledData(framesToSmells)
    features_scaled = scaleData(features)
    features_train,features_test,labels_train,labels_test = train_test_split(
            features_scaled, labels, test_size=0.33)
    clf = tree.DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    print(clf.score(features_test, labels_test))
    if visualization:
        visualize_tree(clf)

def getTestSmells():
    import pprint
    framesToSmells = {'cakephp': {'www.bowser.com': {'smell1': 2, 'smell2': 3}, 'www.mario.com': {'smell1': 5, 'smell2': 3}, 'www.peach.com': {'smell1': 18, 'smell2': 28}}, 'django': {'www.wario.com' : {'smell1': 4, 'smell2': 91}, 'www.luigi.com': {'smell1': 8, 'smell2': 0}, 'www.dasiy.com': {'smell1': 0, 'smell2':1002}}}
    pprint.pprint(framesToSmells)
    return framesToSmells

def main():
    sitesDir = checkArgs()
    #framesToSmells = loadCodeSmells(sitesDir)
    framesToSmells = getTestSmells()
    #smells = list(list(framesToSmells.values())[0].values())[0].keys()
    #print(smells)
    meanTable(framesToSmells)
    #outputLatexTable(framesToSmells)
    run_kmeans(framesToSmells, len(framesToSmells.keys()), visualization=False)
    #run_decision_tree(framesToSmells, visualization=True)
    #run_logistic_regression(framesToSmells)
    #calcFrameworkMeansAndDeviation(framesToSmells)
main()

#sitesDir = checkArgs()
#printSmellCounts(sitesDir)
