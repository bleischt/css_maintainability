import os, sys

from collections import defaultdict
import math

import CodeSmellParser

import numpy as np
from prettytable import PrettyTable

from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score

import graphviz
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
    smell_frame_avg = calcFrameworkAverages(framesToSmells)
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
    smell_frame_avg = calcFrameworkAverages(framesToSmells)
    frameworks = list(list(smell_frame_avg.values())[0].keys())
    subColHeaders = ['smellName']
    smellToAvg = defaultdict(list)
    for frame in frameworks:
        subColHeaders.append(frame + '-mean')
        subColHeaders.append(frame + '-std-dev')
        subColHeaders.append(frame + '-median')
        subColHeaders.append(frame + '-%')

    table = PrettyTable(subColHeaders)
    for smell,frameToAvg in smell_frame_avg.items():
        row = [smell]
        for frame in frameworks:
            avgs = frameToAvg[frame]
            row.extend([str(round(avgs['mean'], 2)), str(round(avgs['standard_deviation'], 2)), str(round(avgs['median'], 2)), str(round(avgs['percent'], 2))])
        table.add_row(row) 
    print(table)

def calcFrameworkAverages(framesToSmells):
    smell_frame_values = aggregateSmells(framesToSmells)
    smell_frame_avg = defaultdict(lambda:defaultdict(dict))
    for smell,framesToValues in smell_frame_values.items():
        for frame,valueList in framesToValues.items():
            array = np.array([value for value in list(valueList)])
            percent = len([val for val in array if val != 0]) / len(array)
            smell_frame_avg[smell][frame] = {'mean': array.mean(), 'median':np.median(array), 'percent':percent, 'standard_deviation': array.std()} 
    #print(smell_frame_avg)
    return smell_frame_avg

def get_smell_vectors(smell_frame_avg):
    vectors = []
    for smell,frames in smell_frame_avg.items():
        means = [avgDict['mean'] for frame,avgDict in sorted(list(frames.items()))]
        vectors.append(means) 
    return vectors

def get_framework_vectors(smell_frame_avg):
    vectors = []
    frames = list(smell_frame_avg.values())[0].keys()
    for frame in frames:
        means = []
        for smell in smell_frame_avg.keys():
            means.append(smell_frame_avg[smell][frame]['mean'])
        vectors.append(means)
    return vectors
        
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

    colors = ['#FF0000', '#EB984E', '#FFFF00', '#008000', '#0000FF', '#800080', 
            '#000080', '#FF00FF', '#00FF00', '#800000', '#808080', '#FA8072',
            '#F4D03F', '#48C9B0', '#D7BDE2', '#0B5345', '#FDEBD0', '#A04000', 
            '#8D6E63']

    #print(reduced_data)
    #for i,point in enumerate(reduced_data):
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    #    plt.plot(point[0], point[1], 'k.', color=colors[i], markersize=2)
        

    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.legend()
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('plots/plot-pca{}'.format(numClusters))
    return plt
    #plt.show()

def run_kmeans(data, numClusters, visualization=False):
    kmeans = KMeans(n_clusters=numClusters)
    kmeans.fit(data_scaled)
    print('labels:', kmeans.labels_)
    print('cluster centers:', kmeans.cluster_centers_)
    if visualization:
        visualize_kmeans(data_scaled, numClusters, 'Code Smell Cluster')
    return kmeans

def run_kmeans_determine_clusters(data, minClusters=2, maxClusters=10, visualization=False):
    for n_clusters in range(minClusters, maxClusters+1):
        kmeans = KMeans(n_clusters=n_clusters)
        #print(data)
        kmeans.fit(data)
        if visualization:
            labels = kmeans.labels_
            plot_silhouettes(data, kmeans, n_clusters)
            visualize_kmeans(data, n_clusters, 'Framework Cluster')
        silhouette_avg = metrics.silhouette_score(data, labels, metric='euclidean')
        print('for {} clusters, avg silhouette score is: {}'.format(n_clusters, silhouette_avg))

def plot_silhouettes(data, clusterer, n_clusters):
    # Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
    
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    #clusterer = KMeans(n_clusters=n_clusters)
    #cluster_labels = clusterer.fit_predict(data)
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(reduced_data)
    #cluster_labels = clusterer.labels_
    cluster_labels = kmeans.labels_

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(data, cluster_labels)
    #print("For n_clusters =", n_clusters,
    #"The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='.', s=30, lw=0, alpha=0.7,
        c=colors, edgecolor='k')

    # Labeling the clusters
    centers = kmeans.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
        c="white", alpha=1, s=80, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
            s=50, edgecolor='k')

    ax2.set_title("The visualization of the PCS-reduced clustered data.")
    #ax2.set_xlabel("Feature space for the 1st feature")
    #ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
          "with n_clusters = %d" % n_clusters),
         fontsize=14, fontweight='bold')

    plt.savefig('plots/plot{}'.format(n_clusters))

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
    #data = getUnlabeledData(framesToSmells)
    #data_scaled = scaleData(data)
    #plot_silhouettes(data_scaled)
    framesToSmells = getTestSmells()
    #smells = list(list(framesToSmells.values())[0].values())[0].keys()
    #print(smells)
    meanTable(framesToSmells)
    exit()
    #outputLatexTable(framesToSmells)
    #run_kmeans(framesToSmells, len(framesToSmells.keys()), visualization=False)
    #run_decision_tree(framesToSmells, visualization=False)
    #run_logistic_regression(framesToSmells)
    averages = calcFrameworkAverages(framesToSmells)
    #vectors = get_smell_vectors(averages)
    vectors = get_framework_vectors(averages)
    data_scaled = scaleData(vectors)
    print(len(data_scaled))
    run_kmeans_determine_clusters(data_scaled, visualization=True)
main()

#sitesDir = checkArgs()
#printSmellCounts(sitesDir)
