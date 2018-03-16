import os, sys

from collections import defaultdict
import math, random

import CodeSmellParser

import numpy as np
from prettytable import PrettyTable

from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn import metrics
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

import graphviz
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import AgglomerativeClustering
import ete3

smell_based = {'InlinedRules', 'EmbeddedRules', 'TooLongRules', 'SelectorswithIDandatleastoneclassorelement', 'PropertieswithValueEqualtoNoneorZero', 'SelectorswithInvalidSyntax', 'PropertieswithHard-CodedValues', 'UniversalSelectors', 'TooSpecificSelectorsTypeII', 'TooSpecificSelectorsTypeI', 'DangerousSelectors', 'EmptyCatchRules', 'TooLongRules'}
metric_based = {'ExternalRules', 'Effective', 'FileswithCSScode', 'LOC(CSS)', 'Ineffective', 'TotalDefinedCSSselectors', 'TotalDefinedCSSrules', 'Matched', 'TotalDefinedCSSProperties', '>UndefinedClasses', 'Unmatched', 'Ignored', 'IgnoredProperties', 'UnusedProperties'}
rule_based = {'InlinedRules', 'EmbeddedRules', 'ExternalRules', 'TooLongRules', 'TotalDefinedCSSrules', 'EmptyCatchRules'}
selector_based = {'TooSpecificSelectorsTypeI', 'TooSpecificSelectorsTypeII', 'SelectorswithInvalidSyntax', 'DangerousSelectors', 'UniversalSelectors', 'TotalDefinedCSSselectors', 'Ignored', 'SelectorswithIDandatleastoneclassorelement', 'Matched', 'Effective', 'Unmatched', 'Ineffective', '>UndefinedClasses'}
property_based = {'PropertieswithHard-CodedValues', 'PropertieswithValueEqualtoNoneorZero', 'TotalDefinedCSSProperties', 'UnusedProperties', 'IgnoredProperties'}
file_based = {'FileswithCSScode', 'LOC(CSS)'}


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
def loadCodeSmells(sitesDir, smellsToKeep=None):
    siteDirs = [name for name in os.listdir(sitesDir) if os.path.isdir('{}/{}'.format(sitesDir, name))]
    absSiteDir = os.path.abspath(sitesDir)
    framesToSmells = dict()
    for siteDir in siteDirs:
        smells = CodeSmellParser.readCodeSmells('{}/{}'.format(absSiteDir, siteDir), removeConnectionRefused=False, smellsToKeep=smellsToKeep)
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

def outputPercentLatexTable(framesToSmells):
    smell_frame_avg = calcFrameworkAverages(framesToSmells)
    frameworks = sorted(list(list(smell_frame_avg.values())[0].keys()))
    #frameworks = frameworks[:]
    rows = []
    numSubColumns = 1
    for smell,frameToAvg in smell_frame_avg.items():
        row = [smell]
        for index,frame in enumerate(frameworks):
            avgs = frameToAvg[frame]
            row.extend([str(round(avgs['percent'] * 100, 2))])
        rows.append(row)

    column_structure = ['|c' for i in range(len(frameworks) * numSubColumns + 1)]
    column_structure[-1] += '|'
    table = "\\begin{table*}\n\\centering\n\\begin{sideways}\n\\begin{tabular}{ " 
    table += ''.join(column_structure) + " }"
    #table += \n\\cline{2-" + str(len(frameworks) * numSubColumns + 1) + "}\n"
    #table += "\multicolumn{1}{c}{} & "
    #table += ' & '.join(["\multicolumn{" + str(numSubColumns) + "}{|c|}{" + frame + "}" for frame in frameworks])
    table += '\n\\hline\nCode Smell & ' + ' & '.join(frameworks)
    table += ' \\\\\n\\hline\n'
    table += '\\\\\n'.join([' & '.join(row) for row in rows]) + ' \\\\\n\\hline\n'
    table += '\\end{tabular}\n\\end{sideways}\n\\caption{The percentages of the website for a specific framework that contain at least one instance of the specified code smell.}\n\\label{tab:smell_percents}\n\\end{table*}\n'

    return table

def outputAveragesLatexTable(framesToSmells):
    smell_frame_avg = calcFrameworkAverages(framesToSmells)
    frameworks = sorted(list(list(smell_frame_avg.values())[0].keys()))
    frameworks = frameworks[15:]
    rows = []
    numSubColumns = 3
    for smell,frameToAvg in smell_frame_avg.items():
        row = [smell]
        means = preprocessing.scale([frameToAvg[frame]['mean'] for frame in frameworks])
        for index,frame in enumerate(frameworks):
            avgs = frameToAvg[frame]
            row.extend([str(round(avgs['mean'], 2)), str(round(avgs['standard_deviation'], 2)), str(round(means[index], 2))])
        rows.append(row)

    column_structure = ['|c' for i in range(len(frameworks) * numSubColumns + 1)]
    column_structure[-1] += '|'
    table = "\\begin{table*}\n\\centering\n\\resizebox{18cm}{!}{\n\\begin{tabular}{ " 
    table += ''.join(column_structure) + " }\n\\cline{2-" + str(len(frameworks) * numSubColumns + 1) + "}\n"
    table += "\multicolumn{1}{c}{} & "
    table += ' & '.join(["\multicolumn{" + str(numSubColumns) + "}{|c|}{" + frame + "}" for frame in frameworks])
    table += ' \\\\\n\\hline\nCode Smell & '
    table += ' & '.join(['mean & std-dev & z-mean' for frame in frameworks])
    table += ' \\\\\n\\hline\n'
    table += '\\\\\n'.join([' & '.join(row) for row in rows]) + ' \\\\\n\\hline\n'
    table += '\\end{tabular}\n}\n\\caption{}\n\\end{table*}\n'

    return table

def outputMegaAveragesLatexTable(framesToSmells):
    smell_frame_avg = calcFrameworkAverages(framesToSmells)
    frameworks = sorted(list(list(smell_frame_avg.values())[0].keys()))
    smellToZ = dict()
    rows = []

    mean_rows = []
    deviation_rows = []
    z_score_cols = []
    z_score_rows = []
    
    for frame in frameworks:
        means = [smell_frame_avg[smell][frame]['mean'] for smell in sorted(smell_frame_avg.keys())]
        deviations = [smell_frame_avg[smell][frame]['standard_deviation'] for smell in sorted(smell_frame_avg.keys())]
        mean_rows.append(means)
        deviation_rows.append(deviations)

    # z-scores have to be calcualted by smell which is a column on this table
    # so need to restructure means into columns (line1) and then calc zscores (line2)
    for index in range(len(smell_frame_avg.keys())):
        means_by_smell = [means[index] for means in mean_rows]
        z_score_cols.append(preprocessing.scale(means_by_smell))

    # need to convert the zscore columns to rows now to print in the latex table
    for index in range(len(frameworks)):
        z_score_rows.append([scores[index] for scores in z_score_cols])

    for index, frame in enumerate(frameworks):
        means = mean_rows[index]
        deviations = deviation_rows[index]
        z_scores = z_score_rows[index]

        row = ['\multirow{3}{*}{' + frame + '}', '\multicolumn{1}{c|}{mean}']
        means = [str(round(mean, 2)) for mean in means]
        row.extend(means)
        rows.append(' & '.join(row) + ' \\\\')
        row = ['\multicolumn{1}{c|}{std-dev}']
        deviations = [str(round(deviation, 2)) for deviation in deviations]
        row.extend(deviations)
        rows.append('& ' + ' & '.join(row) + ' \\\\')
        row = ['\multicolumn{1}{c|}{z-score}']
        z_scores = [str(round(z, 2)) for z in z_scores]
        row.extend(z_scores)
        rows.append('& ' + ' & '.join(row) + ' \\\\\n\\hline')

    column_structure = ['c' for i in range(2 + len(smell_frame_avg.keys()))]
    table = "\\begin{table}\n\\begin{sideways}\n\\resizebox{23cm}{!} {\n\\begin{tabular}{ " 
    table += ''.join(column_structure) + " }\n"
    table += "& & " + ' & '.join(['\\rot{90}{' + smell + '}' for smell in sorted(smell_frame_avg.keys())])
    table += ' \\\\\n\\hline\n'
    table += '\n'.join(rows)
    table += '\n\\end{tabular}\n}\n\\end{sideways}\n\\caption{}\n\\label{}\n\\end{table}\n'

    return table

def meanTable(framesToSmells):
    smell_frame_avg = calcFrameworkAverages(framesToSmells)
    frameworks = sorted(list(list(smell_frame_avg.values())[0].keys()))
    subColHeaders = ['smellName']
    smellToAvg = defaultdict(list)
    for frame in frameworks:
        subColHeaders.append(frame + '-mean')
        subColHeaders.append(frame + '-median')
        subColHeaders.append(frame + '-std-dev')
        subColHeaders.append(frame + '-z-score')
        subColHeaders.append(frame + '-%')

    table = PrettyTable(subColHeaders)
    for smell,frameToAvg in smell_frame_avg.items():
        row = [smell]
        means = preprocessing.scale([frameToAvg[frame]['mean'] for frame in frameworks])
        for index,frame in enumerate(frameworks):
            avgs = frameToAvg[frame]
            row.extend([str(round(avgs['mean'], 2)), str(round(avgs['median'], 2)), str(round(avgs['standard_deviation'], 2)), str(round(means[index], 2)), str(round(avgs['percent'], 2))])
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
    labels = []
    for smell,frames in sorted(list(smell_frame_avg.items())):
        labels.append(smell)
        print(smell)
        means = [avgDict['mean'] for frame,avgDict in sorted(list(frames.items()))]
        vectors.append(means) 
    return vectors,labels

def get_framework_vectors(smell_frame_avg):
    vectors = []
    frames = sorted(list(smell_frame_avg.values())[0].keys())
    for frame in frames:
        means = []
        for smell in sorted(list(smell_frame_avg.keys())):
            means.append(smell_frame_avg[smell][frame]['mean'])
        vectors.append(means)
    return vectors,frames
        
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

def visualize_kmeans(data, numClusters, title, folderName):
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
    plt.savefig('{}/plot-pca{}'.format(folderName, numClusters))
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

def run_kmeans_determine_clusters(data, labels, minClusters=2, maxClusters=10, visualization=False, latex_vector_plot=False, latex_site_plot=False, folderName='plots', title=''):
    for n_clusters in range(minClusters, maxClusters+1):
        kmeans = KMeans(n_clusters=n_clusters)
        #print(data)
        kmeans.fit(data)
        cluster_labels = kmeans.labels_
        print('---', n_clusters, '---')
        clusterToFrame = defaultdict(list)
        for label,cluster in zip(labels, cluster_labels):
            clusterToFrame[cluster].append(label)
        for cluster in clusterToFrame.keys():
            print('cluster:', cluster)
            for label in set(labels):
                print('{}: {}'.format(label, clusterToFrame[cluster].count(label))) 
        #print('label:', label, 'cluster:', cluster)
        if visualization:
            plot_silhouettes(data, kmeans, n_clusters, folderName)
            visualize_kmeans(data, n_clusters, title, folderName)
        if latex_vector_plot or latex_site_plot:
            reduced_data = PCA(n_components=2).fit_transform(data)
            reduced_kmeans = KMeans(n_clusters=n_clusters)
            reduced_kmeans.fit(reduced_data)
            cluster_labels = reduced_kmeans.labels_
            #print(reduced_data)
            if latex_vector_plot:
                print(output_latex_vector_plot(reduced_data, labels, cluster_labels))
            else:
                print(output_latex_site_plot(reduced_data, labels, cluster_labels, n_clusters, folderName))

        silhouette_avg = metrics.silhouette_score(data, cluster_labels, metric='euclidean')
        print('for {} clusters, avg silhouette score is: {}'.format(n_clusters, silhouette_avg))

def output_latex_site_plot(points, labels, cluster_labels, n_clusters, folderName):
    marks = ['*', 'square*', 'traingle*', 'pentagon*', 'oplus*', 'otimes*',
        '-', 'x', '+', 'asterisk', 'star']
    frames = list(set(labels))
    plot = '''
    \\begin{tikzpicture}
    \\begin{axis}[enlargelimits=0.2]
        \\addplot[
          scatter,mark=\mark,
          point meta=\\thisrow{color},
          visualization depends on={value \\thisrow{mark} \\as \mark},
    ]
    table {plotdata/site_clusters.dat};
    \end{axis}
\end{tikzpicture}
'''

    with open('{}/site_clusters-{}.txt'.format(folderName, n_clusters), 'w') as f:
        title = 'x y color mark'
        lines = []
        for point,label,cluster in zip(points, labels, cluster_labels):
            lines.append('\n{} {} {} {}'.format(point[0], point[1], frames.index(label)+1, marks[cluster]))
         
        f.write(title + '\n'.join(random.sample(lines, len(lines) // 2)))

    return plot

def output_latex_vector_plot(points, labels, cluster_labels):
    plot = '''
    \\begin{tikzpicture}
    \\begin{axis}[enlargelimits=0.2]
        \\addplot[
          scatter,mark=*,only marks,
          point meta=\\thisrow{color},
          nodes near coords*={\myvalue},
          visualization depends on={value \\thisrow{myvalue} \\as \myvalue},
    ]
    table {
    x y color myvalue
    '''
    for point,label,cluster in zip(points, labels, cluster_labels):
        plot += '\n{} {} {} {}'.format(point[0], point[1], cluster+1, label)

    plot += '''
        };
    \end{axis}
\end{tikzpicture}
'''
    return plot


def plot_silhouettes(data, clusterer, n_clusters, folderName):
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
        c="white", alpha=1, s=400, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
            s=50, edgecolor='k')

    ax2.set_title("PCA-Reduced Website Code Smells / Metrics")
    ax2.set_xlabel("PCA 1")
    ax2.set_ylabel("PCA 2")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
          "with n_clusters = %d" % n_clusters),
         fontsize=14, fontweight='bold')

    plt.savefig('{}/plot{}'.format(folderName, n_clusters))

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
    return clf


def run_neural_net(framesToSmells, precision_recall=False, crossValidation=None):
    features,labels = getLabeledData(framesToSmells)
    features = scaleData(features)

    features_train,features_test,labels_train,labels_test = train_test_split(
            features, labels, test_size=0.33)

    clf = MLPClassifier(alpha=1)
    #clf.fit(scaleData(features_train), labels_train)
    clf.fit(features_train, labels_train)
    print('score:', clf.score(features_test, labels_test))

    if precision_recall:
        predict = clf.predict(features_test)
        #print('precision, recall, fscore, support:', precision_recall_fscore_support(labels_test,predict, average=None, labels=sorted(list(set(labels)))))
        #print('labels:', set(labels))
        print(classification_report(labels_test, predict, sorted(list(set(labels)))))

    if crossValidation:
        clf_x = MLPClassifier(alpha=1)
        scores = cross_val_score(clf_x, features, labels, cv=crossValidation)
        print('average:', scores.mean())
        print('scores:', scores)

    #for i in range(10):
    #    rand = random.randint(0,len(features_test)-1)
    #    print('actual: {}'.format(labels_test[rand]))
    #    print('predicted: {}'.format(clf.predict(np.array([features_test[rand]]))))
    #    print('for:', features_test[rand])

def getTestSmells():
    import pprint
    framesToSmells = {'cakephp': {'www.bowser.com': {'smell1': 2, 'smell2': 3}, 'www.mario.com': {'smell1': 5, 'smell2': 3}, 'www.peach.com': {'smell1': 18, 'smell2': 28}}, 'django': {'www.wario.com' : {'smell1': 4, 'smell2': 91}, 'www.luigi.com': {'smell1': 8, 'smell2': 0}, 'www.dasiy.com': {'smell1': 0, 'smell2':1002}}}
    pprint.pprint(framesToSmells)
    return framesToSmells

def classifier_comparison(framesToSmells, crossValidation=None):
    data,labels = getLabeledData(framesToSmells)
    data = scaleData(data)

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", #"Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()
    ]

    data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.33)
    for name, clf in zip(names, classifiers):
        clf.fit(data_train, label_train)
        score = clf.score(data_test, label_test)
        print('{}: {}'.format(name, score))
        if crossValidation:
            scores = cross_val_score(clf, data, labels, cv=crossValidation)
            print('scores:', scores)
            print('average:', scores.mean())
            print()

def build_Newick_tree(children,n_leaves,X,leaf_labels,spanner):
    """
    build_Newick_tree(children,n_leaves,X,leaf_labels,spanner)

    Get a string representation (Newick tree) from the sklearn
    AgglomerativeClustering.fit output.

    Input:
        children: AgglomerativeClustering.children_
        n_leaves: AgglomerativeClustering.n_leaves_
        X: parameters supplied to AgglomerativeClustering.fit
        leaf_labels: The label of each parameter array in X
        spanner: Callable that computes the dendrite's span

    Output:
        ntree: A str with the Newick tree representation

    """
    return go_down_tree(children,n_leaves,X,leaf_labels,len(children)+n_leaves-1,spanner)[0]+';'

def go_down_tree(children,n_leaves,X,leaf_labels,nodename,spanner):
    """
    go_down_tree(children,n_leaves,X,leaf_labels,nodename,spanner)

    Iterative function that traverses the subtree that descends from
    nodename and returns the Newick representation of the subtree.

    Input:
        children: AgglomerativeClustering.children_
        n_leaves: AgglomerativeClustering.n_leaves_
        X: parameters supplied to AgglomerativeClustering.fit
        leaf_labels: The label of each parameter array in X
        nodename: An int that is the intermediate node name whos
            children are located in children[nodename-n_leaves].
        spanner: Callable that computes the dendrite's span

    Output:
        ntree: A str with the Newick tree representation

    """
    nodeindex = nodename-n_leaves
    if nodename<n_leaves:
        return leaf_labels[nodeindex],np.array([X[nodeindex]])
    else:
        node_children = children[nodeindex]
        branch0,branch0samples = go_down_tree(children,n_leaves,X,leaf_labels,node_children[0],spanner)
        branch1,branch1samples = go_down_tree(children,n_leaves,X,leaf_labels,node_children[1],spanner)
        node = np.vstack((branch0samples,branch1samples))
        branch0span = spanner(branch0samples)
        branch1span = spanner(branch1samples)
        nodespan = spanner(node)
        branch0distance = nodespan-branch0span
        branch1distance = nodespan-branch1span
        nodename = '({branch0}:{branch0distance},{branch1}:{branch1distance})'.format(branch0=branch0,branch0distance=branch0distance,branch1=branch1,branch1distance=branch1distance)
        return nodename,node

def get_cluster_spanner(aggClusterer):
    """
    spanner = get_cluster_spanner(aggClusterer)

    Input:
        aggClusterer: sklearn.cluster.AgglomerativeClustering instance

    Get a callable that computes a given cluster's span. To compute
    a cluster's span, call spanner(cluster)

    The cluster must be a 2D numpy array, where the axis=0 holds
    separate cluster members and the axis=1 holds the different
    variables.

    """
    if aggClusterer.linkage=='ward':
        if aggClusterer.affinity=='euclidean':
            spanner = lambda x:np.sum((x-aggClusterer.pooling_func(x,axis=0))**2)
    elif aggClusterer.linkage=='complete':
        if aggClusterer.affinity=='euclidean':
            spanner = lambda x:np.max(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2))
        elif aggClusterer.affinity=='l1' or aggClusterer.affinity=='manhattan':
            spanner = lambda x:np.max(np.sum(np.abs(x[:,None,:]-x[None,:,:]),axis=2))
        elif aggClusterer.affinity=='l2':
            spanner = lambda x:np.max(np.sqrt(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2)))
        elif aggClusterer.affinity=='cosine':
            spanner = lambda x:np.max(np.sum((x[:,None,:]*x[None,:,:]))/(np.sqrt(np.sum(x[:,None,:]*x[:,None,:],axis=2,keepdims=True))*np.sqrt(np.sum(x[None,:,:]*x[None,:,:],axis=2,keepdims=True))))
        else:
            raise AttributeError('Unknown affinity attribute value {0}.'.format(aggClusterer.affinity))
    elif aggClusterer.linkage=='average':
        if aggClusterer.affinity=='euclidean':
            spanner = lambda x:np.mean(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2))
        elif aggClusterer.affinity=='l1' or aggClusterer.affinity=='manhattan':
            spanner = lambda x:np.mean(np.sum(np.abs(x[:,None,:]-x[None,:,:]),axis=2))
        elif aggClusterer.affinity=='l2':
            spanner = lambda x:np.mean(np.sqrt(np.sum((x[:,None,:]-x[None,:,:])**2,axis=2)))
        elif aggClusterer.affinity=='cosine':
            spanner = lambda x:np.mean(np.sum((x[:,None,:]*x[None,:,:]))/(np.sqrt(np.sum(x[:,None,:]*x[:,None,:],axis=2,keepdims=True))*np.sqrt(np.sum(x[None,:,:]*x[None,:,:],axis=2,keepdims=True))))
        else:
            raise AttributeError('Unknown affinity attribute value {0}.'.format(aggClusterer.affinity))
    else:
        raise AttributeError('Unknown linkage attribute value {0}.'.format(aggClusterer.linkage))
    return spanner

def dendrogram(data, leaf_labels):
    clusterer = AgglomerativeClustering(n_clusters=2,compute_full_tree=True) # You can set compute_full_tree to 'auto', but I left it this way to get the entire tree plotted
    clusterer.fit(data) # X for whatever you want to fit
    spanner = get_cluster_spanner(clusterer)
    if any("LOC" in s for s in leaf_labels):
        leaf_labels[leaf_labels.index('LOC(CSS)')] = 'LOC' 
    newick_tree = build_Newick_tree(clusterer.children_,clusterer.n_leaves_,data,leaf_labels,spanner) # leaf_labels is a list of labels for each entry in X
    tree = ete3.Tree(newick_tree)
    tree.show()

def smell_groups(sitesDir):
    smell_based_smells = loadCodeSmells(sitesDir, smell_based)
    smells = list(list(smell_based_smells .values())[0].values())[0].keys()
    print(smells)
    print('size:', len(smells))
    metric_based_smells = loadCodeSmells(sitesDir, metric_based)
    smells = list(list(metric_based_smells .values())[0].values())[0].keys()
    print(smells)
    print('size:', len(smells))
    property_based_smells = loadCodeSmells(sitesDir, property_based)
    smells = list(list(property_based_smells .values())[0].values())[0].keys()
    print(smells)
    print('size:', len(smells))
    rule_based_smells = loadCodeSmells(sitesDir, rule_based)
    smells = list(list(rule_based_smells .values())[0].values())[0].keys()
    print(smells)
    print('size:', len(smells))
    selector_based_smells = loadCodeSmells(sitesDir, selector_based)
    smells = list(list(selector_based_smells.values())[0].values())[0].keys()
    print(smells)
    print('size:', len(smells))
    file_based_smells = loadCodeSmells(sitesDir, file_based)
    smells = list(list(file_based_smells.values())[0].values())[0].keys()
    print(smells)
    print('size:', len(smells))

    print('smell-based:')
    run_neural_net(smell_based_smells, crossValidation=10)
    print()
    print('metric-based:')
    run_neural_net(metric_based_smells, crossValidation=10)
    print()
    print('property-based:')
    run_neural_net(property_based_smells, crossValidation=10)
    print()
    print('rule-based:')
    run_neural_net(rule_based_smells, crossValidation=10)
    print()
    print('selector-based:')
    run_neural_net(selector_based_smells, crossValidation=10)
    print()
    print('file-based:')
    run_neural_net(file_based_smells, crossValidation=10)


def main():
    sitesDir = checkArgs()
    #smell_groups(sitesDir)
    framesToSmells = loadCodeSmells(sitesDir)
    #print(len(framesToSmells))
    #data = getUnlabeledData(framesToSmells)
    #framesToSmells = getTestSmells()
    #data,labels = getLabeledData(framesToSmells)
    #data = scaleData(data)
    #plot_silhouettes(data_scaled)
    #smells = list(list(framesToSmells.values())[0].values())[0].keys()
    #print(smells)
    #meanTable(framesToSmells)
    #print(outputAveragesLatexTable(framesToSmells))
    #print(outputPercentLatexTable(framesToSmells))
    print(outputMegaAveragesLatexTable(framesToSmells))
    #run_kmeans(framesToSmells, len(framesToSmells.keys()), visualization=True)
    #run_decision_tree(framesToSmells, visualization=False)
    #run_logistic_regression(framesToSmells)
    #averages = calcFrameworkAverages(framesToSmells)
    #vectors,labels = get_framework_vectors(averages)
    #vectors,labels = get_smell_vectors(averages)
    #data = scaleData(vectors)
    #print(len(data))
    #print(labels)
    #dendrogram(data, labels)
    #print('clustering by site')
    #run_kmeans_determine_clusters(data, labels, visualization=True, latex_vector_plot=False, latex_site_plot=False, folderName='plots/plots_site', title='Site Clusters')
    #print('clustering by framework')
    #run_kmeans_determine_clusters(data, labels, visualization=True, latex_plot=True, folderName='plots/plots_frame', title='Framework Clusters')
    #print('clustering by smells')
    #run_kmeans_determine_clusters(data, labels, visualization=True, latex_plot=True, folderName='plots/plots_smell', title='Smell Clusters')
    #run_neural_net(framesToSmells, precision_recall=True, crossValidation=10)
    #classifier_comparison(framesToSmells, crossValidation=10)    
main()

#sitesDir = checkArgs()
#printSmellCounts(sitesDir)
