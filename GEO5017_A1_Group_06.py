"""
This demo shows how to visualize the designed features. Currently, only 2D feature space visualization is supported.
I use the same data for A2 as my input.
Each .xyz file is initialized as one urban object, from where a feature vector is computed.
6 features are defined to describe an urban object.
Required libraries: numpy, scipy, scikit learn, matplotlib, tqdm 
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from sklearn.neighbors import KDTree
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import ConvexHull
from sklearn.feature_selection import SequentialFeatureSelector
from tqdm import tqdm
from os.path import exists, join
from os import listdir
from itertools import combinations


class urban_object:
    """
    Define an urban object
    """
    def __init__(self, filenm):
        """
        Initialize the object
        """
        # obtain the cloud name
        self.cloud_name = filenm.split('/\\')[-1][-7:-4]

        # obtain the cloud ID
        self.cloud_ID = int(self.cloud_name)

        # obtain the label
        self.label = math.floor(1.0*self.cloud_ID/100)

        # obtain the points
        self.points = read_xyz(filenm)

        # initialize the feature vector
        self.feature = []

    def compute_features(self):
        """
        Compute the features, here we provide two example features. You're encouraged to design your own features
        """
        # calculate the height
        height = np.amax(self.points[:, 2])
        self.feature.append(height)

        # get the root point and top point
        root = self.points[[np.argmin(self.points[:, 2])]]
        top = self.points[[np.argmax(self.points[:, 2])]]

        # construct the 2D and 3D kd tree
        kd_tree_2d = KDTree(self.points[:, :2], leaf_size=5)
        kd_tree_3d = KDTree(self.points, leaf_size=5)

        # compute the root point planar density
        radius_root = 0.2
        count = kd_tree_2d.query_radius(root[:, :2], r=radius_root, count_only=True)
        root_density = 1.0*count[0] / len(self.points)
        self.feature.append(root_density)

        # compute the 2D footprint and calculate its area
        hull_2d = ConvexHull(self.points[:, :2])
        hull_area = hull_2d.volume
        self.feature.append(hull_area)

        # get the hull shape index
        hull_perimeter = hull_2d.area
        shape_index = 1.0 * hull_area / hull_perimeter
        self.feature.append(shape_index)

        # obtain the point cluster near the top area
        k_top = max(int(len(self.points) * 0.005), 100)
        idx = kd_tree_3d.query(top, k=k_top, return_distance=False)
        idx = np.squeeze(idx, axis=0)
        neighbours = self.points[idx, :]

        # obtain the covariance matrix of the top points
        cov = np.cov(neighbours.T)
        w, _ = np.linalg.eig(cov)
        w.sort()

        # calculate the linearity and sphericity
        linearity = (w[2]-w[1]) / (w[2] + 1e-5)
        sphericity = w[0] / (w[2] + 1e-5)
        self.feature += [linearity, sphericity]

        #verticality
        e_z = np.array([0, 0, 1])
        evals, evecs = np.linalg.eig(np.cov(self.points.T))
        normal = evecs[:, np.argmin(evals)]
        cos_a = np.clip(np.dot(normal, e_z), -1.0, 1.0)
        self.feature.append(abs(math.pi / 2 - np.arccos(cos_a)))

        #desnity
        hull_3d = ConvexHull(self.points)
        point_count = len(self.points)
        density = point_count / hull_3d.volume
        self.feature.append(density)

        #omnivariacne
        omnivariance = np.cbrt(np.prod(evals))
        self.feature.append(omnivariance)

        # local planarity ratio (ratio of points fitting a local plane)
        bbox_diag = np.linalg.norm(self.points.max(axis=0) - self.points.min(axis=0))
        radius = bbox_diag * 0.1

        sample = self.points[::5]
        planarity_scores = []

        for p in sample:
            idx = kd_tree_3d.query_radius([p], r=radius)[0]
            if len(idx) < 4:
                continue
            neighbors = self.points[idx]
            cov_local = np.cov(neighbors.T)
            evals_local, evecs_local = np.linalg.eig(cov_local)
            evals_local = np.sort(np.abs(evals_local))
            normal_local = evecs_local[:, np.argmin(np.abs(evals_local))]
            dists = np.abs((neighbors - neighbors.mean(axis=0)) @ normal_local)
            soft_score = (evals_local[1] - evals_local[0]) / (evals_local[2] + 1e-5)
            neighbor_fit = (dists < np.std(dists)).mean()
            planarity_scores.append(soft_score * neighbor_fit)
        local_planarity = np.mean(planarity_scores) if planarity_scores else 0.0
        self.feature.append(local_planarity)

        # volume occupancy (point cloud fill ratio)
        hull_3d = ConvexHull(self.points)
        bbox_volume = np.prod(self.points.max(axis=0) - self.points.min(axis=0))
        volume_occupancy = hull_3d.volume / bbox_volume
        self.feature.append(volume_occupancy)

        # vertical density ratio (top half vs bottom half)
        z_mid = (self.points[:, 2].max() + self.points[:, 2].min()) / 2
        top_count = np.sum(self.points[:, 2] > z_mid)
        bottom_count = np.sum(self.points[:, 2] <= z_mid)
        vertical_density_ratio = top_count / (bottom_count + 1e-5)
        self.feature.append(vertical_density_ratio)

def read_xyz(filenm):
    """
    Reading points
        filenm: the file name
    """
    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points


def feature_preparation(data_path):
    """
    Prepare features of the input point cloud objects
        data_path: the path to read data
    """
    # check if the current data file exist
    data_file = 'data.txt'
    if exists(data_file):
        return

    # obtain the files in the folder
    files = sorted(listdir(data_path))

    # initialize the data
    input_data = []

    # retrieve each data object and obtain the feature vector
    for file_i in tqdm(files, total=len(files)):
        # obtain the file name
        file_name = join(data_path, file_i)

        # read data
        i_object = urban_object(filenm=file_name)

        # calculate features
        i_object.compute_features()

        # add the data to the list
        i_data = [i_object.cloud_ID, i_object.label] + i_object.feature
        input_data += [i_data]

    # transform the output data
    outputs = np.array(input_data).astype(np.float32)

    # write the output to a local file
    data_header = 'ID,label,height,root_density,area,shape_index,linearity,sphericity,verticality,density,omnivariance,local_planarity,volume_occupancy,vertical_density_ratio'
    np.savetxt(data_file, outputs, fmt='%10.5f', delimiter=',', newline='\n', header=data_header)


def data_loading(data_file='data.txt'):
    """
    Read the data with features from the data file
        data_file: the local file to read data with features and labels
    """
    # load data
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=',', comments='#')

    # extract object ID, feature X and label Y
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)

    return ID, X, y


def feature_visualization(X):
    """
    Visualize the features
        X: input features. This assumes classes are stored in a sequential manner
    """
    # initialize a plot
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("feature subset visualization of 5 classes", fontsize="small")

    # define the labels and corresponding colors
    colors = ['firebrick', 'grey', 'darkorange', 'dodgerblue', 'olivedrab']
    labels = ['building', 'car', 'fence', 'pole', 'tree']

    # plot the data with first two features (trying with 2 specific features now)
    for i in range(5):
        ax.scatter(X[100*i:100*(i+1), 0], X[100*i:100*(i+1), 8], marker="o", c=colors[i], edgecolor="k", label=labels[i])

    # show the figure with labels
    """
    Replace the axis labels with your own feature names
    """
    ax.set_xlabel('height')
    ax.set_ylabel('omnivariance')
    ax.legend()
    plt.show()


def SVM_classification(X, y, kernel='linear'):
    """
    Conduct SVM classification
        X: features
        y: labels
        kernel: kernel function
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    #Scaling
    scale = StandardScaler()
    X_train_scaled = scale.fit_transform(X_train)
    X_test_scaled = scale.transform(X_test)

    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train_scaled, y_train)
    y_preds = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_preds)
    print("SVM accuracy: %5.2f" % acc)
    print("confusion matrix")
    conf = confusion_matrix(y_test, y_preds)
    print(conf)


def RF_classification(X, y):
    """
    Conduct RF classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

    #Scaling
    scale = StandardScaler()
    X_train_scaled = scale.fit_transform(X_train)
    X_test_scaled = scale.transform(X_test)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_preds = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_preds)
    print("RF accuracy: %5.2f" % acc)
    print("confusion matrix")
    conf = confusion_matrix(y_test, y_preds)
    print(conf)

def sequential_feature_selection(X, y, names, clf, direction='backward', max_features=4):
    for n in range(1, max_features + 1):
        sfs = SequentialFeatureSelector(clf, n_features_to_select=n, direction=direction)
        sfs.fit(X, y)
        selected = [name for name, s in zip(names, sfs.get_support()) if s]
        print(f"  {n} features: {selected}")

if __name__=='__main__':
    # specify the data folder
    """"Here you need to specify your own path"""
    path = 'Data/pointclouds'

    # conduct feature preparation
    print('Start preparing features')
    feature_preparation(data_path=path)

    # load the data
    print('Start loading data from the local file')
    ID, X, y = data_loading()

    # visualize features
    print('Visualize the features')
    feature_visualization(X=X)

    # SVM classification
    # print('Start SVM classification')
    # SVM_classification(X, y)

    ### Figuring out the best combinations of features for each classifier:

    names = ['height', 'root_density', 'area', 'shape_index', 'linearity',
             'sphericity', 'verticality', 'density', 'omnivariance', 'local_planarity', 'volume_occupancy', 'vertical_density_ratio']

    classifiers = [
        svm.SVC(kernel='linear'),
        svm.SVC(kernel='rbf'),
        svm.SVC(kernel='poly'),
        RandomForestClassifier(n_estimators=100, random_state=42)
    ]
    # for clf in tqdm(classifiers):
    #     print(f'\nSequential backward selection — {clf.__class__.__name__} kernel={getattr(clf, "kernel", "N/A")}')
    #     sequential_feature_selection(X, y, names, clf, direction='backward', max_features=6)

    # Sequential feature selection performed worse than manual -> brute force search
    feature_preparation(data_path=path)
    ID, X, y = data_loading()
    feature_visualization(X=X)

    names = ['height', 'root_density', 'area', 'shape_index', 'linearity',
             'sphericity', 'verticality', 'density', 'omnivariance',
             'local_planarity', 'volume_occupancy', 'vertical_density_ratio']

    classifiers = [
        ('SVM linear', svm.SVC(kernel='linear')),
        ('SVM rbf',    svm.SVC(kernel='rbf')),
        ('SVM poly',   svm.SVC(kernel='poly')),
        ('RF',         RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    # shared fixed split for feature selection (same across all classifiers)
    X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X, y, test_size=0.4, random_state=1)
    sc_fs   = StandardScaler()
    X_tr_fs = sc_fs.fit_transform(X_train_fs)
    X_te_fs = sc_fs.transform(X_test_fs)

    n_runs = 200
    for label, clf in classifiers:
        # brute-force: find best 4 features
        best_score, best_idx = 0, None
        for combo in combinations(range(len(names)), 4):
            combo = list(combo)
            clf.fit(X_tr_fs[:, combo], y_train_fs)
            score = accuracy_score(y_test_fs, clf.predict(X_te_fs[:, combo]))
            if score > best_score:
                best_score, best_idx = score, combo
        print(f'\n{label} — best features: {[names[i] for i in best_idx]}  (OA={best_score:.4f})')

        # averaged runs with best features
        X_best = X[:, best_idx]
        accs = []
        for seed in range(n_runs):
            X_tr, X_te, y_tr, y_te = train_test_split(X_best, y, test_size=0.4, random_state=seed)
            sc = StandardScaler()
            clf.fit(sc.fit_transform(X_tr), y_tr)
            accs.append(accuracy_score(y_te, clf.predict(sc.transform(X_te))))
        print(f'  mean OA ({n_runs} runs): {np.mean(accs):.4f} ± {np.std(accs):.4f}')

    # RF classification
    # print('Start RF classification')
    # RF_classification(X, y)

    # best4 = {
    #     'linear': ['height', 'verticality', 'density', 'local_planarity'],
    #     'rbf':    ['height', 'density', 'omnivariance', 'vertical_density_ratio'],
    #     'poly':   ['height', 'density', 'omnivariance', 'vertical_density_ratio'],
    # }
    #
    # for kernel, features in best4.items():
    #     idx = [names.index(n) for n in features]
    #     X_best4 = X[:, idx]
    #     print(f'\nSVM {kernel} with best 4 features {features}')
    #     SVM_classification(X_best4, y, kernel=kernel)
    #
    # additional_linear = [
    #     ['height', 'area', 'verticality', 'density'],
    #     ['height', 'area', 'verticality', 'omnivariance'],
    #     ['height', 'verticality', 'density', 'omnivariance'],
    #     ['height', 'shape_index', 'verticality', 'density'],
    #     ['height', 'verticality', 'omnivariance', 'local_planarity'],
    #     ['height', 'density', 'omnivariance', 'vertical_density_ratio'],
    #     ['height', 'verticality', 'density', 'vertical_density_ratio'],
    #     ['height', 'area', 'density', 'omnivariance'],
    # ]
    #
    # for features in additional_linear:
    #     idx = [names.index(n) for n in features]
    #     X_combo = X[:, idx]
    #     print(f'\nSVM linear with {features}')
    #     SVM_classification(X_combo, y, kernel='linear')

    # best4_rf = ['height', 'density', 'omnivariance', 'local_planarity']
    # idx = [names.index(n) for n in best4_rf]
    # X_best4_rf = X[:, idx]
    # print(f'\nRF with best 4 features {best4_rf}')
    # RF_classification(X_best4_rf, y)

    #AVERAGED RUNS FOR BEST FEATURES
    # configs = [
    #     ('RF',  ['height', 'density', 'omnivariance', 'local_planarity'],         'rf'),
    #     ('SVM', ['height', 'verticality', 'omnivariance', 'local_planarity'],     'linear'),
    #     ('SVM', ['height', 'shape_index', 'verticality', 'density'],              'linear'),
    # ]
    #
    # n_runs = 200
    # for clf_type, features, kernel in configs:
    #     idx = [names.index(n) for n in features]
    #     X_sub = X[:, idx]
    #     accs = []
    #     for seed in range(n_runs):
    #         X_train, X_test, y_train, y_test = train_test_split(X_sub, y, test_size=0.4, random_state=seed)
    #         scale = StandardScaler()
    #         X_train_scaled = scale.fit_transform(X_train)
    #         X_test_scaled = scale.transform(X_test)
    #         if clf_type == 'RF':
    #             clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    #         else:
    #             clf = svm.SVC(kernel=kernel)
    #         clf.fit(X_train_scaled, y_train)
    #         accs.append(accuracy_score(y_test, clf.predict(X_test_scaled)))
    #     print(f'{clf_type} {kernel} {features}')
    #     print(f'  mean accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}')

    ###////////////////////!!!!!!!!!!!!!!!!//////////////////###
    ### NOTE TO US:
    ### for testing with different parameters always delete data.txt first
    ### when adding features do these things with the new feature name:
    ### compute_features() — append the new value
    ### data_header in feature_preparation — add the name to the string
    ### names list in main — add the name in the same order
