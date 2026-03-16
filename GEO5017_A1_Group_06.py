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


def SVM_classification_hyperparamters(X, y, kernel='linear'):
    test_sizes = [0.4]
    C_values   = [0.01, 0.1, 1, 10, 100]

    best_acc, best_params = 0, {}
    for test_size in test_sizes:
        for C in C_values:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=1)
            scale = StandardScaler()
            clf = svm.SVC(kernel=kernel, C=C)
            clf.fit(scale.fit_transform(X_train), y_train)
            acc = accuracy_score(y_test, clf.predict(scale.transform(X_test)))
            if acc > best_acc:
                best_acc, best_params = acc, {'test_size': test_size, 'C': C}

    print(f"SVM ({kernel}) best: {best_params}  OA={best_acc:.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=best_params['test_size'], random_state=1)
    scale = StandardScaler()
    clf = svm.SVC(kernel=kernel, C=best_params['C'])
    clf.fit(scale.fit_transform(X_train), y_train)
    print(confusion_matrix(y_test, clf.predict(scale.transform(X_test))))

def SVM_classification(X, y):
    """
    Conduct SVM classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf = svm.SVC(kernel='linear', C=0.1)
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)
    acc = accuracy_score(y_test, y_preds)
    print("SVM accuracy: %5.2f" % acc)
    print("confusion matrix")
    conf = confusion_matrix(y_test, y_preds)
    print(conf)

def RF_classification_hyperparameters(X, y):
    param_grid = {
        'n_estimators':      [100, 200, 500, 800],
        'max_features':      [1, 2, 3, 4, None],
        'max_depth':         [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],}
    best_acc, best_params = 0, {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    scale = StandardScaler()
    X_tr = scale.fit_transform(X_train)
    X_te = scale.transform(X_test)

    for n_est in param_grid['n_estimators']:
        for max_feat in param_grid['max_features']:
            for max_dep in param_grid['max_depth']:
                for min_split in param_grid['min_samples_split']:
                    for min_leaf in param_grid['min_samples_leaf']:
                        clf = RandomForestClassifier(
                            n_estimators=n_est, max_features=max_feat,
                            max_depth=max_dep, min_samples_split=min_split,
                            min_samples_leaf=min_leaf, random_state=42)
                        clf.fit(X_tr, y_train)
                        acc = accuracy_score(y_test, clf.predict(X_te))
                        if acc > best_acc:
                            best_acc = acc
                            best_params = {
                                'n_estimators': n_est, 'max_features': max_feat,
                                'max_depth': max_dep, 'min_samples_split': min_split,
                                'min_samples_leaf': min_leaf}

    print(f"RF best: {best_params}  OA={best_acc:.4f}")
    clf = RandomForestClassifier(**best_params, random_state=42)
    clf.fit(X_tr, y_train)
    print(confusion_matrix(y_test, clf.predict(X_te)))

def RF_classification(X, y):
    # best hyperparameters found from prior grid search

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    #Scaling
    scale = StandardScaler()
    X_train_scaled = scale.fit_transform(X_train)
    X_test_scaled = scale.transform(X_test)
    clf = RandomForestClassifier(n_estimators=500, max_depth=5,
                                 min_samples_split=2, max_features=1,
                                 min_samples_leaf=4, random_state=1
                                 )
    clf.fit(X_train_scaled, y_train)
    y_preds = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_preds)
    print("RF accuracy: %5.2f" % acc)
    print("confusion matrix")
    conf = confusion_matrix(y_test, y_preds)
    print(conf)

def learning_curve(clf, X, y):
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    train_errors, test_errors = [], []

    for train_size in train_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=1)
        scale = StandardScaler()
        X_tr = scale.fit_transform(X_train)
        X_te = scale.transform(X_test)

        clf.fit(X_tr, y_train)
        train_errors.append(1 - accuracy_score(y_train, clf.predict(X_tr)))
        test_errors.append(1 - accuracy_score(y_test, clf.predict(X_te)))

    # plot
    n_train_samples = [int(s * len(y)) for s in train_sizes]
    plt.plot(n_train_samples, train_errors, label='Training error')
    plt.plot(n_train_samples, test_errors,  label='Test error')
    plt.xlabel('Number of training samples')
    plt.ylabel('Error rate')
    plt.title('Learning curve')
    plt.legend()
    plt.show()

def compute_j_scores(X, y, names):
    classes = np.unique(y)
    N = len(y)
    j_scores = []

    for f in range(X.shape[1]):
        x = X[:, f]
        mu_all = np.mean(x)

        sw, sb = 0.0, 0.0
        for c in classes:
            x_c = x[y == c]
            Nk = len(x_c)
            mu_k = np.mean(x_c)
            sw += (Nk / N) * np.var(x_c)
            sb += (Nk / N) * (mu_k - mu_all) ** 2

        j_scores.append(sb / (sw + 1e-10))

    # print ranked table
    ranked = sorted(zip(names, j_scores), key=lambda x: x[1], reverse=True)
    print(f'\n{"Feature":<25} {"J-score":>8}')
    print('-' * 35)
    for name, j in ranked:
        print(f'{name:<25} {j:>8.4f}')

    return ranked

def sequential_feature_selection(X, y, names, clf, direction='backward', max_features=4):
    for n in range(1, max_features + 1):
        sfs = SequentialFeatureSelector(clf, n_features_to_select=n, direction=direction)
        sfs.fit(X, y)
        selected = [name for name, s in zip(names, sfs.get_support()) if s]
        print(f"  {n} features: {selected}")

def error_analysis(clf, X, y, clf_name, test_size=0.5):
    class_names = ['building', 'car', 'fence', 'pole', 'tree']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=1)

    scale = StandardScaler()
    X_train_s = scale.fit_transform(X_train)
    X_test_s  = scale.transform(X_test)

    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    cm = confusion_matrix(y_test, y_pred)

    #Overall Accuracy
    OA = accuracy_score(y_test, y_pred)

    #Mean Per-Class Accuracy
    per_class_acc = []
    for i in range(len(class_names)):
        acc_i = cm[i, i] / cm[i, :].sum()
        per_class_acc.append(acc_i)
    mA = np.mean(per_class_acc)

    #Per-class Precision, Recall, F1
    precisions, recalls, f1s = [], [], []
    for i in range(len(class_names)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP   # others predicted as class i
        FN = cm[i, :].sum() - TP   # class i predicted as others

        p  = TP / (TP + FP + 1e-10)
        r  = TP / (TP + FN + 1e-10)
        f1 = 2 * p * r / (p + r + 1e-10)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    macro_precision = np.mean(precisions)
    macro_recall    = np.mean(recalls)
    macro_f1        = np.mean(f1s)



    print(f'  Error Analysis: {clf_name}')


    print(f'\nConfusion Matrix:')
    print(f'{"":12}', end='')
    for name in class_names:
        print(f'{name:>10}', end='')
    print()
    for i, name in enumerate(class_names):
        print(f'{name:<12}', end='')
        for j in range(len(class_names)):
            print(f'{cm[i,j]:>10}', end='')
        print()

    print(f'\n{"Class":<12} {"Precision":>10} {"Recall":>8} {"F1":>8} {"Per-class Acc":>15}')
    print('-'*55)
    for i, cls in enumerate(class_names):
        print(f'{cls:<12} {precisions[i]:>10.3f} {recalls[i]:>8.3f} '
              f'{f1s[i]:>8.3f} {per_class_acc[i]:>15.3f}')

    print('-'*55)
    print(f'{"Macro-avg":<12} {macro_precision:>10.3f} {macro_recall:>8.3f} {macro_f1:>8.3f} {mA:>15.3f}')
    print(f'\nOverall Accuracy (OA):      {OA:.4f}')
    print(f'Mean per-class Accuracy (mA): {mA:.4f}')
    print(f'Macro-avg Precision:         {macro_precision:.4f}')
    print(f'Macro-avg Recall:            {macro_recall:.4f}')
    print(f'Macro-avg F1:                {macro_f1:.4f}')




    #Confusion matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(f'Confusion Matrix — {clf_name}')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{clf_name}.png', dpi=150)
    plt.show()

    return OA, mA, macro_f1, cm


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
    # print('Visualize the features')
    # feature_visualization(X=X)

    names = ['height', 'root_density', 'area', 'shape_index', 'linearity',
             'sphericity', 'verticality', 'density', 'omnivariance',
             'local_planarity', 'volume_occupancy', 'vertical_density_ratio']

    best_features = ['height', 'density', 'omnivariance', 'local_planarity']
    idx = [names.index(n) for n in best_features]
    X_best = X[:, idx]

    #J-scores
    compute_j_scores(X, y, names)

    # learning curves
    learning_curve(svm.SVC(kernel='linear', C=0.1), X_best, y)
    learning_curve(RandomForestClassifier(n_estimators=500, max_depth=5,
                                 min_samples_split=2, max_features=1,
                                 min_samples_leaf=4, random_state=1
                                 ), X_best, y)

    # SVM classification
    print('Start SVM classification')
    SVM_classification(X, y)

    # RF classification
    print('Start RF classification')
    RF_classification(X, y)

    # SVM classification with hyperparameter grid search
    # print('Start SVM classifier grid search')
    # SVM_classification_hyperparamters(X, y)

    # RF classification with hyperparamter grid search
    # print('Start RF classifier grid search')
    # RF_classification_hyperparameters(X, y)

    #AVERAGED RUNS FOR BEST FEATURES
    # SVM error analysis

    svm_features = ['height', 'shape_index', 'density', 'local_planarity']
    svm_idx = [names.index(n) for n in svm_features]
    X_svm = X[:, svm_idx]
    OA_svm, mA_svm, f1_svm, cm_svm = error_analysis(
        svm.SVC(kernel='linear', C=0.1), X_svm, y, 'SVM', test_size=0.5)

    # RF error analysis
    rf_features = ['height', 'density', 'omnivariance', 'local_planarity']
    rf_idx = [names.index(n) for n in rf_features]
    X_rf = X[:, rf_idx]
    OA_rf, mA_rf, f1_rf, cm_rf = error_analysis(
        RandomForestClassifier(n_estimators=500, max_depth=5,
                               min_samples_split=2, max_features=1,
                               min_samples_leaf=4, random_state=1),
        X_rf, y, 'RF', test_size=0.4)

    #Final comparison table
    print(f'  FINAL COMPARISON')

    print(f'{"Metric":<30} {"SVM":>6} {"RF":>6}')
    print('-' * 45)
    print(f'{"Overall Accuracy (OA)":<30} {OA_svm:>6.3f} {OA_rf:>6.3f}')
    print(f'{"Mean per-class Accuracy (mA)":<30} {mA_svm:>6.3f} {mA_rf:>6.3f}')
    print(f'{"Macro-avg F1":<30} {f1_svm:>6.3f} {f1_rf:>6.3f}')


    ###////////////////////!!!!!!!!!!!!!!!!//////////////////###
    ### NOTE TO US:
    ### for testing with different parameters always delete data.txt first
    ### when adding features do these things with the new feature name:
    ### compute_features() — append the new value
    ### data_header in feature_preparation — add the name to the string
    ### names list in main — add the name in the same order