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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
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

def SVM_classification_gridsearch(X, y):
    # Split once: keep test set untouched
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1, stratify=y
    )

    # Pipeline: scaling + model
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', svm.SVC())
    ])

    # Hyperparameter grid
    param_grid = {
        'svc__C': [0.01, 0.1, 1, 10, 100],
        'svc__kernel': ['linear', 'rbf', 'poly'],
        'svc__gamma': [0.001, 0.01, 0.1]
    }

    # Cross-validation setup
    # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    # Grid search
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring='accuracy',
        n_jobs=-1
    )

    # Fit on training data only
    grid.fit(X_train, y_train)

    # Best result
    print(f"SVM, best: {grid.best_params_}  OA={grid.best_score_:.4f}")

    # Final evaluation on test set
    y_pred = grid.predict(X_test)
    print("confusion matrix")
    print(confusion_matrix(y_test, y_pred))

def SVM_classification(X, y):
    """
    Conduct SVM classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    #Scaling
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)
    X_test = scale.transform(X_test)
    clf = svm.SVC(kernel='linear', C=0.1)
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)
    acc = accuracy_score(y_test, y_preds)
    print("SVM accuracy: %5.2f" % acc)
    print("confusion matrix")
    conf = confusion_matrix(y_test, y_preds)
    print(conf)

def SVM_cv_scores(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1, stratify=y
    )
    # test set remains untouched
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', svm.SVC(kernel='linear', C=0.1))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')

    print("CV scores:", scores)
    print("Mean CV accuracy: %5.4f" % np.mean(scores))
    print("Std CV accuracy:  %5.4f" % np.std(scores))

def RF_classification_gridsearch(X, y):
    # Split once: keep test set untouched
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1, stratify=y
    )

    # Pipeline: scaling + model
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier())
    ])

    # Hyperparameter grid
    param_grid = {
        'rf__n_estimators': [100, 200, 500, 800],
        'rf__max_features': [1, 2, 3, 4, None],
        'rf__max_depth': [None, 5, 10, 20],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    # Grid search
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring='accuracy',
        n_jobs=-1,
        cv = cv
    )

    # Fit on training data only
    grid.fit(X_train, y_train)

    # Best result
    print(f"best features: {grid.best_params_}  best score={grid.best_score_:.4f}")

    # Final evaluation on test set
    y_pred = grid.predict(X_test)
    print("confusion matrix")
    print(confusion_matrix(y_test, y_pred))


def RF_cv_scores(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1, stratify=y
    )
    # test set remains untouched
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(
            max_depth = None,
            max_features = 1,
            min_samples_leaf = 1,
            min_samples_split = 2,
            n_estimators = 200,
            random_state=1))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')

    print("CV scores:", scores)
    print("Mean CV accuracy: %5.4f" % np.mean(scores))
    print("Std CV accuracy:  %5.4f" % np.std(scores))

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


def plot_learning_curve(clf, X, y):
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

    n_train_samples = [int(s * len(y)) for s in train_sizes]
    print(f'\n{"n_train":<10} {"train_err":>10} {"test_err":>10} ')
    print('-' * 32)
    for n, tr, te in zip(n_train_samples, train_errors, test_errors):
        print(f'{n:<10} {tr:>10.4f} {te:>10.4f} {te-tr:>10.4f}')
    # plot

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

# NOT USED
def sequential_feature_selection(X, y, names, clf, direction, n_features=4):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1, stratify=y
    )
    sfs = SequentialFeatureSelector(clf, n_features_to_select=n_features, direction=direction)
    sfs.fit(X_train, y_train)
    selected = [name for name, s in zip(names, sfs.get_support()) if s]

    # evaluate accuracy with selected features
    idx = [names.index(n) for n in selected]
    scale = StandardScaler()
    X_train_sel = scale.fit_transform(X_train[:, idx])
    X_test_sel = scale.transform(X_test[:, idx])
    clf.fit(X_train_sel, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_sel))

    print(f"  Selected: {selected}  |  OA = {acc:.4f}")
    return selected, acc

# NOT USED
def backward_elimination(X, y, names, clf):
    from sklearn.model_selection import cross_val_score
    remaining = list(names)
    print(f'  {len(remaining)} features: {remaining}')
    while len(remaining) > 1:
        scores = []
        for name in remaining:
            candidate = [n for n in remaining if n != name]
            idx = [names.index(n) for n in candidate]
            score = cross_val_score(clf, X[:, idx], y, cv=5).mean()
            scores.append((score, name))
        scores.sort(reverse=True)
        worst = scores[-1][1]
        remaining.remove(worst)
        print(f'  {len(remaining)} features: {remaining}  (removed: {worst})')

# NOT USED
def forward_selection(X, y, names, clf, max_features=6):
    from sklearn.model_selection import cross_val_score
    remaining = list(names)
    selected = []
    while len(selected) < max_features:
        scores = []
        for name in remaining:
            candidate = selected + [name]
            idx = [names.index(n) for n in candidate]
            score = cross_val_score(clf, X[:, idx], y, cv=5).mean()
            scores.append((score, name))
        scores.sort(reverse=True)
        best = scores[0][1]
        selected.append(best)
        remaining.remove(best)
        print(f'  {len(selected)} features: {selected}')

def plot_learning_curve_manual(clf_name, clf, X, y):
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    train_accs = []
    test_accs = []
    train_sizes = []

    for ratio in ratios:
        test_ratio = 1 - ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=1, stratify=y
        )
        scale = StandardScaler()
        X_train_scaled = scale.fit_transform(X_train)
        X_test_scaled = scale.transform(X_test)

        clf.fit(X_train_scaled, y_train)
        train_accs.append(1 - accuracy_score(y_train, clf.predict(X_train_scaled)))
        test_accs.append(1 - accuracy_score(y_test, clf.predict(X_test_scaled)))
        train_sizes.append(len(X_train))

    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_accs, '-', label='Training error')
    plt.plot(train_sizes, test_accs, '-', label='Test error')
    plt.xlabel('Number of training samples')
    plt.ylabel('Error rate')
    plt.title('Learning curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'learning_curve_{clf_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

def error_analysis(clf_name, clf, X, y):
    """Classification results with confusion matrix and per-class metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1, stratify=y
    )
    scale = StandardScaler()
    X_train_scaled = scale.fit_transform(X_train)
    X_test_scaled = scale.transform(X_test)

    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    labels = ['building', 'car', 'fence', 'pole', 'tree']
    cm = confusion_matrix(y_test, y_pred)

    # print results
    print(f"\n=== {clf_name} Error Analysis ===")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)

    # per-class accuracy
    print(f"\n{'Class':<12} {'Correct':<10} {'Total':<8} {'Accuracy':<10}")
    print('-' * 40)
    for i, label in enumerate(labels):
        correct = cm[i, i]
        total = cm[i, :].sum()
        print(f"{label:<12} {correct:<10} {total:<8} {correct/total:<10.4f}")

    # plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {clf_name}')

    # add numbers in cells
    for i in range(5):
        for j in range(5):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color)

    plt.colorbar(im)
    plt.savefig(f'confusion_matrix_{clf_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

def tune_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1, stratify=y
    )
    scale = StandardScaler()
    X_train_scaled = scale.fit_transform(X_train)
    X_test_scaled = scale.transform(X_test)

    print(f"\n{'Kernel':<10} {'C':<8} {'Gamma':<10} {'Train OA':<10} {'Test OA':<10}")
    print('-' * 48)

    best_acc, best_params = 0, {}
    for kernel in ['linear', 'rbf', 'poly']:
        for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
            gammas = [0.001, 0.01, 0.1, 1.0] if kernel != 'linear' else ['scale']
            for gamma in gammas:
                clf = svm.SVC(kernel=kernel, C=C, gamma=gamma)
                clf.fit(X_train_scaled, y_train)
                train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
                test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
                print(f"{kernel:<10} {C:<8} {str(gamma):<10} {train_acc:<10.4f} {test_acc:<10.4f}")

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_params = {'kernel': kernel, 'C': C, 'gamma': gamma}

    print(f"\nBest SVM: {best_params}  |  OA = {best_acc:.4f}")
    return best_params


def tune_rf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1, stratify=y
    )
    scale = StandardScaler()
    X_train_scaled = scale.fit_transform(X_train)
    X_test_scaled = scale.transform(X_test)

    print(f"\n{'n_est':<8} {'depth':<8} {'min_split':<12} {'min_leaf':<10} {'max_feat':<10} {'Train OA':<10} {'Test OA':<10}")
    print('-' * 68)

    best_acc, best_params = 0, {}
    for n_est in [100, 200, 500]:
        for max_depth in [5, 10, 20, None]:
            for min_split in [2, 5, 10]:
                for min_leaf in [1, 2, 4]:
                    for max_feat in [1, 2, 3, None]:
                        clf = RandomForestClassifier(
                            n_estimators=n_est, max_depth=max_depth,
                            min_samples_split=min_split, min_samples_leaf=min_leaf,
                            max_features=max_feat, random_state=1
                        )
                        clf.fit(X_train_scaled, y_train)
                        train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
                        test_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
                        print(f"{n_est:<8} {str(max_depth):<8} {min_split:<12} {min_leaf:<10} {str(max_feat):<10} {train_acc:<10.4f} {test_acc:<10.4f}")

                        if test_acc > best_acc:
                            best_acc = test_acc
                            best_params = {
                                'n_estimators': n_est, 'max_depth': max_depth,
                                'min_samples_split': min_split, 'min_samples_leaf': min_leaf,
                                'max_features': max_feat
                            }

    print(f"\nBest RF: {best_params}  |  OA = {best_acc:.4f}")
    return best_params


if __name__=='__main__':

    """"Here you need to specify your own path"""
    path = 'Data/pointclouds'

    # conduct feature preparation
    print('Start preparing features')
    feature_preparation(data_path=path)

    # load the data
    print('Start loading data from the local file')
    ID, X, y = data_loading()

    # All the features we defined
    names = ['height', 'root_density', 'area', 'shape_index', 'linearity',
             'sphericity', 'verticality', 'density', 'omnivariance',
             'local_planarity', 'volume_occupancy', 'vertical_density_ratio']

    # J-scores
    compute_j_scores(X, y, names)


    run_featureselector = True
    run_tuning = False
    run_learning_curve = False

    if run_featureselector:

        print("=== SVM Feature Selection ===")
        for kernel in ['linear', 'rbf', 'poly']:
            for C in [0.1, 1.0, 10.0]:
                if kernel == 'rbf':
                    clf = svm.SVC(kernel=kernel, C=C, gamma=0.1)
                elif kernel == 'poly':
                    clf = svm.SVC(kernel=kernel, C=C, gamma='scale', degree=2)
                else:
                    clf = svm.SVC(kernel=kernel, C=C)

                print(f"\nKernel={kernel}, C={C}")
                svm_features = sequential_feature_selection(X, y, names, clf, direction='forward', n_features=4)

        print("\n=== RF Feature Selection ===")
        for n_est in [100, 200]:
            for max_depth in [5, 10, 15]:
                print(f"\nn_estimators={n_est}, max_depth={max_depth}")
                clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=1)
                rf_features = sequential_feature_selection(X, y, names, clf, direction='forward', n_features=4)

    # we now choose our 4 final features. These are the following:
    best_features = ['height', 'shape_index', 'density', 'local_planarity']
    #best_features = ['height', 'root_density', 'density', 'local_planarity']

    # selecting only the 4 best features from our data.txt
    idx = [names.index(n) for n in best_features]
    X_best = X[:, idx]

    if run_tuning:
        print("\n=== SVM Hyperparameter Tuning ===")
        best_svm_params = tune_svm(X_best, y)

        print("\n=== RF Hyperparameter Tuning ===")
        best_rf_params = tune_rf(X_best, y)

    if run_learning_curve:
        # Learning curves
        print("\n=== Learning Curves ===")
        plot_learning_curve_manual('SVM', svm.SVC(kernel='rbf', C=1.0, gamma=1.0), X_best, y)
        plot_learning_curve_manual('RF', RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=2,
            min_samples_leaf=1, max_features=1, random_state=1), X_best, y)

        # Error analysis
        print("\n=== Error Analysis ===")
        error_analysis('SVM', svm.SVC(kernel='rbf', C=1.0, gamma=1.0), X_best, y)
        error_analysis('RF', RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=2,
            min_samples_leaf=1, max_features=1, random_state=1), X_best, y)

