from sklearn.datasets import load_digits, load_wine
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score, accuracy_score
from sklearn.decomposition import PCA, FastICA
from sklearn import preprocessing, datasets
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import time


def kMeans(n, x, y):
    n_cluster = []
    kmeans = []
    v_scores = []
    for i in range(1, n):
        n_cluster.append(i)
    #for i in range(1, n):
        kmean = KMeans(n_clusters=i, random_state=12345).fit(x)
        kmeans.append(kmean)
        labels = kmean.predict(x)
        v_scores.append(v_measure_score(y, labels))
    #print(kmeans)
    #print(v_scores)
    return n_cluster, v_scores, kmeans

def ExpMax(n, x, y):
    n_component = []
    exp_maxs = []
    v_scores = []
    for i in range(1, n):
        n_component.append(i)
        exp_max = GaussianMixture(n_components=i, random_state=12345).fit(x)
        exp_maxs.append(exp_max)
        labels = exp_max.predict(x)
        v_scores.append(v_measure_score(y, labels))
    #print(v_scores)
    return n_component, v_scores, exp_maxs

def PCA_algo(n, x, y):
    n_component = []
    pcas = []
    for i in range(1, n):
        n_component.append(i)
        pca = PCA(n_components=i, random_state=12345)
        pca.fit(x)
        pcas.append(pca)
    # component contribution
    #print(pca.explained_variance_ratio_)
    return n_component, list(pca.explained_variance_ratio_), pcas

def PCA_cluster(n_pca, x, y, n_kmean_best, n_ExpMax_best):
    n_component = []
    v_scores_1 = []
    v_scores_2 = []
    kmean_best = KMeans(n_clusters=n_kmean_best, random_state=12345)
    exp_max_best = GaussianMixture(n_components=n_ExpMax_best, random_state=12345)
    for i in range(1, n_pca):
        n_component.append(i)
        pca = PCA(n_components=i, random_state=12345)
        pca.fit(x)
        x_trans1 = pca.transform(x)
        x_trans2 = pca.transform(x)
        # v_scores_1 = fit_vscore(kmean_best, x_trans1, y)
        # print(v_scores_1)
        # v_scores_2 = fit_vscore(exp_max_best, x_trans2, y)
        kmean_best.fit(x_trans1)
        labels1 = kmean_best.predict(x_trans1)
        v_scores_1.append(v_measure_score(y, labels1))
        #print(v_scores_1)
        exp_max_best.fit(x_trans2)
        labels2 = exp_max_best.predict(x_trans2)
        v_scores_2.append(v_measure_score(y, labels2))
    return n_component, v_scores_1, v_scores_2

def ICA_algo(n, x, y):
    n_component = []
    icas = []
    kurts = []
    for i in range(1, n):
        ica = FastICA(n_components=i, random_state=12345)
        ica.fit(x)
        icas.append(ica)
    ica_n = FastICA(n_components=n, random_state=12345)
    ica_n.fit(x)
    x_trans = ica_n.transform(x)
    print(x_trans.shape)
    for i in range(x_trans.shape[1]):
        n_component.append(i)
        n = x_trans.shape[0]
        mean = np.sum(x_trans[:, i] ** 1 / n)
        variance = np.sum((x_trans[:, i] - mean) ** 2) / n
        kurt = np.sum((x_trans[:, i] - mean) ** 4) / n
        kurt2 = kurt / (variance ** 2) - 3
        kurts.append(kurt2)
    return n_component, kurts, icas

def ICA_cluster(n_ica, x, y, n_kmean_best, n_ExpMax_best):
    n_component = []
    v_scores_1 = []
    v_scores_2 = []
    kmean_best = KMeans(n_clusters=n_kmean_best, random_state=12345)
    exp_max_best = GaussianMixture(n_components=n_ExpMax_best, random_state=12345)
    for i in range(1, n_ica):
        n_component.append(i)
        ica = FastICA(n_components=i, random_state=12345)
        ica.fit(x)
        x_trans1 = ica.transform(x)
        x_trans2 = ica.transform(x)
        # v_scores_1 = fit_vscore(kmean_best, x_trans1, y)
        # v_scores_2 = fit_vscore(exp_max_best, x_trans2, y)
        kmean_best.fit(x_trans1)
        labels = kmean_best.predict(x_trans1)
        v_scores_1.append(v_measure_score(y, labels))
        exp_max_best.fit(x_trans2)
        labels = exp_max_best.predict(x_trans2)
        v_scores_2.append(v_measure_score(y, labels))
        
    return n_component, v_scores_1, v_scores_2


def RandProj(n, x, y, k):
    n_component = []
    randProjs = []
    v_scores = []
    kmean = KMeans(n_clusters=k, random_state=12345)
    for i in range(1, n):
        n_component.append(i)
        randProj = GaussianRandomProjection(n_components=i, random_state=12345).fit(x)
        randProjs.append(randProj)
        x_trans = randProj.transform(x)
        kmean.fit(x_trans)
        labels = kmean.predict(x_trans)
        v_scores.append(v_measure_score(y, labels))
    #print(v_scores)
    return n_component, v_scores, randProjs

def RandProj_cluster(n, x, y, n_kmean_best, n_ExpMax_best):
    n_component = []
    v_scores_1 = []
    v_scores_2 = []
    kmean_best = KMeans(n_clusters=n_kmean_best, random_state=12345)
    exp_max_best = GaussianMixture(n_components=n_ExpMax_best, random_state=12345)
    for i in range(1, n):
        n_component.append(i)
        randProj = GaussianRandomProjection(n_components=i, random_state=12345).fit(x)
        x_trans1 = randProj.transform(x)
        x_trans2 = randProj.transform(x)
        kmean_best.fit(x_trans1)
        labels = kmean_best.predict(x_trans1)
        v_scores_1.append(v_measure_score(y, labels))
        #print(v_scores_1)
        exp_max_best.fit(x_trans2)
        labels = exp_max_best.predict(x_trans2)
        v_scores_2.append(v_measure_score(y, labels))
    return n_component, v_scores_1, v_scores_2

def LDA_algo(n, x, y, k):
    n_component = []
    ldas = []
    v_scores = []
    kmean = KMeans(n_clusters=k, random_state=12345)
    for i in range(1, n):
        n_component.append(i)
        lda = LinearDiscriminantAnalysis(n_components=i).fit(x, y)
        ldas.append(lda)
        x_trans = lda.transform(x)
        kmean.fit(x_trans)
        labels = kmean.predict(x_trans)
        v_scores.append(v_measure_score(y, labels))
    #print(v_scores)
    return n_component, v_scores, ldas

def LDA_cluster(n, x, y, n_kmean_best, n_ExpMax_best):
    n_component = []
    v_scores_1 = []
    v_scores_2 = []
    kmean_best = KMeans(n_clusters=n_kmean_best, random_state=12345)
    exp_max_best = GaussianMixture(n_components=n_ExpMax_best, random_state=12345)
    for i in range(1, n):
        n_component.append(i)
        lda = LinearDiscriminantAnalysis(n_components=i).fit(x, y)
        x_trans1 = lda.transform(x)
        x_trans2 = lda.transform(x)
        kmean_best.fit(x_trans1)
        labels = kmean_best.predict(x_trans1)
        v_scores_1.append(v_measure_score(y, labels))
        #print(v_scores_1)
        exp_max_best.fit(x_trans2)
        labels = exp_max_best.predict(x_trans2)
        v_scores_2.append(v_measure_score(y, labels))
    return n_component, v_scores_1, v_scores_2


def v_score_kmeans_DR(x_w, y_w, best_n_kmeans_wine, n_w_PCA, n_w_LDA, n):
    v_score_kmeans_w = []
    kmean_best_w = KMeans(n_clusters=best_n_kmeans_wine, random_state=12345).fit(x_w)
    labels = kmean_best_w.predict(x_w)
    v_score_kmeans_w.append(v_measure_score(y_w, labels))
    pca = PCA(n_components=n_w_PCA, random_state=12345).fit(x_w)
    x_trans = pca.transform(x_w)
    kmean_best_w.fit(x_trans)
    labels = kmean_best_w.predict(x_trans)
    v_score_kmeans_w.append(v_measure_score(y_w, labels))
    ica = FastICA(n_components=n, random_state=12345).fit(x_w)
    x_trans = ica.transform(x_w)
    kmean_best_w.fit(x_trans)
    labels = kmean_best_w.predict(x_trans)
    v_score_kmeans_w.append(v_measure_score(y_w, labels))
    randProj = GaussianRandomProjection(n_components=n, random_state=12345).fit(x_w)
    x_trans = randProj.transform(x_w)
    kmean_best_w.fit(x_trans)
    labels = kmean_best_w.predict(x_trans)
    v_score_kmeans_w.append(v_measure_score(y_w, labels))
    lda = LinearDiscriminantAnalysis(n_components=n_w_LDA).fit(x_w, y_w)
    x_trans = lda.transform(x_w)
    kmean_best_w.fit(x_trans)
    labels = kmean_best_w.predict(x_trans)
    v_score_kmeans_w.append(v_measure_score(y_w, labels))
    return v_score_kmeans_w

def v_score_expmax_DR(x_w, y_w, best_n_ExpMax_wine, n_w_PCA, n_w_LDA, n):
    v_score_kmeans_w = []
    expmax_best_w = GaussianMixture(n_components=best_n_ExpMax_wine, random_state=12345).fit(x_w)
    labels = expmax_best_w.predict(x_w)
    v_score_kmeans_w.append(v_measure_score(y_w, labels))
    pca = PCA(n_components=n_w_PCA, random_state=12345).fit(x_w)
    x_trans = pca.transform(x_w)
    expmax_best_w.fit(x_trans)
    labels = expmax_best_w.predict(x_trans)
    v_score_kmeans_w.append(v_measure_score(y_w, labels))
    ica = FastICA(n_components=n, random_state=12345).fit(x_w)
    x_trans = ica.transform(x_w)
    expmax_best_w.fit(x_trans)
    labels = expmax_best_w.predict(x_trans)
    v_score_kmeans_w.append(v_measure_score(y_w, labels))
    randProj = GaussianRandomProjection(n_components=n, random_state=12345).fit(x_w)
    x_trans = randProj.transform(x_w)
    expmax_best_w.fit(x_trans)
    labels = expmax_best_w.predict(x_trans)
    v_score_kmeans_w.append(v_measure_score(y_w, labels))
    lda = LinearDiscriminantAnalysis(n_components=n_w_LDA).fit(x_w, y_w)
    x_trans = lda.transform(x_w)
    expmax_best_w.fit(x_trans)
    labels = expmax_best_w.predict(x_trans)
    v_score_kmeans_w.append(v_measure_score(y_w, labels))
    return v_score_kmeans_w

def nn_w_clustering(x, y, clusters, hidden_layer_sizes):
    accuracy_scores = []
    data_new = np.c_[x, clusters]
    x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(data_new, y, random_state=12345)
    nn_new = MLPClassifier(solver='adam', hidden_layer_sizes=hidden_layer_sizes, max_iter=500, random_state=12345)
    nn_new.fit(x_train_new, y_train_new)
    accuracy_score1 = accuracy_score(nn_new.predict(x_test_new), y_test_new)
    accuracy_scores.append(accuracy_score1)
    return nn_new, accuracy_scores


def nn_w_DR(x, y):
    accuracy_scores, run_time = [], []
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123456)
    nn_new = MLPClassifier(solver='adam', hidden_layer_sizes=10, max_iter=500, random_state=12345)
    start_time = time.time()
    nn_new.fit(x_train, y_train)
    accuracy_scores = accuracy_score(nn_new.predict(x_test), y_test)
    gap_time_test = time.time() - start_time
    run_time.append(gap_time_test)
    print(f'time for Training: {gap_time_test:.3f}s')
    return accuracy_scores, run_time

"""
                                                     ###
                                            Figure Export and Data Processing Function Separator     
                                                     ###
"""
def fig_plot2_scatter(x1, y1, x2, y2, y0_1, y0_2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(fig_title, fontsize=18)
    ax1.scatter(x1, y1, c=y0_1)
    ax1.set_title(title1)
    ax1.set_ylim(ylim1, ylim2)
    ax1.set(xlabel=label_x, ylabel=label_y)
    ax2.scatter(x2, y2, c=y0_2)
    ax2.set_title(title2)
    ax2.set_ylim(ylim1, ylim2)
    ax2.set(xlabel=label_x, ylabel=label_y)
    plt.subplots_adjust(top=0.85, wspace=0.4)
    plt.show()
    fig.savefig(fig_name)


def fig_plot_1(data_x1, data_y1, xlabel, ylabel, title, fig_name, y1, y2):
    plt.plot(data_x1, data_y1, marker='s')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=18)
    plt.ylim(y1, y2)
    plt.legend()
    plt.savefig(fig_name)
    plt.show()

def fig_plot_2(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(fig_title, fontsize=18)
    ax1.plot(x1, y1, 's-', color='orange')
    ax1.set_title(title1)
    ax1.set_ylim(ylim1, ylim2)
    ax1.set(xlabel=label_x, ylabel=label_y)
    ax2.plot(x2, y2, 's-', color='green')
    ax2.set_title(title2)
    ax2.set_ylim(ylim1, ylim2)
    ax2.set(xlabel=label_x, ylabel=label_y)
    plt.subplots_adjust(top=0.85, wspace=0.4)
    plt.show()
    fig.savefig(fig_name)

def fig_barplot_1(label1, y1, ylabel, fig_title, fig_name, color):
    x1 = np.arange(len(label1))
    plt.xticks(x1, label1)
    plt.bar(x1, y1, 0.5, color=color)
    plt.title(fig_title, fontsize=18)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.savefig(fig_name)
    plt.show()

def fig_plot_2_noylim(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, fig_name):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(fig_title, fontsize=18)
    ax1.plot(x1, y1, 's-', color='orange')
    ax1.set_title(title1)
    ax1.set(xlabel=label_x, ylabel=label_y)
    ax2.plot(x2, y2, 's-', color='green')
    ax2.set_title(title2)
    ax2.set(xlabel=label_x, ylabel=label_y)
    plt.subplots_adjust(top=0.85, wspace=0.4)
    plt.show()
    fig.savefig(fig_name)

def data_processing(load_data):
    x, y = load_data(return_X_y=True)
    x = preprocessing.scale(x)
    return x, y

def data_processing2(load_data):
    x, y = load_data(return_X_y=True)
    x = preprocessing.scale(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=12345)
    return x_train, x_test, y_train, y_test



