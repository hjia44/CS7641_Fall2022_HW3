
from A3_USL_DimensionReduction_lib import *


def main():
    print('A3 Unsupervise Learning and Dimension Reduction')
    data_wine = datasets.load_wine()
    #print(dir(data_wine), len(data_wine.feature_names))
    n = 25
    x_w, y_w = data_processing(load_data=load_wine)
    x_d, y_d = data_processing(load_data=load_digits)
    y_array_w = np.array(y_w)
    class_num_w = len(np.unique(y_array_w))
    y_array_d = np.array(y_d)
    class_num_d = len(np.unique(y_array_d))
    print(class_num_w, class_num_d)
    """
                                                         ###
                                                Single Algorithm Study     
                                                         ###
    # """
    ### K means
    k_w, v_scores_w, kmeansModel_w = kMeans(n, x_w, y_w)
    k_d, v_scores_d, kmeansModel_d = kMeans(n, x_d, y_d)
    #print(k_w, v_scores_w)
    print("kmeans Wine Max. v_measure_score index: " + str(v_scores_w.index(max(v_scores_w))))
    print("kmeans Digits Max. v_measure_score index: " + str(v_scores_d.index(max(v_scores_d))))
    # kmeans Wine Max. v_measure_score index: 2
    # kmeans Digits Max. v_measure_score index: 8
    best_n_kmeans_wine = v_scores_w.index(max(v_scores_w))
    best_n_kmeans_digits = v_scores_w.index(max(v_scores_w))
    x1, y1, x2, y2 = k_w, v_scores_w, k_d, v_scores_d
    fig_title, title1, title2 = 'kmeans: v_measure_score vs. n_cluster', 'Wine', 'Digits'
    label_x, label_y, ylim1, ylim2, fig_name = 'n_cluster', 'v_measure_score', 0, 1.0, 'kMeans_v_measure_score_vs_n_cluster.png'
    fig_plot_2(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)

    ### Expectation Maximization
    nComponent_w, v_scores_w, expmaxModel_w = ExpMax(n, x_w, y_w)
    nComponent_d, v_scores_d, expmaxModel_d = ExpMax(n, x_d, y_d)
    #print(nComponent_w, v_scores_w)
    print("ExpMax Wine Max. v_measure_score index: " + str(v_scores_w.index(max(v_scores_w))))
    print("ExpMax Digits Max. v_measure_score index: " + str(v_scores_d.index(max(v_scores_d))))
    best_n_ExpMax_wine = v_scores_w.index(max(v_scores_w))
    best_n_ExpMax_digits = v_scores_w.index(max(v_scores_w))
    # ExpMax Wine Max. v_measure_score index: 2
    # ExpMax Digits Max. v_measure_score index: 6
    x1, y1, x2, y2 = nComponent_w, v_scores_w, nComponent_d, v_scores_d
    fig_title, title1, title2 = 'Exp Max: v_measure_score vs. n_component', 'Wine', 'Digits'
    label_x, label_y, ylim1, ylim2, fig_name = 'n_component', 'v_measure_score', 0, 1.0, 'ExpMax_v_measure_score_vs_n_compoenent.png'
    fig_plot_2(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)

    ### PCA
    n_w = np.minimum(x_w.shape[0], x_w.shape[1])
    n_d = np.minimum(x_d.shape[0], x_d.shape[1])
    nComponent_w, exp_var_r_w, pcas_w = PCA_algo(n_w, x_w, y_w)
    nComponent_d, exp_var_r_d, pcas_d = PCA_algo(n_d, x_d, y_d)
    print(nComponent_w, sum(exp_var_r_d[:20]))
    x1, y1, x2, y2 = nComponent_w, exp_var_r_w, nComponent_d, exp_var_r_d
    fig_title, title1, title2 = 'PCA: explained_variance_ratio vs. n_component', 'Wine', 'Digits'
    label_x, label_y, ylim1, ylim2, fig_name = 'n_component', 'explained_variance_ratio', 0, 0.5, 'PCA_explained_variance_ratio_vs_n_compoenent.png'
    fig_plot_2(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)
    x_trans_w = pcas_w[1].transform(x_w)
    x_trans_d = pcas_d[1].transform(x_d)
    #print(x_trans_w.shape)
    x1, y1, x2, y2 = x_trans_w[:, 0], x_trans_w[:, 1], x_trans_d[:, 0], x_trans_d[:, 1]
    fig_title, title1, title2 = 'PCA: Reduced 2-D Plots', 'Wine', 'Digits'
    label_x, label_y, ylim1, ylim2, fig_name = ' ', ' ', -10, 10, 'PCA_reduced_2D_plots.png'
    fig_plot2_scatter(x1, y1, x2, y2, y_w, y_d, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)

    ### ICA
    nComponent_w, kurts_w, icas_w = ICA_algo(n, x_w, y_w)
    nComponent_d, kurts_d, icas_d = ICA_algo(n, x_d, y_d)
    #print(nComponent_w, kurts_w)
    x1, y1, x2, y2 = nComponent_w, kurts_w, nComponent_d, kurts_d
    fig_title, title1, title2 = 'ICA: Kurts vs. n_component', 'Wine', 'Digits'
    label_x, label_y, ylim1, ylim2, fig_name = 'n_component', 'Kurtosis', 0, 1500, 'ICA_kurts_vs_n_compoenent.png'
    fig_plot_2_noylim(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, fig_name)
    x_trans_w = icas_w[1].transform(x_w)
    x_trans_d = icas_d[1].transform(x_d)
    #print(x_trans_w.shape)
    x1, y1, x2, y2 = x_trans_w[:, 0], x_trans_w[:, 1], x_trans_d[:, 0], x_trans_d[:, 1]
    fig_title, title1, title2 = 'ICA: Reduced 2-D Plots', 'Wine', 'Digits'
    label_x, label_y, ylim1, ylim2, fig_name = ' ', ' ', -0.2, 0.2, 'ICA_reduced_2D_plots.png'
    fig_plot2_scatter(x1, y1, x2, y2, y_w, y_d, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)
    ## Random Projection
    k_w, k_d = 2, 6
    nComponent_w, v_scores_w, randproj_w = RandProj(n, x_w, y_w, k_w)
    nComponent_d, v_scores_d, randproj_d = RandProj(n, x_d, y_d, k_d)
    #print(nComponent_w, v_scores_w)
    print("Rand Proj Wine Max. v_measure_score index: " + str(v_scores_w.index(max(v_scores_w))))
    print("Rand Proj Digits Max. v_measure_score index: " + str(v_scores_d.index(max(v_scores_d))))
    # Rand Proj Wine Max. v_measure_score index: 6
    # Rand Proj Digits Max. v_measure_score index: 4
    x1, y1, x2, y2 = nComponent_w, v_scores_w, nComponent_d, v_scores_d
    fig_title, title1, title2 = 'Rand Proj: v_measure_score vs. n_component', 'Wine', 'Digits'
    label_x, label_y, ylim1, ylim2, fig_name = 'n_component', 'v_measure_score', 0, 1.0, 'Rand_Proj_v_measure_score_vs_n_compoenent.png'
    fig_plot_2(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)
    ### LDA
    k_w, k_d = 2, 6
    #n_w, n_d = x_w.shape[1]-1, x_d.shape[1]-1
    n_w = np.minimum(class_num_w - 1 , x_w.shape[1])
    n_d = np.minimum(class_num_d - 1, x_d.shape[1])
    nComponent_w, v_scores_w, ldas_w = LDA_algo(n_w, x_w, y_w, k_w)
    nComponent_d, v_scores_d, ldas_d = LDA_algo(n_d, x_d, y_d, k_d)
    #print(nComponent_w, v_scores_w)
    print("LDA Wine Max. v_measure_score index: " + str(v_scores_w.index(max(v_scores_w))))
    print("LDA Proj Digits Max. v_measure_score index: " + str(v_scores_d.index(max(v_scores_d))))
    # LDA Wine Max. v_measure_score index: 0
    # LDA Proj Digits Max. v_measure_score index: 6
    x1, y1, x2, y2 = nComponent_w, v_scores_w, nComponent_d, v_scores_d
    fig_title, title1, title2 = 'LDA: v_measure_score vs. n_component', 'Wine', 'Digits'
    label_x, label_y, ylim1, ylim2, fig_name = 'n_component', 'v_measure_score', 0, 1.0, 'LDA_v_measure_score_vs_n_compoenent.png'
    fig_plot_2(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)

    """
                                                         ###
                                                Combination Algorithm Study     
                                                         ###
    """
    best_n_kmeans_wine, best_n_kmeans_digits, best_n_ExpMax_wine, best_n_ExpMax_digits = 2, 8, 2, 6
    ###Rerun Cluster on PCA data
    n_w = np.minimum(x_w.shape[0], x_w.shape[1])
    n_d = np.minimum(x_d.shape[0], x_d.shape[1])
    nComponent_w, v_scores_kmean_w, v_scores_expmax_w = PCA_cluster(n_w, x_w, y_w, best_n_kmeans_wine, best_n_ExpMax_wine)
    nComponent_d, v_scores_kmean_d, v_scores_expmax_d = PCA_cluster(n_d, x_d, y_d, best_n_kmeans_digits, best_n_ExpMax_digits)
    x1, y1, x2, y2 = nComponent_w, v_scores_kmean_w, nComponent_w, v_scores_expmax_w
    fig_title, title1, title2 = 'Wine PCA: v_measure_score vs. n_cluster', 'kmeans', 'expection maximum'
    label_x, label_y, ylim1, ylim2, fig_name = 'n_cluster', 'v_measure_score', 0, 1.0, 'Wine_PCA_v_measure_score_vs_n_cluster.png'
    fig_plot_2(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)
    x1, y1, x2, y2 = nComponent_d, v_scores_kmean_d, nComponent_d, v_scores_expmax_d
    fig_title, title1, title2 = 'Digits PCA: v_measure_score vs. n_cluster', 'kmeans', 'expection maximum'
    label_x, label_y, ylim1, ylim2, fig_name = 'n_cluster', 'v_measure_score', 0, 1.0, 'Digits_PCA_v_measure_score_vs_n_cluster.png'
    fig_plot_2(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)
    # best_n_PCA_wine = v_scores_w.index(max(v_scores_w))
    # best_n_PCA_digits = v_scores_w.index(max(v_scores_w))

    ###Rerun Cluster on ICA data
    nComponent_w, v_scores_kmean_w, v_scores_expmax_w = ICA_cluster(n, x_w, y_w, best_n_kmeans_wine, best_n_kmeans_digits)
    nComponent_d, v_scores_kmean_d, v_scores_expmax_d = ICA_cluster(n, x_d, y_d, best_n_kmeans_wine, best_n_kmeans_digits)
    x1, y1, x2, y2 = nComponent_w, v_scores_kmean_w, nComponent_w, v_scores_expmax_w
    fig_title, title1, title2 = 'Wine ICA: v_measure_score vs. n_cluster', 'kmeans', 'expection maximum'
    label_x, label_y, ylim1, ylim2, fig_name = 'n_cluster', 'v_measure_score', 0, 1.0, 'Wine_ICA_v_measure_score_vs_n_cluster.png'
    fig_plot_2(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)
    x1, y1, x2, y2 = nComponent_d, v_scores_kmean_d, nComponent_d, v_scores_expmax_d
    fig_title, title1, title2 = 'Digits ICA: v_measure_score vs. n_cluster', 'kmeans', 'expection maximum'
    label_x, label_y, ylim1, ylim2, fig_name = 'n_cluster', 'v_measure_score', 0, 1.0, 'Digits_ICA_v_measure_score_vs_n_cluster.png'
    fig_plot_2(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)

    ###Rerun Cluster on Random Projection data
    nComponent_w, v_scores_kmean_w, v_scores_expmax_w = RandProj_cluster(n, x_w, y_w, best_n_kmeans_wine, best_n_kmeans_digits)
    nComponent_d, v_scores_kmean_d, v_scores_expmax_d = RandProj_cluster(n, x_d, y_d, best_n_kmeans_wine, best_n_kmeans_digits)
    x1, y1, x2, y2 = nComponent_w, v_scores_kmean_w, nComponent_w, v_scores_expmax_w
    fig_title, title1, title2 = 'Wine Random Projection: score vs. n_cluster', 'kmeans', 'expection maximum'
    label_x, label_y, ylim1, ylim2, fig_name = 'n_cluster', 'v_measure_score', 0, 1.0, 'Wine_Random_Projection_v_measure_score_vs_n_cluster.png'
    fig_plot_2(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)
    x1, y1, x2, y2 = nComponent_d, v_scores_kmean_d, nComponent_d, v_scores_expmax_d
    fig_title, title1, title2 = 'Digits Random Projection: score vs. n_cluster', 'kmeans', 'expection maximum'
    label_x, label_y, ylim1, ylim2, fig_name = 'n_cluster', 'v_measure_score', 0, 1.0, 'Digits_Random_Projection_v_measure_score_vs_n_cluster.png'
    fig_plot_2(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)

    ###Rerun Cluster on LDA data
    n_w = np.minimum(class_num_w - 1, x_w.shape[1])
    n_d = np.minimum(class_num_d - 1, x_d.shape[1])
    nComponent_w, v_scores_kmean_w, v_scores_expmax_w = LDA_cluster(n_w, x_w, y_w, best_n_kmeans_wine, best_n_ExpMax_wine)
    nComponent_d, v_scores_kmean_d, v_scores_expmax_d = LDA_cluster(n_d, x_d, y_d, best_n_kmeans_digits, best_n_ExpMax_digits)
    x1, y1, x2, y2 = nComponent_w, v_scores_kmean_w, nComponent_w, v_scores_expmax_w
    fig_title, title1, title2 = 'Wine LDA: score vs. n_cluster', 'kmeans', 'expection maximum'
    label_x, label_y, ylim1, ylim2, fig_name = 'n_cluster', 'v_measure_score', 0, 1.0, 'Wine_LDA_v_measure_score_vs_n_cluster.png'
    fig_plot_2(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)
    x1, y1, x2, y2 = nComponent_d, v_scores_kmean_d, nComponent_d, v_scores_expmax_d
    fig_title, title1, title2 = 'Digits LDA: score vs. n_cluster', 'kmeans', 'expection maximum'
    label_x, label_y, ylim1, ylim2, fig_name = 'n_cluster', 'v_measure_score', 0, 1.0, 'Digits_LDA_v_measure_score_vs_n_cluster.png'
    fig_plot_2(x1, y1, x2, y2, fig_title, title1, title2, label_x, label_y, ylim1, ylim2, fig_name)

    ### Comparion of all combinations
    v_score_kmeans_w, v_score_expmaxs_d = [], []
    best_n_kmeans_wine, best_n_kmeans_digits, best_n_ExpMax_wine, best_n_ExpMax_digits = 2, 8, 2, 6
    n_w_PCA = np.minimum(x_w.shape[0], x_w.shape[1])
    n_d_PCA = np.minimum(x_d.shape[0], x_d.shape[1])
    n_w_LDA = np.minimum(class_num_w - 1, x_w.shape[1])
    n_d_LDA = np.minimum(class_num_d - 1, x_d.shape[1])
    v_score_kmeans_w = v_score_kmeans_DR(x_w, y_w, best_n_kmeans_wine, n_w_PCA, n_w_LDA, n)
    print(v_score_kmeans_w)
    v_score_kmeans_d = v_score_kmeans_DR(x_d, y_d, best_n_kmeans_digits, n_d_PCA, n_d_LDA, n)
    print(v_score_kmeans_d)
    v_score_expmaxs_w = v_score_expmax_DR(x_w, y_w, best_n_ExpMax_wine, n_w_PCA, n_w_LDA, n)
    print(v_score_expmaxs_w)
    v_score_expmaxs_d = v_score_kmeans_DR(x_d, y_d, best_n_ExpMax_digits, n_d_PCA, n_d_LDA, n)
    print(v_score_expmaxs_d)
    label1, y1, = ['no DR', 'PCA', 'ICA', 'Rand Proj', 'LDA'], v_score_kmeans_w
    fig_title, label_y, fig_name = 'Wine: Kmeans & Dimension Reduction', 'v_measure_score', 'Wine_kmeans_Dimension_Reduction.png'
    fig_barplot_1(label1, y1, label_y, fig_title, fig_name, 'orange')
    label2, y2, = ['no DR', 'PCA', 'ICA', 'Rand Proj', 'LDA'], v_score_expmaxs_w
    fig_title, label_y, fig_name= 'Wine: Exp Max & Dimension Reduction', 'v_measure_score', 'Wine_exp_max_Dimension_Reduction.png'
    fig_barplot_1(label2, y2, label_y, fig_title, fig_name, 'green')

    label1, y1, = ['no DR', 'PCA', 'ICA', 'Rand Proj', 'LDA'], v_score_kmeans_d
    fig_title, label_y, fig_name = 'Digits: Kmeans & Dimension Reduction', 'v_measure_score', 'Digits_kmeans_Dimension_Reduction.png'
    fig_barplot_1(label1, y1, label_y, fig_title, fig_name, 'orange')
    label2, y2, = ['no DR', 'PCA', 'ICA', 'Rand Proj', 'LDA'], v_score_expmaxs_d
    fig_title, label_y, fig_name = 'Digits: Exp Max & Dimension Reduction', 'v_measure_score', 'Digits_exp_max_Dimension_Reduction.png'
    fig_barplot_1(label2, y2, label_y, fig_title, fig_name, 'green')

    """
                                                         ###
                                                Netural Networks with Dimension Reduction     
                                                         ###
    """


    x_pca = PCA(n_components=n_w_PCA, random_state=12345).fit_transform(x_d)
    x_ica = FastICA(n_components=n, random_state=12345).fit_transform(x_d)
    x_randProj = GaussianRandomProjection(n_components=n, random_state=12345).fit_transform(x_d)
    x_lda = LinearDiscriminantAnalysis(n_components=n_w_LDA).fit_transform(x_d, y_d)
    accu_scores_noDR, runtime_noDR = nn_w_DR(x_d, y_d)
    accu_scores_pca, runtime_pca = nn_w_DR(x_pca, y_d)
    accu_scores_ica, runtime_ica = nn_w_DR(x_ica, y_d)
    accu_scores_randProj, runtime_randProj = nn_w_DR(x_randProj, y_d)
    accu_scores_lda, runtime_lda = nn_w_DR(x_lda, y_d)
    #print(accu_scores_noDR, accu_scores_pca, accu_scores_ica, accu_scores_randProj, accu_scores_lda)
    accu_scores_nn_DC = [accu_scores_noDR, accu_scores_pca, accu_scores_ica, accu_scores_randProj, accu_scores_lda]
    label1, y1, = ['no DR', 'PCA', 'ICA', 'Rand Proj', 'LDA'], accu_scores_nn_DC
    fig_title, label_y, fig_name = 'Digits: NN & Dimension Reduction ', 'Accuracy_score', 'Digits_Score_NN_Dimension_Reduction.png'
    fig_barplot_1(label1, y1, label_y, fig_title, fig_name, 'orange')
    time_nn_DC = runtime_noDR + runtime_pca + runtime_ica + runtime_randProj + runtime_lda
    print(time_nn_DC)
    label1, y1, = ['no DR', 'PCA', 'ICA', 'Rand Proj', 'LDA'], time_nn_DC
    fig_title, label_y, fig_name = 'Digits: NN & Dimension Reduction ', 'Time', 'Digits_Time_NN_Dimension_Reduction.png'
    fig_barplot_1(label1, y1, label_y, fig_title, fig_name, 'green')

    """
                                                         ###
                                                Netural Networks with clustering data     
                                                         ###
    """
    kmean_clusters_d = KMeans(n_clusters=best_n_kmeans_digits, random_state=12345).fit_predict(x_d)
    expmax_clusters_d = GaussianMixture(n_components=best_n_ExpMax_digits, random_state=12345).fit_predict(x_d)
    nn_kmean, accu_score_kmean_d = nn_w_clustering(x_d, y_d, kmean_clusters_d, 10)
    nn_expmax, accu_score_expmax_d = nn_w_clustering(x_d, y_d, expmax_clusters_d, 10)
    accu_scores_noDR_d = []
    x_train, x_test, y_train, y_test = data_processing2(load_data=load_digits)
    nn_noDR_d = MLPClassifier(solver='adam', hidden_layer_sizes=10, max_iter=500, random_state=12345)
    nn_noDR_d.fit(x_train, y_train)
    accu_score_noDR_d = accuracy_score(nn_noDR_d.predict(x_test), y_test)
    accu_scores_noDR_d.append(accu_score_noDR_d)
    print(accu_scores_noDR_d, accu_score_kmean_d, accu_score_expmax_d)
    label2, y2, = ['no DR', 'kmeans', 'Exp Max'], accu_scores_noDR_d+accu_score_kmean_d+accu_score_expmax_d
    fig_title, label_y, fig_name = 'Digits: NN with Clustering Data', 'Accuracy score', 'Digits_NN_with_clustering_data.png'
    fig_barplot_1(label2, y2, label_y, fig_title, fig_name, 'orange')

"""
                                                     ###
                                            Main Function Separator     
                                                     ###
"""

if __name__ == '__main__':
    print('Assignment 3 USL and Dimension Reduction')
    main()