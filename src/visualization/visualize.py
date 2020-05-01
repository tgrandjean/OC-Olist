import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram

sns.set()


def distplot(series):
    # Cut the window in 2 parts
    kwrgs = {"height_ratios": (.15, .85)}
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, figsize=(8, 8),
                                        gridspec_kw=kwrgs)

    # Add a graph in each part
    sns.boxplot(series, ax=ax_box)
    sns.distplot(series, ax=ax_hist)

    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    return f


def piechart(series):

    f, ax = plt.subplots(1, figsize=(5, 5))

    modal = series.value_counts().to_dict()
    labels = [x for x in modal.keys()]
    sizes = [x for x in modal.values()]
    explode = [0.2 if x != max(sizes) else 0.0 for x in sizes]

    ax.pie(sizes, labels=labels, autopct='%1.1f%%', explode=explode)
    ax.axis('equal')

    plt.show()


def barplot(series, normalize=False):

    f, ax = plt.subplots(1, figsize=(8, 8))

    modal = series.value_counts().to_dict()
    labels = [x for x in modal.keys()]
    sizes = np.array([x for x in modal.values()])
    if normalize:
        sizes = (sizes / sum(sizes)) * 100

    ax.bar(labels, sizes)
    # ax.set_xticks(range(int(max(labels) + 1)))

    plt.show()


def group_repartition(df, column='group', ax=None):
    count = df.groupby(column).count()\
        .sort_values(by='group').to_dict()['index']
    if not ax:
        f, ax = plt.subplots(1, figsize=(12, 8))
    ax.barh(y=range(len(count.keys())), width=list(count.values()),
            height=1.0, align="edge")
    # ax.set_yticks(list(count.keys()))
    return ax


def group_composition(df, column='group', ax=None):
    df = df.copy()
    # col_to_log = ['monetary', 'clothing', 'food',
    #               'high-tech', 'home', 'other']
    # for col in col_to_log:
    #     df[col] = df[col].apply(np.expm1)

    means = df.groupby(column).mean()
    scaled_means = means.copy()
    for col in scaled_means.select_dtypes(exclude='object').columns:
        if col != 'group':
            scaled_means[col] = MinMaxScaler()\
                .fit_transform(scaled_means[col].values.reshape((-1, 1)))

    if not ax:
        f, ax = plt.subplots(1, figsize=(12, 9))
    ax = sns.heatmap(scaled_means, cbar=False,
                     annot=means, fmt='.3f', cmap='RdBu_r', ax=ax)
    return ax


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def group_analysis(df, column='group'):
    df.sort_values(list(df.columns.values[1:-1]), inplace=True, ascending=True)
    kwargs = {'width_ratios': (0.7, 0.3)}
    f, (ax_heatmap, ax_bar) = plt.subplots(1, 2, sharey=True,
                                           gridspec_kw=kwargs,
                                           figsize=(14, 8))
    group_composition(df, column=column, ax=ax_heatmap)
    group_repartition(df, column=column, ax=ax_bar)
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.show()
