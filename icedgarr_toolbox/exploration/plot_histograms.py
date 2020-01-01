import matplotlib.pyplot as plt


def plot_features_histograms(train_data, test_data, features, normalized=True, bins=100):
    for feature in features:
        plot_train_test_histogram(train_data[feature], test_data[feature], feature, normalized=normalized, bins=bins)


def plot_train_test_histogram(train_feature, test_feature, title, normalized=True, bins=100):
    plt.figure()
    train_feature.hist(bins=bins, normed=normalized, alpha=0.5, label='test')
    test_feature.hist(bins=bins, normed=normalized, alpha=0.5, label='train')
    plt.title(title)
    plt.legend()
    plt.show()
