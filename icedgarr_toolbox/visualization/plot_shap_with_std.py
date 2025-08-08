import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_shap_values_with_std(data, shap_values_df, feature, xlabel, order_plot):
    data = data.reset_index(drop=True)
    feature_shape_df = pd.concat([data[feature], shap_values_df[feature]], axis=1)
    feature_shape_df.columns = ['feature', 'shap_value']
    feature_mean_values = feature_shape_df.groupby('feature')['shap_value'].agg(['mean', 'std'])

    plt.figure(figsize=(12, 8))
    plt.scatter(feature_shape_df['feature'], feature_shape_df['shap_value'], marker='x', s=1.5, c='#1f78b4',
                label='Observation')
    plt.plot(list(feature_mean_values.index), feature_mean_values['mean'], c='#1f78b4')
    xlim = plt.xlim()
    plt.scatter(xlim[0] * np.ones(data[feature].isnull().sum()),
                shap_values_df.loc[data[feature].isnull(), feature], c='grey', label='NaN observation', s=1.5,
                marker='x')
    plt.fill_between(list(feature_mean_values.index), feature_mean_values['mean'] - feature_mean_values['std'],
                     feature_mean_values['mean'] + feature_mean_values['std'], color='#a6cee3', alpha=0.6,
                     label='Mean +- std')
    nan_mean = shap_values_df.loc[data[feature].isnull(), feature].mean()
    nan_std = shap_values_df.loc[data[feature].isnull(), feature].std()
    plt.errorbar(xlim[0], nan_mean, nan_std, linestyle='None', marker='o', color='#a6cee3')
    plt.fill_between([xlim[0] - 0.01, xlim[0] + 0.01], [nan_mean - nan_std] * 2,
                     [nan_mean + nan_std] * 2, color='#a6cee3', alpha=0.6)

    plt.legend(markerscale=4)
    # b.legendHandles[0]._legmarker.set_markersize(6)
    # plt.xticks(np.arange(feature_shape_df['feature'].min(), feature_shape_df['feature'].max(), 20))
    plt.xlabel(xlabel)
    plt.ylabel('SHAP value')
    # plt.savefig(f'nature_medicine_paper/shap_plots/shap_with_std/{i}_{feature}.png')
    # plt.show()




"""
Example usage:
i=7
feature = np.abs(shap_values_df).mean().sort_values(ascending=False).index[i-1]
print(feature)
plot_shap_values_with_std(test_data.reset_index(), shap_values_df, feature, xlabel=rename_important_features[feature], 
                          order_plot=i)"""


def plot_shap_values_with_std_set_ticks_and_max_value(data, shap_values_df, feature, xlabel, order_plot,
                                                      xticks, xticks_labels, max_value=None):
    data = data.reset_index(drop=True)
    feature_shape_df = pd.concat([data[feature].copy(), shap_values_df[feature]].copy(), axis=1)
    feature_shape_df.columns = ['feature', 'shap_value']
    if max_value:
        feature_shape_df.loc[feature_shape_df['feature'] >= max_value, 'feature'] = max_value
    feature_mean_values = feature_shape_df.groupby('feature')['shap_value'].agg(['mean', 'std'])

    ax = plt.figure(figsize=(12, 8))
    plt.scatter(feature_shape_df['feature'], feature_shape_df['shap_value'], marker='x', s=1.5, c='#1f78b4',
                label='Observation')
    plt.plot(list(feature_mean_values.index), feature_mean_values['mean'], c='#1f78b4')
    xlim = plt.xlim()
    plt.scatter(xlim[0] * np.ones(data[feature].isnull().sum()),
                shap_values_df.loc[data[feature].isnull(), feature], c='grey', label='NaN observation', s=1.5,
                marker='x')
    plt.fill_between(list(feature_mean_values.index), feature_mean_values['mean'] - feature_mean_values['std'],
                     feature_mean_values['mean'] + feature_mean_values['std'], color='#a6cee3', alpha=0.6,
                     label='Mean +- std')
    nan_mean = shap_values_df.loc[data[feature].isnull(), feature].mean()
    nan_std = shap_values_df.loc[data[feature].isnull(), feature].std()
    plt.errorbar(xlim[0], nan_mean, nan_std, linestyle='None', marker='o', color='#a6cee3')
    plt.fill_between([xlim[0] - 0.01, xlim[0] + 0.01], [nan_mean - nan_std] * 2,
                     [nan_mean + nan_std] * 2, color='#a6cee3', alpha=0.6)
    plt.plot([xlim[0], xlim[1]], [0, 0], linestyle='--', lw=2, color='black',
             label='No effect', alpha=.3)

    plt.legend(markerscale=4)
    # b.legendHandles[0]._legmarker.set_markersize(6)
    # plt.xticks(np.arange(feature_shape_df['feature'].min(), feature_shape_df['feature'].max(), 20))
    plt.xlabel(xlabel)
    ax.axes[0].set_xticks([xlim[0]] + xticks)
    ax.axes[0].set_xticklabels(['NaN'] + xticks_labels)
    # ax.axes[0].set_xticks(xticks)
    # ax.axes[0].set_xticklabels(xticks_labels)
    plt.ylabel('SHAP value')
    plt.savefig(f'nature_medicine_paper/shap_plots/shap_with_std/{i}_{feature}.png')
    # plt.show()

"""
Example usage:
plot_shap_values_with_std_set_ticks_and_max_value(test_data, shap_values_df, feature, 
                                                  xlabel=rename_important_features[feature], 
                                                  order_plot=i, xticks=xticks_list, xticks_labels=xticks_labels, 
                                                  max_value=10)"""