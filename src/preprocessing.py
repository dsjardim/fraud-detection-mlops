import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler

warnings.filterwarnings("ignore")
np.random.seed(203)
COLORS = ["#0101DF", "#DF0101"]


def subsample_dataset(df, debug=False):
    if debug:
        print('\nNo Frauds', round(df['Class'].value_counts()[0] / len(df) * 100, 2), '% of the original dataset')
        print('Frauds', round(df['Class'].value_counts()[1] / len(df) * 100, 2), '% of the original dataset\n')

    # Since most of our data has already been scaled we should scale the columns that are left to scale
    # (Amount and Time)

    # RobustScaler is less prone to outliers.
    std_scaler = StandardScaler()
    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))

    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']

    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    # Amount and Time are Scaled!
    if debug:
        df.head()

    X = df.drop('Class', axis=1)
    y = df['Class']

    sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    for train_index, test_index in sss.split(X, y):
        if debug:
            print("Train:", train_index, "Test:", test_index)
        original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
        original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    # We already have X_train and y_train for undersample data thats why I am using original to distinguish and to
    # not overwrite these variables. original_Xtrain, original_Xtest, original_ytrain, original_ytest =
    # train_test_split(X, y, test_size=0.2, random_state=42)

    # Check the Distribution of the labels
    # Turn into an array
    original_Xtrain = original_Xtrain.values
    original_Xtest = original_Xtest.values
    original_ytrain = original_ytrain.values
    original_ytest = original_ytest.values

    # See if both the train and test label distribution are similarly distributed
    train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
    test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

    if debug:
        print('-' * 100)
        print('Label Distributions:')
        print(train_counts_label / len(original_ytrain))
        print(test_counts_label / len(original_ytest))

    # Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of
    # the classes. Lets shuffle the data before creating the subsamples

    df = df.sample(frac=1)

    # amount of fraud classes 492 rows.
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df = df.loc[df['Class'] == 0][:492]

    normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

    # Shuffle dataframe rows
    new_df = normal_distributed_df.sample(frac=1, random_state=42)

    if debug:
        print("\nNew dataset:")
        print('No Frauds', round(new_df['Class'].value_counts()[0] / len(new_df) * 100, 2),
              '% of the subsample dataset')
        print('Frauds', round(new_df['Class'].value_counts()[1] / len(new_df) * 100, 2),
              '% of the subsample dataset\n')
        new_df.head()

        print('\nDistribution of the Classes in the subsample dataset')
        print(new_df['Class'].value_counts() / len(new_df))

    return new_df


def correlation_analysis(original_df, subsample_df):
    # Make sure we use the subsample in our correlation
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 20))

    # Entire DataFrame
    corr = original_df.corr()
    sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size': 20}, ax=ax1)
    ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)

    sub_sample_corr = subsample_df.corr()
    sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size': 20}, ax=ax2)
    ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
    plt.show()

    f, axes = plt.subplots(ncols=4, figsize=(20, 4))

    # Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
    sns.boxplot(x="Class", y="V17", data=subsample_df, palette=COLORS, ax=axes[0])
    axes[0].set_title('V17 vs Class Negative Correlation')

    sns.boxplot(x="Class", y="V14", data=subsample_df, palette=COLORS, ax=axes[1])
    axes[1].set_title('V14 vs Class Negative Correlation')

    sns.boxplot(x="Class", y="V12", data=subsample_df, palette=COLORS, ax=axes[2])
    axes[2].set_title('V12 vs Class Negative Correlation')

    sns.boxplot(x="Class", y="V10", data=subsample_df, palette=COLORS, ax=axes[3])
    axes[3].set_title('V10 vs Class Negative Correlation')

    plt.show()

    f, axes = plt.subplots(ncols=4, figsize=(20, 4))

    # Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
    sns.boxplot(x="Class", y="V11", data=subsample_df, palette=COLORS, ax=axes[0])
    axes[0].set_title('V11 vs Class Positive Correlation')

    sns.boxplot(x="Class", y="V4", data=subsample_df, palette=COLORS, ax=axes[1])
    axes[1].set_title('V4 vs Class Positive Correlation')

    sns.boxplot(x="Class", y="V2", data=subsample_df, palette=COLORS, ax=axes[2])
    axes[2].set_title('V2 vs Class Positive Correlation')

    sns.boxplot(x="Class", y="V19", data=subsample_df, palette=COLORS, ax=axes[3])
    axes[3].set_title('V19 vs Class Positive Correlation')

    plt.show()


def outlier_treatment(df, debug=False):
    # # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
    v14_fraud = df['V14'].loc[df['Class'] == 1].values
    q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
    v14_iqr = q75 - q25

    v14_cut_off = v14_iqr * 1.5
    v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off

    outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
    new_df = df.drop(df[(df['V14'] > v14_upper) | (df['V14'] < v14_lower)].index)

    if debug:
        print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
        print('iqr: {}'.format(v14_iqr))
        print('Cut Off: {}'.format(v14_cut_off))
        print('V14 Lower: {}'.format(v14_lower))
        print('V14 Upper: {}'.format(v14_upper))
        print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))
        print('V10 outliers:{}'.format(outliers))
        print('----' * 44)

    # -----> V12 removing outliers from fraud transactions
    v12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
    q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
    v12_iqr = q75 - q25

    v12_cut_off = v12_iqr * 1.5
    v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
    outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
    new_df = new_df.drop(new_df[(new_df['V12'] > v12_upper) | (new_df['V12'] < v12_lower)].index)

    if debug:
        print('V12 Lower: {}'.format(v12_lower))
        print('V12 Upper: {}'.format(v12_upper))
        print('V12 outliers: {}'.format(outliers))
        print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))
        print('Number of Instances after outliers removal: {}'.format(len(new_df)))
        print('----' * 44)

    # Removing outliers V10 Feature
    v10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
    q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
    v10_iqr = q75 - q25

    v10_cut_off = v10_iqr * 1.5
    v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
    outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
    new_df = new_df.drop(new_df[(new_df['V10'] > v10_upper) | (new_df['V10'] < v10_lower)].index)

    if debug:
        print('V10 Lower: {}'.format(v10_lower))
        print('V10 Upper: {}'.format(v10_upper))
        print('V10 outliers: {}'.format(outliers))
        print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))
        print('Number of Instances after outliers removal: {}'.format(len(new_df)))
        print('----' * 44)

    return new_df


def dimensionality_reduction(X, algorithm="pca"):
    if algorithm.lower() == "pca":
        return PCA(n_components=2, random_state=42).fit_transform(X.values)
    elif algorithm.lower() == "tsne":
        return TSNE(n_components=2, random_state=42).fit_transform(X.values)
    else:
        return TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)


def prepare_dataset(df, debug=False, dim_reduction=False, dim_reduction_algo="pca"):
    subsample_df = subsample_dataset(df)

    if debug:
        correlation_analysis(df, subsample_df)

    new_df = outlier_treatment(subsample_df)

    X = new_df.drop('Class', axis=1)
    y = new_df['Class']

    if dim_reduction:
        X = dimensionality_reduction(X, dim_reduction_algo)

    from sklearn.model_selection import train_test_split

    # This is explicitly used for undersampling.
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.5)

    data_split = {
        "X_train": X_train.values,
        "X_valid": X_valid.values,
        "X_test": X_test.values,
        "y_train": y_train.values,
        "y_valid": y_valid.values,
        "y_test": y_test.values
    }

    return data_split
