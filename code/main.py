import os
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from bureau import merge_bureau

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better.

    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance

    Returns:
        shows a plot of the 15 most importance features

        df (dataframe): feature importances sorted by importance (highest to lowest)
        with a column for normalized importance
        """

    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))),
            df['importance_normalized'].head(15),
            align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))

    # Plot labeling
    plt.xlabel('Normalized Importance');
    plt.title('Feature Importances')
    plt.show()

    return df

def model(features, test_features, model_type='lgb', n_folds=5):
    """Train and test models using cross validation.
    Parameters
    --------
        features (pd.DataFrame):
            dataframe of training features to use
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame):
            dataframe of testing features to use
            for making predictions with the model.
    Return
    --------
        submission (pd.DataFrame):
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame):
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame):
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
    """

    # Extract the ids and labels
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    labels = features['TARGET']

    # Remove the ids and target
    features = features.drop(columns=['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns=['SK_ID_CURR'])

    # One Hot Encoding / Simplex Representation
    features = pd.get_dummies(features)
    test_features = pd.get_dummies(test_features)
    # Align the dataframes by the columns
    features, test_features = features.align(test_features, join='inner', axis=1)
    cat_indices = 'auto'

    # Extract feature names
    feature_names = list(features.columns)

    if model_type == 'lgb':
        features = np.array(features)
        test_features = np.array(test_features)
    else:
        # Fill Missing Value and Standardize to [0,1] (additional)
        features = features.copy()
        test_features = test_features.copy()
        imputer = SimpleImputer(strategy='median')
        scaler = MinMaxScaler(feature_range=(0, 1))

        imputer.fit(features)
        features = imputer.transform(features)
        test_features = imputer.transform(test_features)

        scaler.fit(features)
        features = scaler.transform(features)
        test_features = scaler.transform(test_features)

    print('*** After preprocessing ***')
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)

    feature_importance_values = np.zeros(len(feature_names))
    test_predictions = np.zeros(test_features.shape[0])
    out_of_fold = np.zeros(features.shape[0])
    valid_scores = []
    train_scores = []

    # Create the kfold object
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)

    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        # Data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]

        if model_type == 'lgb':
            # Create the model
            model = lgb.LGBMClassifier(n_estimators=10000, objective='binary',
                                       class_weight='balanced', learning_rate=0.05,
                                       reg_alpha=0.1, reg_lambda=0.1,
                                       subsample=0.8, n_jobs=-1, random_state=50)

            # Train the model
            model.fit(train_features, train_labels, eval_metric='auc',
                      eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                      eval_names=['valid', 'train'], categorical_feature=cat_indices)

            # Record the best iteration and feature importance
            best_iteration = model.best_iteration_
            feature_importance_values += model.feature_importances_ / k_fold.n_splits

            # Make predictions
            test_predictions += model.predict_proba(test_features, num_iteration=best_iteration)[:, 1] / k_fold.n_splits

            # Record the out of fold predictions
            out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration=best_iteration)[:, 1]

            # Record the best score
            valid_score = model.best_score_['valid']['auc']
            train_score = model.best_score_['train']['auc']

        else:
            if model_type == 'logreg':
                model = LogisticRegression(C=0.0001)
            else:
                model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=50)
            model.fit(train_features, train_labels)

            if model_type != 'logreg': feature_importance_values += model.feature_importances_ / k_fold.n_splits
            test_predictions += model.predict_proba(test_features)[:, 1] / k_fold.n_splits
            out_of_fold[valid_indices] = model.predict_proba(valid_features)[:, 1]

            valid_score = roc_auc_score(valid_labels, out_of_fold[valid_indices]) #model.score(valid_features, valid_labels)
            train_score = roc_auc_score(train_labels, model.predict_proba(train_features)[:, 1]) #train_score = model.score(train_features, train_labels)

        valid_scores.append(valid_score)
        train_scores.append(train_score)

        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})

    # Make the feature importance dataframe
    if model_type == 'logreg':
        feature_importances = None
    else:
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

    # # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)

    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))

    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')

    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})

    return submission, feature_importances, metrics

if __name__ == '__main__':
    model_type = 'logreg' # (lgb or rf or logreg) lgb: LightBGM; rf: RandomForest; logreg: LogisticRegression

    ###### Loading data ######
    app_train = pd.read_csv('./data/application_train.csv')
    app_test = pd.read_csv('./data/application_test.csv')
    print('*** Raw data ***')
    print('Training data shape: ', app_train.shape)
    print('Testing data shape: ', app_test.shape)

    ###### Combine with Bureau data
    app_train, app_test = merge_bureau(app_train, app_test)

    ###### Train and Test ######
    submission, fi, metrics = model(app_train, app_test, model_type=model_type)
    print('\nBaseline metrics')
    print(metrics)

    ###### Plot Feature Importance ######
    if model_type != 'logreg': fi_sorted = plot_feature_importances(fi)

    ###### Save Result ######
    submission.to_csv('results/{}_combined.csv'.format(model_type), index=False)

    ###### Looking at the kde plot of particular feature (IF need) ######
    # import seaborn as sns
    # def kde_target(var_name, df):
    #     # Calculate the correlation coefficient between the new variable and the target
    #     corr = df['TARGET'].corr(df[var_name])
    #
    #     # Calculate medians for repaid vs not repaid
    #     avg_repaid = df.ix[df['TARGET'] == 0, var_name].median()
    #     avg_not_repaid = df.ix[df['TARGET'] == 1, var_name].median()
    #
    #     plt.figure(figsize=(12, 6))
    #
    #     # Plot the distribution for target == 0 and target == 1
    #     sns.kdeplot(df.ix[df['TARGET'] == 0, var_name], label='TARGET == 0')
    #     sns.kdeplot(df.ix[df['TARGET'] == 1, var_name], label='TARGET == 1')
    #
    #     # label the plot
    #     plt.xlabel(var_name);
    #     plt.ylabel('Density');
    #     plt.title('%s Distribution' % var_name)
    #     plt.legend();
    #
    #     # print out the correlation
    #     print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    #     # Print out average values
    #     print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    #     print('Median value for loan that was repaid =     %0.4f' % avg_repaid)
    # kde_target('EXT_SOURCE_3', app_train)