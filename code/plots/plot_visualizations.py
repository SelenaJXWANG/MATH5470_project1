import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_visuals():
    """
    Loads the application_train data, calculates correlations with the TARGET,
    and plots a bar chart for the most correlated features and a heatmap
    for their inter-correlations.
    """
    # Load the data
    try:
        app_train = pd.read_csv('/Users/wjx/Downloads/home-credit-default-risk/data/application_train.csv')
    except FileNotFoundError:
        print("Error: '/Users/wjx/Downloads/home-credit-default-risk/data/application_train.csv' not found. Make sure the path is correct.")
        return

    # --- 1. Calculate Correlations with TARGET (on numeric columns only) ---
    # Select only numeric columns to avoid errors with categorical data
    numeric_app_train = app_train.select_dtypes(include=np.number)
    correlations = numeric_app_train.corr()['TARGET'].sort_values()

    # Remove the self-correlation of TARGET
    correlations = correlations.drop('TARGET')

    # Get the 15 most positive and 15 most negative correlations
    most_negative_corr = correlations.head(15)
    most_positive_corr = correlations.tail(15)
    top_corr_features = pd.concat([most_negative_corr, most_positive_corr])

    # --- 2. Plot Bar Chart ---
    plt.figure(figsize=(12, 10))
    top_corr_features.plot(kind='barh')
    plt.title('Top 30 Feature Correlations with TARGET', fontsize=16)
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    print("Displaying bar chart of feature correlations with TARGET...")
    plt.show()

    # --- 3. Plot Heatmap for Top 10 Features + TARGET ---
    # Get the absolute correlation values to find the most impactful features
    abs_correlations = correlations.abs().sort_values(ascending=False)

    # Get the names of the top 10 most correlated features
    top_10_features = abs_correlations.head(10).index.tolist()

    # Add 'TARGET' to the list for the heatmap
    heatmap_features = top_10_features + ['TARGET']

    # Create a correlation matrix for these specific features
    corr_matrix = numeric_app_train[heatmap_features].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Heatmap of Top 10 Correlated Features and TARGET', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    print("\nDisplaying heatmap for top 10 features and TARGET...")
    plt.show()

if __name__ == '__main__':
    plot_correlation_visuals()
