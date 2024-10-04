
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler  # or MinMaxScaler
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, roc_auc_score, make_scorer
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier, VotingRegressor
import category_encoders as ce
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc


# Detailed and Quick parameter grids for classification
models_classification = {
    'CatBoost': (CatBoostClassifier(verbose=0), {
        'detailed': {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8]
        },
        'quick': {
            'iterations': [50, 100],
            'learning_rate': [0.1],
            'depth': [4, 6]
        }
    }),
    'LightGBM': (LGBMClassifier(), {
        'detailed': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127]
        },
        'quick': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1],
            'num_leaves': [31, 63]
        }
    }),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
        'detailed': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 9]
        },
        'quick': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1],
            'max_depth': [3, 6]
        }
    })
}

# Detailed and Quick parameter grids for regression
models_regression = {
    'CatBoost': (CatBoostRegressor(verbose=0), {
        'detailed': {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8]
        },
        'quick': {
            'iterations': [50, 100],
            'learning_rate': [0.1],
            'depth': [4, 6]
        }
    }),
    'LightGBM': (LGBMRegressor(), {
        'detailed': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 63, 127]
        },
        'quick': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1],
            'num_leaves': [31, 63]
        }
    }),
    'XGBoost': (XGBRegressor(use_label_encoder=False, eval_metric='logloss'), {
        'detailed': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 9]
        },
        'quick': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1],
            'max_depth': [3, 6]
        }
    })
}


def df_summary(df):
    """
    Function to display basic information about a DataFrame and return the results in a dictionary.

    Parameters:
    df (pd.DataFrame): The DataFrame to summarize.

    Returns:
    dict: A dictionary containing the summary statistics, counts of NaN, inf, and -inf values,
          correlations, and unique counts for categorical columns.
    """

    summary_dict = {}
    df = df.copy()

    # Basic info
    print("Basic Information:")
    df.info()

    # Summary statistics for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    print("\nSummary Statistics for Numeric Columns:")
    summary_statistics = numeric_df.describe()
    summary_dict['summary_statistics'] = summary_statistics
    print(summary_statistics)

    # Check for NaN values
    nan_counts = df.isna().sum()
    columns_with_nan = nan_counts[nan_counts > 0]
    summary_dict['columns_with_nan'] = columns_with_nan.to_dict()
    print("\nColumns with NaN Values (count per column):")
    print(summary_dict['columns_with_nan'])

    # Check for Inf values
    inf_counts = numeric_df.apply(lambda x: (x == np.inf).sum())
    neg_inf_counts = numeric_df.apply(lambda x: (x == -np.inf).sum())

    # Combine inf and -inf counts into a single dictionary
    combined_inf_counts = {
        col: inf_counts[col] + neg_inf_counts[col]
        for col in inf_counts.index if inf_counts[col] > 0 or neg_inf_counts[col] > 0
    }
    summary_dict['columns_with_inf'] = combined_inf_counts
    print("\nCombined Inf and -Inf Values (count per column):")
    print(summary_dict['columns_with_inf'])

    # Unique values count for categorical columns
    categorical_df = df.select_dtypes(include=['object', 'category'])
    unique_counts = {col: categorical_df[col].nunique() for col in categorical_df}
    summary_dict['unique_counts'] = unique_counts
    print("\nUnique Values Count for Categorical Columns:")
    print(unique_counts)

    # Correlation matrix for numeric columns
    print("\nCorrelation Matrix for Numeric Columns:")
    correlation_matrix = numeric_df.corr()
    summary_dict['correlation_matrix'] = correlation_matrix
    print(correlation_matrix)

    # Value counts for categorical columns
    value_counts = {col: categorical_df[col].value_counts().to_dict() for col in categorical_df}
    summary_dict['value_counts'] = value_counts

    return summary_dict

def univariate_analysis(df, sample_size=None, max_unique_categories=20):
    """
    Function to perform univariate analysis and generate plots for both
    numeric and categorical columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    sample_size (int, optional): If specified, use a sample of this size for analysis.
    max_unique_categories (int, optional): Maximum number of unique values in categorical columns to plot.
    """
    # Set the aesthetic style of the plots
    df = df.copy()
    sns.set(style="darkgrid")

    # Sample the data if a sample size is specified
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=1)

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create a figure for numeric plots
    num_plots = len(numeric_cols)
    plt.figure(figsize=(15, num_plots * 4))  # Adjust height based on number of numeric columns

    # Histogram and Box Plot for Numeric Columns
    for i, col in enumerate(numeric_cols):
        plt.subplot(num_plots, 2, 2*i + 1)
        sns.histplot(df[col], bins=15, kde=False)  # Reduced number of bins and disabled KDE
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

        plt.subplot(num_plots, 2, 2*i + 2)
        sns.boxplot(x=df[col])
        plt.title(f'Box Plot of {col}')
        plt.xlabel(col)

    plt.tight_layout(pad=3.0)  # Add padding for the numerical plots
    plt.show()

    # Track categorical columns that will not be plotted
    skipped_categorical_cols = []

    # Count Plot for Categorical Columns
    for col in categorical_cols:
        if df[col].nunique() <= max_unique_categories:
            plt.figure(figsize=(10, 5))
            sns.countplot(y=df[col], order=df[col].value_counts().index)
            plt.title(f'Count Plot of {col}')
            plt.xlabel('Count')
            plt.ylabel(col)
            plt.tight_layout(pad=3.0)  # Add padding
            plt.show()
        else:
            skipped_categorical_cols.append(col)

    # Print skipped categorical columns
    if skipped_categorical_cols:
        print(f"Skipped categorical columns with more than {max_unique_categories} unique values: {skipped_categorical_cols}")

def bivariate_analysis(df, target_column, sample_size=None, max_unique_categories=20,
                       normalize=True, iqr_percentage=150):
    """
    Function to perform bivariate analysis between a target column
    and all other columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    target_column (str): The name of the target column for analysis.
    sample_size (int, optional): If specified, use a sample of this size for analysis.
    max_unique_categories (int, optional): Maximum number of unique values in categorical columns to plot.
    normalize (bool, optional): If True, normalize the counts in count plots.
    iqr_percentage (float, optional): Percentage of IQR to include for outlier removal (default is None).
    """
    # Check if the target column exists
    if target_column not in df.columns:
        print(f"Column '{target_column}' not found in DataFrame.")
        return

    df = df.copy()

    # Sample the data if a sample size is specified
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=1)

    # Identify the type of the target column
    target_dtype = df[target_column].dtype
    if np.issubdtype(target_dtype, np.number) and df[target_column].nunique() <= max_unique_categories:
        df[target_column] = df[target_column].astype('str')
        target_dtype = df[target_column].dtype

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove the target column from the lists
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    elif target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # Track skipped categorical columns
    skipped_categorical_cols = []

    # Function to remove outliers based on IQR
    def remove_outliers(data, column, iqr_percentage):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (iqr_percentage / 100) * IQR
        upper_bound = Q3 + (iqr_percentage / 100) * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

    # Bivariate analysis depending on the type of the target column
    if np.issubdtype(target_dtype, np.number):  # Target is numeric
        # Scatter plots with numeric columns
        for col in numeric_cols:
            plt.figure(figsize=(10, 5))
            sns.scatterplot(x=df[col], y=df[target_column])
            plt.title(f'Scatter Plot: {target_column} vs {col}')
            plt.xlabel(col)
            plt.ylabel(target_column)
            plt.tight_layout()
            plt.show()

        # Box plots with categorical columns
        for col in categorical_cols:
            if df[col].nunique() <= max_unique_categories:
                plot_data = df.copy()

                # Remove outliers only if iqr_percentage is less than 100
                if iqr_percentage is not None:
                    plot_data = remove_outliers(plot_data, target_column, iqr_percentage)

                plt.figure(figsize=(10, 5))
                sns.boxplot(x=plot_data[col], y=plot_data[target_column])
                plt.title(f'Box Plot: {target_column} by {col}')
                plt.xlabel(col)
                plt.ylabel(target_column)
                plt.tight_layout()
                plt.show()
            else:
                skipped_categorical_cols.append(col)

    else:  # Target is categorical
        # Box plots with numeric columns
        for col in numeric_cols:
            plot_data = df.copy()

            # Remove outliers only if iqr_percentage is less than 100
            if iqr_percentage is not None:
                plot_data = remove_outliers(plot_data, col, iqr_percentage)

            plt.figure(figsize=(10, 5))
            sns.boxplot(x=plot_data[target_column], y=plot_data[col])
            plt.title(f'Box Plot: {col} by {target_column}')
            plt.xlabel(target_column)
            plt.ylabel(col)
            plt.tight_layout()
            plt.show()

        # Bar plots with categorical columns
        for col in categorical_cols:
            if col != target_column:  # Ensure we don't plot against itself
                if df[col].nunique() <= max_unique_categories:
                    plt.figure(figsize=(10, 5))
                    count_data = df.groupby([col, target_column]).size().reset_index(name='Counts')

                    if normalize:
                        count_data['Proportion'] = count_data.groupby(col)['Counts'].transform(lambda x: x / x.sum())
                        sns.barplot(x=col, y='Proportion', hue=target_column, data=count_data)
                        plt.ylabel('Proportion')
                    else:
                        sns.countplot(x=col, hue=target_column, data=df)
                        plt.ylabel('Count')

                    plt.title(f'Count Plot: {col} colored by {target_column}')
                    plt.xlabel(col)
                    plt.tight_layout()
                    plt.show()
                else:
                    skipped_categorical_cols.append(col)

    # Print skipped categorical columns
    if skipped_categorical_cols:
        print(
            f"Skipped categorical columns with more than {max_unique_categories} unique values: {skipped_categorical_cols}")


def fast_correlation_heatmap(df, columns=None):
    """
    Generates a fast correlation heatmap for specified numeric variables in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing numeric variables.
    columns (list, optional): List of column names to include in the heatmap.
                              If None, includes all numeric columns.
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # If specific columns are provided, filter the DataFrame
    if columns is not None:
        # Ensure all specified columns are numeric and exist in the DataFrame
        columns = [col for col in columns if col in numeric_df.columns]
        if not columns:
            print("No valid numeric columns provided.")
            return
        numeric_df = numeric_df[columns]

    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))

    # Generate a heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                square=True, cbar_kws={"shrink": .8}, linewidths=.5,
                vmax=1, vmin=-1)  # Set the color limits to speed up rendering

    # Set title
    plt.title('Correlation Heatmap', fontsize=16)

    # Show the plot
    plt.tight_layout()
    plt.show()

def impute_outliers(df, method_dict=None, k_neighbors=10, drop_outlier_cols=True, knn_features=None, normalize=True):
    """
    Imputes or drops outliers from the DataFrame based on the specified methods for each column,
    using the outlier indicators appended to the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing numeric variables with outlier indicators.
    method_dict (dict): A dictionary where keys are tuples of column names and values are methods
                        to handle outliers ('drop', 'mean', 'median', 'knn').
    k_neighbors (int): Number of neighbors for KNN imputation.
    drop_outlier_cols (bool): If True, drops the outlier indicator columns after handling.
    knn_features (dict or None): A dictionary where keys are the columns to be imputed and values are lists of
                                  column names to be used for KNN imputation. If None, uses all numeric columns.
    normalize (bool): If True, normalizes the features before KNN imputation.

    Returns:
    pd.DataFrame: The DataFrame after handling outliers.
    """
    df_copy = df.copy()

    # Identify numeric columns
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()

    # Filter out columns that contain '_outlier'
    numeric_cols = [c for c in numeric_cols if '_outlier' not in c]

    # Create outlier indicator columns
    outlier_cols = [f"{col}_outlier" for col in numeric_cols]

    # Handle outliers according to the specified methods in method_dict
    for cols, method in method_dict.items():
        for col in cols:
            if col in numeric_cols:
                outlier_col = f"{col}_outlier"
                outlier_indices = df_copy[df_copy[outlier_col] == 1].index.tolist()

                if method == 'drop':
                    df_copy = df_copy.drop(index=outlier_indices)
                elif method in ['mean', 'median']:
                    if outlier_indices:
                        value = df_copy[col].mean() if method == 'mean' else df_copy[col].median()
                        df_copy.loc[outlier_indices, col] = value
                elif method == 'knn':
                    # Only proceed if there are outlier indices
                    print(f'process knn for {col}')
                    if outlier_indices:
                        # Set outlier values to NaN in the target column
                        df_copy.loc[outlier_indices, col] = np.nan

                        # Determine features to use for KNN
                        if knn_features is None:
                            features = numeric_cols  # Use all numeric columns if knn_features is None
                        else:
                            features = knn_features.get(col,
                                                        numeric_cols)  # Use specified features or use all numeric columns
                            features = list(set(features + [col]))

                        # Normalize the features if needed
                        if normalize:
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(df_copy[features])
                        else:
                            scaled_data = df_copy[features].values

                        # Prepare the data for KNN
                        imputer = KNNImputer(n_neighbors=k_neighbors)

                        # Fit and transform KNN on the data, excluding outliers
                        imputed_values = imputer.fit_transform(scaled_data)

                        if normalize:
                            imputed_values = scaler.inverse_transform(imputed_values)

                        # Fill the target column with the imputed values
                        df_copy.loc[outlier_indices, col] = imputed_values[
                            outlier_indices, df_copy.columns.get_loc(col)]

    # Optionally drop the outlier indicator columns
    if drop_outlier_cols:
        drop_cols = [c for c in outlier_cols if c in df_copy.columns]
        df_copy = df_copy.drop(columns=drop_cols)

    return df_copy


def univariate_feature_selection(df, target_column):
    df = df.copy()
    results = {}

    # Identify target variable type
    target_type = df[target_column].dtype

    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Analyzing Numeric Features
    if np.issubdtype(target_type, np.number):
        # If target is numeric, use correlation coefficients
        for col in numeric_cols:
            if col != target_column:
                # Pearson correlation
                pearson_corr, pearson_p = stats.pearsonr(df[col], df[target_column])
                # Spearman correlation
                spearman_corr, spearman_p = stats.spearmanr(df[col], df[target_column])
                results[col] = {
                    'Pearson Correlation': pearson_corr,
                    'Pearson P-Value': pearson_p,
                    'Spearman Correlation': spearman_corr,
                    'Spearman P-Value': spearman_p,
                    'Significant': (pearson_p < 0.05) or (spearman_p < 0.05)
                }

    elif target_type in ['category', 'object']:
        # If target is categorical, use ANOVA for numeric features
        for col in numeric_cols:
            if col != target_column:
                f_value, p_value = stats.f_oneway(*[df[col][df[target_column] == category] for category in df[target_column].unique()])
                results[col] = {
                    'F-Value': f_value,
                    'P-Value': p_value,
                    'Significant': p_value < 0.05
                }

    # Analyzing Categorical Features
    if target_type in ['category', 'object']:
        for col in categorical_cols:
            if col != target_column:
                contingency_table = pd.crosstab(df[col], df[target_column])
                chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
                results[col] = {
                    'Chi-Squared': chi2,
                    'P-Value': p_value,
                    'Significant': p_value < 0.05
                }
    elif np.issubdtype(target_type, np.number) and df[target_column].nunique() == 2:
        # If target is binary, use Point Biserial Correlation for categorical features
        for col in categorical_cols:
            if col != target_column:
                # Convert categorical to numeric (0, 1)
                df[col] = pd.Categorical(df[col]).codes
                correlation, p_value = stats.pointbiserialr(df[target_column], df[col])
                results[col] = {
                    'Point Biserial Correlation': correlation,
                    'P-Value': p_value,
                    'Significant': p_value < 0.05
                }

    return pd.DataFrame(results).T


def encode_categorical_features(X_train, X_test, one_hot_cols, target_encode_cols, y_train):
    # Get all categorical columns if no target encode columns are specified
    if target_encode_cols == 'all':
        target_encode_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # One-hot encoding
    if one_hot_cols:
        X_train = pd.get_dummies(X_train, columns=one_hot_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=one_hot_cols, drop_first=True)

        # Align columns after one-hot encoding
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Target encoding
    if target_encode_cols:
        encoder = ce.TargetEncoder(cols=target_encode_cols)

        X_train = encoder.fit_transform(X_train, y_train)
        X_test = encoder.transform(X_test)

    return X_train, X_test


def train_baseline_model(X_train, y_train, is_classification=True, l1_param=1.0):
    """
    Train a baseline model with optional L1 regularization and scaling.

    Parameters:
    - X_train: Features for training
    - y_train: Target variable for training
    - is_classification: Boolean indicating if the task is classification (True) or regression (False)
    - l1_param: Regularization strength for L1 (default is 1.0, set to 0 for no regularization)

    Returns:
    - model: Trained model
    """
    if is_classification:
        if l1_param > 0:
            model = LogisticRegression(penalty='l1', C=1 / l1_param, solver='liblinear')
        else:
            model = LogisticRegression(penalty='none')  # No regularization
    else:
        if l1_param > 0:
            model = Lasso(alpha=l1_param)  # Use Lasso for regression with L1 regularization
        else:
            model = LinearRegression()  # No regularization for Linear Regression

    # Create a pipeline with scaling and the model
    pipeline = make_pipeline(StandardScaler(), model)

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    return pipeline

def baseline_coefficient(pipeline):
    # Access the model coefficients
    if isinstance(pipeline.named_steps['logisticregression'], LogisticRegression):
        model = pipeline.named_steps['logisticregression']
    elif isinstance(pipeline.named_steps['lasso'], Lasso):
        model = pipeline.named_steps['lasso']
    else:
        model = pipeline.named_steps['linearregression']

    # Get coefficients
    coefficients = model.coef_
    intercept = model.intercept_

    print("Coefficients:", coefficients)
    print("Intercept:", intercept)
    return coefficients, intercept


def performance_report(y_test, y_pred, is_classification):
    if is_classification:
        # Convert probabilities to class predictions
        y_pred_class = (y_pred[:, 1] >= 0.5).astype(int)  # Assuming binary classification

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred_class)
        print(f'Accuracy: {accuracy:.2f}')

        # Classification report
        cl_report = classification_report(y_test, y_pred_class)
        print('Classification Report:\n', cl_report)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_class)
        print('Confusion Matrix:\n', conf_matrix)

        return accuracy, cl_report, conf_matrix

    else:
        # Calculate regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'Mean Absolute Error: {mae:.2f}')
        print(f'Mean Squared Error: {mse:.2f}')
        print(f'R-squared: {r2:.2f}')

        return mae, mse, r2


def performance_visualize(y_test, y_pred, is_classification):
    if is_classification:
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        conf_matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))  # Use argmax for class predictions
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred[:, 1])  # Use probabilities for positive class
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    else:
        # Predicted vs Actual
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted')
        plt.show()

        # Residuals
        residuals = y_test - y_pred

        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs. Predicted Values')
        plt.show()

    return


def evaluate_baseline_model(model, X_train, y_train, X_test, y_test, is_classification):
    if is_classification:
        train_preds = model.predict_proba(X_train)[:, 1]
        test_preds = model.predict_proba(X_test)[:, 1]
        train_metric = roc_auc_score(y_train, train_preds)
        test_metric = roc_auc_score(y_test, test_preds)
    else:
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        train_metric = mean_squared_error(y_train, train_preds)
        test_metric = mean_squared_error(y_test, test_preds)

    return train_metric, test_metric

def check_overfitting(train_metric, test_metric, ratio = 1.1):
    if train_metric > test_metric * ratio:  # Adjust the threshold as needed
        print("Warning: Possible overfitting detected!")
    else:
        print("Model performance is consistent between training and test sets.")


# Custom logging function for GridSearchCV
def log_loss_function(y_true, y_pred):
    loss = mean_squared_error(y_true, y_pred)
    print(f"Current Loss: {loss:.4f}")
    return loss


# Step 3: Define and Optimize Ensemble Models with Two Hyperparameter Sets
def optimize_model(model, param_grid, X_train, y_train, is_classification, tscv = False):
    if tscv:
        cvs = TimeSeriesSplit(n_splits=3)
    else:
        cvs = 3
    grid_search = GridSearchCV(model, param_grid, cv=cvs,
                               scoring='neg_mean_squared_error' if not is_classification else 'roc_auc', n_jobs=-1,
                               verbose=10)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


# Function to optimize models with a specified hyperparameter set
def optimize_all_models(models, param_set, X_train, y_train, is_classification, tscv = False):
    best_models = {}
    for model_name, (model, params) in models.items():
        best_model, best_params = optimize_model(model, params[param_set], X_train, y_train, is_classification,
                                                 tscv = tscv)
        best_models[model_name] = best_model
        print(f'Best {model_name} Parameters ({param_set}): {best_params}')
    return best_models

def ensemble_predict(models, X, is_classification):
    # Generate predictions using each individual model
    if is_classification:
        # Get predicted probabilities for classification models
        predictions = np.array([model.predict_proba(X)[:, 1] for model in models.values()])
        ensemble_preds = np.mean(predictions, axis=0)  # Average probabilities for final prediction
    else:
        # Get predictions for regression models
        predictions = np.array([model.predict(X) for model in models.values()])
        ensemble_preds = np.mean(predictions, axis=0)  # Average predictions for final output

    return ensemble_preds

# Step 5: Evaluate Ensemble Model and Individual Models
def evaluate_multi_models(best_models, y_train, y_test, train_ensemble_preds, test_ensemble_preds,
                          X_train_encoded, X_test_encoded, is_classification):
    results = {}

    # Evaluate ensemble model
    if is_classification:
        ensemble_train_metric = roc_auc_score(y_train, train_ensemble_preds)
        ensemble_test_metric = roc_auc_score(y_test, test_ensemble_preds)
    else:
        ensemble_train_metric = mean_squared_error(y_train, train_ensemble_preds)
        ensemble_test_metric = mean_squared_error(y_test, test_ensemble_preds)

    results['Ensemble'] = {
        'Train Metric': ensemble_train_metric,
        'Test Metric': ensemble_test_metric,
        'Train Predictions': train_ensemble_preds,
        'Test Predictions': test_ensemble_preds
    }

    # Evaluate individual models
    for model_name, model in best_models.items():
        if is_classification:
            train_preds_model = model.predict_proba(X_train_encoded)[:, 1]
            test_preds_model = model.predict_proba(X_test_encoded)[:, 1]
            train_metric = roc_auc_score(y_train, train_preds_model)
            test_metric = roc_auc_score(y_test, test_preds_model)
        else:
            train_preds_model = model.predict(X_train_encoded)
            test_preds_model = model.predict(X_test_encoded)
            train_metric = mean_squared_error(y_train, train_preds_model)
            test_metric = mean_squared_error(y_test, test_preds_model)

        results[model_name] = {
            'Train Metric': train_metric,
            'Test Metric': test_metric,
            'Train Predictions': train_preds_model,
            'Test Predictions': test_preds_model
        }

    return results


# Step 5: Plotting Results
def plot_model_performance(results, is_classification):
    metrics = []

    for model_name, metrics_data in results.items():
        metrics.append({
            'Model': model_name,
            'Train Metric': metrics_data['Train Metric'],
            'Test Metric': metrics_data['Test Metric']
        })

    metrics = pd.DataFrame(metrics)
    metrics.set_index('Model').plot(kind='bar', figsize=(10, 5))
    plt.title('Model Performance Comparison')
    plt.ylabel('AUC' if is_classification else 'MSE')
    plt.xticks(rotation=45)
    plt.axhline(y=0.5, color='r', linestyle='--')  # Reference line for AUC
    plt.show()

    return metrics


def identify_outliers(df, visualize=False):
    """
    Identifies outliers in the DataFrame using the IQR method and appends
    outlier indicators to the original DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing numeric variables.
    visualize (bool): If True, visualizes the outliers for each numeric feature.

    Returns:
    pd.DataFrame: The original DataFrame with added columns indicating outliers.
    """

    df_outliers = df.copy()

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Create a new column indicating outliers
        outlier_col_name = f"{col}_outlier"
        df_outliers[outlier_col_name] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)

        # Count the number of outliers found
        num_outliers = df_outliers[outlier_col_name].sum()
        print(f"Number of outliers found in '{col}': {num_outliers}")

        if visualize:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=df[col])
            plt.title(f'Box plot for {col} - Outliers marked')
            plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
            plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound')
            plt.legend()
            plt.show()

    return df_outliers


def evaluate_model(model, X_train, y_train, X_test, y_test, is_classification):
    model.fit(X_train, y_train)  # Suppress output for quicker runs
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred) if is_classification else np.sqrt(mean_squared_error(y_test, y_pred))


def create_param_grid(is_classification, unique_classes):
    common_params = {
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': -1,
        'min_data_in_leaf': 10,
        'feature_fraction': 1.0,
        'n_estimators': 50
    }

    if is_classification:
        if len(unique_classes) > 2:  # More than 2 classes
            return {**common_params,
                    'objective': 'multiclass',
                    'metric': 'multi_logloss'}, LGBMClassifier()
        else:  # Binary classification
            return {**common_params,
                    'objective': 'binary',
                    'metric': 'binary_logloss'}, LGBMClassifier()
    else:
        return {**common_params,
                'objective': 'regression',
                'metric': 'rmse'}, LGBMRegressor()


def iterative_feature_selection(X_train, y_train, is_classification, core_features=None, tscv = False):
    # Perform stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train,
        shuffle= not tscv
    )

    # Check unique classes in training set
    if is_classification:
        unique_classes = np.unique(y_train)
        if len(unique_classes) <= 1:
            raise ValueError("Training data must contain more than one class for classification.")
        print("Unique classes in training set:", unique_classes)

    # Initial model with core features
    X_train_core = X_train[core_features]
    X_val_core = X_val[core_features]

    # Create parameter grid and model
    params_grid, model = create_param_grid(is_classification, unique_classes)

    # Fit the model with the parameters set during initialization
    model.fit(X_train_core, y_train)

    # Evaluate initial model on validation set
    initial_score = evaluate_model(model, X_train_core, y_train, X_val_core, y_val, is_classification)
    print(f'Initial Score with Core Features on Validation Set: {initial_score:.2f}')

    # Test additional features
    additional_features = [col for col in X_train.columns if col not in core_features]
    best_score = initial_score
    improved_features = []

    for feature in additional_features:
        print(f'Trying {feature}')
        X_train_new = X_train_core.join(X_train[feature])
        model_new = model.__class__(**params_grid)  # Create a new model with the same params
        new_score = evaluate_model(model_new, X_train_new, y_train, X_val_core.join(X_val[feature]), y_val,
                                   is_classification)

        print(f'Score with {feature} added: {new_score:.2f}')

        if new_score > best_score if is_classification else new_score < best_score:
            improved_features.append(feature)
            best_score = new_score

    print(f'Features that improved performance: {improved_features}')
    return model, best_score, improved_features, initial_score

def lgb_feature_importance(best_model, plot = False):
    # Plot feature importance for the best model
    feature_names = best_model.feature_name_
    importance = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    if plot:
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title('Feature Importance for Best Model')
        plt.gca().invert_yaxis()  # Reverse the y-axis
        plt.show()

    return feature_importance_df