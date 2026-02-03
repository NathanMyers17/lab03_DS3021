import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from io import StringIO
import requests

# %% [markdown]
# ## Function to Drop Columns from a DataFrame

# %%
def drop_columns(df, columns):
    """
    Drops specified columns from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to drop.
    
    Returns:
    pd.DataFrame: DataFrame with specified columns dropped.
    """
    print(df.info())
    return df.drop(columns=columns)

# %% [markdown]
# ## Function to convert categorical columns to categoriy dtype

# %%
def convert_to_category(df, columns):
    """
    This function converts the columns given to a category dtype (from string or int or other dtypes).
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to convert to category dtype.
    
    Returns: 
    pd.DataFrame: DataFrame with specified columns converted to category dtype.
    """
    for col in columns:
        df[col] = df[col].astype('category')
    return df

# %% [markdown]
# ## Function to convert columsn to boolean if they contain two unique values.
# These values could be X and NaN, 0 and 1, True and False, etc.

# %%
def convert_to_boolean(df, columns = None):
    """
    This function converts columns with exactly two unique values to boolean dtype.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: DataFrame with applicable columns converted to boolean dtype.
    """
    for col in df.columns:
        if df[col].nunique() == 2:  # this selects the columns that only have 2 unique values to be converted to boolean
            df[col] = df[col].astype('bool')
    return df

# %% [markdown]
# ## Function to handle missing values
# We need to get the columns that have missing values and then decide what to do with them based on how many values they are, what the column is, etc.

# %%
def handle_missing_values(df, threshold=0):
    """
    This function handles missing values by giving columns with more than the 'threshold' number of missing values
    a median imput if they are numeric and dropping the rows for columns with missing values less than or equal to the threshold.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    threshold (int): The number of missing values above which we use median imputation. The default threshold is 0.
    
    Returns:
    pd.DataFrame: DataFrame with no missing values. (some converted to median and some rows dropped)
    """
    missing_stats = df.isna().sum() # get the count of missing values per column
    
    # the columns that will receive median imputation instead of being dropped have more missing values than the threshold
    # filtering to only the numeric columns so that categories are not included
    median_cols = missing_stats[(missing_stats > threshold)].index.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    cols_above_threshold = [col for col in median_cols if col in numeric_cols]
    
    # this loop applies the median to the selected columns, filling every Nan  with the median value of that column
    for col in cols_above_threshold:
        median_value = df[col].median() # selecting the median value of the column indexed by 'col'
        df[col] = df[col].fillna(median_value)
    
    # finally drop all remaining rows with any NaN values, which are the cols with missing values less than or equal to the threshold
    # and the non-numeric columns with missing values above the threshold
    df = df.dropna()
    
    return df

# %% [markdown]
# ## Function to scale numerical columns
# In this function we will scale based on min-max scaling

# %%
def scale_numerical_columns(df, target_column = None, columns=None):
    """
    This function scales numerical columns using Min-Max scaling.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_col (str): The name of the target variable to exclude from scaling. Default is no excluded variable. 
    columns (list): List of column names to scale. If None, all numerical columns will be scaled.
    
    Returns:
    pd.DataFrame: DataFrame with specified numerical columns scaled, excluding the target variable if specified.
    """
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    if target_column is not None and target_column in columns:
        columns.remove(target_column)   # this makes sure that the target column is not scaled if it is specified
        
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])

    return df      
  
# %% [markdown]
# ## Function to one-hot-encode categorical columns
# In this function we will use pandas get_dummies to one-hot-encode categorical columns

# %%
def one_hot_encode(df, columns=None):
    """
    This function one-hot-encodes specified categorical columns using pandas get_dummies.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to one-hot-encode. If no columns are specified, all categorical columns will be encoded.
        if there are specified columns, they should be of categorical dtype.
            
    Returns:
    pd.DataFrame: DataFrame with specified categorical columns one-hot-encoded.
    """
    if columns is None:
        columns = df.select_dtypes(include=['category']).columns.tolist()   # this converts all categorical columns if none are specified
    
    df = pd.get_dummies(df, columns=columns, drop_first=True)   # drop_first=True to avoid having multiple columns that provide the same info
    
    return df

# %% [markdown]
# ## Function to Calculate Prevalence of Target Variable
# This function will output the prevalence of the positive class in the target variable.

# %%
def calculate_prevalence(df, target_column):
    """
    This function calculates the prevalence of the positive class in the target variable.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target variable column to calculate prevalence for.
    
    Returns:
    the prevalence of the positive class in the target variable.
    """
    if df[target_column].nunique() == 2:    # If the target variable is binary, the prevalence will equal the mean
        prevalence = df[target_column].mean()
        print(f"Classification Prevalence: {prevalence:.2%}")
        return prevalence
    
    else:   # for regression or non-binary target variables, we will calculate the mean as a baseline to compare to (like with the college expenditure variable)
        avg_val = df[target_column].mean()
        print(f"Regression Baseline (Mean): ${avg_val:,.2f}")
        return avg_val

# %% [markdown]
# ## Function to Split Data into Train, Tune, and Test Sets

# %% 
def split_data(df, target_column, stratify_column=None, test_size=0.2, tune_size=0.2, random_state=42):
    """
    This function splits the DataFrame into train, tune, and test sets.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target variable column.
    test_size (float): Proportion of the dataset to include in the test set. Default is 0.2.
    tune_size (float): Proportion of the training set to include in the tune set. Default is 0.2.
    random_state (int): Random seed for reproducibility. Default is 42 (random number)
    
    Returns:
    The train, tune, and test DataFrames.
    """
    # Decide which column to use for stratification - either the specified stratify_column or the target_column if no stratify_column is provided
    s_values = df[stratify_column] if stratify_column is not None else df[target_column]
    
    # First I split the data into a combined train+tune set and a test set, stratifying by the chosen column
    train_tune_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=s_values, 
        random_state=random_state
    )
    
    # Identify stratification values for the second split from the remaining data
    s_values_rel = train_tune_df[stratify_column] if stratify_column is not None else train_tune_df[target_column]
    
    # Calculate relative size for tune set
    tune_rel = tune_size / (1 - test_size)
    
    # Split the (train + tune) set into separate train and tune sets
    train_df, tune_df = train_test_split(
        train_tune_df, 
        test_size=tune_rel, 
        stratify=s_values_rel, 
        random_state=random_state
    )
    
    return train_df, tune_df, test_df