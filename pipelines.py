# %% [markdown]
# # College Expenditure Efficiency Prediction Pipeline
# ## Step One: 
# Based on institutional and demographic characteristics, can we predict a college's 
# expenditure efficiency?
# * The target variable is `exp_award_value`, or expenditures per award.
# * This is a regression problem where we will predict a continuous numeric value.
# * By predicting expenditure efficiency, colleges can experiment with resource allocation. 
# * A business metric could potentially be how much money a college saves by optimizing their
#   expenditure per award without negatively impacting student outcomes.

# %% [markdown]
# ### Imports

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from io import StringIO
import requests
college = pd.read_csv("https://github.com/UVADS/DS-3021/raw/main/data/cc_institution_details.csv")
college.head()
college.info()
college.describe()

# %% [markdown]
# ## Step Two: 

# %% [markdown]
# ### Data Cleaning and Preprocessing

# %%
vsa_cols = [col for col in college.columns if 'vsa' in col]
# all of the vsa columns have >40% missing values, so we will drop them

# Define specific columns to remove based on leakage, noise, or sparsity
manual_drop = [
    'index', 'unitid', 'chronname', 'nicknames', 'site', 'city',
    'similar',             # Identifiers
    'exp_award_percentile',
    'exp_award_state_value',
    'exp_award_natl_value',    # Leakage: These contain the answer/target
    'long_x', 'lat_y',
    'basic', 'counted_pct',  # Redundant or low-variance noise
    'med_sat_value',
    'med_sat_percentile',
    'endow_value',
    'endow_percentile'     # Sparse data (>40% missing)
]

# Combining lists into a single removal set
final_drop_list = vsa_cols + manual_drop

# Creat the cleaned df
college_clean = college.drop(columns=final_drop_list)
college_clean.info()

# %% [markdown]
# ### Convert Categorical Variables to category dtype

# %%
# Collapse state column
print(college_clean.state.value_counts())
# Collapse low occurance states into 'Other' category
state_counts = college_clean['state'].value_counts()

# Define a threshold for top states (I set it to 100 so there are 12 states plus Other)
top_states = state_counts[state_counts > 100].index.tolist()

# Create a new column with collapsed states
college_clean['state_group'] = college_clean['state'].apply(
    lambda x: x if x in top_states else 'Other'
).astype('category')

# Drop the original state column
college_clean = college_clean.drop(columns=['state'])
# view value counts of new state_group column
college_clean['state_group'].value_counts()

# %%
# Convert other categorical columns to the 'category' data type
cat_cols = ['state_group', 'level', 'control']
college_clean[cat_cols] = college_clean[cat_cols].astype('category')

# %%
# View the value counts of the new categorical columns
college_clean.level.value_counts() # 2 cats 
college_clean.control.value_counts() # 3 cats

# %% [markdown]
# ### Boolean Conversions

# %%
# The hbcu and flagship columns currently have 'X' for True and NaN for False.
# We convert the X to a 1 and the NaN to a 0.
bool_cols = ['hbcu', 'flagship']
for col in ['hbcu', 'flagship']:
    college_clean[col] = college_clean[col].fillna(0).replace('X', 1).astype(int)
college_clean.hbcu.value_counts()
college_clean.flagship.value_counts()


# %% [markdown]
# ### Handle Missing Values

# %%
missing_stats = college_clean.isna().sum()
print(missing_stats[missing_stats > 0])     
# For columns with a large number of missing values, we will impute with the median to 
# preserve more data.
# I use 13 because there is a large jump in missing values after that point (13 to 263).
median_cols = missing_stats[missing_stats > 13].index.tolist()
for col in median_cols:
    median_value = college_clean[col].median()
    college_clean[col] = college_clean[col].fillna(median_value)
# For the remaining columns with only a few missing values, we will drop those rows.
college_clean = college_clean.dropna()
college_clean.isna().sum()  # confirm no missing values remain

# %% [markdown]
# ### Feature Scaling

# %%
# Scale numeric columns using Min-Max Scaling
numeric_cols = list(college_clean.select_dtypes('number'))  # gives all float and int cols
college_clean[numeric_cols] = MinMaxScaler().fit_transform(college_clean[numeric_cols])
 # rewrite target variable so that it is not scaled
college_clean['exp_award_value'] = college['exp_award_value']
college_clean.head() # view the new scaled values 

# %% [markdown]
# ### One-Hot Encoding

# %%
category_list = list(college_clean.select_dtypes('category'))  # Find categorical columns
# drop_first=True to avoid having multiple columns that provide the same info
college_encoded = pd.get_dummies(college_clean, columns=category_list, drop_first=True)
print(college_encoded.columns)

# %% [markdown]
# ### Calculate Prevalence of Target Variable

# %%
# Visualize the distribution of the target variable
college_encoded.boxplot(column='exp_award_value', vert=False, grid=False)
plt.title('Boxplot of Expenditure per Award')
plt.show()

# %%
# Display summary statistics of the target variable
stats = college_encoded.exp_award_value.describe()
print(stats)

# %% 
# calculate the threshold for what constitutes a "High Expenditure" school.
# since the target variable isn't a binary class, we will separate the positive class 
# (high expenditure schools) based on the upper quartile of the distribution
upper_quartile = stats['75%']
print(f"Top Quartile Threshold: ${upper_quartile:,.2f} per award")

# %% 
# calculate the prevalence of the positive class, those that spend
# more than the average per award given.
mean_val = stats['mean']
# count how many schools are above the mean
above_avg_count = len(college_encoded[college_encoded.exp_award_value > mean_val])
prevalence_above_avg = above_avg_count / len(college_encoded)

print(f"Baseline (Mean Expenditure): ${mean_val:,.2f}")
print(f"Percentage of schools above the mean: {prevalence_above_avg:.2%}")

# %% [markdown]
# ### Train, Tune, Test Split

# %%
# First, create a binary feature for extremely high expenditure schools that
# are seen in the boxplot so that # they can be stratified in the splits.
college_encoded['high_exp_f'] = pd.cut(college_encoded.exp_award_value,
                                       bins=[-1, 77043, 6000000],   # splitting 75th percentile and above
                                       labels=[0, 1])   # 0 is not high exp, 1 is high exp
X = college_encoded.drop(['exp_award_value'], axis=1)
y = college_encoded['exp_award_value']  # y is the target variable

# %%
# Data partitions
# The first split is 60% train, 40% rest (tune + test)
train, rest = train_test_split(
    college_encoded, # We split the whole DF so we can keep features and targets aligned
    train_size=0.60,
    random_state=42,
    stratify=college_encoded.high_exp_f # Stratify based on high expenditure flag
)
# The second split splits the 'rest' into equal parts tune and test (20% each of the whole)
tune, test = train_test_split(
    rest,
    train_size=0.50,
    random_state=42,
    stratify=rest.high_exp_f
)
#. The high exp flag is no longer needed, so we drop it from all datasets.
for dataset in [train, tune, test]:
    dataset.drop('high_exp_f', axis=1, inplace=True)
    
# %%
# Verifying prevalence in each split (the mean and std)
train_mean = train.exp_award_value.mean()
train_std = train.exp_award_value.std()
print("Training set dist:")
print(f"Count: {len(train)}")
print(f"Mean:  ${train_mean:,.2f}")
print(f"Std:   ${train_std:,.2f}")

tune_mean = tune.exp_award_value.mean()
tune_std = tune.exp_award_value.std()
print("\nTuning set distribution:")
print(f"Count: {len(tune)}")
print(f"Mean:  ${tune_mean:,.2f}")
print(f"Std:   ${tune_std:,.2f}")

test_mean = test.exp_award_value.mean()
test_std = test.exp_award_value.std()
print("\nTest set distribution:")
print(f"Count: {len(test)}")
print(f"Mean:  ${test_mean:,.2f}")
print(f"Std:   ${test_std:,.2f}")

# %% [markdown]
"""
## Step Three:
My instincts tell me that this data may not be able to fully address the problem
of predicting expenditure efficiency and the value of spending per degree awarded.
Many schools that spend a lot per award may do so because they have high-quality
costly programs, like a medical school, or a low-spending school may not have 
efficient spending, but rather low-cost programs that do not require much funding.
I am slightly concerned about the high standard deviation of the target variable
in my data splits as well, for which I could consider dropping the extreme outliers. 
Finally, the median implementation of missing values may have introduced bias
into the dataset that could affect model performance, such as creating an 
artificial cluster of similar schools that really aren't similar at all.
"""


# %% [markdown]
# # Job Placement Prediction Pipeline
# ## Step One: 
# Based on academic performance and demographic cahracteristics, can we predict a 
# student's job placement status (placed vs not placed)?
# * The target variable is `status`, which indicates whether a student was placed in a job or not.
# * This is a classification problem where we will predict a categorical value.
# * By predicting job placement status, educational institutions can better address student needs for the workforce.
# * A business metric for this problem could be how placement rates increase across 
# instituions that utilize this model to address career placement and training.

# %%
# Imports
jobs = pd.read_csv("https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv")
jobs.head()
jobs.info()
jobs.describe()

# %% [markdown]
# ## Step Two: 

# %% [markdown]
# ### Data Cleaning and Preprocessing

# %%
jobs_drop = ['sl_no', 'salary']  # identifiers and salary reveals placement status
# create the cleaned df
jobs_clean = jobs.drop(columns=jobs_drop)
jobs_clean.info()

# %% [markdown]
# ### Convert Categorical Variables to category dtype

# %%
category_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
jobs_clean[category_cols] = jobs_clean[category_cols].astype('category')
jobs_clean.info()

# %%
# Make sure that the categorical variables don't have too many unique values
for col in category_cols:
    print(f"Value counts for {col}:")
    print(jobs_clean[col].value_counts())
    print("\n")

# %% [markdown]
# ### Handle Missing Values

# %%
missing_stats = jobs_clean.isna().sum()
print(missing_stats[missing_stats > 0])    
# This shows that there are no missing values in the dataset

# %% [markdown]
# ### Feature Scaling

# %%
# Scale numeric columns using Min-Max Scaling
num_cols = list(jobs_clean.select_dtypes('number'))  # gives all float and int cols
jobs_clean[num_cols] = MinMaxScaler().fit_transform(jobs_clean[num_cols])

jobs_clean.head() # view the new scaled values 

# %% [markdown]
# ### One-Hot Encoding

# %%
# Specifically encoding the target variable column so that it does not appear as two separate columns in the
# final encoded dataframe.
jobs_clean['status'] = jobs_clean['status'].map({'Placed': 1, 'Not Placed': 0})

cat_list = list(jobs_clean.select_dtypes('category').columns)   # Find categorical columns
if 'status' in cat_list:
    cat_list.remove('status') # Make sure target isn't in the get_dummies list
jobs_encoded = pd.get_dummies(jobs_clean, columns=cat_list, drop_first=True)
print(jobs_encoded.columns)

# %% [markdown]
# ### Calculate Prevalence of Target Variable

# %%
# Visualize the distribution of the target variable
sns.countplot(x='status', data=jobs_encoded)
plt.title('Count of Job Placement Status')
plt.show()

# %%
# Calculate the counts for placement status
status_counts = jobs_encoded['status'].value_counts()
total_students = len(jobs_encoded)

# %% 
# Calculate the prevalence (the Positive Class of placed students).
placed_count = status_counts[1] # Count of students who were placed
prevalence_placed = placed_count / total_students

print(f"Total Students: {total_students}")
print(f"Number of Students Placed: {placed_count}")
print(f"Placement Prevalence (Baseline Accuracy): {prevalence_placed:.2%}")

# %% [markdown]
# ### Train, Tune, Test Split

# %%
# Separate features and target variable
X = jobs_encoded.drop(['status'], axis=1)
y = jobs_encoded['status']

# %%
# Data partitions
# The first split is 60% train, 40% rest (tune + test)
train, rest = train_test_split(
    jobs_encoded, 
    train_size=0.60,
    random_state=42,
    stratify=jobs_encoded.status # Stratify directly on the target!
)
# The second split splits the 'rest' into equal parts tune and test (20% each of the whole)
tune, test = train_test_split(
    rest,
    train_size=0.50,
    random_state=42,
    stratify=rest.status
)
    
# %%
# Verify prevalence in each split (the mean and std)
train_counts = train.status.value_counts(normalize=True)
print("Training set distribution:")
print(f"  Count: {len(train)}")
print(f"  Placement Prevalence (Class 1): {train_counts.get(1, 0):.2%}")

tune_counts = tune.status.value_counts(normalize=True)
print("\nTuning set distribution:")
print(f"  Count: {len(tune)}")
print(f"  Placement Prevalence (Class 1): {tune_counts.get(1, 0):.2%}")

test_counts = test.status.value_counts(normalize=True)
print("\nTest set distribution:")
print(f"  Count: {len(test)}")
print(f"  Placement Prevalence (Class 1): {test_counts.get(1, 0):.2%}")

print( "\nData partitioning is pretty good, with prevalence close to the overall dataset in all splits." )

# %% [markdown]
"""
## Step Three:
My instincts tell me that this data will be able to fairly accurately address
the problem of student job placement prediction. Through the given degree types, specializations
and grades throughout their academic career, I think trends will generally appear
that dictate whether a student is likely to be placed in a job or not. However,
there is definitely the potential for confounding factors since this target is 
just a binary placement indicator, and does not take into account the possibility
of students having skills or economic circumstances outside of this dataset that may help them 
get placed in a job.
"""
