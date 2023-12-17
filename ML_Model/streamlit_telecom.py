#!/usr/bin/env python
# coding: utf-8
#Basic libraries
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
warnings.filterwarnings('ignore')
import psycopg2
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
# PostgreSQL connection parameters
host = "localhost"
user = "postgres"
password = "mariam21"
database = "telecom_db"
table_name = "xdr_data"
# Construct the connection string
conn_str = f"postgresql+psycopg2://{user}:{password}@{host}/{database}"

# Create a database engine
engine = create_engine(conn_str)

# Example query to retrieve data into a pandas DataFrame
query = f"SELECT * FROM {table_name};"
df = pd.read_sql(query, engine)
print('number of rows', df.shape[0])
print('number of columns', df.shape[1])

df.head()

df.tail()
df.columns

df.info()
df.describe()

df.dtype
# list of numerical variables
numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
print('Number of numerical variables: ', len(numerical_features))
# visualise the numerical variables
df[numerical_features].sample(5)

df.shape

df.isnull().any()

df.isnull().sum() 

# we should imput the missing avlue
# numerical features imput by mean and categorical features are imputed by mode


# In[16]:


# Enumerate Function in Python to check unickness of  all columns
for l, k in enumerate(df.columns):
      print(k, ' == >> ',df[k].unique())


# In[17]:

# Replace 'path_to_your_dataset.csv' with the actual path or URL to your dataset
# df = pd.read_csv('path_to_your_dataset.csv')

# Create a heatmap to visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.title('Missing Values Heatmap')
plt.show()

#After idetfing the numerical and categorical feature I can iput like this
# Loop through each column
for column_mean in ['Bearer Id', 'Start ms',  'End ms', 'Dur. (ms)',
       'MSISDN/Number', 'Avg RTT DL (ms)',
       'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
       'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
       'DL TP < 50 Kbps (%)', '50 Kbps < DL TP < 250 Kbps (%)',
       '250 Kbps < DL TP < 1 Mbps (%)', 'DL TP > 1 Mbps (%)',
       'UL TP < 10 Kbps (%)', '10 Kbps < UL TP < 50 Kbps (%)',
       '50 Kbps < UL TP < 300 Kbps (%)', 'UL TP > 300 Kbps (%)',
       'HTTP DL (Bytes)', 'HTTP UL (Bytes)', 'Activity Duration DL (ms)',
       'Activity Duration UL (ms)', 'Dur. (ms).1',
       'Nb of sec with 1250B < Vol UL < 6250B',
       'Nb of sec with 31250B < Vol DL < 125000B',
       'Nb of sec with 37500B < Vol UL',
       'Nb of sec with 6250B < Vol DL < 31250B','IMEI','IMSI',
       'Nb of sec with 6250B < Vol UL < 37500B',
       'Nb of sec with Vol DL < 6250B', 'Nb of sec with Vol UL < 1250B',
       'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
       'Google DL (Bytes)', 'Google UL (Bytes)', 'Email DL (Bytes)',
       'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
       'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)',
       'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)',
       'Total UL (Bytes)', 'Total DL (Bytes)']:
    try:
        # Convert the column to numeric (ignore errors)
        df[column_mean] = pd.to_numeric(df[column_mean], errors='coerce')
        # Calculate the mean, ignoring NaN values
        mean_value = df[column_mean].mean(skipna=True)
        # Impute missing values with the mean
        df[column_mean].fillna(mean_value, inplace=True)
    except Exception as e:
        print(f"Error processing column '{column_mean}': {e}")


# In[19]:


df.isnull().sum()

df['Start'].unique()
df['End'].unique()

df['Handset Type'].unique()

df['Last Location Name'].unique()

df['Handset Manufacturer'].unique(
# Let us Imput the categorical features given below using mode iputation technique
# Loop through each column
for column in ['Start', 'End', 'Last Location Name', 'Handset Type', 'Handset Manufacturer']:
    # Calculate the mode
    mode_value = df[column].mode()[0]
    # Impute missing values with the mode
    df[column].fillna(mode_value, inplace=True)

# After Properlly handlling the missing values we should check the the missing values
df.isnull().sum()
import plotly.express as px
import pandas as pd

# Assuming df is your DataFrame
# Create a DataFrame with boolean values indicating missing values
missing_values = pd.DataFrame(df.isnull())

# Create a heatmap using Plotly Express
fig = px.imshow(missing_values, labels=dict(color='Missing Values'), 
                color_continuous_scale="Viridis", 
                title='Missing Values in DataFrame')
# Show the plot
fig.show()

df['Nb of sec with 1250B < Vol UL < 6250B'].isnull().sum()

#after this we should convert all feature types into numeric datatypes for easy of machinelearning 
"""converting features to numeric types is a common preprocessing step to ensure 
compatibility with machine learning algorithms and to leverage the efficiency and 
capabilities of numerical representations"""

# for easly learnig machine learning algorithms all features should covert into numeric
df[x] = df[x].astype("Int64")
df.dtypes
for x in df:
    if df[x].dtypes == "float":
        df[x] = df[x].astype("int64")


df.dtypes

# Convert 'Start' and 'End' columns to datetime
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])


#Explaratory Data Analisis 

import seaborn as sns
# Example for 'Dur. (ms)'
plt.figure(figsize=(10, 6))
sns.histplot(df['Dur. (ms)'], bins=30, kde=True)
plt.title('Distribution of Duration (ms)')
plt.show()
# Convert 'Start' and 'End' to datetime
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

# Plot histograms for 'Start' and 'End'
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(df['Start'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Start Times')
plt.xlabel('Start Time')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.hist(df['End'], bins=30, color='salmon', edgecolor='black')
plt.title('Distribution of End Times')
plt.xlabel('End Time')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Select only numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns

# Set up subplots
fig, axes = plt.subplots(nrows=len(numeric_columns), ncols=1, figsize=(10, 5 * len(numeric_columns)))

# Loop through numeric columns and create box plots
for i, column in enumerate(numeric_columns):
    sns.boxplot(x=df[column], ax=axes[i])
    axes[i].set_title(f'Box Plot of {column}')

plt.tight_layout()
plt.show()


# In[49]:


df.head()

#xDR sessions per user
# Aggregate the number of xDR sessions per user
user_xdr_sessions = df.groupby('MSISDN/Number')['Bearer Id'].count().reset_index()
user_xdr_sessions.columns = ['MSISDN/Number', 'Num_xDR_Sessions']

# Display the resulting DataFrame
print(user_xdr_sessions)

#Session duration
# Convert 'Dur. (ms)' to seconds for session duration
df['Session_Duration'] = df['Dur. (ms)'] / 1000

# Aggregate the session duration per user
user_session_duration = df.groupby('MSISDN/Number')['Session_Duration'].sum().reset_index()
user_session_duration.columns = ['MSISDN/Number', 'Total_Session_Duration']

# Display the resulting DataFrame
print(user_session_duration)


# Aggregate the total download and upload data per user
user_total_data = df.groupby('MSISDN/Number').agg({
    'Total DL (Bytes)': 'sum',  # Total download data
    'Total UL (Bytes)': 'sum'   # Total upload data
}).reset_index()

# Rename columns for clarity
user_total_data.columns = ['MSISDN/Number', 'Total_DL_Data', 'Total_UL_Data']

# Display the resulting DataFrame
print(user_total_data)


# List of application columns
application_columns = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)',
                        'Email DL (Bytes)', 'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                        'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
                        'Other DL (Bytes)', 'Other UL (Bytes)']

# Add a new column 'Total_Data_Volume' to store the sum of data volume for each session
df['Total_Data_Volume'] = df[application_columns].sum(axis=1)

# Aggregate the total data volume per user
user_total_data_volume = df.groupby('MSISDN/Number')['Total_Data_Volume'].sum().reset_index()

# Rename columns for clarity
user_total_data_volume.columns = ['MSISDN/Number', 'Total_Data_Volume']

# Display the resulting DataFrame
print(user_total_data_volume)


# Assuming df is your DataFrame
basic_metrics = df.describe()

# Print the basic metrics
print(basic_metrics)


"""numerical_variables = df.select_dtypes(include=['int64', 'float64']).columns

fig, axes = plt.subplots(nrows=len(numerical_variables), ncols=1, figsize=(10, 4 * len(numerical_variables)))

for i, variable in enumerate(numerical_variables):
    sns.histplot(df[variable], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {variable}')

plt.tight_layout()
plt.show()"""


import streamlit as st
# Function to identify the top N handsets
def top_handsets(n):
    top_handsets_df = df['Handset Type'].value_counts().head(n)
    return top_handsets_df

# Function to identify the top N handset manufacturers
def top_manufacturers(n):
    top_manufacturers_df = df['Handset Manufacturer'].value_counts().head(n)
    return top_manufacturers_df

# Function to identify the top N handsets for each manufacturer
def top_handsets_per_manufacturer(manufacturer, n):
    manufacturer_df = df[df['Handset Manufacturer'] == manufacturer]
    top_handsets_df = manufacturer_df['Handset Type'].value_counts().head(n)
    return top_handsets_df

# Streamlit app
def main():
    st.title("Telecom User Overview Analysis")

    # Sidebar with options
    st.sidebar.header("Options")
    analysis_option = st.sidebar.selectbox("Select Analysis", ["Top Handsets", "Top Manufacturers", "Top Handsets per Manufacturer"])

    # Display the analysis based on the selected option
    if analysis_option == "Top Handsets":
        st.header("Top 10 Handsets")
        top_10_handsets = top_handsets(10)
        st.bar_chart(top_10_handsets)

    elif analysis_option == "Top Manufacturers":
        st.header("Top 3 Handset Manufacturers")
        top_3_manufacturers = top_manufacturers(3)
        st.bar_chart(top_3_manufacturers)

    elif analysis_option == "Top Handsets per Manufacturer":
        st.header("Top 5 Handsets per Top 3 Manufacturers")
        # Select top 3 manufacturers
        top_3_manufacturers = top_manufacturers(3)
        for manufacturer in top_3_manufacturers.index:
            st.subheader(f"Top 5 Handsets for {manufacturer}")
            top_5_handsets_per_manufacturer = top_handsets_per_manufacturer(manufacturer, 5)
            st.bar_chart(top_5_handsets_per_manufacturer)




# Filter relevant columns for xDR applications
xdr_columns = ['MSISDN/Number', 'Dur. (ms).1', 'Dur. (ms)',
               'Total UL (Bytes)', 'Total DL (Bytes)',
               'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
               'Google DL (Bytes)', 'Google UL (Bytes)',
               'Email DL (Bytes)', 'Email UL (Bytes)',
               'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
               'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
               'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
               'Other DL (Bytes)', 'Other UL (Bytes)']

xdr_data = df[xdr_columns]

# Group by MSISDN/Number (user) and aggregate required metrics
user_behavior_overview = xdr_data.groupby('MSISDN/Number').agg({
    'Dur. (ms).1': 'count',  # Number of xDR sessions
    'Dur. (ms)': 'sum',      # Session duration
    'Total UL (Bytes)': 'sum',
    'Total DL (Bytes)': 'sum',
    'Social Media DL (Bytes)': 'sum', 'Social Media UL (Bytes)': 'sum',
    'Google DL (Bytes)': 'sum', 'Google UL (Bytes)': 'sum',
    'Email DL (Bytes)': 'sum', 'Email UL (Bytes)': 'sum',
    'Youtube DL (Bytes)': 'sum', 'Youtube UL (Bytes)': 'sum',
    'Netflix DL (Bytes)': 'sum', 'Netflix UL (Bytes)': 'sum',
    'Gaming DL (Bytes)': 'sum', 'Gaming UL (Bytes)': 'sum',
    'Other DL (Bytes)': 'sum', 'Other UL (Bytes)': 'sum'
}).reset_index()

# Rename columns for clarity
user_behavior_overview.columns = ['MSISDN/Number', 'Number of xDR Sessions', 'Session Duration',
                                  'Total UL (Bytes)', 'Total DL (Bytes)',
                                  'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                                  'Google DL (Bytes)', 'Google UL (Bytes)',
                                  'Email DL (Bytes)', 'Email UL (Bytes)',
                                  'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
                                  'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
                                  'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
                                  'Other DL (Bytes)', 'Other UL (Bytes)']

# Display the user behavior overview
print(user_behavior_overview)


# In[61]:


# Task 2.2

# Display basic information about the dataset
print("Basic Information about the Dataset:")
print(df.info())


# Summary statistics for numeric columns
print("\nSummary statistics for numeric columns:")
print(df.describe())

# Select only numeric columns for mean calculation
numeric_columns = df.select_dtypes(include=['float64', 'int64'])

# Calculate mean and median for relevant columns
mean_values = numeric_columns.mean()
median_values = numeric_columns.median()

# Display the mean and median values
print("Mean Values:")
print(mean_values)

print("\nMedian Values:")
print(median_values)


# Display mean and median
print("\nMean values for relevant columns:")
print(mean_values)

print("\nMedian values for relevant columns:")
print(median_values)



# Select quantitative columns for analysis
quantitative_columns = df.select_dtypes(['float64', 'int64']).columns

# Initialize an empty list to store dictionaries
dispersion_results = []

# Compute dispersion parameters for each quantitative variable
for column in quantitative_columns:
    # Calculate range
    data_range = df[column].max() - df[column].min()
    
    # Calculate variance
    column_variance = df[column].var()
    
    # Calculate standard deviation
    column_std_dev = df[column].std()
    
    # Append results to the list
    dispersion_results.append({
        'Variable': column,
        'Range': data_range,
        'Variance': column_variance,
        'Standard Deviation': column_std_dev
    })

# Convert the list of dictionaries to a DataFrame
dispersion_results_df = pd.DataFrame(dispersion_results)

# Display the dispersion results
print(dispersion_results_df)


# In[66]:


top_manufacturers = df['Handset Manufacturer'].value_counts().head(3).index
top_handsets_per_manufacturer = df[df['Handset Manufacturer'].isin(top_manufacturers)]

sns.countplot(x='Handset Type', hue='Handset Manufacturer', data=top_handsets_per_manufacturer)
plt.xticks(rotation=45, ha='right')
plt.title('Top Handsets per Top Handset Manufacturer')
plt.show()

# Scatter plot for Social Media
sns.scatterplot(x='Social Media DL (Bytes)', y='Social Media UL (Bytes)', data=df)
plt.title('Scatter Plot for Social Media')
plt.xlabel('Social Media DL (Bytes)')
plt.ylabel('Social Media UL (Bytes)')
plt.show()


correlation_social_media = df['Social Media DL (Bytes)'].corr(df['Social Media UL (Bytes)'])
print(f"Correlation for Social Media: {correlation_social_media}")


sns.pairplot(df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Total DL (Bytes)']])
plt.title('Pair Plot for Applications and Total DL (Bytes)')
plt.show()


#Variable transformations

# Calculate total duration per user
total_duration_per_user = df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index()

# Calculate deciles based on total duration
total_duration_per_user['duration_decile'] = pd.qcut(total_duration_per_user['Dur. (ms)'], q=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=False, duplicates='drop')

# Calculate total data (DL+UL) per user
total_data_per_user = df.groupby('MSISDN/Number')[['Total UL (Bytes)', 'Total DL (Bytes)']].sum().reset_index()

# Merge the two DataFrames on 'MSISDN/Number'
merged_df = pd.merge(total_duration_per_user, total_data_per_user, on='MSISDN/Number')

# Calculate total data (DL+UL) per decile class
total_data_per_decile = merged_df.groupby('duration_decile')[['Total UL (Bytes)', 'Total DL (Bytes)']].sum().reset_index()

# Display the result
print(total_data_per_decile)


# Create a subset DataFrame with the specified variables
subset_df = df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']]

# Compute the correlation matrix
correlation_matrix = subset_df.corr()

# Display the correlation matrix
print(correlation_matrix)


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Drop any rows with missing values for simplicity
df.dropna(inplace=True)

# Identify and drop non-numeric columns
non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
df_numeric = df.drop(non_numeric_columns, axis=1)


# Standardize the data (important for PCA)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(df_numeric)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_standardized)

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

# Plot the explained variance
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Choose the number of components based on the plot or a desired threshold (e.g., 95% variance)
num_components = 3  # Adjust as needed

# Retain the selected number of components
X_pca_selected = X_pca[:, :num_components]

# Interpretation of principal components
principal_components_df = pd.DataFrame(pca.components_, columns=df_numeric.columns)
print("Principal Component Loadings:")
print(principal_components_df)

# Real-world interpretation of principal components
# Add your interpretation based on the loadings and context

# Visualize in the reduced-dimensional space (for 2D or 3D)
if num_components == 2:
    plt.scatter(X_pca_selected[:, 0], X_pca_selected[:, 1], marker='o')
    plt.title('PCA: Reduced 2D Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()
elif num_components == 3:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_pca_selected[:, 0], X_pca_selected[:, 1], X_pca_selected[:, 2], marker='o')
    ax.set_title('PCA: Reduced 3D Space')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.show()



import pandas as pd

# Assuming df is your DataFrame containing the provided dataset
# Convert "Start" and "End" columns to datetime format for easier calculations
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

# 1. Sessions Frequency
sessions_frequency = df.groupby(['MSISDN/Number', 'Bearer Id']).size().reset_index(name='Sessions')

# 2. Duration of the Session
df['Session_Duration'] = (df['End'] - df['Start']).dt.total_seconds()

# 3. Sessions Total Traffic
df['Total_Traffic'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']

# Group by 'MSISDN/Number' and 'Bearer Id' to get total traffic for each session
sessions_total_traffic = df.groupby(['MSISDN/Number', 'Bearer Id'])['Total_Traffic'].sum().reset_index()

# Display the results
print("1. Sessions Frequency:")
print(sessions_frequency)

print("\n2. Duration of the Session:")
print(df[['MSISDN/Number', 'Bearer Id', 'Session_Duration']])

print("\n3. Sessions Total Traffic:")
print(sessions_total_traffic)


# Assuming df is your DataFrame containing the provided dataset
# Convert "Start" and "End" columns to datetime format for easier calculations
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

# Calculate sessions frequency
sessions_frequency = df.groupby(['MSISDN/Number', 'Bearer Id']).size().reset_index(name='Sessions Frequency')

# Display the result
print("Sessions Frequency:")
print(sessions_frequency)


# Assuming df is your DataFrame containing the provided dataset
# Convert "Start" and "End" columns to datetime format for easier calculations
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

# Calculate the duration of the session
df['Session Duration'] = (df['End'] - df['Start']).dt.total_seconds()

# Display the result
print("Duration of the Session:")
print(df[['MSISDN/Number', 'Bearer Id', 'Session Duration']])


# Assuming df is your DataFrame containing the provided dataset

# Calculate sessions total traffic
df['Total Traffic'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']

# Group by 'MSISDN/Number' and 'Bearer Id' to get total traffic for each session
sessions_total_traffic = df.groupby(['MSISDN/Number', 'Bearer Id'])['Total Traffic'].sum().reset_index()

# Display the result
print("Sessions Total Traffic:")
print(sessions_total_traffic)

# Assuming df is your DataFrame containing the provided dataset
# Convert "Start" and "End" columns to datetime format for easier calculations
df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

# Calculate sessions frequency
sessions_frequency = df.groupby(['MSISDN/Number', 'Bearer Id']).size().reset_index(name='Sessions Frequency')

# Calculate the duration of the session
df['Session Duration'] = (df['End'] - df['Start']).dt.total_seconds()

# Calculate sessions total traffic
df['Total Traffic'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
sessions_total_traffic = df.groupby(['MSISDN/Number', 'Bearer Id'])['Total Traffic'].sum().reset_index()

# Aggregate metrics per customer (MSISDN)
customer_metrics = pd.DataFrame({
    'Sessions Frequency': sessions_frequency.groupby('MSISDN/Number')['Sessions Frequency'].sum(),
    'Total Session Duration': df.groupby('MSISDN/Number')['Session Duration'].sum(),
    'Total Traffic': sessions_total_traffic.groupby('MSISDN/Number')['Total Traffic'].sum()
}).reset_index()

# Report the top 10 customers per engagement metric
top_10_frequency = customer_metrics.nlargest(10, 'Sessions Frequency')
top_10_duration = customer_metrics.nlargest(10, 'Total Session Duration')
top_10_traffic = customer_metrics.nlargest(10, 'Total Traffic')

# Display the results
print("Top 10 Customers by Sessions Frequency:")
print(top_10_frequency)

print("\nTop 10 Customers by Total Session Duration:")
print(top_10_duration)

print("\nTop 10 Customers by Total Traffic:")
print(top_10_traffic)


# Aggregate metrics per customer (MSISDN)
customer_metrics = pd.DataFrame({
    'Sessions Frequency': sessions_frequency.groupby('MSISDN/Number')['Sessions Frequency'].sum(),
    'Total Session Duration': df.groupby('MSISDN/Number')['Session Duration'].sum(),
    'Total Traffic': sessions_total_traffic.groupby('MSISDN/Number')['Total Traffic'].sum()
}).reset_index()

# Report the top 10 customers per engagement metric
top_10_frequency = customer_metrics.nlargest(10, 'Sessions Frequency')
top_10_duration = customer_metrics.nlargest(10, 'Total Session Duration')
top_10_traffic = customer_metrics.nlargest(10, 'Total Traffic')

# Display the results
print("Top 10 Customers by Sessions Frequency:")
print(top_10_frequency)


print("\nTop 10 Customers by Total Session Duration:")
print(top_10_duration)

print("\nTop 10 Customers by Total Traffic:")
print(top_10_traffic)

#3.1.Aggregate the above metrics per customer id (MSISDN) and report the top 10 customers per engagement metric 
agg_metrics = df.groupby('MSISDN/Number').agg({
    'Avg RTT DL (ms)': 'mean',
    'Avg RTT UL (ms)': 'mean',
    'Avg Bearer TP DL (kbps)': 'mean',
    'Avg Bearer TP UL (kbps)': 'mean',
    'TCP DL Retrans. Vol (Bytes)': 'sum',
    'TCP UL Retrans. Vol (Bytes)': 'sum'
})

# Report the top 10 customers for each engagement metric
top_10_rtt_dl = agg_metrics['Avg RTT DL (ms)'].nlargest(10)
top_10_rtt_ul = agg_metrics['Avg RTT UL (ms)'].nlargest(10)
top_10_tp_dl = agg_metrics['Avg Bearer TP DL (kbps)'].nlargest(10)
top_10_tp_ul = agg_metrics['Avg Bearer TP UL (kbps)'].nlargest(10)
top_10_retrans_dl = agg_metrics['TCP DL Retrans. Vol (Bytes)'].nlargest(10)
top_10_retrans_ul = agg_metrics['TCP UL Retrans. Vol (Bytes)'].nlargest(10)

# Plotting the top 10 customers for each engagement metric
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

top_10_rtt_dl.plot(kind='bar', ax=axes[0, 0], title='Top 10 Avg RTT DL (ms) per Customer')
top_10_rtt_ul.plot(kind='bar', ax=axes[0, 1], title='Top 10 Avg RTT UL (ms) per Customer')
top_10_tp_dl.plot(kind='bar', ax=axes[1, 0], title='Top 10 Avg Bearer TP DL (kbps) per Customer')
top_10_tp_ul.plot(kind='bar', ax=axes[1, 1], title='Top 10 Avg Bearer TP UL (kbps) per Customer')
top_10_retrans_dl.plot(kind='bar', ax=axes[2, 0], title='Top 10 TCP DL Retrans. Vol (Bytes) per Customer')
top_10_retrans_ul.plot(kind='bar', ax=axes[2, 1], title='Top 10 TCP UL Retrans. Vol (Bytes) per Customer')

plt.tight_layout()
plt.show()


#.1.Normalize each engagement metric and run a k-means (k=3) to classify customers in three groups of engagement.
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Normalize the engagement metrics using StandardScaler
scaler = StandardScaler()
normalized_metrics = scaler.fit_transform(customer_metrics[['Sessions Frequency', 'Total Session Duration', 'Total Traffic']])

# Run k-means clustering (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
customer_metrics['Cluster'] = kmeans.fit_predict(normalized_metrics)

# Display the results
print("Customer Clusters:")
print(customer_metrics[['MSISDN/Number', 'Cluster']])

# Optional: Display cluster centroids (normalized values)
print("\nCluster Centroids (Normalized):")
print(scaler.inverse_transform(kmeans.cluster_centers_))


#3.1 Compute minimum, maximum, average, and total non-normalized metrics for each cluster
cluster_metrics_summary = customer_metrics.groupby('Cluster').agg({
    'Sessions Frequency': ['min', 'max', 'mean', 'sum'],
    'Total Session Duration': ['min', 'max', 'mean', 'sum'],
    'Total Traffic': ['min', 'max', 'mean', 'sum']
}).reset_index()

# Display the results
print("Cluster Metrics Summary:")
print(cluster_metrics_summary)
plt.scatter(
    customer_metrics['Total Session Duration'],
    customer_metrics['Total Traffic'],
    c=customer_metrics['Cluster'],
    cmap='viridis',
    alpha=0.5,
    marker='o',
)
plt.title('Customer Engagement Clusters')
plt.xlabel('Total Session Duration')
plt.ylabel('Total Traffic')
plt.show()


df.head()


# List of application columns
app_columns = ['Social Media DL (Bytes)', 'Social Media UL (Bytes)',
               'Google DL (Bytes)', 'Google UL (Bytes)',
               'Email DL (Bytes)', 'Email UL (Bytes)',
               'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
               'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
               'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
               'Other DL (Bytes)', 'Other UL (Bytes)']

# Create a new column 'Total_Traffic' for each application
for app_column in app_columns:
    df[app_column.replace(' ', '_')] = df[app_column]

# Aggregate total traffic per application per user
user_total_traffic = df.groupby('MSISDN/Number')[app_columns].sum()

# Create a new column 'Total_Traffic' representing the total traffic for each user
user_total_traffic['Total_Traffic'] = user_total_traffic.sum(axis=1)

# Get the top 10 most engaged users per application
top_users_per_app = {}
for app_column in app_columns:
    top_users_per_app[app_column] = user_total_traffic.sort_values(by=app_column, ascending=False).head(10)

# Print or use the top users per application as needed
for app_column, top_users in top_users_per_app.items():
    print(f"Top 10 users for {app_column}:\n{top_users[['Total_Traffic', app_column]]}\n")


import matplotlib.pyplot as plt

# Assuming you already have the user_total_traffic DataFrame
top_3_apps = user_total_traffic.sum().nlargest(3).index

# Subset the DataFrame for the top 3 apps
top_3_apps_traffic = user_total_traffic[top_3_apps]

# Plotting
plt.figure(figsize=(12, 8))
top_3_apps_traffic.sum().sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Top 3 Most Used Applications')
plt.xlabel('Applications')
plt.ylabel('Total Traffic (Bytes)')
plt.show()



# Select the engagement metrics for clustering
selected_metrics = customer_metrics

# Initialize an empty list to store the sum of squared distances
sse = []

# Assume you want to try values of k from 1 to 10
k_values = range(1, 11)

# Fit k-means clustering for each value of k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(selected_metrics)
    sse.append(kmeans.inertia_)

# Plot the elbow plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, sse, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances (SSE)')
plt.show()


#4.1 a: Aggregate, per customer Average TCP retransmission

# Convert relevant columns to numeric values
numeric_columns = ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Group by 'MSISDN/Number'
grouped_df = df.groupby('MSISDN/Number')

# Calculate averages for each customer
aggregated_df = grouped_df.agg({
    'TCP DL Retrans. Vol (Bytes)': 'mean',
    'TCP UL Retrans. Vol (Bytes)': 'mean',
}).reset_index()

# Rename columns for clarity
aggregated_df.columns = ['MSISDN/Number', 'Avg TCP Retrans. DL', 'Avg TCP Retrans. UL']

# Display the aggregated DataFrame
print(aggregated_df)


#4.1 b: Aggregate, per customer Average RTT
# Convert relevant columns to numeric values
numeric_columns = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Group by 'MSISDN/Number'
grouped_df = df.groupby('MSISDN/Number')

# Calculate averages for each customer
aggregated_df = grouped_df.agg({
    'Avg RTT DL (ms)': 'mean',
    'Avg RTT UL (ms)': 'mean',
}).reset_index()

# Rename columns for clarity
aggregated_df.columns = ['MSISDN/Number', 'Avg RTT DL', 'Avg RTT UL']

# Display the aggregated DataFrame
print(aggregated_df)


#4.1 c: Aggregate, per customer Handset Type

# Group by 'MSISDN/Number' and select the first 'Handset Type' for each customer
aggregated_df = df.groupby('MSISDN/Number')['Handset Type'].first().reset_index()

# Rename columns for clarity
aggregated_df.columns = ['MSISDN/Number', 'Handset Type']

# Display the aggregated DataFrame
print(aggregated_df)


#4.1 d: Aggregate, per customer Average Throughput

# Group by 'MSISDN/Number' and calculate the mean of 'Avg Bearer TP DL (kbps)' and 'Avg Bearer TP UL (kbps)'
aggregated_df = df.groupby('MSISDN/Number')[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].mean().reset_index()

# Calculate the overall average throughput by averaging the downlink and uplink throughput
aggregated_df['Average Throughput'] = (aggregated_df['Avg Bearer TP DL (kbps)'] + aggregated_df['Avg Bearer TP UL (kbps)']) / 2

# Keep only the relevant columns
aggregated_df = aggregated_df[['MSISDN/Number', 'Average Throughput']]

# Display the aggregated DataFrame
print(aggregated_df)



#4.1 Aggregate, per customer Average TCP retransmission, Average RTT, Handset type, Average throughput

# Convert relevant columns to numeric values
numeric_columns = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Group by 'MSISDN/Number'
grouped_df = df.groupby('MSISDN/Number')

# Calculate averages for each customer
aggregated_df = grouped_df.agg({
    'Avg RTT DL (ms)': 'mean',
    'Avg RTT UL (ms)': 'mean',
    'Avg Bearer TP DL (kbps)': 'mean',
    'Avg Bearer TP UL (kbps)': 'mean',
    'TCP DL Retrans. Vol (Bytes)': 'mean',
    'TCP UL Retrans. Vol (Bytes)': 'mean',
    'Handset Type': 'first',  # Assumes the handset type remains constant for each customer
}).reset_index()

# Rename columns for clarity
aggregated_df.columns = ['MSISDN/Number', 'Avg RTT DL', 'Avg RTT UL', 'Avg Throughput DL', 'Avg Throughput UL', 'Avg TCP Retrans. DL', 'Avg TCP Retrans. UL', 'Handset Type']

# Display the aggregated DataFrame
print(aggregated_df)


# Task 4.2 Compute an list 10 of the top, bottom and most frequent:
# Assuming df is your DataFrame
# Replace 'TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)' with the actual column names
tcp_values = df['TCP DL Retrans. Vol (Bytes)']
rtt_values = df['Avg RTT DL (ms)']
throughput_values = df['Avg Bearer TP DL (kbps)']

# Compute top values
top_tcp_values = tcp_values.nlargest(10)
top_rtt_values = rtt_values.nlargest(10)
top_throughput_values = throughput_values.nlargest(10)

# Compute bottom values
bottom_tcp_values = tcp_values.nsmallest(10)
bottom_rtt_values = rtt_values.nsmallest(10)
bottom_throughput_values = throughput_values.nsmallest(10)

# Compute most frequent values
most_frequent_tcp_values = tcp_values.value_counts().nlargest(10)
most_frequent_rtt_values = rtt_values.value_counts().nlargest(10)
most_frequent_throughput_values = throughput_values.value_counts().nlargest(10)

# Display the results
print("Top TCP Values:")
print(top_tcp_values)
print("\nBottom TCP Values:")
print(bottom_tcp_values)
print("\nMost Frequent TCP Values:")
print(most_frequent_tcp_values)

print("\nTop RTT Values:")
print(top_rtt_values)
print("\nBottom RTT Values:")
print(bottom_rtt_values)
print("\nMost Frequent RTT Values:")
print(most_frequent_rtt_values)

print("\nTop Throughput Values:")
print(top_throughput_values)
print("\nBottom Throughput Values:")
print(bottom_throughput_values)
print("\nMost Frequent Throughput Values:")
print(most_frequent_throughput_values)


# Convert relevant columns to numeric values
numeric_columns = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Group by 'Handset Type'
grouped_by_handset = df.groupby('Handset Type')

# Calculate averages for each handset type
avg_retransmission_per_handset = grouped_by_handset[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].mean().reset_index()

# Display the aggregated DataFrame
print(avg_retransmission_per_handset)


# Task 4.3 - Compute & report

avg_throughput_per_handset = grouped_by_handset[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].mean().reset_index()



bar_width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

# Plotting average download throughput
ax.bar(avg_throughput_per_handset['Handset Type'], avg_throughput_per_handset['Avg Bearer TP DL (kbps)'],
       width=bar_width, label='Download Throughput')

# Plotting average upload throughput
ax.bar(avg_throughput_per_handset['Handset Type'], avg_throughput_per_handset['Avg Bearer TP UL (kbps)'],
       width=bar_width, label='Upload Throughput', bottom=avg_throughput_per_handset['Avg Bearer TP DL (kbps)'])

ax.set_xlabel('Handset Type')
ax.set_ylabel('Average Throughput (kbps)')
ax.set_title('Average Throughput per Handset Type')
ax.legend()

plt.show()


# Convert relevant columns to numeric values
numeric_columns = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Group by 'Handset Type'
grouped_by_handset = df.groupby('Handset Type')

# Calculate averages for each handset type
avg_retransmission_per_handset = grouped_by_handset[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].mean().reset_index()

# Display the aggregated DataFrame
print(avg_retransmission_per_handset)


# Task 4.4 kemeans clustering
# Group by 'Handset Type'
grouped_by_handset = df.groupby('Handset Type')

# Calculate averages for each handset type
avg_throughput_per_handset = grouped_by_handset[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].mean().reset_index()

# Plotting average throughput per handset type
plt.figure(figsize=(12, 6))
sns.barplot(x='Handset Type', y='Avg Bearer TP DL (kbps)', data=avg_throughput_per_handset, label='Downlink', color='blue')
sns.barplot(x='Handset Type', y='Avg Bearer TP UL (kbps)', data=avg_throughput_per_handset, label='Uplink', color='green')
plt.title('Average Throughput per Handset Type')
plt.xlabel('Handset Type')
plt.ylabel('Average Throughput (kbps)')
plt.legend()
plt.show()


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame
# Select experience metrics
experience_metrics = df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
                         'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
                         'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']]

# Handle missing values by filling with the mean
experience_metrics.fillna(experience_metrics.mean(), inplace=True)

# Standardize the data
scaler = StandardScaler()
normalized_metrics = scaler.fit_transform(experience_metrics)

# Run k-means clustering (k = 3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Experience_Cluster'] = kmeans.fit_predict(normalized_metrics)

# Brief description of each cluster
cluster_descriptions = {
    0: "High Throughput, Low Retransmission, Low RTT",
    1: "Moderate Throughput, Moderate Retransmission, Moderate RTT",
    2: "Low Throughput, High Retransmission, High RTT"
}

# Map cluster labels to descriptions
df['Cluster_Description'] = df['Experience_Cluster'].map(cluster_descriptions)

# Display the results
print("Cluster Descriptions:")
print(df[['MSISDN/Number', 'Experience_Cluster', 'Cluster_Description']])



from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame
# Select experience metrics
experience_metrics = df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
                         'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
                         'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']]

# Handle missing values by filling with the mean
experience_metrics.fillna(experience_metrics.mean(), inplace=True)

# Standardize the data
scaler = StandardScaler()
normalized_metrics = scaler.fit_transform(experience_metrics)

# Run k-means clustering (k = 3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Experience_Cluster'] = kmeans.fit_predict(normalized_metrics)

# Display the results
print("User Experience Clusters:")
print(df[['MSISDN/Number', 'Experience_Cluster']])


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import numpy as np

# Assuming you have the engagement clusters already assigned to the DataFrame
# Let's use 'Experience_Cluster' as the column with cluster assignments

# Select engagement metrics for clustering
engagement_metrics = customer_metrics[['Sessions Frequency', 'Total Session Duration', 'Total Traffic']]

# Handle missing values by filling with the mean
engagement_metrics.fillna(engagement_metrics.mean(), inplace=True)

# Normalize engagement metrics using StandardScaler
scaler = StandardScaler()
normalized_metrics = scaler.fit_transform(engagement_metrics)

# Run k-means clustering (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
customer_metrics['Engagement_Cluster'] = kmeans.fit_predict(normalized_metrics)

# Calculate Euclidean distance from each user to cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
user_data = normalized_metrics[:, :3]  # Assuming the first 3 columns are the engagement metrics

# Calculate Euclidean distance between each user and the less engaged cluster
less_engaged_cluster = np.argmin(np.sum((centroids - user_data.mean(axis=0))**2, axis=1))
customer_metrics['Engagement_Score'] = np.linalg.norm(user_data - centroids[less_engaged_cluster], axis=1)

# Display the results
print("Engagement Scores:")
print(customer_metrics[['MSISDN/Number', 'Engagement_Score']])


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming df is your DataFrame
# Select experience metrics
experience_metrics = df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
                         'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
                         'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']]

# Handle missing values by filling with the mean
experience_metrics.fillna(experience_metrics.mean(), inplace=True)

# Standardize the data
scaler = StandardScaler()
normalized_metrics = scaler.fit_transform(experience_metrics)

# Run k-means clustering (k = 3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Experience_Cluster'] = kmeans.fit_predict(normalized_metrics)

# Calculate Euclidean distance from each user to cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
user_data = normalized_metrics  # Assuming all selected columns are used as experience metrics

# Calculate Euclidean distance between each user and the worst experience cluster
worst_experience_cluster = np.argmax(np.sum((centroids - user_data.mean(axis=0))**2, axis=1))
df['Experience_Score'] = np.linalg.norm(user_data - centroids[worst_experience_cluster], axis=1)

# Display the results
print("Experience Scores:")
print(df[['MSISDN/Number', 'Experience_Score']])



from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming df is your DataFrame
# Select experience metrics
experience_metrics = df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
                         'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
                         'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']]

# Handle missing values by filling with the mean
experience_metrics.fillna(experience_metrics.mean(), inplace=True)

# Standardize the data
scaler = StandardScaler()
normalized_metrics = scaler.fit_transform(experience_metrics)

# Run k-means clustering (k = 3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Experience_Cluster'] = kmeans.fit_predict(normalized_metrics)

# Calculate Euclidean distance from each user to cluster centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
user_data = normalized_metrics  # Assuming all selected columns are used as experience metrics

# Calculate Euclidean distance between each user and the worst experience cluster
worst_experience_cluster = np.argmax(np.sum((centroids - user_data.mean(axis=0))**2, axis=1))
df['Experience_Score'] = np.linalg.norm(user_data - centroids[worst_experience_cluster], axis=1)

# Display the results
print("Experience Scores:")
print(df[['MSISDN/Number', 'Experience_Score']])


# customer_metrics is your DataFrame with 'Engagement_Score' and 'Experience_Score' columns
customer_metrics['Satisfaction_Score'] = (customer_metrics['Engagement_Score'] + df['Experience_Score']) / 2

# Display the top 10 satisfied customers
top_satisfied_customers = customer_metrics.nlargest(10, 'Satisfaction_Score')[['MSISDN/Number', 'Satisfaction_Score']]
print("Top 10 Satisfied Customers:")
print(top_satisfied_customers)

# Assuming X is the feature matrix and y is the target variable (Satisfaction_Score)
X = customer_metrics[['Sessions Frequency', 'Total Session Duration', 'Total Traffic']]
y = customer_metrics['Satisfaction_Score']

# Handle missing values by filling with the mean
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but recommended for some models)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
predictions = model.predict(X_test_scaled)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Display the model performance
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Optionally, you can inspect the coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)


# 'Engagement_Score' is in customer_metrics and 'Experience_Score' is in df
engagement_metrics = customer_metrics[['Engagement_Score']]
experience_metrics = df[['Experience_Score']]

# Handle missing values by filling with the mean
engagement_metrics.fillna(engagement_metrics.mean(), inplace=True)
experience_metrics.fillna(experience_metrics.mean(), inplace=True)

# Normalize the data using StandardScaler
scaler = StandardScaler()
normalized_engagement = scaler.fit_transform(engagement_metrics)
normalized_experience = scaler.fit_transform(experience_metrics)

# Run k-means clustering separately for engagement and experience (k=2 for both)
kmeans_engagement = KMeans(n_clusters=2, random_state=42)
kmeans_experience = KMeans(n_clusters=2, random_state=42)

customer_metrics['Engagement_Cluster'] = kmeans_engagement.fit_predict(normalized_engagement)
df['Experience_Cluster'] = kmeans_experience.fit_predict(normalized_experience)

# Display the results
print("Customer Clusters based on Engagement Score:")
print(customer_metrics[['MSISDN/Number', 'Engagement_Cluster']])

print("Customer Clusters based on Experience Score:")
print(df[['MSISDN/Number', 'Experience_Cluster']])


# Aggregate average satisfaction score per engagement cluster
avg_satisfaction_engagement = customer_metrics.groupby('Engagement_Cluster')['Satisfaction_Score'].mean()

# Aggregate average experience score per experience cluster
avg_experience_experience = df.groupby('Experience_Cluster')['Experience_Score'].mean()

# Display the results
print("Average Satisfaction Score per Engagement Cluster:")
print(avg_satisfaction_engagement)

print("\nAverage Experience Score per Experience Cluster:")
print(avg_experience_experience)


import psycopg2
from sqlalchemy import create_engine

# I have PostgreSQL installed locally and running
username = "postgres"
password = "mariam21"
host = "localhost"
port = "5432"
database_name = "telecom_db"

# Create a PostgreSQL connection string
postgres_conn_str = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"

# Create a SQLAlchemy engine
engine = create_engine(postgres_conn_str)

# Assuming 'MSISDN/Number', 'Engagement_Score', 'Experience_Score' are the relevant columns
customer_metrics_table = customer_metrics[['MSISDN/Number', 'Engagement_Score']]
df_table = df[['MSISDN/Number', 'Experience_Score']]

# Export the tables to PostgreSQL
customer_metrics_table.to_sql('customer_metrics_table', engine, index=False, if_exists='replace')
df_table.to_sql('df_table', engine, index=False, if_exists='replace')
# Run a select query
query = "SELECT * FROM xdr_data"
result = pd.read_sql_query(query, engine)

# Display the result (for verification)
print(result)



import psycopg2
from sqlalchemy import create_engine

# Assuming 'Engagement_Score' is in customer_metrics and 'Experience_Score' is in df
engagement_metrics = customer_metrics[['MSISDN/Number', 'Engagement_Score']]
experience_metrics = df[['MSISDN/Number', 'Experience_Score']]

# Merge dataframes on 'MSISDN/Number'
merged_df = pd.merge(engagement_metrics, experience_metrics, on='MSISDN/Number', how='inner')

# Fill missing values only for numerical columns
merged_df[['Engagement_Score', 'Experience_Score']] = merged_df[['Engagement_Score', 'Experience_Score']].fillna(merged_df.mean())

# Create a PostgreSQL connection string
username = "postgres"
password = "mariam21"
host = "localhost"
port = "5432"
database_name = "telecom_db"

# Create a SQLAlchemy engine
postgres_conn_str = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"
engine = create_engine(postgres_conn_str)

# Assuming 'MSISDN/Number', 'Engagement_Score', 'Experience_Score' are the relevant columns
final_table = merged_df[['MSISDN/Number', 'Engagement_Score', 'Experience_Score']]

# Export the final table to PostgreSQL
final_table.to_sql('your_table_name', engine, index=False, if_exists='replace')

# Run a select query
select_query = "SELECT * FROM your_table_name LIMIT 10"
result = pd.read_sql(select_query, con=engine)

# Display the result
print("Select Query Result:")
print(result)



# Thank You

