import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset with st.cache_data for caching
@st.cache_data
def load_data():
    data = pd.read_csv('healthcare-dataset-stroke-data.csv')
    data.replace("Unknown", pd.NA, inplace=True)
    return data

data = load_data()

# App title
st.title('Stroke Prediction Dataset Exploration')

# Introduction Section
st.header('Introduction')
st.write("""
This application presents an exploration of the Stroke Prediction Dataset from Kaggle. 
The dataset includes various health-related parameters, such as age, glucose levels, and BMI, 
to predict whether a patient is likely to have a stroke. The purpose of this exploration is 
to uncover trends and relationships within the data that may provide insights into stroke prediction.
""")

# Show raw data if checkbox selected
if st.checkbox('Show raw data'):
    st.write(data.head())

# Descriptive statistics
st.subheader('Descriptive Statistics')
st.write(data.describe())

# Visualizations Section
st.header('Visualizations')

# Age, Glucose, BMI Histograms
st.subheader('Distributions of Age, Average Glucose Level, and BMI')
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Age Distribution
ax[0].hist(data['age'], bins=20, color='skyblue', edgecolor='black')
ax[0].set_title('Age Distribution')
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Frequency')

# Glucose Level Distribution
ax[1].hist(data['avg_glucose_level'], bins=20, color='lightgreen', edgecolor='black')
ax[1].set_title('Average Glucose Level Distribution')
ax[1].set_xlabel('Avg Glucose Level')
ax[1].set_ylabel('Frequency')

# BMI Distribution
ax[2].hist(data['bmi'], bins=20, color='salmon', edgecolor='black')
ax[2].set_title('BMI Distribution')
ax[2].set_xlabel('BMI')
ax[2].set_ylabel('Frequency')

st.pyplot(fig)

# Box Plots for Age, Average Glucose Level, and BMI
st.subheader('Box Plots of Age, Average Glucose Level, and BMI by Stroke Status')
filtered_data = data[['age', 'avg_glucose_level', 'bmi', 'stroke']]

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Age Box Plot
sns.boxplot(x='stroke', y='age', data=filtered_data, palette="coolwarm", ax=ax[0])
ax[0].set_title('Age Box Plot')
ax[0].set_ylabel('Age')

# Average Glucose Level Box Plot
sns.boxplot(x='stroke', y='avg_glucose_level', data=filtered_data, palette="coolwarm", ax=ax[1])
ax[1].set_title('Average Glucose Level Box Plot')
ax[1].set_ylabel('Avg Glucose Level')

# BMI Box Plot
sns.boxplot(x='stroke', y='bmi', data=filtered_data, palette="coolwarm", ax=ax[2])
ax[2].set_title('BMI Box Plot')
ax[2].set_ylabel('BMI')

st.pyplot(fig)

# Correlation Heatmap
st.subheader('Correlation Matrix')
selected_columns = ['age', 'bmi', 'avg_glucose_level']
correlation_matrix = data[selected_columns].corr()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Conclusion Section
st.header('Conclusion')
st.write("""
From the visualizations and descriptive statistics, we can observe the following trends:
- Age, average glucose level, and BMI have varying distributions, with older age and higher glucose levels possibly correlating with stroke incidents.
- The box plots further show the relationship between these factors and stroke status, indicating potential trends.
- Further model-building and analysis can help explore these relationships in more detail, particularly when building predictive models using TensorFlow or other machine learning libraries.
""")
