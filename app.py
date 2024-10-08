import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

@st.cache_data
def load_data():
    data = pd.read_csv('healthcare-dataset-stroke-data.csv')
    data.replace("Unknown", pd.NA, inplace=True)
    return data

data = load_data()

st.title('Stroke Prediction Dataset Exploration')

# Introduction Section
st.header('Introduction')
st.write("""
This application presents an exploration of the Stroke Prediction Dataset from Kaggle. 
The dataset includes various health-related parameters, such as age, glucose levels, and BMI, 
to predict whether a patient is likely to have a stroke. The purpose of this exploration is 
to uncover trends and relationships within the data that may provide insights into stroke prediction.
""")

# Sidebar for navigation
st.sidebar.header('Chart Selection')
chart_type = st.sidebar.selectbox(
    "Select the chart you want to view",
    ("Distributions of Age, Glucose, and BMI", "Box Plots", "Correlation Matrix", "Pie Charts")
)

if st.checkbox('Show raw data'):
    st.write(data.head())

# Descriptive statistics
st.subheader('Descriptive Statistics')
st.write(data.describe())

# Visualizations Section
st.header('Visualizations')

# Age, Glucose, BMI Histograms
if chart_type == "Distributions of Age, Glucose, and BMI":
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
elif chart_type == "Box Plots":
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
elif chart_type == "Correlation Matrix":
    st.subheader('Correlation Matrix')
    selected_columns = ['age', 'bmi', 'avg_glucose_level']
    correlation_matrix = data[selected_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Pie Charts for Worktype, Residence Type, and Age
elif chart_type == "Pie Charts":
    st.subheader('Pie Charts for Worktype, Residence Type, and Age')

    # Worktype Pie Chart
    worktype_counts = data['work_type'].value_counts(dropna=False)
    fig1, ax1 = plt.subplots()
    ax1.pie(worktype_counts, labels=worktype_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax1.set_title('Work Type Distribution')
    st.pyplot(fig1)
    st.write("""
    **Insights**: 
    This pie chart shows the distribution of different work types among the participants. The majority of the dataset may fall 
    under categories such as "Private," "Self-employed," or "Government Job." Understanding the distribution of work types 
    can help analyze whether certain occupational groups are more susceptible to stroke, as work environments and lifestyles 
    may play a role in health outcomes.
    """)

    # Residence Type Pie Chart
    residence_counts = data['Residence_type'].value_counts(dropna=False)
    fig2, ax2 = plt.subplots()
    ax2.pie(residence_counts, labels=residence_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax2.set_title('Residence Type Distribution')
    st.pyplot(fig2)
    st.write("""
    **Insights**: 
    This pie chart displays the proportion of participants living in urban or rural areas. This distinction is crucial, as 
    healthcare access, lifestyle, and environmental factors differ between rural and urban areas. Analyzing this data might 
    uncover whether living in a particular type of residence is correlated with an increased risk of stroke.
    """)

    # Age Category Pie Chart
    age_bins = pd.cut(data['age'], bins=[0, 18, 35, 50, 65, 80, 100])
    age_group_counts = age_bins.value_counts(sort=False)
    fig3, ax3 = plt.subplots()
    ax3.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax3.set_title('Age Group Distribution')
    st.pyplot(fig3)
    st.write("""
    **Insights**: 
    The Age Group pie chart segments the dataset into age groups, such as children, young adults, middle-aged, and older adults. 
    Since age is a key risk factor for stroke, this chart helps visualize which age groups are most prevalent in the dataset. 
    It can aid in determining if strokes are more common in older populations, as expected, or if any younger age groups show unexpected trends.
    """)

# Conclusion Section
st.header('Conclusion')
st.write("""
From the visualizations and descriptive statistics, we can observe the following trends:
- Age, average glucose level, and BMI have varying distributions, with older age and higher glucose levels possibly correlating with stroke incidents.
- The box plots further show the relationship between these factors and stroke status, indicating potential trends.
- Further model-building and analysis can help explore these relationships in more detail, particularly when building predictive models using TensorFlow or other machine learning libraries.
""")
