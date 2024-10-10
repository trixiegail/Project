import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

@st.cache_data
def load_data():
    data = pd.read_csv('healthcare-dataset-stroke-data.csv')
    # Replace string "Unknown"
    # *Note: "Unknown" in smoking_status means that the information is unavailable for this patient
    data.replace("Unknown", pd.NA, inplace=True)
    # Drop rows with Nan values
    data.dropna(inplace = True)
    # Reset index after dropping rows
    data.reset_index(drop=True, inplace=True)
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
    ("Distributions of Age, Glucose, and BMI", "Box Plots", "Pie Charts", "Correlation Matrix")
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

    st.write("""
    **Insights**:
    The Age Distribution histogram illustrates the range of ages among the dataset participants, highlighting the crucial role of age in stroke risk assessment. 
    As age is a key risk factor for strokes, this visualization pinpoint the age groups that are most vulnerable. The chart reveals a notable rise in 
    frequency from younger to middle ages, peaking in the middle-aged demographic, before gradually tapering off among the elderly. This pattern underscores a 
    significant presence of middle-aged and elderly individuals within the dataset, emphasizing the importance of focusing on these age groups for stroke prevention and prediction efforts.
    """)

    # Glucose Level Distribution
    ax[1].hist(data['avg_glucose_level'], bins=20, color='lightgreen', edgecolor='black')
    ax[1].set_title('Average Glucose Level Distribution')
    ax[1].set_xlabel('Avg Glucose Level')
    ax[1].set_ylabel('Frequency')

    st.write(""" 
    **Insights**:
    The Average Glucose Level Distribution histogram displays the distribution of average glucose levels among the participants. 
    High glucose levels are a risk factor for stroke, and this chart identify individuals with elevated glucose levels. 
    The distribution reveal whether the dataset contains a significant number of individuals with high glucose levels, 
    which could indicate a higher risk of stroke in the population. The peak near the normal glucose range, with a tail extending towards higher values, 
    suggests that while most individuals have glucose levels within a normal range, a subset has elevated levels which are a risk factor for diabetes and potentially stroke.
    """)

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

    st.write(""" 
    **Insights**\n
    Age Box Plot:

    Individuals who have had a stroke (1) tend to be older than those who haven't (0). The median age of the stroke group is significantly higher than that of the non-stroke group. The range of ages is also broader in the stroke group, suggesting age is a considerable factor in stroke incidence.
    
    Average Glucose Level Box Plot:

    There is a noticeable increase in the median glucose level in individuals who have had a stroke compared to those who haven't. The glucose level distribution is more variable and generally higher in the stroke group, which indicates higher glucose levels could be associated with an increased risk of stroke.
    
    BMI Box Plot:

    The BMI distributions between those who have had a stroke and those who have not are quite similar in terms of median BMI. However, the stroke group displays slightly more variability and has outliers indicating higher BMI values. This suggests that while BMI may be a factor in stroke risk, it is not as strongly differentiated between the two groups as age or glucose levels.
    """)

# Correlation Heatmap
elif chart_type == "Correlation Matrix":
    st.subheader('Correlation Matrix')
    numeric_data = data.select_dtypes(include='number').drop(columns=['id'])

    # Compute correlation matrix
    correlation_matrix = numeric_data.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
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

    #Stroke Distribution for Individuals Living in Urban Areas and Rural Areas
    # Calculate stroke and non-stroke counts for urban and rural residents
    # Calculate stroke and non-stroke counts for urban and rural residents
    urban_stroke_count = data[(data['stroke'] == 1) & (data['Residence_type'] == 'Urban')].shape[0]
    urban_no_stroke_count = data[(data['stroke'] == 0) & (data['Residence_type'] == 'Urban')].shape[0]

    rural_stroke_count = data[(data['stroke'] == 1) & (data['Residence_type'] == 'Rural')].shape[0]
    rural_no_stroke_count = data[(data['stroke'] == 0) & (data['Residence_type'] == 'Rural')].shape[0]
    # Calculate the total counts for urban and rural
    total_urban = urban_stroke_count + urban_no_stroke_count
    total_rural = rural_stroke_count + rural_no_stroke_count

    # Calculate percentages for urban and rural areas
    urban_stroke_percent = (urban_stroke_count / total_urban) * 100
    urban_no_stroke_percent = (urban_no_stroke_count / total_urban) * 100

    rural_stroke_percent = (rural_stroke_count / total_rural) * 100
    rural_no_stroke_percent = (rural_no_stroke_count / total_rural) * 100

    # Prepare data for pie charts
    labels = ['Stroke', 'No Stroke']
    colors = ['#fe346e', '#512b58']

    # Create subplots for the two pie charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Pie chart for individuals living in urban areas
    axes[0].pie([urban_stroke_percent, urban_no_stroke_percent], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0].set_title('Stroke Distribution for Individuals Living in Urban Areas')

    # Pie chart for individuals living in rural areas
    axes[1].pie([rural_stroke_percent, rural_no_stroke_percent], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].set_title('Stroke Distribution for Individuals Living in Rural Areas')

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

    # Hypertension and Stroke Pie Charts
    # Prepare data for pie charts
    hyper_stroke_counts = data.groupby(['hypertension', 'stroke']).size().unstack()

    # Calculate percentages for visualization
    hyper_stroke_percentages = hyper_stroke_counts.div(hyper_stroke_counts.sum(axis=1), axis=0) * 100

    # Calculate the counts of stroke and no-stroke for each hypertension status
    stroke_yes = hyper_stroke_counts.loc[1, 1]
    no_stroke_yes = hyper_stroke_counts.loc[1, 0]
    stroke_no = hyper_stroke_counts.loc[0, 1]
    no_stroke_no = hyper_stroke_counts.loc[0, 0]

    labels = ['Stroke', 'No Stroke']
    colors = ['#fe346e', '#512b58']

    # Create subplots for the two pie charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Pie chart for individuals with hypertension
    axes[0].pie([stroke_yes, no_stroke_yes], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0].set_title('Stroke Distribution for Hypertensive Individuals')

    # Pie chart for individuals without hypertension
    axes[1].pie([stroke_no, no_stroke_no], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].set_title('Stroke Distribution for Non-Hypertensive Individuals')

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

    #Stroke Distribution for individuals with unhealthy and healthy heart
        # Prepare data for pie charts
    hyper_stroke_counts = data.groupby(['heart_disease', 'stroke']).size().unstack()

    # Calculate percentages for visualization
    hyper_stroke_percentages = hyper_stroke_counts.div(hyper_stroke_counts.sum(axis=1), axis=0) * 100

    # Calculate the counts of stroke and no-stroke for each heart disease status
    stroke_yes = hyper_stroke_counts.loc[1, 1]
    no_stroke_yes = hyper_stroke_counts.loc[1, 0]
    stroke_no = hyper_stroke_counts.loc[0, 1]
    no_stroke_no = hyper_stroke_counts.loc[0, 0]

    labels = ['Stroke', 'No Stroke']
    colors = ['#fe346e', '#512b58']

    # Create subplots for the two pie charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Pie chart for individuals with heart disease
    axes[0].pie([stroke_yes, no_stroke_yes], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0].set_title('Stroke Distribution for individuals with unhealthy heart')

    # Pie chart for individuals without heart disease
    axes[1].pie([stroke_no, no_stroke_no], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].set_title('Stroke Distribution for individuals with healthy heart')

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

    #Stroke Distribution for Female and Male Individuals
    # Stroke counts for each gender
    stroke_female = data[data['gender'] == 'Female']['stroke'].sum()
    no_stroke_female = data[data['gender'] == 'Female']['stroke'].count() - stroke_female

    stroke_male = data[data['gender'] == 'Male']['stroke'].sum()
    no_stroke_male = data[data['gender'] == 'Male']['stroke'].count() - stroke_male


    # Labels and colors
    labels = ['Stroke', 'No Stroke']
    colors = ['#fe346e', '#512b58']

    # Create subplots for pie charts by gender
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Pie chart for Female individuals
    axes[0].pie([stroke_female, no_stroke_female], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0].set_title('Stroke Distribution for Female Individuals')

    # Pie chart for Male individuals
    axes[1].pie([stroke_male, no_stroke_male], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].set_title('Stroke Distribution for Male Individuals')

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

    #Stroke Distribution for individuals for each worktype
    # Calculate stroke and non-stroke counts for each work type
    govt_stroke_count = data[(data['stroke'] == 1) & (data['work_type'] == 'Govt_job')].shape[0]
    govt_no_stroke_count = data[(data['stroke'] == 0) & (data['work_type'] == 'Govt_job')].shape[0]

    private_stroke_count = data[(data['stroke'] == 1) & (data['work_type'] == 'Private')].shape[0]
    private_no_stroke_count = data[(data['stroke'] == 0) & (data['work_type'] == 'Private')].shape[0]

    self_employed_stroke_count = data[(data['stroke'] == 1) & (data['work_type'] == 'Self-employed')].shape[0]
    self_employed_no_stroke_count = data[(data['stroke'] == 0) & (data['work_type'] == 'Self-employed')].shape[0]

    children_stroke_count = data[(data['stroke'] == 1) & (data['work_type'] == 'children')].shape[0]
    children_no_stroke_count = data[(data['stroke'] == 0) & (data['work_type'] == 'children')].shape[0]

    never_worked_stroke_count = data[(data['stroke'] == 1) & (data['work_type'] == 'Never_worked')].shape[0]
    never_worked_no_stroke_count = data[(data['stroke'] == 0) & (data['work_type'] == 'Never_worked')].shape[0]
    # Total counts for each work type
    total_govt = govt_stroke_count + govt_no_stroke_count
    total_private = private_stroke_count + private_no_stroke_count
    total_self_employed = self_employed_stroke_count + self_employed_no_stroke_count
    total_children = children_stroke_count + children_no_stroke_count
    total_never_worked = never_worked_stroke_count + never_worked_no_stroke_count

    # Calculate percentages
    govt_percentages = [govt_stroke_count / total_govt * 100, govt_no_stroke_count / total_govt * 100]
    private_percentages = [private_stroke_count / total_private * 100, private_no_stroke_count / total_private * 100]
    self_employed_percentages = [self_employed_stroke_count / total_self_employed * 100, self_employed_no_stroke_count / total_self_employed * 100]
    children_percentages = [children_stroke_count / total_children * 100, children_no_stroke_count / total_children * 100]
    never_worked_percentages = [never_worked_stroke_count / total_never_worked * 100, never_worked_no_stroke_count / total_never_worked * 100]

    # Prepare data for pie charts
    labels = ['Stroke', 'No Stroke']
    colors = ['#fe346e', '#512b58']

    # Create subplots for the pie charts
    fig, axes = plt.subplots(1, 5, figsize=(12, 6))

    # Pie chart for each work type
    axes[0].pie(govt_percentages, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0].set_title('Govt Job')

    axes[1].pie(private_percentages, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].set_title('Private Job')

    axes[2].pie(self_employed_percentages, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[2].set_title('Self-employed')

    axes[3].pie(children_percentages, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[3].set_title('Children')

    axes[4].pie(never_worked_percentages, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[4].set_title('Never Worked')

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)


    #Stroke Distribution for individuals with different smoking status
    # Calculate stroke and non-stroke counts for each smoking status
    # Calculate stroke and non-stroke counts for each smoking status
    formerly_smoked_stroke_count = data[(data['stroke'] == 1) & (data['smoking_status'] == 'formerly smoked')].shape[0]
    formerly_smoked_no_stroke_count = data[(data['stroke'] == 0) & (data['smoking_status'] == 'formerly smoked')].shape[0]

    never_smoked_stroke_count = data[(data['stroke'] == 1) & (data['smoking_status'] == 'never smoked')].shape[0]
    never_smoked_no_stroke_count = data[(data['stroke'] == 0) & (data['smoking_status'] == 'never smoked')].shape[0]

    smokes_stroke_count = data[(data['stroke'] == 1) & (data['smoking_status'] == 'smokes')].shape[0]
    smokes_no_stroke_count = data[(data['stroke'] == 0) & (data['smoking_status'] == 'smokes')].shape[0]



# Total counts for each smoking status
    total_formerly_smoked = formerly_smoked_stroke_count + formerly_smoked_no_stroke_count
    total_never_smoked = never_smoked_stroke_count + never_smoked_no_stroke_count
    total_smokes = smokes_stroke_count + smokes_no_stroke_count

    # Calculate percentages
    formerly_smoked_percentages = [
        formerly_smoked_stroke_count / total_formerly_smoked * 100,
        formerly_smoked_no_stroke_count / total_formerly_smoked * 100
    ]
    never_smoked_percentages = [
        never_smoked_stroke_count / total_never_smoked * 100,
        never_smoked_no_stroke_count / total_never_smoked * 100
    ]
    smokes_percentages = [
        smokes_stroke_count / total_smokes * 100,
        smokes_no_stroke_count / total_smokes * 100
    ]

    # Prepare data for pie charts
    labels = ['Stroke', 'No Stroke']
    colors = ['#fe346e', '#512b58']

    # Create subplots for the pie charts
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    # Pie chart for each smoking status
    axes[0].pie(formerly_smoked_percentages, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0].set_title('Formerly Smoked')

    axes[1].pie(never_smoked_percentages, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].set_title('Never Smoked')

    axes[2].pie(smokes_percentages, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[2].set_title('Currently Smokes')


    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

# Conclusion Section
st.header('Conclusion')
st.write("""
From the visualizations and descriptive statistics, we can observe the following trends:
- Age, average glucose level, and BMI have varying distributions, with older age and higher glucose levels possibly correlating with stroke incidents.
- The box plots further show the relationship between these factors and stroke status, indicating potential trends.
- Further model-building and analysis can help explore these relationships in more detail, particularly when building predictive models using TensorFlow or other machine learning libraries.
""")
