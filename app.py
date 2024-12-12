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

st.header('Purpose of Exploration')
st.write("""
The purpose of the exploration of the Stroke Prediction Dataset from Kaggle is to delve into the relationships 
         and trends among various health-related parameters, such as age, glucose levels, BMI, hypertension, and 
         heart disease, to enhance our understanding and predictive capabilities regarding stroke occurrence. 
         By analyzing these variables through methods like correlation matrices, we can identify significant 
         predictors and their interactions, which are crucial for developing robust models for stroke prediction. 
         This exploration aims to uncover insights that could inform healthcare professionals and researchers 
         about potential risk factors, ultimately leading to better preventive measures and treatment strategies 
         tailored to individual risk profiles. This investigation not only aids in the advancement of medical 
         research but also contributes to the development of targeted interventions that could mitigate the risk 
         of strokes in vulnerable populations.
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
    **Insights**:\n
    Age Distribution:

    The age histogram shows a broad distribution with a notable concentration of individuals in their 50s and 60s. 
    The frequency of younger age groups (below 30) appears considerably lower, indicating that the dataset primarily
    includes middle-aged and elderly individuals. This distribution suggests that middle-aged and elderly groups are 
    a key demographic for analyzing stroke risk, reflecting known epidemiological data that stroke risk increases with age.
    """)

    # Glucose Level Distribution
    ax[1].hist(data['avg_glucose_level'], bins=20, color='lightgreen', edgecolor='black')
    ax[1].set_title('Average Glucose Level Distribution')
    ax[1].set_xlabel('Avg Glucose Level')
    ax[1].set_ylabel('Frequency')

    st.write(""" 
    Average Glucose Level Distribution:

    The histogram for glucose levels is skewed left, with a high concentration of individuals in the 50 to 125 mg/dL 
    range, peaking around 75 to 100 mg/dL. The left skewness indicates a subset of the population with higher 
    glucose levels, which could be an indicator of diabetes or prediabetes conditions, both of which are risk 
    factors for stroke.
    """)

    # BMI Distribution
    ax[2].hist(data['bmi'], bins=20, color='salmon', edgecolor='black')
    ax[2].set_title('BMI Distribution')
    ax[2].set_xlabel('BMI')
    ax[2].set_ylabel('Frequency')

    st.write("""
    BMI Distribution:

    The BMI distribution is prominently left-skewed, with the highest frequency around the 25 to 30 range, 
    categorizing this peak within the overweight classification. The tail extending towards higher BMI values 
    indicates the presence of a significant number of obese individuals, which is another important stroke risk factor.
             """)
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
    st.write(""" 
    The correlation matrix provided visualizes the relationships between various health-related variables, 
             indicating several notable associations. Age shows moderate positive correlations with stroke, 
             hypertension, and heart disease, suggesting that as individuals age, their risk for these conditions 
             tends to increase. Interestingly, average glucose levels also show a mild positive correlation with 
             stroke, indicating that higher glucose levels might be linked to an increased risk of stroke. Notably, 
             body mass index (BMI) shows very little to no correlation with stroke or heart disease, suggesting 
             that in this dataset, BMI alone is not a significant direct indicator of these conditions. The matrix 
             uses a color gradient from deep red to deep blue to represent the strength of correlation from strong 
             positive to strong negative, respectively, with neutral correlations appearing closer to white. Overall, 
             this matrix is a valuable tool for identifying and understanding potential risk factors in healthcare, 
             potentially aiding in predictive modeling and risk assessment.
""")

# Pie Charts for Worktype, Residence Type, and Age
elif chart_type == "Pie Charts":
    st.subheader('Pie Charts for Worktype, Residence Type, and Age')

    # Worktype Pie Chart
    worktype_counts = data['work_type'].value_counts(dropna=False)

    fig1, ax1 = plt.subplots()
    wedges, texts, autotexts = ax1.pie(worktype_counts, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))

    # Set title for the pie chart
    ax1.set_title('Work Type Distribution')

    # Improve legibility of autopct labels
    plt.setp(autotexts, size=8, weight="bold", color="white")  # Adjust the color to white for better visibility on pastel colors

    # Create a legend with labels placed in a similar manner to your example
    ax1.legend(wedges, [f'{label}, {prop*100:.1f}%' for label, prop in zip(worktype_counts.index, worktype_counts/worktype_counts.sum())],
            title="Work Type",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()  # Adjust layout to make room for the legend
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

    # Define age bins
    age_bins = pd.cut(data['age'], bins=[0, 18, 35, 50, 65, 80, 100])
    age_group_counts = age_bins.value_counts(sort=False)

    fig3, ax3 = plt.subplots()
    wedges, texts, autotexts = ax3.pie(age_group_counts, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))

    # Set title for the pie chart
    ax3.set_title('Age Group Distribution')

    # Improve legibility of autopct labels
    plt.setp(autotexts, size=8, weight="bold", color="white")  # Adjust the color to white for better visibility on pastel colors

    # Create a legend with labels and their respective percentages
    ax3.legend(wedges, [f'{label}, {prop*100:.1f}%' for label, prop in zip(age_group_counts.index, age_group_counts/age_group_counts.sum())],
            title="Age Groups",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()  # Adjust layout to make room for the legend
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
    st.subheader('Stroke Distribution for Individuals Living in Urban Areas and Rural Areas')
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
    colors = ['#a7bed3', '#dab894']

    # Create subplots for the two pie charts
    fig, axes = plt.subplots(1, 2, figsize=(8, 15))

    # Pie chart for individuals living in urban areas
    axes[0].pie([urban_stroke_percent, urban_no_stroke_percent], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0].set_title('Stroke Distribution for Individuals Living in Urban Areas\n\n')

    # Pie chart for individuals living in rural areas
    axes[1].pie([rural_stroke_percent, rural_no_stroke_percent], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].set_title('Stroke Distribution for Individuals Living in Rural Areas')

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

    # Hypertension and Stroke Pie Charts
    # Prepare data for pie charts
    st.subheader('Pie Chart for Individuals with Hypertension and Stroke')
    hyper_stroke_counts = data.groupby(['hypertension', 'stroke']).size().unstack()

    # Calculate percentages for visualization
    hyper_stroke_percentages = hyper_stroke_counts.div(hyper_stroke_counts.sum(axis=1), axis=0) * 100

    # Calculate the counts of stroke and no-stroke for each hypertension status
    stroke_yes = hyper_stroke_counts.loc[1, 1]
    no_stroke_yes = hyper_stroke_counts.loc[1, 0]
    stroke_no = hyper_stroke_counts.loc[0, 1]
    no_stroke_no = hyper_stroke_counts.loc[0, 0]

    labels = ['Stroke', 'No Stroke']
    colors = ['#f1ffc4', '#ffcaaf']

    # Create subplots for the two pie charts
    fig, axes = plt.subplots(1, 2, figsize=(8, 15))

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
    st.subheader('Stroke Distribution for individuals with unhealthy and healthy heart')
    hyper_stroke_counts = data.groupby(['heart_disease', 'stroke']).size().unstack()

    # Calculate percentages for visualization
    hyper_stroke_percentages = hyper_stroke_counts.div(hyper_stroke_counts.sum(axis=1), axis=0) * 100

    # Calculate the counts of stroke and no-stroke for each heart disease status
    stroke_yes = hyper_stroke_counts.loc[1, 1]
    no_stroke_yes = hyper_stroke_counts.loc[1, 0]
    stroke_no = hyper_stroke_counts.loc[0, 1]
    no_stroke_no = hyper_stroke_counts.loc[0, 0]

    labels = ['Stroke', 'No Stroke']
    colors = ['#d0d0fe', '#f9deff']

    # Create subplots for the two pie charts
    fig, axes = plt.subplots(1, 2, figsize=(8, 15))

    # Pie chart for individuals with heart disease
    axes[0].pie([stroke_yes, no_stroke_yes], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0].set_title('Stroke Distribution for individuals with unhealthy heart\n\n')

    # Pie chart for individuals without heart disease
    axes[1].pie([stroke_no, no_stroke_no], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].set_title('Stroke Distribution for individuals with healthy heart')

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

    #Stroke Distribution for Female and Male Individuals
    # Stroke counts for each gender
    st.subheader('Stroke Distribution for Female and Male Individuals')
    stroke_female = data[data['gender'] == 'Female']['stroke'].sum()
    no_stroke_female = data[data['gender'] == 'Female']['stroke'].count() - stroke_female

    stroke_male = data[data['gender'] == 'Male']['stroke'].sum()
    no_stroke_male = data[data['gender'] == 'Male']['stroke'].count() - stroke_male


    # Labels and colors
    labels = ['Stroke', 'No Stroke']
    colors = ['#fb6f92', '#f6d7e8']

    # Create subplots for pie charts by gender
    fig, axes = plt.subplots(1, 2, figsize=(8, 15))

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
    st.subheader('Stroke Distribution for Individuals for Each Work Type')
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
    colors = ['#c7ceea', '#f28ece']

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
    st.subheader('Stroke Distribution for Individuals with Different Smoking Status')
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
    colors = ['#fac3a5', '#cb8d9a']

    # Create subplots for the pie charts
    fig, axes = plt.subplots(1, 3, figsize=(8,15))

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


    # Total counts for each marital status
    st.subheader('Stroke Distribution for Individuals according to Marital Status')
    # Calculate stroke and non-stroke counts for each marital status
    married_stroke_count = data[(data['stroke'] == 1) & (data['ever_married'] == 'Yes')].shape[0]
    married_no_stroke_count = data[(data['stroke'] == 0) & (data['ever_married'] == 'Yes')].shape[0]

    not_married_stroke_count = data[(data['stroke'] == 1) & (data['ever_married'] == 'No')].shape[0]
    not_married_no_stroke_count = data[(data['stroke'] == 0) & (data['ever_married'] == 'No')].shape[0]

    # Calculate total counts for both groups
    total_married = married_stroke_count + married_no_stroke_count
    total_not_married = not_married_stroke_count + not_married_no_stroke_count

    # Calculate percentages for married and not married individuals
    married_stroke_percent = (married_stroke_count / total_married) * 100
    married_no_stroke_percent = (married_no_stroke_count / total_married) * 100

    not_married_stroke_percent = (not_married_stroke_count / total_not_married) * 100
    not_married_no_stroke_percent = (not_married_no_stroke_count / total_not_married) * 100

    # Prepare data for pie charts
    labels = ['Stroke', 'No Stroke']
    colors = ['#f1ffc4', '#ffcaaf']

    # Create subplots for the two pie charts
    fig, axes = plt.subplots(1, 2, figsize=(8,15))

    # Pie chart for individuals who have ever been married
    axes[0].pie([married_stroke_percent, married_no_stroke_percent], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[0].set_title('Stroke Distribution for Married Individuals\n')

    # Pie chart for individuals who have never been married
    axes[1].pie([not_married_stroke_percent, not_married_no_stroke_percent], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].set_title('Stroke Distribution for Not Married Individuals')

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)
# Conclusion Section
st.header('Conclusion')
st.write("""
From the visualizations and descriptive statistics, several trends are noticeable. Age, average glucose levels, 
    and BMI exhibit diverse distributions, where older age groups and higher glucose levels appear to 
    correlate with increased stroke incidents. The box plots further elucidate the relationship between these 
    factors and stroke occurrence, highlighting potential trends that could be instrumental for deeper analysis. 
    Model-building and further analytical exploration, especially using tools like TensorFlow or other machine 
    learning libraries, could significantly enhance our understanding of these relationships. Overall, 
    the data suggests that health conditions such as hypertension and heart health, alongside lifestyle 
    choices like smoking, are major influencers of stroke risk. However, demographic factors like gender 
    and marital status also play a role, albeit to a lesser extent.
""")
