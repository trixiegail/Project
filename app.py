import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Load the dataset with st.cache_data for caching
@st.cache_data
def load_data():
    data = pd.read_csv('healthcare-dataset-stroke-data.csv')
    data.replace("Unknown", pd.NA, inplace=True)  # Replace "Unknown" with pd.NA
    return data

data = load_data()

# Initialize cleaned data variable
data_cleaned = data.copy()

# Sidebar navigation for the project lifecycle
sections = [
    "Data Discovery",
    "Data Preparation",
    "Model Planning",
    # "Model Building",
    "Communication of Results",
    # "Operationalization",
    "Conclusion"
]
selection = st.sidebar.radio("Navigate through the lifecycle:", sections)

# Data Discovery Section
if selection == "Data Discovery":
    st.header('Data Discovery')
    st.write("This section focuses on understanding the dataset, identifying relevant features, and forming hypotheses.")

    # Show raw data if checkbox selected
    if st.checkbox('Show raw data'):
        st.write(data.head())

    # Descriptive statistics
    st.subheader('Descriptive Statistics')
    st.write(data.describe())

    # Create a dropdown for selecting the type of analysis to display
    analysis_type = st.selectbox('Select the analysis type', 
                                  ['Central Tendency', 
                                   'Spread of the Data', 
                                   'Min, Max, and Range', 
                                   'Percentiles'])

    numerical_cols = ['age', 'avg_glucose_level', 'bmi']

    # Initialize an empty DataFrame for the analysis results
    results = pd.DataFrame()

    if analysis_type == 'Central Tendency':
        st.subheader('Central Tendency')
        results['Metric'] = ['Mean', 'Median', 'Mode']
        for col in numerical_cols:
            results[col] = [data[col].mean(), data[col].median(), data[col].mode()[0]]
        st.table(results)

    elif analysis_type == 'Spread of the Data':
        st.subheader('Spread of the Data')
        results['Metric'] = ['Standard Deviation', 'Variance']
        for col in numerical_cols:
            results[col] = [data[col].std(), data[col].var()]
        st.table(results)

    elif analysis_type == 'Min, Max, and Range':
        st.subheader('Min, Max, and Range')
        results['Metric'] = ['Min', 'Max', 'Range']
        for col in numerical_cols:
            results[col] = [data[col].min(), data[col].max(), data[col].max() - data[col].min()]
        st.table(results)

    elif analysis_type == 'Percentiles':
        st.subheader('Percentiles')
        results['Metric'] = ['25th Percentile', '50th Percentile', '75th Percentile']
        for col in numerical_cols:
            results[col] = [np.percentile(data[col].dropna(), 25), 
                            np.percentile(data[col].dropna(), 50), 
                            np.percentile(data[col].dropna(), 75)]
        st.table(results)

# Data Preparation Section (Improved Data Cleaning)
elif selection == "Data Preparation":
    st.header('Data Preparation')
    st.write("""
    In this step, we clean the data by handling missing values and preparing it for modeling.
    """)

    # Step-by-step data cleaning
    st.subheader('Data Cleaning Process')

    # Check for missing values before cleaning
    st.write('### Missing Values Before Cleaning:')
    st.write(data.isna().sum())

    # Data cleaning operations
    st.write('Cleaning Data: Replacing "Unknown" with NA, dropping rows with missing values...')

    # Replace string "Unknown" and drop NaN values
    data_cleaned.replace("Unknown", pd.NA, inplace=True)  # Already handled in load_data but mentioned for clarity
    data_cleaned.dropna(inplace=True)  # Drop rows with missing values
    data_cleaned.reset_index(drop=True, inplace=True)  # Reset index after dropping rows

    # Show cleaned dataset statistics
    st.write('### Data After Cleaning:')
    st.write(f"Number of rows after cleaning: {len(data_cleaned)}")

    # Check for missing values after cleaning
    st.write('### Missing Values After Cleaning:')
    st.write(data_cleaned.isna().sum())

    # Display first 5 rows of cleaned data
    st.write('### First 5 Rows of Cleaned Data:')
    st.write(data_cleaned.head())

# Model Planning Section
elif selection == "Model Planning":
    st.header('Model Planning')
    st.write("In this step, we plan the model-building process, selecting algorithms and features.")

    st.subheader('Correlation Matrix for Feature Selection')
    selected_columns = ['age', 'bmi', 'avg_glucose_level']
    correlation_matrix = data_cleaned[selected_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("""
    The correlation matrix above shows the relationships among the selected features: age, BMI, and average glucose level. 
    Here's a brief interpretation of these correlations:

    1. **Age and Average Glucose Level**: 
       Typically, there is a positive correlation, meaning that as age increases, average glucose levels also tend to increase. This trend suggests that older individuals may have a higher risk for diabetes and related conditions.

    2. **Age and BMI**: 
       The correlation between age and BMI can vary. In some cases, it is positive, indicating that BMI tends to increase with age. However, in older populations, it might be negative due to weight loss related to health issues.

    3. **BMI and Average Glucose Level**: 
       There is often a positive correlation between these two variables. Higher BMI values are associated with obesity, which can lead to insulin resistance and elevated glucose levels. This highlights the importance of maintaining a healthy weight to mitigate the risk of metabolic diseases.
    """)


# # Model Building Section
# elif selection == "Model Building":
#     st.header('Model Building')
#     st.write("""
#     In this step, we train machine learning models using the prepared data.
#     """)

# Communication of Results Section
elif selection == "Communication of Results":
    st.header('Communication of Results')
    st.write("""
    This step focuses on presenting the findings through visualizations and reports to stakeholders.
    Use the dropdown menu to view different visualizations.
    """)

    # Dropdown for selecting the chart to display
    chart_type = st.selectbox('Select chart to display', 
                              ['Distributions of Age, Glucose Level, and BMI',
                               'Box Plots by Stroke Status',
                               'Pie Chart: Work Type',
                               'Pie Chart: Residence Type',
                               'Pie Chart: Age Distribution'])

    if chart_type == 'Distributions of Age, Glucose Level, and BMI':
        st.subheader('Distributions of Age, Average Glucose Level, and BMI')
        st.write("""
        This chart displays the distribution of three key numerical variables: 
        Age, Average Glucose Level, and BMI. Each histogram helps us understand the 
        spread and frequency of these variables within the dataset. Observing these 
        distributions is important to detect any skewness or outliers that may affect 
        model performance in later stages.
        """)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Age Distribution
        ax[0].hist(data_cleaned['age'], bins=20, color='skyblue', edgecolor='black')
        ax[0].set_title('Age Distribution')
        ax[0].set_xlabel('Age')
        ax[0].set_ylabel('Frequency')

        # Glucose Level Distribution
        ax[1].hist(data_cleaned['avg_glucose_level'], bins=20, color='lightgreen', edgecolor='black')
        ax[1].set_title('Average Glucose Level Distribution')
        ax[1].set_xlabel('Avg Glucose Level')
        ax[1].set_ylabel('Frequency')

        # BMI Distribution
        ax[2].hist(data_cleaned['bmi'], bins=20, color='salmon', edgecolor='black')
        ax[2].set_title('BMI Distribution')
        ax[2].set_xlabel('BMI')
        ax[2].set_ylabel('Frequency')

        st.pyplot(fig)

    elif chart_type == 'Box Plots by Stroke Status':
        st.subheader('Box Plots of Age, Average Glucose Level, and BMI by Stroke Status')
        st.write("""
        The box plots compare the distribution of Age, Average Glucose Level, and BMI across 
        stroke status (either 0 for no stroke, or 1 for stroke). These box plots allow us to 
        visualize the spread of the variables and detect potential relationships between 
        the variables and stroke occurrence. For example, we may observe whether older age 
        or higher glucose levels are more prevalent among stroke patients.
        """)
        filtered_data = data_cleaned[['age', 'avg_glucose_level', 'bmi', 'stroke']]

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

    elif chart_type == 'Pie Chart: Work Type':
        st.subheader('Pie Chart of Work Type Distribution')
        st.write("""
        This pie chart shows the distribution of different work types in the dataset. 
        Understanding the prevalence of different work types can help identify if 
        certain occupations have higher or lower risks of stroke, which may be useful 
        in designing preventive health measures.
        """)
        work_type_counts = data_cleaned['work_type'].value_counts()

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(work_type_counts, labels=work_type_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
        ax.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.

        st.pyplot(fig)

    elif chart_type == 'Pie Chart: Residence Type':
        st.subheader('Pie Chart of Residence Type Distribution')
        st.write("""
        This pie chart shows the proportion of individuals living in urban or rural areas. 
        It helps to see whether the dataset includes more people from urban or rural 
        settings and if this could influence stroke prediction. Differences in healthcare 
        access and lifestyle may vary by residence type.
        """)
        residence_counts = data_cleaned['Residence_type'].value_counts()

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(residence_counts, labels=residence_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3"))
        ax.axis('equal')

        st.pyplot(fig)

    elif chart_type == 'Pie Chart: Age Distribution':
        st.subheader('Pie Chart of Age Group Distribution')
        st.write("""
        This pie chart categorizes the patients into age groups and displays the proportion 
        of each group in the dataset. Grouping the ages helps us to easily identify whether 
        the dataset contains more younger or older individuals and if age can be a significant 
        factor for stroke risk prediction.
        """)
        age_groups = pd.cut(data_cleaned['age'], bins=[0, 18, 35, 55, 80, 120], labels=['0-18', '19-35', '36-55', '56-80', '81+'])
        age_group_counts = age_groups.value_counts()

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("coolwarm", len(age_group_counts)))
        ax.axis('equal')

        st.pyplot(fig)


# # Operationalization Section
# elif selection == "Operationalization":
#     st.header('Operationalization')
#     st.write("In this phase, the model is deployed for practical use, such as creating APIs or embedding it into applications.")

# Conclusion Section
elif selection == "Conclusion":
    st.header('Conclusion')
    st.write("""
    From the visualizations and descriptive statistics, we can observe the following trends:
    - Age, average glucose level, and BMI have varying distributions, with older age and higher glucose levels possibly correlating with stroke incidents.
    - The box plots further show the relationship between these factors and stroke status, indicating potential trends.
    - Further model-building and analysis can help explore these relationships in more detail, particularly when building predictive models using TensorFlow or other machine learning libraries.
    """)
