import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data():
    data = pd.read_csv('healthcare-dataset-stroke-data.csv')
    data.replace("Unknown", pd.NA, inplace=True)
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

data = load_data()

# Sidebar for navigation
st.sidebar.header('Navigation')
page = st.sidebar.selectbox(
    "Select the page",
    ["Introduction", "Visualizations", "Logistic Regression Analysis"]
)

# Introduction Section
if page == "Introduction":
    st.title('Stroke Prediction Dataset Exploration')
    st.header('Introduction')
    st.write("""
    This application presents an exploration of the Stroke Prediction Dataset from Kaggle. 
    The dataset includes various health-related parameters, such as age, glucose levels, and BMI, 
    to predict whether a patient is likely to have a stroke.
    """)

    if st.checkbox('Show raw data'):
        st.write(data.head())

    st.subheader('Descriptive Statistics')
    st.write(data.describe())

# Visualizations Section
elif page == "Visualizations":
    st.title("Visualizations")
    st.write("Explore different visualizations for the dataset.")

    # Age, Glucose, BMI Histograms
    st.subheader('Distributions of Age, Average Glucose Level, and BMI')
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].hist(data['age'], bins=20, color='skyblue', edgecolor='black')
    ax[0].set_title('Age Distribution')
    ax[0].set_xlabel('Age')
    ax[0].set_ylabel('Frequency')

    ax[1].hist(data['avg_glucose_level'], bins=20, color='lightgreen', edgecolor='black')
    ax[1].set_title('Average Glucose Level Distribution')
    ax[1].set_xlabel('Avg Glucose Level')
    ax[1].set_ylabel('Frequency')

    ax[2].hist(data['bmi'], bins=20, color='salmon', edgecolor='black')
    ax[2].set_title('BMI Distribution')
    ax[2].set_xlabel('BMI')
    ax[2].set_ylabel('Frequency')

    st.write("""
    **Insights**:\n
    Age Distribution:

    The age histogram shows a broad distribution with a notable concentration of individuals in their 50s and 60s. 
    The frequency of younger age groups (below 30) appears considerably lower, indicating that the dataset primarily
    includes middle-aged and elderly individuals. This distribution suggests that middle-aged and elderly groups are 
    a key demographic for analyzing stroke risk, reflecting known epidemiological data that stroke risk increases with age.
    """)

    st.write(""" 
    Average Glucose Level Distribution:

    The histogram for glucose levels is skewed left, with a high concentration of individuals in the 50 to 125 mg/dL 
    range, peaking around 75 to 100 mg/dL. The left skewness indicates a subset of the population with higher 
    glucose levels, which could be an indicator of diabetes or prediabetes conditions, both of which are risk 
    factors for stroke.
    """)

    st.write("""
    BMI Distribution:

    The BMI distribution is prominently left-skewed, with the highest frequency around the 25 to 30 range, 
    categorizing this peak within the overweight classification. The tail extending towards higher BMI values 
    indicates the presence of a significant number of obese individuals, which is another important stroke risk factor.
    """)



    st.pyplot(fig)

# Logistic Regression Analysis Section
elif page == "Logistic Regression Analysis":
    st.title("Logistic Regression Analysis")
    st.write("Exploring the predictors of stroke using logistic regression.")

    # Preprocessing for logistic regression
    data = data.dropna(subset=['bmi'])
    data = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=False)
    data = data.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)

    # Define X and y
    X = data.drop(columns=['id', 'stroke'])
    y = data['stroke']

    # Scale predictors
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X = sm.add_constant(X)  # Add constant after scaling

    print(X.dtypes)

    # Add constant to X
    X = sm.add_constant(X)

    # Fit logistic regression model
    model = sm.Logit(y, X)
try:
    result = model.fit()
except np.linalg.LinAlgError as e:
    st.error(f"Error fitting model: {e}")
    result = model.fit_regularized(method='l1', alpha=1.0)

    print(result.summary())

    # Calculate and filter odds ratios
    odds_ratios = np.exp(result.params)
    print(odds_ratios)

    odds_ratios_filtered = odds_ratios[(odds_ratios > 0.01) & (odds_ratios < 1000)]

    # Print filtered odds ratios for inspection
    print(odds_ratios_filtered)

    fig, ax = plt.subplots(figsize=(10, 6))
    odds_ratios_filtered.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Odds Ratios for Predictors of Stroke")
    ax.set_xlabel("Features")
    ax.set_ylabel("Odds Ratio")
    ax.axhline(y=1, color='red', linestyle='--')  # Line indicating no effect
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    st.write("""
    **Insights**:
    - The odds ratios represent the likelihood of stroke occurrence for each feature.
    - Values greater than 1 indicate a positive association, while values less than 1 indicate a negative association.
    - Features with odds ratios close to 1 have minimal impact on stroke prediction.
    """)

    features = ['age', 'avg_glucose_level', 'bmi']  # Add other features of interest

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(features), figsize=(15, 5), sharey=True)

    for i, feature in enumerate(features):
        # Generate a range of values for the feature
        feature_range = np.linspace(data[feature].min(), data[feature].max(), 100)

        # Create a new DataFrame with fixed values for other features
        X_plot = X.mean().to_frame().T
        X_plot = pd.concat([X_plot] * len(feature_range), ignore_index=True)
        X_plot[feature] = feature_range

        # Predict probabilities
        predicted_probs = result.predict(X_plot)

        # Plot the logistic regression curve
        axes[i].plot(feature_range, predicted_probs, color='blue', label=f'{feature}')
        axes[i].scatter(data[feature], data['stroke'], alpha=0.3, label='Actual Data', color='orange')
        axes[i].set_xlabel(feature)
        axes[i].set_title(f'Logistic Regression for {feature}')
        axes[i].grid(alpha=0.7)
        if i == 0:
            axes[i].set_ylabel('Probability of Stroke')
        axes[i].legend()

    plt.tight_layout()
    st.pyplot(fig)
    st.write("""
    **Insights**:
    - These plots illustrate how the probability of stroke changes with each feature (age, glucose level, BMI).
    - A steep curve suggests a stronger relationship between the feature and stroke probability.
    - Outliers and actual data points help validate the model's predictions.
    """)

    categorical_feature = 'gender_Male'  # Replace with your categorical variable of interest

    # Get unique categories for the feature
    categories = [0, 1]  # For one-hot encoded variables (e.g., 0 for Female, 1 for Male)

    # Create a new DataFrame with constant values for other features
    X_plot = X.mean().to_frame().T
    X_plot = pd.concat([X_plot] * len(categories), ignore_index=True)

    # Update the categorical feature with each category
    X_plot[categorical_feature] = categories

    # Predict probabilities
    predicted_probs = result.predict(X_plot)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(categories, predicted_probs, color=['orange', 'blue'], tick_label=['Female', 'Male'])
    ax.set_xlabel('Category')
    ax.set_ylabel('Predicted Probability of Stroke')
    ax.set_title(f'Logistic Regression for {categorical_feature}')
    ax.grid(alpha=0.7)
    st.pyplot(fig)
    st.write("""
    **Insights**:
    - Males appear to have a slightly different stroke probability compared to females.
    - This chart highlights the categorical impact of gender on stroke prediction.
    """)

    smoking_columns = [col for col in X.columns if col.startswith('smoking_status_')]

    # Create a DataFrame for predictions
    X_plot = X.mean().to_frame().T
    X_plot = pd.concat([X_plot] * len(smoking_columns), ignore_index=True)

    # Vary one smoking-related column at a time
    for i, col in enumerate(smoking_columns):
        X_plot[col] = [1 if j == i else 0 for j in range(len(smoking_columns))]

    # Predict probabilities
    predicted_probs = result.predict(X_plot)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(smoking_columns, predicted_probs, color='skyblue')
    ax.set_xlabel('Smoking Categories')
    ax.set_ylabel('Predicted Probability of Stroke')
    ax.set_title('Logistic Regression for Smoking Categories')
    plt.xticks(rotation=45)
    ax.grid(alpha=0.7)
    st.pyplot(fig)
    st.write("""
    **Insights**:
    - Different smoking statuses show varying probabilities for stroke.
    - This indicates that smoking habits could significantly influence stroke risk.
    """)

    # Categorical feature (one-hot encoded column)
    categorical_feature = 'Residence_type_Urban'  # Replace with your actual one-hot encoded column

    # Categories for one-hot encoded column (0 for Rural, 1 for Urban)
    categories = [0, 1]

    # Create a new DataFrame with constant values for other features
    X_plot = X.mean().to_frame().T
    X_plot = pd.concat([X_plot] * len(categories), ignore_index=True)

    # Update the categorical feature with each category
    X_plot[categorical_feature] = categories

    # Predict probabilities
    predicted_probs = result.predict(X_plot)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(categories, predicted_probs, color=['green', 'blue'], tick_label=['Rural', 'Urban'])
    ax.set_xlabel('Residence Type')
    ax.set_ylabel('Predicted Probability of Stroke')
    ax.set_title(f'Logistic Regression for {categorical_feature}')
    ax.grid(alpha=0.7)
    st.pyplot(fig)
    st.write("""
    **Insights**:
    - Individuals living in urban areas have a different stroke probability compared to those in rural areas.
    - This suggests lifestyle or healthcare access differences may play a role.
    """)

    # Categorical feature for marital status
    categorical_feature = 'ever_married_Yes'  # Replace with your actual one-hot encoded column

    # Categories for the one-hot encoded column (0 for No, 1 for Yes)
    categories = [0, 1]

    # Create a new DataFrame with constant values for other features
    X_plot = X.mean().to_frame().T
    X_plot = pd.concat([X_plot] * len(categories), ignore_index=True)

    # Update the categorical feature with each category
    X_plot[categorical_feature] = categories

    # Predict probabilities
    predicted_probs = result.predict(X_plot)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(categories, predicted_probs, color=['orange', 'blue'], tick_label=['Not Married', 'Married'])
    ax.set_xlabel('Marital Status')
    ax.set_ylabel('Predicted Probability of Stroke')
    ax.set_title(f'Logistic Regression for {categorical_feature}')
    ax.grid(alpha=0.7)
    st.pyplot(fig)
    st.write("""
    **Insights**:
    - Marital status appears to have an impact on stroke probability, with married individuals showing a distinct pattern.
    - This could correlate with social or economic factors.
    """)

st.sidebar.info("Explore the dataset, visualizations, and logistic regression results.")