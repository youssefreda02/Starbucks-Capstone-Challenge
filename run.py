# app/app.py

import streamlit as st
import pandas as pd

# Title of the app
st.title("Starbucks Promotional Offers Recommendation System")


# Add LinkedIn and GitHub icons with hyperlinks to your profile
st.markdown(
    """
    <div style="position: fixed; bottom: 2%; left: -2%; width: 100%; padding: 10px; text-align: right;">
        <a href="https://www.linkedin.com/in/youssef-reda-ba3b35194/" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/8/81/LinkedIn_icon.svg" width="30" height="30">
        </a>
        <a href="https://github.com/youssefreda02/Starbucks-Capstone-Challenge" target="_blank" style="margin-left: 10px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" width="30" height="30">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
# Load the offer recommendations DataFrame
@st.cache_data  # Cache the data to improve performance
def load_data():
    # Choose the appropriate loading method based on your saved format
    df = pd.read_csv('/data/offer_recommendations.csv')
    return df

offer_recommendations = load_data()

# Sidebar for user input
st.sidebar.header("User Demographics")

# Collect user input
age = st.sidebar.number_input("Enter your age:", min_value=18, max_value=100, value=30)
gender = st.sidebar.selectbox("Select your gender:", options=["Male", "Female", "Other"])

# Function to determine the cluster based on age and gender
def get_cluster(age, gender):
    if gender.lower() == 'female':
        gen = 0
    elif gender.lower() == 'male':
        gen = 1
    else:
        gen = 2

    # Determine the age group
    if 18 <= age <= 24:
        age_group = '18-24'
    elif 25 <= age <= 34:
        age_group = '25-34'
    elif 35 <= age <= 44:
        age_group = '35-44'
    elif 45 <= age <= 54:
        age_group = '45-54'
    elif 55 <= age <= 64:
        age_group = '55-64'
    elif 65 <= age <= 74:
        age_group = '65-74'
    else:
        age_group = '75+'

    return gen, age_group

# Get the cluster for the input age and gender
gender_code, age_group = get_cluster(age, gender)
 # Filter recommendations based on demographics
recommendation = offer_recommendations[
    (offer_recommendations['gender'] == gender_code) &
    (offer_recommendations['age_group'] == age_group)
]

if recommendation.empty:
    st.warning("No recommendation available for the provided demographics.")
else:
    # Select the highest response offer
    best_offer = recommendation.iloc[0]
    second_offer = recommendation.iloc[1]

    # Generate the response message based on the offer type for the best offer
    if best_offer['offer_type'] == 'informational':
        response = (
            f"{int(best_offer['response_rate'] * 100)}% of the informational offer are completed by people with gender: {gender} "
            f"and aged between {best_offer['age_group']}, within {best_offer['duration']} days. "
            f"\nRecommended channels: {best_offer['channels']}."
        )
    elif best_offer['offer_type'] == 'bogo':
        response = (
            f"{int(best_offer['response_rate'] * 100)}% of the Buy-One-Get-One (BOGO) offer with a reward of \${best_offer['reward']} "
            f"and a minimum spend requirement of \${best_offer['difficulty']}, are completed by people with gender: {gender} "
            f"and aged between {best_offer['age_group']}, within {best_offer['duration']} days. "
            f"\nRecommended channels: {best_offer['channels']}."
        )
    else:
        response = (
            f"{int(best_offer['response_rate'] * 100)}% of the discount offer with a reward of \${best_offer['reward']} "
            f"and a minimum spend requirement of \${best_offer['difficulty']}, are completed by people with gender: {gender} "
            f"and aged between {best_offer['age_group']}, within {best_offer['duration']} days. "
            f"\nRecommended channels: {best_offer['channels']}."
        )

    # Generate the comparison message for the second offer
    if second_offer['offer_type'] == 'informational':
        comparison = (
            f"However, {int(second_offer['response_rate'] * 100)}% of the informational offer are completed within {second_offer['duration']} days. "
            f"\nRecommended channels: {second_offer['channels']}."
        )
    elif second_offer['offer_type'] == 'bogo':
        comparison = (
            f"However, {int(second_offer['response_rate'] * 100)}% of the Buy-One-Get-One (BOGO) offer with a reward of \${second_offer['reward']} "
            f"and a minimum spend requirement of \${second_offer['difficulty']}, are completed within {second_offer['duration']} days. "
            f"\nRecommended channels: {second_offer['channels']}."
        )
    else:
        comparison = (
            f"However, {int(second_offer['response_rate'] * 100)}% of the discount offer with a reward of \${second_offer['reward']} "
            f"and a minimum spend requirement of \${second_offer['difficulty']}, are completed within {second_offer['duration']} days. "
            f"\nRecommended channels: {second_offer['channels']}."
        )

    # Display the responses
    st.success("### Recommended Offer")
    st.write(response)
    st.write(comparison)
    if best_offer['offer_type'] == second_offer['offer_type']:
        concl = f"So, send the first {best_offer['offer_type']} offer."
    elif  abs (int(best_offer['response_rate'] * 100) - int(second_offer['response_rate'] * 100) ) < 5:
        if  (best_offer['offer_type'] == 'informational') or  ( second_offer['offer_type'] == 'informational'):
            concl = (f"\n\n Since the two offers were almost completed with the same percentage, send the informational offer.")
        
        else :
            concl = f"\n\n So, send the {best_offer['offer_type']} offer."
    st.write(concl)