# Starbucks-Capstone-Challenge
This is a Udacity Data Scientist Nanodegree project that aimed for learning and training.
# Starbucks Promotional Offers Recommendation System

## **Project Overview**
This project aims to optimize Starbucks' promotional offers by developing a recommendation system that suggests the most suitable offers to users based on their demographics (age and gender). Leveraging machine learning techniques, including Random Forest classification and K-Means clustering, the system enhances targeted marketing strategies to increase customer engagement and revenue.

## **Problem Statement**
Starbucks seeks to enhance customer engagement and increase revenue by effectively targeting promotional offers. The challenge is to determine which types of offers (BOGO, Discount, Informational) resonate most with specific customer segments defined by demographics.

## **Data Description**
- **Users Data (`profile.json`)**: Contains user demographics and profiles.
- **Offers Data (`portfolio.json`)**: Details of each promotional offer, including reward, duration, difficulty, and channels.
- **Events Data (`transcript.json`)**: Logs of user interactions with offers (received, opened, completed).

## **Methodology**

### **Data Preprocessing**
- **Handling Missing Values:** Imputed missing income values using the median.
- **Removing Duplicates:** Eliminated duplicate event records to ensure data integrity.
- **Encoding Categorical Variables:** Applied one-hot encoding to offer types and channels.
- **Feature Engineering:** Created a binary `response` variable indicating offer completion.
- **Scaling:** Standardized numerical features to improve model performance.

### **Modeling**

#### **Classification: Random Forest**
- **Objective:** Predict whether a user will complete a given offer.
- **Performance:** Achieved 73% accuracy, with higher precision and recall for responsive customers.
- **Confusion Matrix:**

|            | Predicted Negative | Predicted Positive |
|------------|--------------------|--------------------|
| Actual Negative | 1934               | 1144               |
| Actual Positive | 898                | 3572               |

#### **Clustering: K-Means**
- **Objective:** Segment customers into distinct groups based on demographics.
- **Clusters Identified:**
1. ** Age Cluster **
     Ages from 18: 75+, sperated like 18-24, 25-34, 35-44, 45-54, 55-64, 65-75, 75+
2. ** Genders Cluster **
      Genders have only 3 values: (Male, Female, Others)

## **Web Application**

### **Usage Instructions**

1. **Clone the Repository:**
  ```bash
  git clone https://github.com/youssefreda02/Starbucks-Capstone-Challenge.git
  cd Starbucks-Capstone-Challenge
  ```

2. **Install Required Libraries:**
  ```bash
  pip install -r requirements.txt
  ```

3. **Run the Streamlit Application:**
  ```bash
  cd app
  streamlit run app.py
  ```

4. **Access the Web App:**
  Open your browser and navigate to `http://localhost:8501/`.

### ** Changing the dataset **
Add the path for each file and where do u want you csv file.
like the following
```bash
python data/etl.py data/portfolio.json  data/profile.json  data/transcript.json data/offer_recommendations.csv
```
### **Making a Recommendation**

- **Input:** Enter your age and select your gender in the sidebar.
- **Output:** Receive a tailored offer recommendation based on your demographics.

## **Results and Insights**
- **Customer Segmentation:** Three distinct clusters were identified, each with unique offer preferences, enabling personalized marketing strategies.

## **Future Work**
- **Enhanced Feature Engineering:** Incorporate additional demographics or behavioral data to refine predictions.
- **Model Optimization:** Experiment with other algorithms or ensemble methods to improve classification performance.
- **Real-Time Recommendations:** Develop a system that updates recommendations dynamically based on real-time user interactions.
- **Feedback Integration:** Implement mechanisms to collect user feedback on recommendations to further improve the system.

## **Acknowledgments**
- Inspired by the Starbucks Capstone Challenge.
- Thanks to the developers of Pandas, Scikit-Learn, Seaborn, Matplotlib, and Streamlit.
