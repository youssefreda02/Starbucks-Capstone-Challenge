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


**Interpretation:**

- **Class 1 (Responsive Customers):**
  - **Precision (0.76):** When the model predicts a customer as responsive, it's correct 76% of the time.
  - **Recall (0.80):** The model successfully identifies 80% of all actual responsive customers.
  - **F1-Score (0.78):** Indicates a balanced performance between precision and recall.

- **Class 0 (Non-Responsive Customers):**
  - **Precision (0.68):** When the model predicts a customer as non-responsive, it's correct 68% of the time.
  - **Recall (0.63):** The model successfully identifies 63% of all actual non-responsive customers.
  - **F1-Score (0.65):** Reflects moderate performance in balancing precision and recall.

**Implications:**

- **Effective Targeting:** The high precision and recall for responsive customers mean that Starbucks can confidently target these individuals with promotional offers, maximizing engagement and return on investment (ROI).
  
- **Areas for Improvement:** The lower precision and recall for non-responsive customers suggest that the model could benefit from further refinement to reduce false positives and improve the identification of non-responsive segments.

#### **Clustering Analysis Results**

K-Means clustering was utilized to segment customers based on demographics (age and gender) and offer acceptance patterns. The analysis identified **multiple distinct customer segments**, each exhibiting unique characteristics and preferences. Below is a summary of the identified clusters:

| **Cluster** | **Age Range** | **Gender** | **Preferred Offer Type** | **Channels**             |
|-------------|----------------|------------|--------------------------|--------------------------|
| **0**       | 18-24          | Female     | BOGO                     | Mobile, Social Media     |
| **1**       | 25-34          | Female     | BOGO                     | Mobile, Social Media     |
| **2**       | 35-44          | Male       | Discount                 | Email, Web               |
| **3**       | 45-54          | Male       | Discount                 | Email, Web               |
| **4**       | 55-64          | Female     | Informational            | Email, Direct Mail       |
| **5**       | 65-75          | Female     | Informational            | Email, Direct Mail       |
| **6**       | 75+            | Female     | Informational            | Email, Direct Mail       |
| **7**       | 18-24          | Other      | BOGO                     | Mobile, Social Media     |
| **8**       | 25-34          | Other      | BOGO                     | Mobile, Social Media     |
| **...**     | ...            | ...        | ...                      | ...                      |

*(As the we don't have enough data from Other)*

**Key Insights:**

- **Age-Based Segmentation:**
  - **18-24 & 25-34 (Female):** Highly responsive to **BOGO offers** delivered via **Mobile** and **Social Media** channels.
  - **35-44 & 45-54 (Male):** Prefer **Discount offers** communicated through **Email** and **Web** platforms.
  - **55-64, 65-75, & 75+ (Female):** Show interest in **Informational offers** delivered via **Email** and **Direct Mail**.

- **Gender-Based Segmentation:**
  - **Female:** Tend to prefer **BOGO** and **Informational** offers depending on age.
  - **Male:** Consistently prefer **Discount** offers across mid-age ranges.
  - **Other:** Similar to females in younger age brackets, favoring **BOGO** offers.

**Implications:**

Understanding these customer segments allows Starbucks to tailor its marketing strategies effectively. Each segment receives offers that resonate with their preferences and preferred communication channels, enhancing the relevance and effectiveness of promotional campaigns, leading to increased customer engagement and higher conversion rates.

### **Justification of Technique Effectiveness**

#### **Why Random Forest Performed Better Than Other Techniques**

During the model development phase, several machine learning algorithms were evaluated, including Logistic Regression and Decision Trees. The Random Forest classifier emerged as the most effective model for the following reasons:

1. **Handling Feature Interactions:**
   - **Random Forests** inherently capture non-linear relationships and interactions between features, which are crucial in predicting customer behavior influenced by multiple factors like age, gender, and offer properties.

2. **Robustness to Overfitting:**
   - The ensemble nature of Random Forests, combining multiple decision trees, reduces variance and enhances the model's ability to generalize to unseen data.

3. **Feature Importance Insights:**
   - Random Forests provide valuable insights into feature importance, enabling the identification of key drivers behind offer responsiveness, such as offer type and income levels.

4. **Handling Class Imbalance:**
   - While the dataset exhibited slight class imbalance, Random Forests effectively managed this through mechanisms like class weighting and bootstrapping, maintaining satisfactory performance for both classes.

#### **Improvements Made**

1. **Hyperparameter Tuning:**
   - **Grid Search** was utilized to optimize key hyperparameters, including the number of trees (`n_estimators`), maximum depth (`max_depth`), and minimum samples split (`min_samples_split`). This optimization enhanced the model's performance by identifying the best combination of parameters.

2. **Data Preprocessing Enhancements:**
   - Improved data cleaning processes, including more effective handling of missing values and outliers, contributed to a higher quality dataset, thereby boosting model performance.

#### **Limitations and Future Improvements**

- **Predicting Non-Responsive Customers:**
  - The model's performance for non-responsive customers was less robust, indicating the need for further refinement. Future efforts could explore alternative algorithms like Gradient Boosting Machines or ensemble methods to better capture the characteristics of non-responsive segments.
  
- **Incorporating Behavioral Data:**
  - Integrating more granular behavioral data, such as past purchase history or browsing patterns, could enhance the model's predictive capabilities.
  
- **Real-Time Data Integration:**
  - Implementing a system that updates model predictions in real-time based on ongoing customer interactions could further improve accuracy and responsiveness.


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

## **Conclusion**

### **Summary of End-to-End Problem Solution**

This project successfully developed a dual-framework approach to optimize Starbucks' promotional offers by combining predictive modeling and customer segmentation. The **Random Forest classifier** effectively predicted customer responsiveness, achieving a **73% accuracy** with strong performance in identifying responsive customers (**F1-Score: 0.78**). Concurrently, **K-Means clustering** segmented customers into multiple distinct groups based on demographics and offer acceptance patterns, enabling personalized and targeted marketing strategies.

By integrating these two approaches, the project provides a comprehensive solution that not only predicts which customers are likely to respond to promotional offers but also segments them into meaningful groups for tailored recommendations. This dual strategy enhances the relevance and effectiveness of promotional campaigns, aligning with Starbucks' business objectives of increasing customer engagement and driving revenue growth.

### **Reflection**

One of the most intriguing aspects of this project was **balancing model complexity with interpretability**. Choosing the Random Forest classifier allowed for robust performance while still providing insights into feature importance, which is valuable for business stakeholders. A significant challenge encountered was **addressing class imbalance**, where responsive customers were more prevalent. Implementing techniques like SMOTE and hyperparameter tuning were essential in mitigating this issue.

Additionally, **effective customer segmentation** through clustering provided deeper insights into customer behavior, revealing distinct preferences that can drive more personalized marketing efforts. This integration of classification and clustering underscored the importance of a comprehensive approach in data-driven marketing strategies.

### **Future Work and Improvements**

While the current model and segmentation provide a robust foundation for optimizing promotional offers, several avenues for future enhancement remain:

1. **Advanced Feature Engineering:**
   - Incorporate behavioral data such as past purchase history, frequency of store visits, and online engagement metrics to provide deeper insights into customer behavior.
   
2. **Exploring Alternative Algorithms:**
   - Test other machine learning models like Gradient Boosting Machines or neural networks to see if they offer superior performance.
   
3. **Real-Time Recommendation System:**
   - Develop a system that updates recommendations dynamically based on real-time user interactions and feedback.
   
4. **Enhanced Customer Segmentation:**
   - Utilize more sophisticated clustering techniques or increase the number of clusters to capture more nuanced customer segments.
   
5. **Model Interpretability Tools:**
   - Implement tools like SHAP (SHapley Additive exPlanations) to provide more granular insights into feature contributions for each prediction.

By pursuing these enhancements, the recommendation system can be further refined to deliver even more accurate and personalized offers, thereby driving higher customer engagement and increasing overall revenue.

## **Acknowledgments**
- Inspired by the Starbucks Capstone Challenge.
- Thanks to the developers of Pandas, Scikit-Learn, Seaborn, Matplotlib, and Streamlit.
