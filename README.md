# Employee Attrition Prediction

## 📌 Project Overview
This project focuses on predicting employee attrition using machine learning classification models. The objective is to identify employees at high risk of leaving based on historical HR data, enabling proactive decision-making and targeted retention strategies.

## 🚀 Key Features
- **Data Preparation**: Preprocessing and cleaning of HR attrition data.
- **Exploratory Data Analysis (EDA)**: Visualizing trends and performing statistical analysis.
- **Model Training & Evaluation**: Implementing machine learning models, tuning hyperparameters, and optimizing classification thresholds.
- **Predictive Insights**: Identifying key factors influencing attrition and forecasting potential employee turnover.

## 🛠 Tech Stack
- **Programming Language**: Python
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `warnings`, `scipy`
- **Machine Learning Model**: Random Forest Classifier

## 📊 Data Processing Steps
1. **Data Cleaning**: Handling missing values, encoding categorical variables, and normalizing numerical features.
2. **Feature Engineering**: Extracting relevant features and performing statistical correlation analysis.
3. **Model Selection & Training**:
   - Implemented a **Random Forest Classifier**.
   - Tuned the model to an **optimal classification threshold of 0.300**.
4. **Performance Evaluation**:
   - **Training Accuracy**: 98%
   - **Testing Accuracy**: 82%
   - **ROC AUC Score**:
     - **Training**: 0.997
     - **Testing**: 0.784
   - **Key Observations**:
     - High training accuracy indicates a well-fitted model.
     - Moderate testing accuracy suggests room for improvement in generalization.

## 📈 Results & Insights
- **Key Metrics**:
  - **Precision** (Identifying true attrition cases) = 0.43 (Testing)
  - **Recall** (Capturing most attrition cases) = 0.32 (Testing)
  - **F1-Score** = 0.37 (Testing)
- **Business Impact**:
  - Helps HR teams predict employee attrition.
  - Supports workforce planning with data-driven insights.
  - Identifies critical retention factors like job satisfaction, salary, and work-life balance.

## 🛠 How to Run the Project
1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-repo/employee-attrition.git
2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
3. **Run Data Preprocessing**:
   ```sh
   python src/preprocess.py
4. **Train the Model**:
   ```sh
   python src/train_model.py
5. **Run Data Preprocessing**:
   ```sh
   python src/evaluate_model.py

## 📌 Future Improvements
- Implementing additional models like XGBoost for improved accuracy.
- Feature selection and engineering to refine model performance.
- Integration with HR dashboards for real-time attrition monitoring.

## 🚀 Developed by: Bhavya Bhargava
## 📩 For queries, contact: bhargava.bhavya1@gmail.com
