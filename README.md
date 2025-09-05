Hotel Booking Cancellation Prediction – Data Cleaning & Preprocessing

### Project Overview

This project focuses on building a robust data preprocessing pipeline for a hotel booking dataset. The goal is to prepare the data for a machine learning model that predicts booking cancellations. The preprocessing steps include data cleaning, feature engineering, encoding, balancing, and splitting.


### Dataset

- **Source**: hotel_bookings.csv from the Property Management System (PMS)
- **Target Variable**: is_canceled (1 = canceled, 0 = not canceled)


### Preprocessing Steps

##Data Exploration
- Loaded dataset using pandas
- Generated summary statistics with .info() and .describe()
- Visualized missing values using missingno and heatmaps

##Data Cleaning
- Imputed missing values:
  - company, agent: filled with "None"
  - country: filled with mode or "Unknown"
  - children: filled with mode
- Dropped exact duplicate rows
- Converted date columns to datetime format
- Capped outliers in adr and lead_time using thresholds and IQR method

##Feature Engineering
- Created new features:
  - total_guests = adults + children + babies
  - total_nights = stays_in_weekend_nights + stays_in_week_nights
  - booking_time = arrival_date - reservation_status_date
  - is_family, is_solo, is_weekend flags
- Dropped leakage-prone columns: reservation_status, status

##Encoding
- One-hot encoded low-cardinality categorical features (meal, market_segment)
- Frequency encoded high-cardinality features (country)
- Grouped rare countries into "Other"` category
Balancing & Splitting
Checked for imbalance in is_canceled
Applied SMOTE to balance the classes
Split data into training and testing sets:
test_size=0.2
random_state=42
stratify=y

- Next Steps
Train classification models (e.g., Random Forest, Logistic Regression)
Evaluate performance using accuracy, precision, recall, and F1-score
Visualize feature importance and model metrics

- Notes
All preprocessing decisions were documented and justified in the notebook
The pipeline is modular and can be reused for similar datasets
Care was taken to avoid data leakage and overfitting

-Tools Used
Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)
Jupyter Notebook
Missingno
Imbalanced-learn (SMOTE)

-Author
This project was completed as part of the GTC Machine Learning course, Project 1: Data Cleaning & Preprocessing Challenge.
