# Airplane Price Prediction

Airplane Price Prediction: Analysis and Modeling is a regression-based machine learning project that predicts airplane prices from technical and performance specifications using the CRISP-DM methodology. The notebook evaluates multiple linear and ensemble models (Ridge, Lasso, Elastic Net, and Random Forest) with a focus on error minimization and interpretability.

## Project Overview
This project develops a machine learning workflow to estimate airplane prices based on aircraft characteristics such as engine type, power, performance metrics, and physical dimensions. The analysis is structured around the CRISP-DM phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and (optional) Deployment. The work is implemented in a Jupyter Notebook originally designed as my case study for the Airplane Price Prediction dataset on Kaggle.

## Dataset
* Source: “Airplane Price Prediction” / “Plane Price.csv” dataset hosted on Kaggle.
* File used: Plane Price.csv loaded from /kaggle/input/plane-price-prediction/Plane Price.csv in the notebook.
* Shape: 517 rows and 16 columns.
* Target variable: Price (airplane price in monetary units).
​
Example feature columns:​
* Model Name
* Engine Type
* HP or lbs thr ea engine
* Max speed Knots, Rcmnd cruise Knots, Stall Knots dirty
* Fuel gal/lbs
* All eng rate of climb, Eng out rate of climb
* Takeoff over 50ft, Landing over 50ft
* Empty weight lbs, Length ft/in, Wing span ft/in
* Range N.M.

The dataset simulates real-world factors that influence airplane pricing and is suitable for supervised regression tasks.

## Methodology and Workflow
The notebook follows CRISP-DM with the following main steps:
1. Business Understanding
* Define the objective: build a model that accurately predicts airplane prices using technical specifications.
​* Success criteria: minimize RMSE and MAE while maximizing R^2 and keep the final model interpretable and deployable.
​
2. Data Understanding
* Load the CSV into a pandas DataFrame.
* Inspect shape, head, and descriptive statistics to understand distributions and potential data quality issues.
​
3. Data Preparation
* Identify numerical and categorical features.
* Apply appropriate preprocessing (e.g., scaling numeric features and encoding categorical features) using ColumnTransformer, StandardScaler, and encoders.
​
4. Modeling
* Split data into training and test sets with train_test_split.
* ​Build pipelines with the following models:
​Ridge Regression
Lasso Regression
Elastic Net
Random Forest Regressor

* Hyperparameter Tuning and Evaluation
Use GridSearchCV and cross_val_score to tune key hyperparameters and estimate model generalization performance.​

Evaluate models with metrics such as: 
* R^2 score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE) computed from MSE.
​
Model Persistence
* Save the final, best-performing model using pickle for future deployment or integration.

## Installation and Usage
### Requirements
The notebook uses Python 3 with common data science and scikit-learn libraries. A minimal requirements.txt might include:
* ​pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

These match the imports present in the notebook (pandas, numpy, seaborn, matplotlib.pyplot, and various sklearn modules).
​
### Running the Notebook Locally
1. Clone or download the project repository containing the notebook and data.​
2. Create and activate a virtual environment (optional but recommended).
3. Install dependencies:
* pip install -r requirements.txt
4. Place Plane Price.csv in a data/ directory and update the notebook path if necessary (e.g., data/Plane Price.csv instead of the Kaggle path).
​5. Launch Jupyter: jupyter notebook
6. Open airplane-price-prediction-analysis-and-modeling.ipynb and run all cells in order to reproduce the analysis and model training.
​
### Using the Trained Model
After running the modeling and saving section:
* Load the trained pickle file in a separate script or notebook.
* Apply the same preprocessing steps to new airplane specification data using the stored pipeline.
* Call .predict() on the pipeline to obtain price estimates for new airplanes.
​​
This setup enables consistent price prediction and can be extended into an API or simple application for demonstration or business use.


