# Bankruptcy-Prediction-A-Comparative-Study-on-Taiwan-Economic-Journal-Data
Advanced Machine Learning Techniques for Bankruptcy Prediction: A Comparative Study on Taiwan Economic Journal Data

## Project Overview

This project involves the application of various machine learning models to predict bankruptcy using a comprehensive dataset from the Taiwan Economic Journal (1999-2009). The primary objectives are to preprocess the data, experiment with different models, optimize hyperparameters, and evaluate model performance. The results highlight the effectiveness of various models, with the Neural Network model consistently outperforming others.

## Table of Contents
1. Introduction
   - Dataset Overview
   - Research Context
2. Methodology
   - Data Pre-processing
   - Model Building
   - Experimentation
3. Result Evaluation
   - Initial Model Exploration
   - Hyperparameter Tuning
   - Ensemble Voting
   - Evaluation Metrics and Interpretation
4. Conclusion
5. Future Work
6. References

## Installation and Setup

### Prerequisites

- Python 3.x
- Jupyter Notebook (optional, but recommended for interactive exploration)
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repository-url
   cd your-repository-folder
   ```

2. Create a virtual environment (optional but recommended):
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required libraries:
   ```sh
   pip install -r requirements.txt
   ```

## Dataset

### Characteristics
- Multivariate dataset with 95 features and 6819 instances.
- Features include financial metrics and ratios.
- Target variable: Binary indicator of bankruptcy ("Bankrupt?").

### Key Features
- Cost of Interest-bearing Debt
- Cash Reinvestment Ratio
- Current Ratio
- Acid Test Ratio
- Interest Expenses/Total Revenue
- Total Liability/Equity Ratio
- Operating Income/Capital
- Return On Total Assets
- Gross Profit to Net Sales
- Cash Flow from Operating/Current Liabilities
- Return on Total Asset Growth

## Methodology

### Data Pre-processing
1. Load the dataset using `pd.read_csv()`.
2. Check for missing and duplicate values.
3. Remove highly correlated features to avoid multicollinearity.
4. Standardize features using `StandardScaler()`.
5. Apply Principal Component Analysis (PCA) to reduce dimensionality, retaining 45 features to capture at least 90% of the variance.
6. Split the dataset into training, validation, and test sets using `train_test_split()`.

### Model Building
1. Select models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, K-Nearest Neighbors, Naive Bayes, and Neural Network.
2. Perform 5-fold cross-validation to evaluate initial performance.
3. Tune hyperparameters using `GridSearchCV`.
4. Evaluate models using precision, recall, F1-score, and AUC.

### Experimentation
1. Implement `VotingClassifier` for model ensembling with soft and hard voting strategies.
2. Evaluate the performance of individual models and ensemble methods.
3. Select the best-performing model based on weighted average scores.

## Result Evaluation
- Initial model exploration showed promising accuracy across models.
- Hyperparameter tuning significantly improved model performance.
- Ensemble methods (soft and hard voting) did not significantly outperform the best individual models.
- The Neural Network model consistently performed best based on precision, recall, F1-score, and AUC.

## Conclusion
- Hyperparameter tuning is crucial for optimizing model performance.
- Neural Networks are particularly effective for bankruptcy prediction.
- Ensemble methods did not yield substantial improvements in this study.

## Future Work
- Explore additional ensemble techniques (e.g., bagging, stacking).
- Investigate feature engineering strategies to enhance model performance.
- Test models on different datasets for generalizability.

## References
Please refer to the original document for detailed references.

## Usage
1. To run the project, open `main_notebook.ipynb` in Jupyter Notebook.
2. Follow the cells sequentially to understand the workflow and execute the code.
3. For specific model training and evaluation, refer to the respective sections in the notebook.
