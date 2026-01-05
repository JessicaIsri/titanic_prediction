# Titanic Survival Prediction

This project aims to predict the survival of passengers on the Titanic using various machine learning models. The dataset used contains information about individual passengers, including their demographics, ticket details, and survival status.

## 1. Data Loading and Initial Exploration

- **Datasets Loaded**: `train.csv` and `test.csv`.
- **Training Data Shape**: `(891, 12)`
- **Test Data Shape**: `(418, 11)`
- **Initial Columns**: `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.

### Data Information (`train.info()`)
- `Age`, `Cabin`, and `Embarked` columns were identified as having missing values. `Cabin` had a significant number of missing entries.

## 2. Data Preprocessing and Cleaning

- **Columns Dropped**: `Cabin`, `PassengerId`, `Ticket` were removed from both `train` and `test` datasets due to high missing values or irrelevance for initial modeling.
- **Missing Age Values**: Filled with the mean age of the respective datasets.
  - `train['Age'].mean()`: Filled missing Age values in the training set.
  - `test['Age'].mean()`: Filled missing Age values in the test set.

## 3. Exploratory Data Analysis (EDA)

### Survival Rate
- **Survived**: 342 passengers (38%)
- **Died**: 549 passengers (62%)

### Sex Survival Rate
- **Female Survived**: 233 (68% of total survivors)
- **Male Survived**: 109 (32% of total survivors)

### Visualizations
- **`Pclass` vs. `Survived`**: A bar plot showed that passengers in higher classes (`Pclass` 1) had a better survival rate.
- **`SibSp` vs. `Survived`**: A bar plot illustrating survival rates based on the number of siblings/spouses aboard.
- **`Parch` vs. `Survived`**: A bar plot showing survival rates based on the number of parents/children aboard.
- **`Fare` Analysis**:
  - `fare_range` (quartiles of Fare) vs. `Survived`: Bar plot indicated higher survival probability for passengers who paid higher fares.
  - `Fare` Density vs. `Survival`:KDE graphs showed that passengers who paid higher fares had a higher probability of survival, since even though the graph decreased due to the smaller number of people with higher fares, mortality decreased at a steeper rate.
- **`Embarked` vs. `Survived`**: Bar plot indicated differing survival rates depending on the embarkation port.
- **`Sex` vs. `Survived`**: Bar plot clearly showed a significantly higher survival rate for females.

### Feature Engineering (for modeling)
- **`Sex` Encoding**: Converted 'male' to 0 and 'female' to 1.
- **`Embarked` Encoding**: One-hot encoded using `pd.get_dummies`.
- **`Name` Column Dropped**: The `Name` column was dropped as it's not directly used in the current models.

## 4. Correlation Analysis

- A heatmap of the correlation matrix was generated for numerical features including the newly encoded `Sex` and `Embarked` columns, and the target variable `Survived`.

- It is possible to identify a high correlation between the variables Sex and Survived, followed by the fare amount and the location where the person boarded, and finally a low correlation between the number of children on board.

## 5. Model Training and Evaluation

- **Features (X)**: All columns in the `train` DataFrame except 'Survived'.
- **Target (y)**: 'Survived' column.
- **Data Split**: `X_train`, `X_test`, `y_train`, `y_test` (80% train, 20% test, `random_state=42`).

### 5.1 Logistic Regression
- **Model**: `LogisticRegression()`
- **Accuracy Score (without scaling)**: 0.8101
- **Accuracy Score (with StandardScaler)**: 0.8101
- **Confusion Matrix (without scaling)**:
  ```
  [[90 15]
   [19 55]]
  ```
- **Classification Report (without scaling)**:
  ```
                precision    recall  f1-score   support

           0       0.83      0.86      0.84       105
           1       0.79      0.74      0.76        74

    accuracy                           0.81       179
   macro avg       0.81      0.80      0.80       179
weighted avg       0.81      0.81      0.81       179
  ```

### 5.2 Decision Tree
- **Model**: `DecisionTreeClassifier(max_depth=3, random_state=42)`
- **Accuracy Score**: 0.7989
- **Confusion Matrix**:
  ```
  [[92 13]
   [23 51]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support

           0       0.80      0.88      0.84       105
           1       0.80      0.69      0.74        74

    accuracy                           0.80       179
   macro avg       0.80      0.78      0.79       179
weighted avg       0.80      0.80      0.80       179
  ```
- A visualization of the Decision Tree logic was generated.

### 5.3 Random Forest
- **Model**: `RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)`
- **Accuracy Score**: 0.8156
- **Confusion Matrix**:
  ```
  [[95 10]
   [23 51]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support

           0       0.81      0.90      0.85       105
           1       0.84      0.69      0.76        74

    accuracy                           0.82       179
   macro avg       0.82      0.80      0.80       179
weighted avg       0.82      0.82      0.81       179
  ```

### 5.4 XGBoost
- **Model**: `XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)`
- **Accuracy Score**: 0.8212
- **Confusion Matrix**:
  ```
  [[94 11]
   [21 53]]
  ```
- **Classification Report**:
  ```
                precision    recall  f1-score   support

           0       0.82      0.90      0.85       105
           1       0.83      0.72      0.77        74

    accuracy                           0.82       179
   macro avg       0.82      0.81      0.81       179
weighted avg       0.82      0.82      0.82       179
  ```

## Conclusion

Among the models tested, **XGBoost** achieved the highest accuracy of approximately 82.12%, closely followed by Random Forest (81.56%) and Logistic Regression (81.01%). The Decision Tree model performed slightly lower at 79.89%. The analysis highlighted key features such as `Sex`, `Emarked`, and `Fare` as significant indicators of survival on the Titanic.
