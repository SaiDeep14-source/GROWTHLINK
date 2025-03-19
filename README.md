# GROWTHLINK
# Titanic Survival Prediction

## Objective
This project predicts whether a passenger survived the Titanic disaster using machine learning. It involves data preprocessing, feature engineering, and model training.

## Dataset
The dataset contains information such as passenger age, gender, ticket class, fare, and more. You can download it from Kaggle: [Titanic Dataset](https://www.kaggle.com/datasets/brendan45774/test-file)

## Installation
To run this project, install the required dependencies:

```bash
pip install pandas numpy seaborn scikit-learn matplotlib
```

## Steps to Run the Project
1. **Download the Dataset** – Save it as `titanic.csv` in your working directory.
2. **Run the Python Script** – Execute the provided script to train the model:

```bash
python titanic_survival.py
```

3. **Review Model Performance** – Check accuracy, classification report, and confusion matrix.

## Features Used
- `Pclass`: Ticket class (1st, 2nd, 3rd)
- `Sex`: Gender (Encoded)
- `Age`: Passenger age (Missing values filled with median)
- `Fare`: Ticket fare (Scaled)
- `Embarked`: Port of embarkation (Encoded)

## Model & Evaluation
- **Algorithm:** Random Forest Classifier
- **Metrics:** Accuracy, Precision, Recall, Confusion Matrix

## Results
- The model provides a reliable classification of passengers who survived.
- Further improvements can be made using hyperparameter tuning.

## Contributing
Feel free to fork this repository and improve the model with better preprocessing or feature engineering.

## Contact
For queries, reach out to **help.growthlink@gmail.com**

