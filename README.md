# Genetic Algorithm for Air Quality Classification

Machine learning project that uses a Genetic Algorithm (GA) to optimize a Decision Tree classifier for environmental and air quality classification.

The system analyzes meteorological conditions and atmospheric pollutants to automatically classify environmental conditions based on sensor data.

---

# Overview

This project combines: 

- Environmental Data Analysis
- Air Quality Classification
- Decision Trees
- Genetic Algorithms
- Machine Learning
- Model Optimization
- Data Visualization

A Genetic Algorithm searches for the optimal Decision Tree hyperparameters through evolutionary optimization using tournament selection, crossover, and mutation.

---

# Features

- Environmental data preprocessing
- Automatic data cleaning
- Rule-based category generation
- Decision Tree classification
- Genetic Algorithm optimization
- Tournament selection
- Uniform crossover
- Random mutation
- Model serialization using Joblib
- Decision Tree visualization
- Environmental condition prediction

---

# Environmental Variables

The model uses the following variables:

| Variable | Description |
|----------|-------------|
| DV | Wind Direction |
| VV | Wind Velocity |
| PB | Barometric Pressure |
| Temp | Temperature |
| HR | Relative Humidity |
| RS | Solar Radiation |
| PM@10 | PM10 Concentration |
| PM2@5 | PM2.5 Concentration |
| OZONO | Ozone |
| SO2 | Sulfur Dioxide |
| NO | Nitric Oxide |
| NO2 | Nitrogen Dioxide |
| NOX | Nitrogen Oxides |
| CO | Carbon Monoxide |

---

# Environmental Categories

The classifier generates the following environmental categories:

- Buena
- Razonable Buena
- Regular
- Desfavorable
- Muy Favorable
- Extremadamente Desfavorable
- Sin Categoria

The categories are generated using predefined environmental thresholds based on meteorological variables and pollutant concentrations.

---

# Genetic Algorithm

Each chromosome represents a Decision Tree configuration.

## Optimized Hyperparameters

- max_depth
- min_samples_split
- min_samples_leaf

## Evolutionary Process

1. Generate random population
2. Evaluate each chromosome
3. Tournament Selection
4. Crossover
5. Mutation
6. Generate new population
7. Repeat for multiple generations
8. Train the final model using the best chromosome

Current configuration:

```python
population_size = 100
tournament_size = 5
mutation_rate = 0.1
num_generations = 50
```

---

# Machine Learning Model

Classifier:

```python
DecisionTreeClassifier
```

Fitness Function:

```text
Classification Accuracy
```

The chromosome with the highest accuracy is selected to build the final model.

---

# Technologies

- Python
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Joblib

---

# Installation

Clone the repository

```bash
git clone https://github.com/Oscarvdo/REPOSITORY_NAME.git
```

Move into the project directory

```bash
cd REPOSITORY_NAME
```

Install the required packages

```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

---

# Dataset

The program expects a CSV dataset containing the following variables:

```text
DV
VV
PB
Temp
HR
RS
PM@10
PM2@5
OZONO
SO2
NO
NO2
NOX
CO
```

The dataset path can be modified in the script:

```python
data = pd.read_csv('data/data.csv')
```

---

# Running the Project

Execute:

```bash
python main.py
```

The application will:

- Load the dataset
- Clean the data
- Generate environmental categories
- Split the dataset
- Initialize the Genetic Algorithm
- Optimize the Decision Tree
- Train the final model
- Save the model
- Export the Decision Tree visualization

---

# Generated Files

## Trained Model

```text
mejor_modelo.pkl
```

Load the model with:

```python
import joblib

model = joblib.load("mejor_modelo.pkl")
```

---

## Decision Tree

```text
arbol_decision.png
```

The generated image contains the complete Decision Tree structure.

---

# Project Structure

```text
Genetic-Algorithm-Air-Quality/
│
├── data/
│   └── data.csv
│
├── models/
│   └── mejor_modelo.pkl
│
├── outputs/
│   └── arbol_decision.png
│
├── src/
│   └── main.py
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# Future Improvements

- Cross Validation
- Time-Series Validation
- Missing Value Imputation
- Automatic Hyperparameter Search
- Random Forest Optimization
- Explainable AI (XAI)
- Feature Importance Analysis
- Confusion Matrix
- ROC Analysis
- Precision / Recall / F1 Score
- Flask REST API
- Real-Time Prediction Dashboard

---

# Research

This project is part of ongoing research in:

- Artificial Intelligence
- Environmental Analytics
- Evolutionary Computation
- Machine Learning
- Intelligent Environmental Systems
- Air Quality Monitoring

Future work includes extending this approach toward:

- Genetic Programming
- Symbolic Regression
- PM10 Forecasting
- Multi-Horizon Prediction
- Explainable Artificial Intelligence

---

# Author

**Oscar I. Valenzuela Díaz**

 
---

# License

This project is released under the MIT License.
