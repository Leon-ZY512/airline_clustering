# Assignment 5

This repository contains two main projects: a budget tracking application and a customer segmentation analysis using clustering algorithms.

## Projects

### 1. Budget Tracker (`budget.py`)

A command-line budget tracking application that allows users to:
- Set and view monthly budgets
- Add and track expenses
- View expense history
- Delete expenses
- Receive budget alerts when 95% of budget is used

#### Features
- Persistent data storage using JSON files
- Budget alerts
- Interactive command-line interface

#### Usage
```bash
python budget.py
```

The application will prompt you with a menu to:
1. View budget
2. Set budget
3. Add expense
4. View expenses
5. Delete expense
6. Quit

### 2. Customer Segmentation Analysis (`assignment5.py`)

A data analysis project that performs customer segmentation on EastWestAirlines data using clustering techniques.

#### Features
- Data preprocessing and normalization
- Correlation matrix visualization
- K-means clustering
- Hierarchical clustering
- Cluster evaluation using Silhouette Score and Davies-Bouldin Index
- PCA visualization
- Cluster characteristics analysis

#### Requirements
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy
- openpyxl (for reading Excel files)

#### Usage
```bash
python assignment5.py
```

The script will:
1. Load and preprocess the EastWestAirlines dataset
2. Generate correlation matrix heatmap
3. Evaluate clustering models for k values from 2 to 10
4. Visualize clustering results
5. Generate cluster characteristics and comparisons

## Data Files

- `EastWestAirlines.xlsx`: The dataset used for customer segmentation analysis
- `eastwestairline_analysis.pdf`: Analysis report

## Installation

Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy openpyxl
```

## Files

- `budget.py`: Budget tracking application
- `assignment5.py`: Customer segmentation analysis script
- `EastWestAirlines.xlsx`: Dataset for clustering analysis
- `eastwestairline_analysis.pdf`: Analysis documentation

## Author

yizhen

