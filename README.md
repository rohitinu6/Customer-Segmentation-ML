# Customer Segmentation using Unsupervised Learning

This project implements customer segmentation using unsupervised learning techniques, specifically K-Means clustering. The dataset contains various customer attributes which are used to identify distinct customer groups.

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Evaluation Metrics](#evaluation-metrics)
- [License](#license)

## Overview
Customer segmentation is a key strategy for businesses to understand customer behavior and personalize marketing. In this project:

1. We preprocess and clean the dataset.
2. Perform exploratory data analysis (EDA) and visualize key insights.
3. Apply t-SNE for dimensionality reduction.
4. Use the K-Means algorithm to cluster customers.
5. Evaluate the model using the elbow method and silhouette score.

## Technologies Used
- Python 3.13
- Libraries:
    - Pandas (data manipulation)
    - Numpy (numerical computations)
    - Matplotlib & Seaborn (data visualization)
    - Scikit-learn (machine learning)

## Dataset
The dataset contains customer information such as:
- Marital Status
- Income
- Number of Items Purchased
- Date of Joining (Dt_Customer)

Ensure that the dataset is placed in the following directory before running the code:

```
D:/VIT Bhopal/GitHub/ML Projects/Customer Segmentation using Unsupervised Learning/Data.csv
```

## Project Structure
```
├── Customer Segmentation (Main Directory)
    ├── Data.csv (Dataset)
    ├── main.py (Main Python Script)
    └── README.md (Project Documentation)
```

## Installation
Ensure Python 3.13 is installed. Clone this repository and install the required libraries:

```bash
# Clone the repository
git clone https://github.com/rohitinu6/Customer-Segmentation-ML.git
cd customer-segmentation

# Create a virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
Run the main Python script:

```bash
python main.py
```

The script performs:
1. Data cleaning and preprocessing.
2. Exploratory data analysis (EDA).
3. Dimensionality reduction (t-SNE).
4. K-Means clustering.
5. Model evaluation with the silhouette score.

## Results
### Data Visualization
- Distribution of categorical columns.
- Feature correlation heatmap.
- t-SNE projection for dimensionality reduction.

### Clustering Analysis
- Elbow method to determine the optimal number of clusters.
- Visualized customer segments using t-SNE projection.

## Evaluation Metrics
1. **Inertia (Within-Cluster Sum of Squares)**: Measures how tightly data points are grouped within a cluster.
2. **Silhouette Score**: Measures how well clusters are separated; a higher score indicates better-defined clusters.

The silhouette score for K=5 (optimal clusters) is printed during execution.

## License
This project is licensed under the MIT License.

