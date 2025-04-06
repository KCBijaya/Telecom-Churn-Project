# Customer Churn Analysis Project
This repository contains the analysis and implementation of a Customer Churn prediction model for a telecommunications company. The project aims to analyze customer behavior and predict the likelihood of customers leaving the company (churn) based on various factors. The project involves data preparation, clustering analysis, and machine learning techniques.

## Project Structure

The project is organized into the following folders:

### 1. **Data_Preparation**
This folder contains files related to data preparation for analysis, including the preprocessed dataset and separate training and testing sets.

- **Preprocessed Dataset**: The cleaned and preprocessed data used for training and testing the models.
- **Training and Testing Sets**: Files that separate the data into training and testing sets for model training and evaluation.
- **Scaling Techniques Documentation**: Documentation on the scaling techniques applied during data preprocessing, including code snippets.

### 2. **Clustering_Analysis**
This folder contains files and analysis related to the clustering of customer data to find meaningful customer segments.

- **Optimal Number of Clusters**: Documentation or analysis results indicating the optimal number of clusters with supporting visualizations.
- **Trained K-Means Model**: The trained K-Means model used for customer segmentation.
- **Cluster Visualizations and Labeling**: Visualizations of the resulting clusters, with labels and interpretations of the findings.

## Installation

To run this project on your local machine, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/KCBijaya/Telecom-Churn-Project.git

2.  Install required dependencies:

This project requires Python and the following libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

You can install them by running the following command:
pip install -r requirements.txt

3. Run the project:

After setting up the dependencies, you can start running the analysis and models by executing the Python scripts in the respective folders (Data_Preparation and Clustering_Analysis).
###  **Usuage**
Data_Preparation: Preprocess the dataset and create training and testing sets for model training and evaluation.

Clustering_Analysis: Perform clustering analysis using the K-Means algorithm and visualize the resulting clusters to segment customers meaningfully.
