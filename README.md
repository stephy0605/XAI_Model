# Explainable AI for Phishing Email Detection
## Project Overview
This project presents an intelligent and transparent system for the detection of phishing emails. Utilizing a hybrid deep learning model (CNN-BiGRU), the solution achieves a high level of accuracy in identifying malicious emails by learning complex textual patterns. A key innovation of this work is the integration of Explainable AI (XAI) techniques, such as LIME, to provide clear, human-understandable explanations for the model's classifications, thereby building user trust and enhancing threat intelligence.

## Key Features
High-Accuracy Detection: A custom-built CNN-BiGRU model demonstrates superior performance in distinguishing between legitimate and phishing emails.

Comprehensive Data Preprocessing: A robust pipeline cleans raw email data, removes personally identifiable information (PII), and performs feature engineering.

Performance Benchmarking: Traditional machine learning models (Random Forest, Gradient Boosting, etc.) are trained and evaluated to provide a baseline for comparison.

Model Interpretability: Explainable AI (XAI) is integrated to provide insights into the model's decision-making process, highlighting the specific words or phrases that trigger a classification.

Full Lifecycle Implementation: The project follows a complete research-to-solution lifecycle, from initial data exploration (EDA) to model development, evaluation, and documentation.

## Quick Start

### 1) System Requirements and Tool stack 
• OS: Windows 10/11, macOS 12+, or Ubuntu 20.04+
• Software: Python 3.8+ (recommended 3.10)  and Jupyter Notebook or Google Collab pro+.
• CPU RAM: 16 GB (min), 32 GB (recommended)
• Disk: 10 GB free
• GPU (optional): NVIDIA GPU with CUDA 11.8+ for BERT training/inference

 A modern CPU is sufficient for the traditional models. A GPU is highly recommended for faster training of the deep learning (CNN-BiGRU) model.

### 2) Installation and Setup for Environment
This project requires a Python environment. The notebooks are designed to run in a Jupyter or Colaboratory environment.

#### Dependencies
You can install all required libraries using the provided requirements.txt file (if applicable) or by running the following command in your terminal:

pip install pandas scikit-learn tensorflow keras numpy matplotlib seaborn wordcloud

Note: Depending on your specific environment and GPU setup, you may need to install tensorflow-gpu and corresponding CUDA/cuDNN drivers.

### 3) Project Structure
XAI_Model/
│── data/               # Raw datasets (SpamAssasin.csv, CEAS_08.csv)Place this CSV datasets where the notebooks expect them and adjust paths at the top of each notebook
│── model/              # Saved models (cnn_bigru.h5)
│── notebooks/          # Jupyter notebooks for experiments
│── requirements.txt    # Python dependencies to import

### 4) Data
Place your CSV datasets where the notebooks expect them (adjust paths at the top of each notebook if needed). Typical setup:

```
data/
  SpamAssasin.csv
  CEAS_08.csv


### 5) Running the Notebooks
 
- Open each `.ipynb` by below order in Jupyter or Colab and execute all cells from top to bottom.

  **1. EDA.ipynb**: Jupyter notebook for exploratory data analysis.
  **2. lr, xgb, rf** - phishing email detection.ipynb: Notebook for data preprocessing, feature engineering, and traditional ML model training/evaluation.
  **3. CNN-BiGRU.ipynb** :Notebook for the deep learning model (CNN-BiGRU) implementation and evaluation.
  **4. xplainable BERT** for phishing email detection.ipynb: Notebook that implements a BERT-based phishing email detection model with explainable AI techniques to interpret and visualize the model’s predictions. 


- Each notebook is segmented with clear sections: **Data Loading → Preprocessing → Modeling → Evaluation → Explainability**.

### 5) Troubleshooting
1.	Module Not Found Error: This means a required library is not installed. Make sure 
you ran the pip install command successfully.

2. Training is Slow: The CNN-BiGRU model is computationally intensive. If you don't have a GPU, training will take a significant amount of time. You can reduce the 
number of epochs in the model.fit() call to speed this up for testing purposes.

3.  File Not Found: Double-check that your .csv data files are in the same folder as the 
notebooks.
4. Large Notebook Memory: If local Jupyter struggles to open notebooks with outputs, clear outputs and re‑run, or use Google Colab/Colab Pro.
Conclusion



#### Make sure to run the code sequentially, do not run in blocks, train the model first, then run 
it. Better to use Jupyter with a good GPU and V-ram





