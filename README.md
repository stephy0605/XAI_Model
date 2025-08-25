## Quick Start

###1) System Requirements and Tool stack 
1) OS: Windows 10/11, macOS 12+, or Ubuntu 20.04+
2)  Software: Python 3.8+ (recommended 3.10)  and Jupyter Notebook or Google Collab pro+.
3) CPU RAM: 16 GB (min), 32 GB (recommended)
4)  Disk: 10 GB free
5) GPU (optional): NVIDIA GPU with CUDA 11.8+ for BERT training/inference

 A modern CPU is sufficient for the traditional models. A GPU is highly recommended for faster training of the deep learning (CNN-BiGRU) model.

### 2) Installation and Setup for Environment
This project requires a Python environment. The notebooks are designed to run in a Jupyter or Colaboratory environment.

#### Dependencies
You can install all required libraries using the provided requirements.txt file (if applicable) or by running the following command in your terminal:

pip install pandas scikit-learn tensorflow keras numpy matplotlib seaborn wordcloud

Note: Depending on your specific environment and GPU setup, you may need to install tensorflow-gpu and corresponding CUDA/cuDNN drivers.

### Project Structure
XAI_Model/
│── data/               # Raw datasets (SpamAssasin.csv, CEAS_08.csv)Place this CSV datasets where the notebooks expect them and adjust paths at the top of each notebook
│── model/              # Saved models (cnn_bigru.h5)
│── notebooks/          # Jupyter notebooks for experiments
│── requirements.txt    # Python dependencies to import

### 3) Data
Place your CSV datasets where the notebooks expect them (adjust paths at the top of each notebook if needed). Typical setup:

```
data/
  SpamAssasin.csv
  CEAS_08.csv

### 3) Running the Notebooks

- Start Jupyter: `jupyter notebook` (or `jupyter lab`) and open each `.ipynb` in order:
  1. `EDA.ipynb`
  2. `lr, xgb, rf - phishing email detection.ipynb`
  3. `CNN-BiGRU.ipynb`
  4. `xplainable BERT for phishing email detection.ipynb`
