# Sleep Health Disorder Prediction

An end-to-end machine learning and deep learning pipeline for predicting sleep disorders (`No Disorder`, `Sleep Apnea`, and `Insomnia`) from physiological, demographic, and lifestyle metrics using the **Sleep Health and Lifestyle** dataset.

## Repository Structure
```text
├── Sleep_Health_Disorder_Prediction.ipynb   # Main EDA, training, evaluation, and inference workflow
├── Sleep_health_and_lifestyle_dataset.csv # Input dataset
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Ignored cache and binary model files
└── README.md                              # Project documentation
```

## Workflow & Methodology

1. **Data Preprocessing & Feature Engineering**
   - Imputed missing values in the `Sleep Disorder` target column as `No Disorder` to retain complete clinical records.
   - Parsed composite `Blood Pressure` strings into separate numerical features (`Systolic BP` and `Diastolic BP`).
   - Encoded nominal categorical variables via one-hot encoding (`pd.get_dummies`), mapped multi-class target labels using `LabelEncoder`, and standardized numerical features with `StandardScaler`.

2. **Exploratory Data Analysis & Dimensionality Reduction**
   - Analyzed feature relationships and multicollinearity using a correlation heatmap.
   - Applied **Principal Component Analysis (`PCA`)** to project high-dimensional health features into 2D space and visualize cluster separability across sleep disorder classes.

3. **Model Training & Benchmarking**
   - Trained and evaluated classical supervised classifiers:
     - **Logistic Regression** (evaluated with accuracy metrics and confusion matrix)
     - **K-Nearest Neighbors (`KNN`, `n_neighbors=3`)**
     - **Linear Support Vector Classifier (`LinearSVC`)**
     - **Random Forest Classifier (`n_estimators=100`)**
   - Built and trained a multi-layer **Artificial Neural Network (ANN)** using `TensorFlow / Keras` (`ReLU` hidden layers with `Softmax` multi-class output).

4. **Model Serialization & Inference**
   - Saved the trained neural network (`ann_sleep_model.h5`) alongside preprocessing transformers (`scaler.pkl`, `label_encoder.pkl`, `feature_columns.pkl`) using `Joblib` to support reproducible single-record inference.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<Deepak-07Hub>/<Sleep_Health_Disorder_Prediction>.git
   cd <Sleep_Health_Disorder_Prediction>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline:**
   Open `Sleep_Health_Disorder_Prediction.ipynb` in Jupyter Notebook or VS Code (or execute your `.py` script) with `Sleep_health_and_lifestyle_dataset.csv` in the root directory.

## Tech Stack
- **Language:** Python
- **Data Manipulation:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning & Deep Learning:** Scikit-learn, TensorFlow / Keras
- **Serialization:** Joblib