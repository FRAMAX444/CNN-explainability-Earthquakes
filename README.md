![background](media/BG.png)

## Overview  
This project explores the application of explainability techniques in deep learning models trained on seismic data. Inspired by the work of Laurenti et al., we focus on understanding what machine learning models learn when classifying foreshocks and aftershocks or predicting earthquake magnitude. Using SHAP (SHapley Additive exPlanations), we analyze the importance of input features derived from seismic waveforms, converted into spectrograms or feature vectors.  

## Content  

### Preprocessing Notebook  
- **Purpose**: Prepares the raw seismic waveforms for use in training models.  
- **Steps**:  
  - Combines waveforms from multiple stations.  
  - Converts signals into three-channel spectrograms (log-spectrograms or standard spectrograms).  

### CNN Training and SHAP Analysis (Foreshocks and Aftershocks)  
- **Log-Spectrogram Notebook**:  
  - Trains a CNN on three-channel log-spectrograms for binary classification (foreshocks vs. aftershocks).  
  - Uses SHAP to interpret the pixel contributions of the spectrograms to the classification.  

- **Spectrogram Notebook**:  
  - Trains a CNN on standard spectrograms for the same binary classification task.  
  - Applies SHAP to understand the role of frequency and time components.  

### Random Forest Regression for Magnitude Prediction  
- **Purpose**: Predicts earthquake magnitude using extracted features from raw waveforms.  
- **Steps**:  
  - Trains a Random Forest regression model.  
  - Applies SHAP to interpret feature contributions to magnitude predictions.  

## How to Use  

1. **Download the Dataset**
   
   We used the same dataset from Laurenti et al. article (only selecting pre and post waveforms), you can find it [here](https://zenodo.org/records/12795621).

3. **Clone the repository**  
   ```bash  
   git clone https://github.com/FRAMAX444/CNN-explainability-Earthquakes
   cd your_path/CNN-explainability-Earthquakes  
   ```

4. **Create and activate Virtual Environment**
    ```bash  
    python3 -m venv venv
    source venv/bin/activate
   ```

5. **Install dependecies**
    ```bash 
    pip install -r requirements.txt  
    ```

---

## Results

### 1. **Seismic Event Classification**
- **Model**: A CNN trained on spectrograms derived from seismic waveforms.
- **Datasets**: 
  - NRCA dataset (station-specific, 5862 samples): Accuracy = **99.57%**.
  - Full dataset (multi-station, 55,617 samples): Accuracy = **95.75%**.
  - NRCA model applied to the full dataset: Accuracy = **50.19%** (poor generalization).
- **SHAP Insights**:
  - Foreshocks: Significant features in the 10–25 Hz range shortly after P-wave arrivals.
  - Aftershocks: Features more evenly distributed over time, with lower frequency contributions.

### 2. **Earthquake Magnitude Prediction**
- **Model**: Random Forest Regressor with 100 trees.
- **Performance**:
  - R² = **0.8834**
  - MAE = **0.1052**, RMSE = **0.1457**
- **SHAP Insights**:
  - Key features: Mean spectral power, P-wave travel times, and peak counts.
  - Spectral and temporal characteristics strongly influenced predictions.

---

## Observations
- **Generalization Challenges**:
  - CNNs trained on single stations achieve high accuracy locally but fail to generalize to diverse datasets.
  - Dataset diversity is crucial for robust seismic models.
- **Model Transparency**:
  - SHAP analysis provided critical insights into feature importance, bridging the gap between ML predictions and seismic domain knowledge.
- **Future Directions**:
  - Explore domain adaptation, metadata integration, and real-time monitoring applications.

---

## References  

1. **Our Report**  
   - [***What is Machine Learning Teaching Us?** Explainable AI for Seismic Models*](#) (link to be added)  

2. **Project Presentation**  
   - [*Results Presentation*](media/Presentation.pdf)

3. **Laurenti, Paolini et al. (Nature 2024)**  
   - [***Probing the Evolution of Fault Properties During the Seismic Cycle with Deep Learning***](https://www.nature.com/articles/s41467-024-54153-w)


