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

1. **Clone the repository**  
   ```bash  
   git clone https://github.com/FRAMAX444/CNN-explainability-Earthquakes
   cd your_path/CNN-explainability-Earthquakes  
   ```

3. **Create and activate Virtual Environment**
    ```bash  
    python3 -m venv venv
    source venv/bin/activate
   ```

2. **Install dependecies**
    ```bash 
    pip install -r requirements.txt  
    ```

## Observations & Acknowledgments
Work in progres...

## References  

1. **Our Report**  
   - [***What is Machine Learning Teaching Us?** Ecplainable AI for Seismic Models*](#) (link to be added)  

2. **Project Presentation**  
   - [*Results Presentation*](media/Presentation.pdf) (not completed)

3. **Laurenti, Paolini et al. (Nature 2024)**  
   - [***Probing the Evolution of Fault Properties During the Seismic Cycle with Deep Learning***](https://www.nature.com/articles/s41467-024-54153-w)


