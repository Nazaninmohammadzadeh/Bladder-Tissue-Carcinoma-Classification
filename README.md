# Bladder Tissue Carcinoma Prediction

**August 2024**  
**Nazanin Mohammadzadeh**

## 1. Description

Bladder carcinoma refers to cancer that originates in the lining of the bladder and can manifest in various forms based on the type of cells affected and the severity of the disease. Accurate classification is essential for determining the appropriate treatment and management strategies. The main types of bladder carcinoma tissue include:

- **High-Grade Carcinoma (HGC):**  
  A severe form of bladder cancer characterized by aggressive tumor behavior. The cancer cells are highly abnormal, grow rapidly, and are more likely to spread. Early detection and intervention are crucial for improving patient outcomes.

- **Low-Grade Carcinoma (LGC):**  
  Less aggressive compared to high-grade types. The cancer cells in low-grade carcinoma appear closer to normal and tend to grow slowly. While the prognosis is generally better than for high-grade carcinoma, ongoing monitoring and treatment are necessary.

- **Normal Squamous Tissue (NST):**  
  The healthy lining of the bladder composed of flat, thin cells. Although this tissue is not cancerous, its health is important for the bladder's proper function.

- **Normal Transitional Epithelium (NTL):**  
  The flexible, stretching lining of the bladder. It allows the bladder to accommodate varying volumes of urine and plays a vital role in bladder function.

Bladder cancer is a significant health concern, representing about 4% of all cancers in the U.S., with an estimated 83,190 new cases and 16,840 deaths in 2024. This project uses the "Endoscopic Bladder Tissue Classification Dataset" by Lazo et al. (2023) to develop a deep learning model for accurately classifying bladder tissue conditions, aiming to improve diagnostic accuracy. The dataset is supported by the European Union's Horizon 2020 program and is available on Zenodo[1] and Kaggle[2].

## 2. Library Imports and Environment Setup

- **TensorFlow:** 2.16.1
- **Scikit-learn:** 1.2.2
- **Flask:** 3.0.3
- **PIL:** 9.5.0
- **Seaborn:** 0.12.2
- **CV2:** 4.10.0
- **Keras:** 3.3.3
- **Matplotlib:** 3.7.5
- **Pandas:** 2.2.2
- **NumPy:** 1.26.0
- **Python Version:** 3.9.13

## 3. Data Preprocessing

### 3.1 Data Loading and Class Mapping

- **Class Renaming:** Converted non-numeric class labels into numeric values for model compatibility.
- **Tensor Conversion:** Data frames were converted into tensors, ensuring efficient handling during model training.

### 3.2 Image Preprocessing

- **Resizing:** All images were resized to a consistent dimension of 224x224 pixels to match model input requirements.
- **Normalization:** Image pixel values were scaled between -1 and 1 to standardize the input data.
- **Sharpening:** Applied filters to enhance image clarity, improving feature detection during training.

## 4. Data Augmentation

- **Brightness Adjustment:** Variations in image brightness were introduced to make the model resilient to lighting conditions.
- **Contrast Adjustment:** Enhanced the contrast in images to improve the model's ability to differentiate between subtle features.
- **Random Flipping:** Images were flipped horizontally and vertically to diversify the dataset, reducing overfitting.
- **Random Rotation and Zoom:** Applied random rotations and zooming to simulate different perspectives and scales of the tissue samples.

## 5. Model Development

- **Base Models:**  
  EfficientNetB0 was selected for its balance of accuracy and efficiency. The MBConv6 architecture in EfficientNetB0 is a key block that combines efficiency and performance, making it well-suited for image classification tasks, including bladder cancer classification.

- **Sequential Model:**  
  Built a layered architecture that included convolutional layers, pooling layers, and dense layers.

### 5.2 Fine-Tuning

- **Layer Freezing:**  
  Initial layers were frozen to leverage pre-trained weights, focusing training on the final layers.
  
- **Learning Rate Adjustment:**  
  We used a learning rate of 0.001 to optimize the training process. This value strikes a balance between making meaningful progress during training and avoiding the risks of overshooting or getting stuck in local minima.

## 6. Model Evaluation

### 6.1 Performance Metrics

- **Accuracy:**  
  The model's overall accuracy was measured and compared across different epochs and configurations (99%).

- **Confusion Matrix:**  
  A confusion matrix was generated to analyze the model's performance across each class, identifying strengths and weaknesses.

- **Classification Report:**  
  Provided a detailed breakdown of precision, recall, and F1-score for each class, offering deeper insights into model performance.

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| **HGC**      | 1.00      | 1.00   | 1.00     | 120     |
| **LGC**      | 1.00      | 0.99   | 0.99     | 110     |
| **NST**      | 1.00      | 1.00   | 1.00     | 100     |
| **NTL**      | 1.00      | 1.00   | 1.00     | 90      |
| **accuracy** |           |        | 1.00     | 420     |
| **macro avg**| 1.00      | 1.00   | 1.00     | 420     |
| **weighted avg** | 1.00  | 1.00   | 1.00     | 420     |




### 6.2 Model Comparison

- **Pre-trained EfficientNet vs. Custom CNN:**  
  The EfficientNetB0 model achieved superior results (99%) compared to the custom CNN, demonstrating the effectiveness of transfer learning.

## 7. Deployment

### 7.1 Flask Integration

- **Flask Framework:**  
  Deployed the model as a web application using Flask, providing a user-friendly interface for uploading and classifying bladder tissue images.
  
- **Result Visualization:**  
  Implemented a feature to display uploaded images alongside their predicted class labels on the web interface.

### 7.2 Future Work

- **Model Improvement:**  
  Explore additional data augmentation techniques and fine-tuning strategies to further improve model accuracy.
  
- **Dataset Expansion:**  
  Incorporate a larger and more diverse dataset to enhance the model's robustness and generalization capabilities.

---

### References

1. [Zenodo](https://zenodo.org/records/7741476)
2. [Kaggle](https://www.kaggle.com/datasets/aryashah2k/endoscopic-bladder-tissue-classification-
dataset/data)
