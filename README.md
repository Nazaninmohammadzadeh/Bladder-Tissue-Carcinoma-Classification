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

Bladder cancer represents about 4% of all cancers in the U.S. and is the fourth most common cancer in men, though less common in women. In 2024, it's estimated there will be 83,190 new cases (63,070 in men, 20,120 in women) and 16,840 deaths (12,290 in men, 4,550 in women). The incidence rate is 18.2 per 100,000 annually, with a death rate of 4.1 per 100,000, based on data from 2017–2021 and 2018–2022. Lifetime risk for bladder cancer is about 2.2%, with 730,044 people living with it in the U.S. in 2021. At diagnosis, 50% of cases are confined to the bladder's inner layer, 33% have spread deeper, and 5% have distant metastasis. These statistics are derived from SEER data and U.S. Mortality rates, with modeled trend lines calculated using Joinpoint Trend Analysis Software. For more detailed information, consult SEER Cancer Statistics.[1]

The dataset used in this project is derived from the "Endoscopic Bladder Tissue Classification Dataset," as described in the work by Lazo et al. (2023). This dataset, accessible via Zenodo[2], includes endoscopic images of bladder tissue. The dataset was supported by the ATLAS project and received funding from the European Union's Horizon 2020 research and innovation program under Marie Skłodowska-Curie grant agreement No 813782, as well as French State Funds managed by the Agence Nationale de la Recherche (ANR). This dataset was suggested by Semruk Technology[3] for its relevance and quality in classifying bladder tissue types. Besides Zenodo, you can find the dataset on Kaggle[4].

Given the significant impact of bladder tissue conditions on patient health and the importance of accurate diagnosis, we have developed a deep learning model to classify these conditions using endoscopic bladder tissue images. Bladder cancer and related tissue abnormalities are crucial to identify early due to their high prevalence and potential severity. Our model leverages advanced deep learning techniques to analyze images from the "Endoscopic Bladder Tissue Classification Dataset," providing precise classification of tissue types. This approach aims to enhance diagnostic accuracy and support healthcare professionals in delivering timely and effective treatment.

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

## 3. Data Preprocessing

Data preprocessing is a crucial step in any machine learning project, particularly when dealing with image data. The primary purpose of data preprocessing is to prepare the raw data in a form that enhances the model's ability to learn effectively and efficiently. By standardizing and refining the input data, we aim to reduce noise, ensure consistency, and improve the accuracy of the model's predictions.

### 3.1 Data Loading and Class Mapping

- **Class Renaming:** Converted non-numeric class labels into numeric values for model compatibility.
- **Tensor Conversion:** Data frames were converted into tensors, ensuring efficient handling during model training.

### 3.2 Image Preprocessing

- **Resizing:** All images were resized to a consistent dimension of 224x224 pixels to match model input requirements.
- **Normalization:** Image pixel values were scaled between -1 and 1 to standardize the input data.
- **Sharpening:** Applied filters to enhance image clarity, improving feature detection during training.

## 4. Data Augmentation

Data augmentation is a technique used to artificially increase the size and diversity of the training dataset by applying various transformations to the images. The purpose of data augmentation is to improve the model's generalization capabilities, making it more robust to variations in the input data and reducing the risk of overfitting. By simulating different scenarios, such as changes in brightness, contrast, orientation, and scale, data augmentation enables the model to see the same images from different perspectives and dimensions. This exposure helps the model become better equipped to handle real-world variations in endoscopic bladder tissue images, ultimately enhancing its ability to make accurate predictions.

![EfficientNetB0-architecture-36](https://github.com/user-attachments/assets/c11162fc-b27b-4db2-9dc1-f03d337edef0)


### Augmentation Methods

- **Brightness Adjustment:** Variations in image brightness were introduced to make the model resilient to lighting conditions.
- **Contrast Adjustment:** Enhanced the contrast in images to improve the model's ability to differentiate between subtle features.
- **Random Flipping:** Images were flipped horizontally and vertically to diversify the dataset, reducing overfitting.
- **Random Rotation and Zoom:** Applied random rotations and zooming to simulate different perspectives and scales of the tissue samples.

## 5. Model Development

### 5.1 Model Architecture

- **Base Models:**  
  EfficientNetB0 was selected for its balance of accuracy and efficiency. The MBConv6 architecture in EfficientNetB0 is a key block that combines efficiency and performance, making it well-suited for image classification tasks, including bladder cancer classification.


**Overview of MBConv6:**

1. **Expansion Phase:**  
   The input channels are expanded by a factor of 6 using a 1x1 pointwise convolution, increasing the model’s capacity to capture complex features.
   
2. **Depthwise Convolution:**  
   A 3x3 depthwise convolution is applied, which processes each channel separately, reducing computational costs while extracting spatial features. The Swish activation function is then applied.
   
3. **Squeeze-and-Excitation (SE) Module:**  
   This optional module scales the feature map based on global channel importance, helping the network focus on critical features.
   
4. **Projection Phase:**  
   The output is reduced back to the original input size using another 1x1 pointwise convolution, maintaining efficiency.
   
5. **Residual Connection:**  
   If the input and output dimensions match, a skip connection is added, aiding gradient flow and improving model stability.

- **Sequential Model:**  
  Built a layered architecture that included convolutional layers, pooling layers, and dense layers.

### 5.2 Fine-Tuning

Fine-tuning is a critical step in enhancing the performance of a pre-trained model for a specific task. In our project, we employed fine-tuning to adapt a pre-trained EfficientNet model to the task of bladder tissue classification. We began by freezing the first half of the model's layers, preserving the pre-trained weights that capture general features. This approach allowed us to focus the training on the second half of the model, which is more specialized and better suited for our specific task.

By training only the latter half of the model, we could effectively fine-tune the model’s parameters to better align with the characteristics of our dataset without disrupting the foundational features learned from the pre-trained model. To further optimize the training process, we carefully set the learning rate, choosing a lower value to ensure subtle and controlled adjustments to the model’s weights.

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
| **accuracy** | 1.00      | 1.00   | 1.00     | 420     |
| **macro avg**| 1.00      | 1.00   | 1.00     | 420     |
| **weighted avg** | 1.00  | 1.00   | 1.00     | 420     |

### 6.2 Model Comparison

- **Pre-trained EfficientNet vs. Custom CNN:**  
  The EfficientNetB0 model achieved superior results (99%) compared to the custom CNN, demonstrating the effectiveness of transfer learning.

### 6.3 Visualization

- **Accuracy and Loss Curves:**  
  Plotted training and validation accuracy and loss to monitor the model's performance and identify potential overfitting.

## 7. Deployment

### 7.1 Flask Integration

- **Flask Framework:**  
  Deployed the model as a web application using Flask, providing a user-friendly interface for uploading and classifying bladder tissue images.
  
- **Result Visualization:**  
  Implemented a feature to display uploaded images alongside their predicted class labels on the web interface.

  ![__results___38_0](https://github.com/user-attachments/assets/4443e697-dc7a-41b0-8cbd-8c81b1875cb2)


### 7.2 Future Work

- **Model Improvement:**  
  Explore additional data augmentation techniques and fine-tuning strategies to further improve model accuracy.
  
- **Dataset Expansion:**  
  Incorporate a larger and more diverse dataset to enhance the model's robustness and generalization capabilities.

---

### References

1. SEER Cancer Statistics Review, 1975–2020, National Cancer Institute.
2. Zenodo: "Endoscopic Bladder Tissue Classification Dataset" - Jorge F. Lazo et al. (2023).  
3. Semruk Technology: Dataset Recommendation.
4. Kaggle: "Endoscopic Bladder Tissue Classification Dataset."
