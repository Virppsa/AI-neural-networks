# AI-neural-networks
In this repo, I am including all of my work linked to AI. This repository contains three lab assignments that demonstrate different deep learning methods applied to image classification and segmentation tasks. Each notebook builds upon neural network models to solve distinct tasks, with evaluation metrics to validate performance. I included another file of a document that demonstrates our work on detecting buildings in satellite and aerial images.

## Lab Descriptions

### 1. gmm1.ipynb
This notebook addresses the first lab assignment, where I implemented an image classification pipeline using a pre-trained image classification model. Key features include:

- **Data Loading**: An efficient data-loading function to process a selected set of 1000 images from the OpenImages dataset.
- **Metrics Calculation**: Calculated performance metrics, including accuracy, precision, recall, and F1 score, on the selected images.
- **Threshold Adjustment**: Implemented a mechanism to adjust the classification threshold (`T ∈ [0, 1]`) for each class, allowing dynamic evaluation of model performance. The metrics automatically recalculate whenever the threshold is changed.

### 2. gmm2.modelis.ipynb and api.py
In this notebook, I developed a custom image classification model, with the following components:

- **Model Training**: The model was trained on a dataset divided into training and testing sets, with at least three classes selected from the OpenImages V6 dataset.
- **API Implementation**: I created a functional API to allow external use of the model for classification tasks. During evaluation, the instructor provided test images to demonstrate the API’s functionality.
- **Evaluation Metrics**: Computed a confusion matrix along with accuracy, precision, recall, and F1 score metrics on the test set.

### 3. gmm_3.ipynb
The final notebook contains the implementation of an image segmentation model:

- **Model Design and Training**: Built and trained an image segmentation model on a dataset divided into training and testing sets, with a minimum of three classes chosen from the OpenImages V6 segmentation dataset.
- **Evaluation Metrics**: On the test set, I calculated the Dice score, Micro-F1, and Macro-F1 metrics to evaluate the segmentation accuracy across classes.
- **Demonstration**: The instructor provided test images during evaluation to demonstrate the model’s segmentation capabilities in real-time.

### 4. Pastatų_atpažinimas__GMM.pdf
This project, titled **"Detection of Buildings"**, was a collaborative effort to apply deep learning methods for detecting buildings in satellite and aerial images. Key aspects of this project include:

- **Objective**: Accurate recognition of buildings from satellite and aerial imagery, with applications in urban planning and geographic information systems.
- **Data Sources**: We used the Open Buildings dataset from South Sudan and a custom dataset of Google Maps satellite images from the United States and Europe.
- **Model Architecture**: A U-Net convolutional neural network was implemented for semantic segmentation. Two versions were created: one with ImageNet pre-trained weights and one trained from scratch with custom weights.
- **Evaluation Metrics**: The segmentation performance was assessed using Dice coefficient and Intersection over Union (IoU) metrics.
- **Threshold Analysis**: Threshold values were analyzed to find optimal detection thresholds, highlighting the trade-off between sensitivity and specificity in building recognition.
- **Results**: The model with ImageNet weights generally performed more smoothly and demonstrated better stability, though both models faced challenges in distinguishing buildings from shadows and roads.
- **Tools**: Implemented in PyTorch and tested using QGIS and Google Colaboratory Pro (T4 GPU).

This approach successfully demonstrated the potential of deep learning models in handling complex building recognition tasks, although further improvements, such as larger datasets and enhanced data augmentation, were identified as future directions for improvement.

Here are some pictures:
<img width="470" alt="image" src="https://github.com/user-attachments/assets/43aab445-88c7-4e21-bb0d-90e00c3c41f5">


---

Each notebook contains the complete code and explanations of the models, methods, and metrics used. The assignments were completed in accordance with the provided instructions, and the models have been validated using appropriate statistical metrics for each task.
