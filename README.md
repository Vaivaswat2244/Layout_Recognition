# Layout_Recognition

**Layout Organization Recognition** is the process of detecting and classifying structural elements in a document, such as text blocks, headings, images, tables, and footnotes. It utilizes computer vision and machine learning techniques to analyze layouts, enabling better understanding and processing of complex document structures. This technology is crucial in digitizing printed and handwritten materials, improving OCR accuracy, and preserving the original formatting of documents.  

In historical books, Layout Organization Recognition helps in enhancing OCR by accurately segmenting text from decorative elements or marginalia, improving transcription quality. It aids in structuring content by distinguishing headings, paragraphs, and multi-column layouts, ensuring proper digitization for archival purposes. Additionally, it enables automated metadata extraction for indexing, searchability, and even restoration of damaged manuscripts. This makes historical texts more accessible for research, translation, and public use.

## Strategy

For Layout Organization Recognition, we leverage **YOLOv8**, a state-of-the-art object detection model, to detect and classify structural elements in document layouts. Our approach involves training YOLOv8 on the **PubLayNet** dataset, which contains annotated document images with labeled regions such as text blocks, titles, tables, figures, and lists.  

1. **Dataset Preparation:** We preprocess the **PubLayNet** dataset to align with YOLOv8’s input format, ensuring proper annotation conversion and image resizing while maintaining aspect ratios.  
2. **Model Training:** We fine-tune YOLOv8 using transfer learning on PubLayNet, optimizing it for detecting layout elements with high precision and recall.
3. **Testing**: The model is tested on the pdfs present in this [folder](https://bama365-my.sharepoint.com/personal/xgranja_ua_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxgranja%5Fua%5Fedu%2FDocuments%2FUA%2F1%2E%20Research%2FAI%2FHumanAI%2FGSoC%2025%2FTest%2FTest%20sources&ga=1). 

## **Evaluation Metrics**  

To assess the performance of our YOLOv8 model for Layout Organization Recognition, we utilize the key metrics recorded in the `result.csv` file generated during training. These metrics help in monitoring training progress, detecting overfitting, and selecting the best model checkpoint.

#### **1. Training Losses:**  
- `train/box_loss`: Measures localization error in bounding box predictions.  
- `train/cls_loss`: Evaluates classification error in detected elements.  
- `train/dfl_loss`: Distribution Focal Loss, affecting bounding box refinement.  

#### **2. Validation Metrics:**  
- `metrics/precision(B)`: The ability of the model to avoid false positives.  
- `metrics/recall(B)`: The ability of the model to correctly detect true positives.  
- `metrics/mAP50(B)`: Mean Average Precision at IoU=0.5, a standard detection metric.  
- `metrics/mAP50-95(B)`: mAP computed across multiple IoU thresholds (0.5 to 0.95), providing a more comprehensive evaluation.  
- `val/box_loss`, `val/cls_loss`, `val/dfl_loss`: Validation losses indicating how well the model generalizes to unseen data.  

#### **3. Learning Rate Monitoring:**  
- `lr/pg0`, `lr/pg1`, `lr/pg2`: Learning rates for different parameter groups, useful for diagnosing training stability.  

### **Using These for Evaluation**  
- **Early Stopping Criteria:** Training should stop when `val/box_loss` and `val/cls_loss` stop decreasing, and `metrics/mAP50-95(B)` saturates.  
- **Overfitting Detection:** A large gap between training and validation losses suggests overfitting.  
- **Final Model Selection:** The checkpoint with the **highest `metrics/mAP50-95(B)` and lowest `val/box_loss`** should be chosen for deployment.  

These metrics ensure our model effectively recognizes diverse layouts while maintaining high precision and recall.

Demo Notebook:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x02QwVEedDWl3OmbbRmQspkpn2II-aqY?usp=sharing)

<img src="https://github.com/phoeenniixx/Layout_Recognition/blob/main/images/F1_curve.png">
<img src="https://github.com/phoeenniixx/Layout_Recognition/blob/main/images/PR_curve.png">
<img src="https://github.com/phoeenniixx/Layout_Recognition/blob/main/images/confusion_matrix.png">
<img src="https://github.com/phoeenniixx/Layout_Recognition/blob/main/images/confusion_matrix_normalized.png">
<img src="https://github.com/phoeenniixx/Layout_Recognition/blob/main/images/results.png">

