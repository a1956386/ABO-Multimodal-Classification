# Multimodal Product Type Classification Using Amazon Berkeley Objects

## Project Overview

This project explores multimodal machine learning for product type classification using the Amazon Berkeley Objects (ABO) dataset. It combines product images and textual descriptions to predict product categories.

The project compares five machine learning approaches to evaluate the effectiveness of combining image and text features.

## Dataset

**Source:** Amazon Berkeley Objects (ABO)

**Hugging Face Dataset:** `hyper3labs/amazon-berkeley-objects`

The original dataset contains 147,702 product listings.

After data cleaning, a balanced sample of 1,600 products across eight categories was selected.

The dataset was divided into:

- Training: 1,120 products (70%)
- Validation: 240 products (15%)
- Testing: 240 products (15%)

Data preprocessing included handling missing values, removing ambiguous duplicate product IDs, checking image availability and creating leakage-controlled dataset splits.

## Models Implemented

### 1. Text-Only Model
TF-IDF feature extraction with Logistic Regression.

### 2. Image-Only Model
Pretrained ResNet-50 for image feature extraction with Logistic Regression.

### 3. Simple Multimodal Model
Concatenation of TF-IDF and ResNet-50 features followed by Logistic Regression.

### 4. Early Fusion Model
ResNet-50 image embeddings and DistilBERT text embeddings combined through trainable projection layers and a multilayer perceptron classifier.

### 5. Late Fusion Model
Weighted probability combination of independently trained text and image classifiers.

## Final Experimental Results

| Model | Test Accuracy | Macro-F1 |
|---|---:|---:|
| Text-only | 84.17% | 0.8450 |
| Image-only | 78.33% | 0.7896 |
| Simple Multimodal | 78.75% | 0.7927 |
| Early Fusion | 84.17% | 0.8447 |
| Late Fusion | **85.42%** | **0.8569** |

The best-performing approach was weighted late fusion, using 65% text predictions and 35% image predictions.

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro-F1
- Confusion Matrix
- Overfitting Analysis

## Technologies Used

Python, PyTorch, Torchvision, Hugging Face Transformers, Hugging Face Datasets, Scikit-learn, Pandas, NumPy and Google Colab.

## How to Run

1. Download or clone this repository.
2. Open `ABO_Assignment_2_2Final.ipynb` in Google Colab.
3. Enable a GPU runtime if available.
4. Install any missing Python dependencies.
5. Run the notebook cells sequentially to load the dataset, preprocess the data, train the models and evaluate the results.
6. Refer to the corrected experiment results for the final model comparison.

The notebook downloads product images and pretrained model weights. An internet connection is required.

## Limitations

The study uses a relatively small, balanced subset of the ABO dataset. Results may differ on larger or naturally imbalanced product datasets.

Overlapping product categories, multilingual titles and visual similarities between products can affect classification performance.

## Conclusion

The results indicate that combining text and image predictions through weighted late fusion improved classification performance compared with the individual text-only and image-only models in this experiment.

The best model achieved 85.42% test accuracy and a Macro-F1 score of 0.8569.

## Author

**Spoorthi**  
Student ID: a1956386  
Adelaide University  
Assignment 2.2 – Multimodal Machine Learning Final Project
