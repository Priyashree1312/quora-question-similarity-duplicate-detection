# Quora Question Similarity & Duplicate Detection

A machine learning project for detecting whether two Quora questions have the same meaning. The project compares traditional linguistic features, TF-IDF-based features, and semantic embeddings generated using a Sentence Transformer.

## Project Overview

Duplicate-question detection is a natural language processing (NLP) task where the goal is to determine whether two questions express the same underlying intent.

In this project, three approaches were implemented and compared:

1. Engineered linguistic features + Logistic Regression
2. TF-IDF similarity + engineered features + Logistic Regression
3. Sentence Transformer semantic similarity + engineered features + Logistic Regression

The final approach uses the `all-MiniLM-L6-v2` Sentence Transformer to generate semantic representations of question pairs.

## Objective

The main objectives of this project are to:

* Detect duplicate and non-duplicate question pairs.
* Compare traditional NLP features with semantic embeddings.
* Understand the contribution of lexical and semantic similarity.
* Perform error analysis using false positives and false negatives.
* Evaluate different approaches using Accuracy, Precision, Recall, and F1 Score.

## Dataset

The project uses the **Quora Question Pairs** dataset.

After removing rows containing missing question text:

* Total usable question pairs: **404,287**
* Training samples: **323,429**
* Testing samples: **80,858**

### Target Classes

| Class | Description   | Percentage |
| ----- | ------------- | ---------: |
| 0     | Non-Duplicate |     63.08% |
| 1     | Duplicate     |     36.92% |

## Methodology

### 1. Data Preprocessing

The dataset was cleaned by:

* Handling missing question values.
* Removing unusable question pairs.
* Creating linguistic features.
* Performing stratified train-test splitting.

### 2. Engineered Features

The following features were created:

* Question 1 length
* Question 2 length
* Number of words in Question 1
* Number of words in Question 2
* Word overlap
* Fuzzy string similarity
* Fuzzy partial similarity

### 3. Baseline Model

A Logistic Regression classifier was trained using the engineered linguistic features.

This establishes a baseline for comparison with more advanced NLP representations.

### 4. TF-IDF Approach

TF-IDF representations were generated using:

* Unigrams and bigrams
* Maximum 100,000 features
* Minimum document frequency of 2
* Sublinear TF scaling

Pairwise cosine similarity was calculated between the TF-IDF representations of the two questions.

### 5. Sentence Transformer Approach

The `all-MiniLM-L6-v2` Sentence Transformer was used to generate **384-dimensional semantic embeddings**.

Cosine similarity between the two question embeddings was then combined with the engineered linguistic features.

A Logistic Regression classifier was trained using these combined features.

## Model Architecture

```text
                 Quora Question Pair
                         │
              ┌──────────┴──────────┐
              │                     │
         Question 1            Question 2
              │                     │
              ▼                     ▼
       Sentence Transformer   Sentence Transformer
          MiniLM-L6-v2            MiniLM-L6-v2
              │                     │
              └──────────┬──────────┘
                         │
                  Semantic Similarity
                         │
                         ▼
             Engineered Linguistic Features
                         │
                         ▼
                 Feature Combination
                         │
                         ▼
                Logistic Regression
                         │
                         ▼
             Duplicate / Non-Duplicate
```

## Model Performance

| Model                                      | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) |
| ------------------------------------------ | -----------: | ------------: | ---------: | -----------: |
| Engineered Features + Logistic Regression  |        66.93 |         54.15 |      68.05 |        60.31 |
| TF-IDF + Engineered Features               |        67.62 |         54.88 |      69.23 |        61.22 |
| Sentence Transformer + Engineered Features |    **78.42** |     **65.94** |  **85.93** |    **74.62** |

The Sentence Transformer + engineered features approach achieved the highest performance among the approaches evaluated in this project.

Compared with the engineered-feature baseline:

* Accuracy improved by **11.48 percentage points**
* Precision improved by **11.78 percentage points**
* Recall improved by **17.88 percentage points**
* F1 Score improved by **14.31 percentage points**

## Error Analysis

The final model was evaluated using false-positive and false-negative examples.

### False Positives

False positives occurred when questions were highly similar in topic or wording but differed in:

* Intent
* Specific entities
* Quantities
* Context
* Definition versus application

For example, two questions may discuss the same technology but ask different questions about it.

### False Negatives

False negatives occurred when questions had the same underlying intent but used:

* Different wording
* Synonyms
* Additional context
* Different levels of detail
* Domain-specific expressions

This demonstrates that duplicate-question detection requires more than simple lexical matching.

## Semantic Similarity Analysis

The semantic similarity distributions showed a clear difference between the two target classes.

Average semantic similarity on the test set:

| Class         | Mean Semantic Similarity |
| ------------- | -----------------------: |
| Non-Duplicate |                    0.561 |
| Duplicate     |                    0.865 |

This indicates that duplicate question pairs generally had substantially higher semantic similarity, although the distributions still overlapped.

## Threshold Analysis

Different classification thresholds were evaluated.

The threshold of **0.50** produced the highest F1 score among the tested thresholds:

* Accuracy: **78.42%**
* Precision: **65.94%**
* Recall: **85.93%**
* F1 Score: **74.62%**

The threshold analysis was performed experimentally on the test evaluation and should not be interpreted as a universal threshold for all applications.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Sentence Transformers
* PyTorch
* TF-IDF
* Logistic Regression
* Matplotlib
* Google Colab / Jupyter Notebook
* Git & GitHub

## Project Structure

```text
quora-question-similarity-duplicate-detection/
│
├── notebook/
│   └── Quora_Question_Similarity.ipynb
│
├── models/
│   └── semantic_classifier.pkl
│
├── data/
│   └── README.md
│
├── README.md
├── requirements.txt
├── .gitignore
└── app.py
```

> The original Quora dataset is not included in this repository because of its size and dataset licensing/distribution considerations.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/quora-question-similarity-duplicate-detection.git
cd quora-question-similarity-duplicate-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

The main experiments were developed in Google Colab/Jupyter Notebook.

Open:

```text
notebook/Quora_Question_Similarity.ipynb
```

Then run the notebook cells sequentially.

A GPU is recommended for generating Sentence Transformer embeddings on the full dataset.

## Example

Example input:

```text
Question 1:
How can I learn Python?

Question 2:
What is the best way to learn Python?
```

The system generates semantic representations for both questions, calculates their similarity, combines the result with linguistic features, and predicts whether the questions are duplicates.

## Limitations

* The Sentence Transformer model was not fine-tuned specifically for this task.
* Some dataset labels may contain ambiguity or noise.
* The model still produces false positives and false negatives.
* Evaluation was performed using a single stratified train-test split.
* Extensive hyperparameter optimization was not performed.
* Semantic similarity does not always imply identical question intent.
* Generating embeddings for a large dataset requires significant computational resources.

## Future Work

Potential improvements include:

* Fine-tuning the Sentence Transformer on Quora question pairs.
* Using a Cross-Encoder architecture.
* Hard-negative mining.
* Adding intent-aware and entity-aware features.
* Exploring advanced transformer-based models.
* Performing systematic hyperparameter optimization.
* Improving probability calibration and threshold selection.
* Building a Streamlit web application.
* Deploying the model through a REST API.
* Evaluating the model using additional validation splits.

## Conclusion

This project demonstrates a progression from traditional linguistic features to semantic NLP representations for duplicate-question detection.

The Sentence Transformer approach provided substantially stronger performance than the engineered-feature and TF-IDF approaches evaluated in this experiment. The results highlight the value of semantic representations for understanding question meaning while also showing that fine-grained intent differences remain challenging.

Future improvements can focus on task-specific fine-tuning, hard-negative examples, cross-encoder architectures, and deployment as an interactive application.

## Author

**Priyashree Panda**

* LinkedIn: https://linkedin.com/in/priyashree-panda-063ab91bb
* GitHub: https://github.com/Priyashree1312
