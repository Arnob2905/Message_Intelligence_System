# Message Intelligence System

## Project Overview

The Message Intelligence System classifies incoming messages as
**legitimate** or **spam** using machine-learning classifiers. The
supplied dataset contains numerical message summaries and
sender-behavior signals. The target column is `spam_label`:

-   `0` --- Legitimate Message
-   `1` --- Spam Message

The project compares K-Nearest Neighbors (KNN), Support Vector Machine
(SVM) with linear and RBF kernels, and Gaussian Naive Bayes.

## Files

Keep the following files in one project folder:

``` text
Message-Intelligence-System/
├── Message_Intelligence_System_Project.ipynb
├── Message_Intelligence_Dataset.csv
└── README.md
```

## Objectives

-   Inspect the dataset and identify input features and target variable.
-   Preprocess numeric features and scale features where required.
-   Create a stratified training/testing split.
-   Train KNN, compare different values of K and distance metrics, and
    inspect misclassified examples.
-   Train linear and RBF SVM classifiers and examine support vectors.
-   Demonstrate conditional probability and Bayes' Theorem with a small
    manual example.
-   Train Gaussian Naive Bayes and compare model predictions.
-   Compare classifiers using accuracy, precision, recall, F1-score, and
    AUC-ROC.
-   Prepare final conclusions and a deployment recommendation.

## Requirements

-   Python 3
-   Jupyter Notebook, JupyterLab, or Google Colab
-   pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   scikit-learn

Install dependencies in a notebook cell:

``` python
%pip install pandas numpy matplotlib seaborn scikit-learn
```

Or from a terminal:

``` bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

If you install packages while Jupyter is open and imports are not
detected, restart the kernel.

## How to Run

1.  Place `Message_Intelligence_Dataset.csv` in the same folder as the
    notebook.
2.  Open `Message_Intelligence_System_Project.ipynb`.
3.  Check that the dataset path in the setup cell is correct.
4.  Run every cell from top to bottom.
5.  Review the generated tables, confusion matrices, and charts.
6.  Save the executed notebook so it includes outputs.

The notebook calculates metrics from the supplied dataset when executed.
This README does not state performance scores in advance.

## Notebook Sections

-   **Part A --- Probability & Conceptual Foundation:** Conditional
    probability, Bayes' Theorem, Naive Bayes assumptions, KNN, SVM, and
    classifier families.
-   **Part B --- Dataset Understanding & Preparation:** Column
    inspection, target distribution, feature selection, missing-value
    handling, scaling, and stratified splitting.
-   **Part C --- KNN:** Baseline KNN, different K values, distance
    metrics, and misclassified examples.
-   **Part D --- SVM:** Linear and RBF kernels, support-vector analysis,
    and comparison with KNN.
-   **Part E --- Naive Bayes & Probability:** Gaussian Naive Bayes and a
    manual conditional-probability demonstration using two discretized
    signals.
-   **Part F --- Model Comparison:** Accuracy, precision, recall,
    F1-score, AUC-ROC, and high-precision/high-recall comparisons.
-   **Part G --- Final Analysis & Reporting:** Classifier trade-offs,
    assumptions, deployment considerations, and submission checklist.

## Evaluation Metrics

-   **Accuracy:** Proportion of all predictions that are correct.
-   **Precision:** Proportion of messages predicted as spam that are
    actually spam.
-   **Recall:** Proportion of actual spam messages correctly detected.
-   **F1-score:** Harmonic mean of precision and recall.
-   **AUC-ROC:** Measures ranking ability across classification
    thresholds.
-   **Confusion matrix:** Shows true and false predictions for both
    classes.

For spam filtering, high precision helps avoid incorrectly sending
legitimate messages to spam, while high recall helps catch more spam.
The preferred trade-off depends on the application.

## Methodology and Limitations

-   Numeric features are imputed and scaled within model pipelines to
    reduce data leakage.
-   A stratified split preserves the approximate target-class ratio in
    the train and test sets.
-   The model features exclude the identifier `message_id`, raw
    `message_text`, and raw `timestamp`. This follows the assignment's
    focus on numerical summaries and sender-behavior signals. A
    text-based TF-IDF model could be evaluated as a separate extension.
-   The manual probability example uses two simplified signals and is
    not expected to exactly match Gaussian Naive Bayes, which uses all
    selected numeric features and Gaussian likelihoods.
-   Choosing K using test performance is exploratory. For strict model
    selection, use cross-validation on training data and reserve the
    test set for final evaluation.
-   Before deployment, validate on fresh messages, monitor changes in
    message patterns, and provide a way to report false classifications.


