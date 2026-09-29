# A03-Sampling-iym24004

Assignment for OPIM 5512 (Dr. Dave Wanik, University of Connecticut).

## Goal
Try undersampling, oversampling and SMOTE on an imbalanced dataset to see if sampling improves model performance.

## Dataset
California housing data (`california_housing_train.csv` from Colab's sample data).

- Target: `median_house_value`, converted to 1 if the house is under $380k and 0 if it is $380k or more
- Class 0 (expensive houses) is the minority class, about 10% of the data (1,696 out of 17,000 rows)

## What I did
1. Split the data into train (80%) and test (20%) first, using stratify so both keep the same 0/1 ratio. No sampling was done on the test set.
2. Applied three sampling methods to the training data only:
   - Majority undersampling (`RandomUnderSampler`)
   - Minority oversampling (`RandomOverSampler`)
   - SMOTE
3. Trained a Decision Tree (`min_samples_split=10`) for each method and evaluated it on the test set with a classification report and confusion matrix.
4. Compared the three methods.
5. Repeated SMOTE 100 times with a different train/test split each time (`random_state = i`) to see how much the results change.

## Results

| Method        | Accuracy | Precision (class 0) | Recall (class 0) | F1 (class 0) |
|---------------|----------|---------------------|------------------|--------------|
| Undersampling | 0.813    | 0.327               | 0.826            | 0.468        |
| Oversampling  | 0.921    | 0.615               | 0.560            | 0.586        |
| SMOTE         | 0.901    | 0.503               | 0.758            | 0.605        |

- SMOTE had the best F1-score for the minority class.
- Undersampling had the highest recall but very low precision.
- Oversampling had the highest accuracy but the lowest recall.

**Repeated experiment (SMOTE, 100 runs):**
- Average accuracy 0.891, precision 0.471, recall 0.721
- Recall ranged from 0.658 to 0.785, so some splits did much better than others
- One train/test split is not enough to judge a method, so it's better to look at the average over many runs

## Tools
Python, pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn (run in Google Colab)

## How to run
Open `A03_SamplingxAI_5512.ipynb` in Google Colab and click **Runtime → Run all**. The dataset is already included in Colab's `sample_data` folder.
