# SURFY2

This repository constitutes SURFY2 and corresponds to the bioRxiv preprint 'Updating the in silico human surfaceome with meta-ensemble learning and feature engineering' by Daniel Bojar. SURFY2 is a machine learning classifier to predict whether a human transmembrane protein is located at the surface of a cell (the plasma membrane) or in one of the intracellular membranes based on the sequence characteristics of the protein. Making use of the data described in the recent publication from Bausch-Fluck et al. (https://doi.org/10.1073/pnas.1808790115), SURFY2 considerably improves on their reported classifier SURFY in terms of accuracy (95.5%), precision (94.3%), recall (97.6%) and area under ROC curve (0.954) when using a test set never seen by the classifier before.

SURFY2 consists of a layer of 12 base estimators generating 24 new engineered features (class probabilities for both classes) which are appended to the original 253 features. Then, a soft voting classifier with three optimized base estimators (Random Forest, Gradient Boosting and Logistic Regression) and optimized voting weights is trained on this expanded dataset, resulting in the final prediction. The motivation of SURFY2 is to provide an updated and better version of the in silico human surfaceome to facilitate research and drug development on human surface-exposed transmembrane proteins. Additionally, SURFY2 enabled insights into biological properties of these proteins and generated several new hypotheses / ideas for experiments.

The workflow is as following:

1) dataPrep Gets training data from data.xlsx, labels it according to surface class and outputs 'train_data.csv'.

2) split Gets train_data.csv, splits it into training, validation and test data and outputs 'train.csv', 'val.csv', 'test.csv'.

3) main_val Was used for optimizing hyperparameters of base estimators and estimators &amp; weights of voting classifier. Stores all estimators. Evaluates meta-ensemble classifier SURFY2 on validation set.

4) classifier_selection All base estimators and meta-ensemble approaches are tested on the initial dataset as well as the expanded dataset including the engineered features and compared in terms of their cross-validation score.

5) main_test Evaluates SURFY2 on the separate test set (trained on training + validation set).

6) testing_SURFY Evaluates the original SURFY through cross-validation and on validation as well as test set.

7) pred_unlabeled Uses SURFY2 to predict the surface label (+ prediction score) for unlabeled proteins in data.xlsx. Also gets the feature importances of the voting classifier estimators.

8) getting_discrepancies Compare predictions with those made by SURFY ('surfy.xlsx') and store mismatches. Also store the 10 most confident mismatches (by SURFY2 classification score) from each class.

9) feature_importances Plot the 10 most important features for the voting classifier estimators (Random Forest, Gradient Boosting, Logistic Regression) to interpret predictions.

10) base_estimator_importances Plot the 10 most important features for the two most important base estimators (XGBClassifier and Gradient Boosting).

11) comparing_mismatches Separate datasets into shared &amp; discrepant predictions (between SURFY and SURFY2). Compare feature means and select features with the highest class feature mean differences between prediction datasets. Statistically analyze differences in features means between classes in both prediction datasets. Plot 9 representative features with their means grouped according to class and prediction dataset to rationalize discrepant predictions.

12) tSNE_surfy2 Perform nonlinear dimensionality reduction using t-SNE on proteins with predictions from both SURFY and SURFY2. Plot the two t-SNE dimensions and label the proteins according to their prediction class in order to see where discrepant predictions reside in the landscape. Plot surface proteins with most prevalent annotated functional subclasses and label them according to their subclass to enable comparison to class predictions. Functional annotations came from 'surfy.xlsx'.
