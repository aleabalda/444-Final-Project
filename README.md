# Machine Learning Experience
This repository consists of my machine learning experience from an Applied Machine Learning course at the University of Calgary.

## Assignments
Each of the assignments are in their respective folders with their ipynb iles and corresponding pdf files.

### Assignment 1
In this assignment, we dove deep into algorithmic bias. We watched a ted talk video discussing what algorithmic bias is and the consequences of it.

### Assignment 2
In this assignment, we used linear models to perform classification and regression tasks. We went through the typical flow of first retrieving the data, then pre-processing it (checking for missing values in this case), then implementing the chosen model, validating it using training accuracy and validation accuracy, and then finally visualizing the results using a dataframe with pandas.

### Assignment 3
In this assignment, we went through the exact same workflow (data input, pre-processing of data, implementation of ML model, validation, visualization). This time we used **non-linear** models for the data including decision trees, random forest, and gradient boosting. We used the concrete dataset from the yellowbrick library. Validated the model using cross validation by calculating the average training accuracy and validation accuracy using Mean Squared Error (MSE). This was for **Regression**, then we used models for **Classification** such as `SVC` and `DecisionTreeClassifier` from sklearn.

### Assignment 4
In this assignment we used a heart disease dataset which contains information about patients with possible coronary artery disease. This time, we pre-processed the data by checking for missing values and them using sklearn's `SimpleImputer` to fill in the values using the column mean. Then we leveraged sklearn's `ColumnTransformer` to apply different pre-processing steps to different columns (`StandardScaler` for numerical features, `OneHotEncoder` for categorical features, and `passthrough` for binary features). Created a `Pipeline` object to take this column transformer and add one or more models as the subsequemt steps, so for the models I used `LogisticRegression()`, `SVC()`, and `RandomForestClassifier()`. Then I performed a grid search for the best hyperparams for each model. I did this using sklearn's `GridSearchCV` which finds the best hyperparams that maximize the cross-validation score. Finally, I created a stacking classifier to use the previous models as base estimators and a meta-model chosen by me as the `final_estimator`. Then we performed cross-validation on it using `StratifiedKFold` and reported the metrics to then visualize the results.

### Final Project
For the final project the focus was on **Principal Component Analysis (PCA)** and **Clustering**. I went through a similar pre-processing workflow where the data was first scaled, pipeline was created, and a grid search performed to find the ideal hyperparams. We then reduced the dimensions using either PCA or t-SNE, I chose PCA then performed a grid search once again. Plotted the data on a scatter plot to visualize it. Then I created a pipeline which included a scaler and **K-Means Clustering** algorithm. I determined the optimal number of clusters using the KElbowVisualizer then I applied PCA once again to reduce the dimesions to 2D and then created a scatter plot using that data.

## Conclusion
In retrospect, all of these assignments provided me with the knowledge and insight needed to understand datasets and figuring out the best models for certain data. Throughout my ML course I learned a plethora of topics including the fundamentals such as bias and variance, over-fitting and under-fitting, regression and classification, linear models as well as non-linear models, and then finally neural networks. The assignments asked us meaningful reflection questions which forced us to really think about our results and ensure we understand what's going on with our models and data. For example, whether a model is capturing the underlying structure of the dataset or not depending on our calculated metrics such as accuracy or F1-score.
