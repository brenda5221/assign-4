# assign-4
# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))

# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# for code formating
%load_ext nb_black

# read the data and split it into a training and test set
url = "http://bit.ly/wine-quality-lwd"
wine = pd.read_csv(url)
X = wine.drop("quality", axis=1).copy()
y = wine["quality"].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# fill missing values with medians
imputer = SimpleImputer(strategy="median")
X_train_tr = imputer.fit_transform(X_train)
# scale the data
scale = StandardScaler()
X_train_tr = scale.fit_transform(X_train_tr)

# do the same for test data. But here we will not apply the
# fit method only the transform method because we
# do not want our model to learn anything from the test data
X_test_tr = imputer.transform(X_test)
X_test_tr = scale.transform(X_test_tr)
from sklearn.neighbors import KNeighborsRegressor

# initiate the k-nearest neighbors regressor class
knn = KNeighborsRegressor()
# train the knn model on training data
knn.fit(X_train_tr, y_train)
# make predictions on test data
y_pred = knn.predict(X_test_tr)
# measure the performance of the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse)
