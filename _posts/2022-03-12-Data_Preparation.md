Data Preparation for Machine Learning - Data Cleaning, Feature Selection, and Data Transform
P.06) The step before data preparation involves defining the problem. As part of
defining the problem, this may involve many sub-tasks, such as:
 Gather data from the problem domain.
 Discuss the project with subject matter experts.
 Select those variables to be used as inputs and outputs for a predictive model.
 Review the data that has been collected.
 Summarize the collected data using statistical methods.
 Visualize the collected data using plots and charts.
P.07) Model evaluation may involve sub-tasks such as:
 Select a performance metric for evaluating model predictive skill.
 Select a model evaluation procedure.
 Select algorithms to evaluate.
 Tune algorithm hyperparameters.
 Combine predictive models into ensembles.
P.10) Data refers to examples or cases from the domain that characterize the problem you want to solve.
What we call data are observations of real-world phenomena. [...] Each piece of data
provides a small window into a limited aspect of reality.
P.11) Data collected from your domain is referred to as raw data and is collected in the context of a problem you want to solve. This means you must first define what you want to predict, then gather the data that you think will help you best make the predictions.
A feature is a numeric representation of an aspect of raw data. Features sit between data and models in the machine learning pipeline. Feature engineering is the act of extracting features from raw data and transforming them into formats that are suitable for the machine learning model.
2.3.1 Machine Learning Algorithms Expect Numbers
2.3.2 Machine Learning Algorithms Have Requirements
It is a good practice to evaluate a suite of different candidate algorithms systematically and discover what works well or best on our data.
P.12) For example, some algorithms assume each input variable, and perhaps the target variable,
to have a specific probability distribution. This is often the case for linear machine learning
models that expect each numeric input variable to have a Gaussian probability distribution.
This means that if you have input variables that are not Gaussian or nearly Gaussian, you
might need to change them so that they are Gaussian or more Gaussian.
Some algorithms are known to perform worse if there are input variables that are irrelevant or redundant to the target variable. There are also algorithms that are negatively impacted if two or more input variables are highly correlated. In these cases, irrelevant or highly correlated variables may need to be identified and removed, or alternate algorithms may need to be used.
As such, there is an interplay (تفاعل) between the data and the choice of algorithms.
2.3.3 Model Performance Depends on Data
The idea that there are different ways to represent predictors in a model, and that some of these representations are better than others, leads to the idea of feature engineering — the process of creating representations of data that increase the effectiveness of a model.
P.13) 
 Complex Data: Raw data contains compressed complex nonlinear relationships that may need to be exposed
 Messy Data: Raw data contains statistical noise, errors, missing values, and conflicting examples.
P.14) In order for you to be an effective machine learning practitioner, you must know:
 The different types of data preparation to consider on a project.
 The top few algorithms for each class of data preparation technique.
 When to use and how to configure top data preparation techniques.
P.17) Data Preparation Tasks:
 Data Cleaning: Identifying and correcting mistakes or errors in the data.
Once messy, noisy, corrupt, or erroneous observations are identified, they can be addressed. This might involve removing a row or a column. Alternately, it might involve replacing observations with new values. As such, there are general data cleaning operations that can be performed, such as:
	 Using statistics to define normal data and identify outliers (Chapter 6).
	 Identifying columns that have the same value or no variance and removing them (Chapter 5).
	 Identifying duplicate rows of data and removing them (Chapter 5).
	 Marking empty values as missing (Chapter 7).
	 Imputing missing values using statistics or a learned model (Chapters 8, 9 and 10).
Data cleaning is an operation that is typically performed first, prior to other data preparation operations.
 Feature Selection: Identifying those input variables that are most relevant to the task.
P.19) Statistical methods, such as correlation, are popular for scoring input features. The features can then be ranked by their scores and a subset with the largest scores used as input to a model.
 Data Transforms: Changing the scale or distribution of variables.
	 Discretization Transform: Encode a numeric variable as an ordinal variable (Chapter 22).
	 Ordinal Transform: Encode a categorical variable into an integer variable (Chapter 19).
	 One Hot Transform: Encode a categorical variable into binary variables (Chapter 19).
	 Normalization Transform: Scale a variable to the range 0 and 1 (Chapters 17 and 18).
	 Standardization Transform: Scale a variable to a standard Gaussian (Chapter 17).
	 Power Transform: Change the distribution of a variable to be more Gaussian (Chapter 20).
	 Quantile Transform: Impose a probability distribution such as uniform or Gaussian (Ch. 21).
An important consideration with data transforms is that the operations are generally performed separately for each variable. As such, we may want to perform different operations on different variable types. We may also want to use the transform on new data in the future. This can be achieved by saving the transform objects to file along with the final model trained on all available data.


 Feature Engineering: Deriving new variables from available data.
P.22) There are some techniques that can be reused, such as:
 Adding a boolean flag variable for some state.
 Adding a group or global summary statistic, such as a mean.
 Adding new variables for each component of a compound variable, such as a date-time.
	 Polynomial Transform: Create copies of numerical input variables that are raised to a power (Ch. 23).

 Dimensionality Reduction: Creating compact projections of the data.
P.23) The most common approach to dimensionality reduction is to use a matrix factorization technique:
 Principal Component Analysis (Chapter 29).
 Singular Value Decomposition (Chapter 30).
The main impact of these techniques is that they remove linear dependencies between input variables, e.g. correlated variables.
 Linear Discriminant Analysis (Chapter 28).


P.25) Data preparation is the process of transforming raw data into a form that is appropriate for modeling. A naive approach to preparing data applies the transform on the entire dataset before evaluating the performance of the model. This results in a problem referred to as data leakage, where knowledge of the hold-out test set leaks into the dataset used to train the model.
leakage means that information is revealed to the model that gives it an unrealistic advantage to make better predictions. This could happen when test data is leaked into the training set, or when data from the future is leaked to the past. Any time that a model is given information that it shouldn’t have access to when it is making predictions in real time in production, there is leakage.
We get the same type of leakage with almost all data preparation techniques; for example, standardization estimates the mean and standard deviation values from the domain in order to scale the variables. Even models that impute missing values using a model or summary statistics will draw on the full dataset to fill in values in the training dataset.
The solution is straightforward. Data preparation must be fit on the training dataset only.
1. Split Data.
2. Fit Data Preparation on Training Dataset.
3. Apply Data Preparation to Train and Test Datasets.
4. Evaluate Models.
P.29) 4.3.1 Train-Test Evaluation With Naive Data Preparation
# naive approach to normalizing the data before splitting the data and evaluating the model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
random_state=7)						# define dataset
scaler = MinMaxScaler()					# standardize the dataset
X = scaler.fit_transform(X)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
model = LogisticRegression()				# fit the model
model.fit(X_train, y_train)
yhat = model.predict(X_test)				# evaluate the model
accuracy = accuracy_score(y_test, yhat)			# evaluate predictions
print( ' Accuracy: %.3f ' % (accuracy*100))
P.30) 4.3.2 Train-Test Evaluation With Correct Data Preparation 
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
random_state=7)						# define dataset
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
scaler = MinMaxScaler()					# define the scaler
scaler.fit(X_train)						# fit on the training dataset
X_train = scaler.transform(X_train)				# scale the training dataset
X_test = scaler.transform(X_test)				# scale the test dataset
model = LogisticRegression()				# fit the model
model.fit(X_train, y_train)
yhat = model.predict(X_test)					# evaluate the model
accuracy = accuracy_score(y_test, yhat)			# evaluate predictions
print( ' Accuracy: %.3f ' % (accuracy*100))
4.4 Data Preparation With k-fold Cross-Validation
P.31) The k-fold cross-validation procedure generally gives a more reliable estimate of model performance than a train-test split, although it is more computationally expensive given the repeated fitting and evaluation of models.
4.4.1 Cross-Validation Evaluation With Naive Data Preparation
We will use repeated stratified 10-fold cross-validation, which is a best practice for classification. Repeated means that the whole cross-validation procedure is repeated multiple times, three in this case.
P.29)# naive data preparation for model evaluation with k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
random_state=7)						# define dataset
scaler = MinMaxScaler()					# standardize the dataset
X = scaler.fit_transform(X)
model = LogisticRegression()					# fit the model
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model using cross-validation
scores = cross_val_score(model, X, y, scoring= ' accuracy ' , cv=cv, n_jobs=-1)
print( ' Accuracy: %.3f (%.3f) ' % (mean(scores)*100, std(scores)*100))		# report performance

4.4.2 Cross-Validation Evaluation With Correct Data Preparation
from sklearn.pipeline import Pipeline
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5,
random_state=7)						# define dataset
steps = list()							# define the pipeline
steps.append(( ' scaler ' , MinMaxScaler()))
steps.append(( ' model ' , LogisticRegression()))
pipeline = Pipeline(steps=steps)
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate the model using cross-validation
scores = cross_val_score(model, X, y, scoring= ' accuracy ' , cv=cv, n_jobs=-1)
print( ' Accuracy: %.3f (%.3f) ' % (mean(scores)*100, std(scores)*100))		# report performance
P.38) Data cleaning is used to refer to all kinds of tasks and activities to detect and repair errors in the data. The basic data cleaning you should always perform on your dataset:
1. Messy Datasets
2. Identify Columns That Contain a Single Value: You can detect rows that have this property using the unique() NumPy function that will report the number of unique values in each column. A simpler approach is to use the nunique() Pandas function that does the hard work for you.
3. Delete Columns That Contain a Single Value: simply remove the zero-variance predictors.
4. Consider Columns That Have Very Few Values: Depending on the choice of data preparation and modeling algorithms, variables with very few numerical values can also cause errors or unexpected results. For example, I have seen them cause errors when using power transforms for data preparation and when fitting linear models that assume a sensible data probability distribution.
5. Remove Columns That Have A Low Variance: The VarianceThreshold class from the scikit-learn library supports this as a type of feature selection.
6. Identify Rows that Contain Duplicate Data: removing duplicate data will be an important step in ensuring your data can be accurately used. The Pandas function duplicated() will report whether a given row is duplicated or not.
7. Delete Rows that Contain Duplicate Data: Rows of duplicate data should probably be deleted from your dataset prior to modeling; Pandas provides the drop_duplicates() function that achieves exactly this.
P.54) Outlier Identification and Removal: Sometimes a dataset can contain extreme values that are outside the range of what is expected and unlike the other data. These are called outliers and often machine learning modeling and model skill in general can be improved by understanding and even removing these outlier values. You, or a domain expert, must interpret the raw observations and decide whether a value is an outlier or not. A good tip is to consider plotting the identified outlier values, perhaps in the context of non-outlier values to see if there are any systematic relationship or pattern to the outliers. If there is, perhaps they are not outliers and can be explained, or perhaps the outliers themselves can be identified more systematically.
Three standard deviations from the mean is a common cut-off in practice for identifying outliers in a Gaussian or Gaussian-like distribution. For smaller samples of data, perhaps a value of 2 standard deviations (95 percent) can be used, and for larger samples, perhaps a value of 4 standard deviations (99.9 %) can be used.
Listing 6.6: Example of identifying and removing outliers using the standard deviation.
from numpy.random import seed				# identify outliers with standard deviation
from numpy.random import randn
from numpy import mean
from numpy import std
seed(1)								# seed the random number generator
data = 5 * randn(10000) + 50					# generate univariate observations
data_mean, data_std = mean(data), std(data)			# calculate summary statistics
cut_off = data_std * 3						# define outliers
lower, upper = data_mean - cut_off, data_mean + cut_off
outliers = [x for x in data if x < lower or x > upper]		# identify outliers
print( ' Identified outliers: %d ' % len(outliers))
outliers_removed = [x for x in data if x >= lower and x <= upper]	# remove outliers
print( ' Non-outlier observations: %d ' % len(outliers_removed))
P.58) 6.5 Interquartile Range Method IQR:
Listing 6.12: Example of a identifying and removing outliers using the IQR.
from numpy import  percentile
q25, q75 = percentile(data, 25), percentile(data, 75)		# calculate interquartile range
iqr = q75 - q25
print( ' Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f ' % (q25, q75, iqr))
cut_off = iqr * 1.5						# calculate the outlier cutoff
lower, upper = q25 - cut_off, q75 + cut_off
P.62) We can remove outliers from the training dataset by defining the LocalOutlierFactor model and using it to make a prediction on the training dataset, marking each row in the training dataset as normal (1) or an outlier (-1). We will use the default hyperparameters for the outlier detection model, although it is a good idea to tune the configuration to the specifics of your dataset.
Listing 6.21: Example of evaluating a model on the regression dataset with outliers removed from the training dataset.
from pandas import read_csv			# evaluate model on training dataset with outliers removed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error
df = read_csv( ' housing.csv ' , header=None)		# load the dataset
data = df.values						# retrieve the array
X, y = data[:, :-1], data[:, -1]					# split into input and output elements
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print(X_train.shape, y_train.shape)				# summarize the shape of the training dataset
lof = LocalOutlierFactor()					# identify outliers in the training dataset
yhat = lof.fit_predict(X_train)
mask = yhat != -1						# select all rows that are not outliers
X_train, y_train = X_train[mask, :], y_train[mask]
print(X_train.shape, y_train.shape)			# summarize the shape of the updated training dataset
model = LinearRegression()
model.fit(X_train, y_train)					# fit the model
yhat = model.predict(X_test)					# evaluate the model
mae = mean_absolute_error(y_test, yhat)			# evaluate predictions
print( ' MAE: %.3f ' % mae)
P.68) Missing values are frequently indicated by out-of-range entries; perhaps a negative number (e.g., -1) in a numeric field that is normally only positive, or a 0 in a numeric field that can never normally be 0. Many popular predictive models such as support vector machines, the glmnet, and neural networks, cannot tolerate any amount of missing values.
P.72) The simplest approach for dealing with missing values is to remove entire predictor(s) and/or sample(s) that contain missing values. We can do this by creating a new Pandas DataFrame with the dropna() function that can be used to drop either columns or rows with missing data.
Listing 7.16: Example of evaluating a model after rows with missing values are removed.
from numpy import nan	# evaluate model on data after rows with missing data are removed
from pandas import read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
dataset = read_csv( ' pima-indians-diabetes.csv ' , header=None)	# load the dataset
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, nan)		# replace ' 0 ' values with ' nan '
dataset.dropna(inplace=True)					# drop rows with missing values
values = dataset.values					# split dataset into inputs and outputs
X = values[:,0:8]
y = values[:,8]
model = LinearDiscriminantAnalysis()			# define the model
cv = KFold(n_splits=3, shuffle=True, random_state=1)	# define the model evaluation procedure
result = cross_val_score(model, X, y, cv=cv, scoring= ' accuracy ' )	# evaluate the model
print( ' Accuracy: %.3f ' % result.mean())			# report the mean performance
Removing rows with missing values can be too limiting on some predictive modeling problems, an alternative is to impute missing values. Most machine learning algorithms require numeric input values, and a value to be present for each row and column in a dataset. As such, missing values can cause problems for machine learning algorithms. Because of this, it is common to identify missing values in a dataset and replace them with a numeric value. This is called data imputing, or missing data imputation. Common data imputation statistical methods are: column mean value, column mode value, column median value or a constant value.
P.78) Listing 8.5: Example of loading and summarizing a dataset with missing values.
from pandas import read_csv					# summarize the horse colic dataset
dataframe = read_csv( ' horse-colic.csv ' , header=None, na_values= ' ? ' )		# load dataset
print(dataframe.head())					# summarize the first few rows
for i in range(dataframe.shape[1]):	# summarize the number of rows with missing values for columns
	n_miss = dataframe[[i]].isnull().sum()			# count number of rows with missing values
	perc = n_miss / dataframe.shape[0] * 100
	print( ' > %d, Missing: %d (%.1f%%) ' % (i, n_miss, perc))
P.79) The scikit-learn library provides the SimpleImputer class that supports statistical imputation.
Listing 8.11: Example of imputing missing values in the dataset.
from numpy import isnan	# statistical imputation transform for the horse colic dataset
from pandas import read_csv
from sklearn.impute import SimpleImputer
dataframe = read_csv( ' horse-colic.csv ' , header=None, na_values= ' ? ' )		# load dataset
data = dataframe.values					# split into input and output elements
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
print( ' Missing: %d ' % sum(isnan(X).flatten()))		# summarize total missing
imputer = SimpleImputer(strategy= ' mean ' )		# define imputer
imputer.fit(X)							# fit on the dataset
Xtrans = imputer.transform(X)				# transform the dataset
print( ' Missing: %d ' % sum(isnan(Xtrans).flatten()))	# summarize total missing
P.80) It is a good practice to evaluate machine learning models on a dataset using k-fold cross -validation. The Pipeline below uses a SimpleImputer with a ‘mean’ strategy, followed by a random forest model.
P.82) Listing 8.16: Example of comparing model performance with different statistical imputation strategies.
from numpy import mean	# evaluate mean imputation and random forest for the horse colic dataset
from numpy import std
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
dataframe = read_csv( ' horse-colic.csv ' , header=None, na_values= ' ? ' )		# load dataset
data = dataframe.values				# split into input and output elements
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
results = list()						# evaluate each strategy on the dataset
strategies = [ ' mean ' , ' median ' , ' most_frequent ' , ' constant ' ]
for s in strategies:
	# create the modeling pipeline
	pipeline = Pipeline(steps=[( ' i ' , SimpleImputer(strategy=s)), ( ' m ' , RandomForestClassifier())])
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)	# evaluate the model
	scores = cross_val_score(pipeline, X, y, scoring= ' accuracy ' , cv=cv, n_jobs=-1)
	print( ' >%s %.3f (%.3f) ' % (s, mean(scores), std(scores)))

pyplot.boxplot(results, labels=strategies, showmeans=True)	# plot models performance
pyplot.show()
P.84) Listing 8.19: Example of making a prediction on data with missing values.
……
p = Pipeline(steps=[( ' i ' , SimpleImputer(strategy= ' constant ' )), ( ' m ' , RandomForestClassifier())])
p.fit(X, y)						# fit the model
row = [2, 1, 530101, 38.50, 66, 28, 3, 3, nan, 2, 5, 4, 4, nan, nan, nan, 3, 5, 45.00, 8.40, nan, nan, 2, 11300, 00000, 00000, 2]					# define new data
yhat = p.predict([row])				# make a prediction
print( ' Predicted Class: %d ' % yhat[0])		# summarize prediction
P.86) A popular approach to missing data imputation is to use a model to predict the missing values. This requires a model to be created for each input variable that has missing values. The k-nearest neighbor (KNN) algorithm has proven to be generally effective. The scikit-learn machine learning library provides the KNNImputer class that supports nearest neighbor imputation.
P.90) Listing 9.10: Example of using the KNNImputer to impute missing values.
from numpy import isnan				# knn imputation transform for the horse colic dataset
from pandas import read_csv
from sklearn.impute import	 KNNImputer
dataframe = read_csv( ' horse-colic.csv ' , header=None, na_values= ' ? ' )		# load dataset
data = dataframe.values				# split into input and output elements
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
print( ' Missing: %d ' % sum(isnan(X).flatten()))	# summarize total missing
imputer = KNNImputer()				# define imputer
imputer.fit(X)						# fit on the dataset
Xtrans = imputer.transform(X)			# transform the dataset
print( ' Missing: %d ' % sum(isnan(Xtrans).flatten()))	# summarize total missing
It is a good practice to evaluate machine learning models on a dataset using k-fold cross-validation.
Li 9.13: Example of evaluating a model on a dataset transformed with the KNNImputer.
...
model = RandomForestClassifier()			# define modeling pipeline
imputer = KNNImputer()
pipeline = Pipeline(steps=[( ' i ' , imputer), ( ' m ' , model)])
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)	# define model evaluation
scores = cross_val_score(pipeline, X, y, scoring= ' accuracy ' , cv=cv, n_jobs=-1)	# evaluate model
print( ' Mean Accuracy: %.3f (%.3f) ' % (mean(scores), std(scores)))
P.100) The scikit-learn machine learning library provides the IterativeImputer class that supports
iterative imputation.
Listing 10.17: Example of comparing model performance with different data order in the IterativeImputer.
from numpy import mean				from numpy import std
from pandas import read_csv 			from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer	from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline		from matplotlib import pyplot
dataframe = read_csv( ' horse-colic.csv ' , header=None, na_values= ' ? ' )		# load dataset
data = dataframe.values				# split into input and output elements
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]
results = list()						# evaluate each strategy on the dataset
strategies = [ ' ascending ' , ' descending ' , ' roman ' , ' arabic ' , ' random ' ]
for s in strategies:
	# create the modeling pipeline
	p = Pipeline(steps=[( ' i ' , IterativeImputer(imputation_order=s)), ( ' m ' , RandomForestClassifier())])
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)	# evaluate the model
	scores = cross_val_score(p, X, y, scoring= ' accuracy ' , cv=cv, n_jobs=-1)
	results.append(scores)				# store results
	print( ' >%s %.3f (%.3f) ' % (s, mean(scores), std(scores)))
pyplot.boxplot(results, labels=strategies, showmeans=True)# plot model performance for comparison
pyplot.show()
P.112) Feature selection methods are intended to reduce the number of input variables to those that are believed to be most useful to a model in order to predict the target variable.
P.111) Statistical-based feature selection methods involve evaluating the relationship between each input variable and the target variable using statistics and selecting those input variables that have the strongest relationship with the target variable.
	 Unsupervised Selection: Do not use the target variable (e.g. remove redundant variables).
	 Supervised Selection: Use the target variable (e.g. remove irrelevant variables).
Supervised feature selection methods may further be classified into three groups, including intrinsic, wrapper, filter methods.
	 Intrinsic: Algorithms that perform automatic feature selection during training.
	 Filter: Select subsets of features based on their relationship with the target.
	 Wrapper: Search subsets of features that perform according to a predictive model.
P.114) Most of these techniques are univariate, meaning that they evaluate each predictor in isolation. In this case, the existence of correlated predictors makes it possible to select important, but redundant, predictors. The obvious consequences of this issue are that too many predictors are chosen and, as a result, collinearity problems arise.

P.116) It is rare that we have a dataset with just a single input variable data type. One approach to handling different input variable data types is to separately select numerical input variables and categorical input variables using appropriate metrics. This can be achieved using the ColumnTransformer class.
Q. How Do You Filter Input Variables? Using the scikit-learn libraries
	1- Rank all input variables by their score and select the k-top input variables: SelectKBest.
	2- convert the scores into a percentage of the largest score and select the top percentile: SelectPercentile.
Q. What is the Best Feature Selection Method?
This is unknowable. Just like there is no best machine learning algorithm, there is no best feature selection technique. At least not universally. Instead, you must discover what works best for your specific problem using careful systematic experimentation. Try a range of different techniques and discover what works best for your specific problem.
P.124) The scikit-learn machine library provides an implementation of the chi-squared test in the chi2() function. This function can be used in a feature selection strategy, such as selecting the top k most relevant features (largest values) via the SelectKBest class.
L 12.16: Example of applying chi-squared feature selection and summarizing the selected features.
from pandas import read_csv		# example of chi squared feature selection for categorical data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder 
from sklearn.feature_selection import SelectKBest, chi2 
from matplotlib import pyplot
def load_dataset(filename):					# load the dataset
	data = read_csv(filename, header=None)			# load the dataset as a pandas DataFrame
	dataset = data.values					# retrieve numpy array
	X = dataset[:, :-1]						# split into input (X) and output (y) variables
	y = dataset[:,-1]
	X = X.astype(str)						# format all fields as string
	return X, y
def prepare_inputs(X_train, X_test):				# prepare input data
	oe = OrdinalEncoder()
	oe.fit(X_train)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc
def prepare_targets(y_train, y_test):				# prepare target
	le = LabelEncoder()
	le.fit(y_train)
	y_train_enc = le.transform(y_train)
	y_test_enc = le.transform(y_test)
	return y_train_enc, y_test_enc
def select_features(X_train, y_train, X_test):		# feature selection
	fs = SelectKBest(score_func=chi2, k= ' all ' )		# configure to select all features
	fs.fit(X_train, y_train)					# learn relationship from training data
	X_train_fs = fs.transform(X_train)				# transform train input data
	X_test_fs = fs.transform(X_test)				# transform test input data
	return X_train_fs, X_test_fs, fs
X, y = load_dataset( ' breast-cancer.csv ' )			# load the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)# split train test sets
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)	# prepare input data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)	# prepare output data
X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)	# feature selection
for i in range(len(fs.scores_)):				# what are scores for the features
	print( ' Feature %d: %f ' % (i, fs.scores_[i]))
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)	# plot the scores
pyplot.show()
Feature 0: 	0.472553
Feature 1: 	0.029193
Feature 2: 	2.137658
Feature 3: 	29.381059
Feature 4: 	8.222601
Feature 5: 	8.100183
Feature 6:	1.273822
Feature 7: 	0.950682
Feature 8: 	3.699989
L 12.19: Example of applying mutual information feature selection and summarizing the features.
Same as Listing 12.16 except two lines:
from sklearn.feature_selection import SelectKBest, mutual_info_classif
	fs = SelectKBest(score_func=mutual_info_classif, k= ' all ' )		# feature selection
Feature 0: 	0.472553 
Feature 1: 	0.029193
Feature 2: 	2.137658
Feature 3: 	29.381059
Feature 4: 	8.222601
Feature 5: 	8.100183
Feature 6:	1.273822
Feature 7: 	0.950682
Feature 8: 	3.699989
P.130) There are many different techniques for scoring features and selecting features based on scores; how do you know which one to use? A robust approach is to evaluate models using different feature selection methods (and numbers of features) and select the method that results in a model with the best performance.
L 12.19: Example of evaluating a model using features selected by chi-squared statistics.
Same as Listing 12.16 except following lines and without plot function:
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
	fs = SelectKBest(score_func=chi2, k= 4 )			# feature selection
model = LogisticRegression(solver= ' lbfgs ' )		# fit the model
model.fit(X_train_fs, y_train_enc)				# evaluate the model
yhat = model.predict(X_test_fs)				# evaluate predictions
accuracy = accuracy_score(y_test_enc, yhat)
print( ' Accuracy: %.2f ' % (accuracy*100))
In the Book; LogisticRegression model applied for all features and accuracy was: 75.79, however after applying chi-squared class the  accuracy was: 74.74.
L 12.27: Example of evaluating a model using features selected by mutual information statistics.
Same as Listing 12.19 except following lines:
from sklearn.feature_selection import mutual_info_classif
	fs = SelectKBest(score_func=mutual_info_classif, k= 4 )	# feature selection
After applying mutual information class the  accuracy was: 76.84. To be sure that the effect is real, it would be a good idea to repeat each experiment multiple times and compare the mean performance. It may also be a good idea to explore using k-fold cross-validation instead of a simple train/test split.
P.140) ANOVA is an acronym for analysis of variance and is a parametric statistical hypothesis test for determining whether the means from two or more samples of data (often three or more) come from the same distribution or not. An F-statistic, or F-test, is a class of statistical tests that calculate the ratio between variances values, such as the variance from two different samples or the explained and unexplained variance by a statistical test, like ANOVA. The ANOVA method is a type of F-statistic referred to here as an ANOVA F-test. Importantly, ANOVA is used when one variable is numeric and one is categorical, such as numerical input variables and a classification target variable in a classification task. 
The scikit-learn machine library provides an implementation of the ANOVA F-test in the f_classif() function.
The scikit-learn machine learning library provides an implementation of mutual information for feature selection with numeric input and categorical output variables via the mutual_info_classif() function.
Listing 13.9 & 13.12: Example of applying ANOVA F-statistic & mutual information features selection and summarizing the selected features.
from pandas import read_csv		# example of chi squared feature selection for categorical data
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import SelectKBest, f_classif,  mutual_info_classif
from matplotlib import pyplot
def load_dataset(filename):					# load the dataset
	data = read_csv(filename, header=None)			# load the dataset as a pandas DataFrame
	dataset = data.values					# retrieve numpy array
	X = dataset[:, :-1]						# split into input (X) and output (y) variables
	y = dataset[:,-1]
	X = X.astype(str)						# format all fields as string
	return X, y
def select_features(X_train, y_train, X_test):		# feature selection
	fs = SelectKBest(score_func=f_classif, k= ' all ' )		# configure to select all features
	fs = SelectKBest(score_func=mutual_info_classif, k= ' all ' )	# configure to select all features
	fs.fit(X_train, y_train)					# learn relationship from training data
	X_train_fs = fs.transform(X_train)				# transform train input data
	X_test_fs = fs.transform(X_test)				# transform test input data
	return X_train_fs, X_test_fs, fs
X, y = load_dataset( ' pima-indians-diabetes.csv ' )		# load the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)# split train test sets
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)	# prepare input data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)	# prepare output data
X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)	# feature selection
for i in range(len(fs.scores_)):				# what are scores for the features
	print( ' Feature %d: %f ' % (i, fs.scores_[i]))
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)	# plot the scores
pyplot.show()

Feature 0: 16.527385    0.118431 
Feature 1: 131.325562  0.019966 
Feature 2: 0.042371 	   0.041791 
Feature 3: 1.415216 	   0.019858 
Feature 4: 12.778966 	   0.084719 
Feature 5: 49.209523    0.018079 
Feature 6: 13.377142 	   0.033098 
Feature 7: 25.126440 	 

In the Book; LogisticRegression model applied for all features and accuracy was: 77.56, however after applying ANOVA F-test statistics the accuracy was: 78.74.& for mutual information ccuracy was: 77.56.
Listing 13.9 & 13.12: Example of evaluating a model using features selected by ANOVA F-test statistics & mutual information features selection and summarizing the selected features.
Same as Listing 12.19 except following lines:
from sklearn.feature_selection import  f_classif
	fs = SelectKBest(score_func=f_classif, k= 4 )			# feature selection
	fs = SelectKBest(score_func=mutual_info_classif, k= 4 )		# feature selection
model = LogisticRegression(solver= ' liblinear ' )			# fit the model
P.150) In the previous example, we selected four features, but how do we know that is a good or best number of features to select? Instead of guessing, we can systematically test a range of different numbers of selected features and discover which results in the best performing model. This is called a grid search, where the k argument to the SelectKBest class can be tuned. It is good practice to evaluate model configurations on classification tasks using repeated stratified k-fold cross-validation. We will use three repeats of 10-fold cross-validation via the RepeatedStratifiedKFold class.
P.151) Listing 13.26: Example of grid searching the number of features selected by ANOVA.
P.153) Listing 13.28: Example of comparing model performance versus the number of selected features with ANOVA.
According output of Listing 13.28, selecting five features might be an appropriate configuration in this case.
P.157) The make regression() function from the scikit-learn library can be used to define a dataset. In this case, we will define a dataset with 1,000 samples, each with 100 input features where 10 are informative and the remaining 90 are irrelevant. There are two popular feature selection techniques that can be used for numerical input data and a numerical target variable. They are:
	 Correlation Statistics: Correlation is a measure of how two variables change together. Perhaps the most common correlation measure is Pearson’s correlation that assumes a Gaussian distribution to each variable
and reports on their linear relationship. The scikit-learn machine library provides an implementation of the
 correlation statistic in the f_regression() function. 
Listing 14.7: Example of applying correlation feature 
selection and summarizing the selected features.
The plot clearly shows 8 to 10 features are a lot more 
important than the other features. We could set k = 10 When 
configuring the SelectKBest to select these top features.
	 Mutual Information Statistics: Mutual information is
calculated between two variables and measures the reduction in uncertainty for one variable given a known value of the other variable. Mutual information is straightforward when considering the distribution of two discrete (categorical or ordinal) variables, such as categorical input and categorical output data. The scikit-learn machine learning library provides an implementation of mutual information for feature selection with numeric input and output variables via the mutual_info_regression() function.
Listing 14.10: Example of applying mutual information feature selection and summarizing the selected features.
In the Book; LogisticRegression model applied for all features and MAE was: 0.086, however after applying model using features selected by correlation statistics (10) the MAE was: 2.740. This suggests
that although the method has a strong idea of what features to select, building a model from these features alone does not result in a more skillful model. This could be because features that are important to the target are being left out, meaning that the method is being deceived about what is important. Finally, after applying model using most features selected by correlation statistics (88) the MAE was: 0.085.
Listing 14.20: Example of evaluating a model using features selected by mutual information statistics.
P.168)  Using Mutual Information; applying model using most features selected by correlation statistics (88) the MAE was: 0.084.
P.170) L 14.26: Example of grid searching the number of features selected by mutual information.
We can see that the best number of selected features is 81, which achieves a MAE of about 0.082.
Recursive Feature Elimination (RFE) is a popular feature selection algorithm. RFE is a wrapper-type feature selection algorithm. This means that a different machine learning algorithm is given and used in the core of the method, is wrapped by RFE, and used to help select features. This is in contrast to filter-based feature selections that score each feature and select those features with the largest (or smallest) score.
The implementation of RFE for machine learning can be done via the RFE class in scikit-learn.
P.177) It is common to use k-fold cross-validation to evaluate a machine learning algorithm on a dataset. When using cross-validation, it is good practice to perform data transforms like RFE as part of a Pipeline to avoid data leakage. 
P.183) It is also possible to automatically select the number of features chosen by RFE. This can be achieved by performing cross-validation evaluation of different numbers of features as we did in the previous section and automatically selecting the number of features that resulted in the best mean score. The RFECV class implements this. 
Listing 15.17: Example of automatically selecting the number of features selected with RFE.
from numpy import mean, std 		# automatically select the number of features for RFE
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold 
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
rfe = RFECV(estimator=DecisionTreeClassifier())					# create pipeline
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[( ' s ' ,rfe),( ' m ' ,model)])
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)	# evaluate model
n_scores = cross_val_score(pipeline, X, y, scoring= ' accuracy ' , cv=cv, n_jobs=-1)
print( ' Accuracy: %.3f (%.3f) ' % (mean(n_scores), std(n_scores)))		# report performance
P.185) Listing 15.21: Example of comparing the base algorithm used by RFE.
We can see the general trend of good performance with logistic regression, CART and perhaps GBM. This highlights that even thought the actual model used to fit the chosen features is the same in each case, the model used within RFE can make an important difference to which features are selected and in turn the performance on the prediction problem.
P.190) Feature importance refers to techniques that assign a score to input features based on how
useful they are at predicting a target variable. The scores are useful and can be used in a range of situations in a predictive modeling problem, such as:	 Better understanding the data.	
	 Better understanding a model.		 Reducing the number of input features.
The relative scores can highlight which features may be most relevant to the target, and the converse, which features are the least relevant. This may be interpreted by a domain expert and could be used as the basis for gathering more or different data.
P.193) 16.4 Coefficients as Feature Importance: Linear machine learning algorithms fit a model where the prediction is the weighted sum of the input values. Examples include linear regression, logistic regression, and extensions that add regularization, such as Ridge regression, LASSO, and the ElasticNet. All of these algorithms find a set of coefficients to use in the weighted sum in order to make a prediction. These coefficients can be used directly as a crude type of feature importance score.
Listing 16.5: Example of calculating feature importance with linear regression.
from sklearn.datasets import make_regression			# linear regression feature importance
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
X, y = make_regression(n_samples=1000, n_features=10,n_informative=5,
 		random_state=1)		# define dataset
model = LinearRegression()		# define the model
model.fit(X, y)			# fit the model
importance = model.coef_		# get importance
for i,v in enumerate(importance):	# summarize feature importance
print( ' Feature: %0d, Score: %.5f ' % (i,v))
pyplot.bar([x for x in range(len(importance))], importance)	# plot feature importance
pyplot.show()
Listing 16.7: Example of calculating feature importance with logistic regression.
Same as Listing 16.5 except:
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()			# define the model 
importance = model.coef_[0]				# get importance
P.196) Decision tree algorithms like classification and regression trees (CART) offer importance scores based on the reduction in the criterion used to select split points, like Gini or entropy. We can use the CART algorithm for feature importance implemented in scikit-learn as the DecisionTreeRegressor and DecisionTreeClassifier classes. After being fit, the model provides a feature_importances_ property that can be accessed to retrieve the relative importance scores for each input feature.
Listing 16.9: Example of calculating feature importance with CART for regression.
Same as Listing 16.5 except:
from sklearn.linear_model import DecisionTreeRegressor
model = DecisionTreeRegressor()			# define the model 
importance = model.feature_importances_		# get importance 
Listing 16.11: Example of calculating feature importance with CART for classification.
Same as Listing 16.5 except:
from sklearn.datasets import make_classification
from sklearn.linear_model import DecisionTreeClassifier
model = DecisionTreeClassifier()			# define the model 
importance = model.feature_importances_		# get importance 
P.200) We can use the Random Forest algorithm for feature importance implemented in scikit-learn as the RandomForestRegressor and RandomForestClassifier classes.
Listing 16.13: Example of calculating feature importance with Random Forest for regression.
Same as Listing 16.5 except:
from sklearn.linear_model import RandomForestRegressor
model = RandomForestRegressor()			# define the model 
importance = model.feature_importances_		# get importance 
Listing 16.15: Example of calculating feature importance with Random Forest for classification.
Same as Listing 16.5 except:
from sklearn.datasets import make_classification
from sklearn.linear_model import RandomForestClassifier
model = RandomForestClassifier()			# define the model 
importance = model.feature_importances_		# get importance 
Permutation feature importance is a technique for calculating relative importance scores that is independent of the model used. First, a model is fit on the dataset, such as a model that does not support native feature importance scores. Then the model is used to make predictions on a dataset, although the values of a feature (column) in the dataset are scrambled. This is repeated for each feature in the dataset. Then this whole process is repeated 3, 5, 10 or more times. The result is a mean importance score for each input feature (and distribution of scores given the repeats).
P.204) Listing 16.17: Example of calculating permutation feature importance for regression.
P.205) Listing 16.19: Example of calculating permutation feature importance for classification.
P.207) Feature importance scores can be used to help interpret the data, but they can also be used directly to help rank and select features that are most useful to a predictive model.
L 16.21: Example of evaluating a model with all selected features.
L 16.25: Example of evaluating a model with feature selection performed using feature importance.
P.210) Q. What Do The Scores Mean? You can interpret the scores as a specific technique relative importance ranking of the input variables. The importance scores are relative, not absolute. This means you can only compare the input variable scores to each other as calculated by a single method.
Q. How Do You Use The Importance Scores? Some popular uses for feature importance scores include:
	 Data interpretation.			 Model interpretation.		 Feature selection.
Q. Which Is The Best Feature Importance Method? This is unknowable. If you are interested in the best model performance, then you can evaluate model performance on features selected using the importance from many different techniques and use the feature importance technique that results in the best performance for your data set on your model.
P.213) How to Scale Numerical Data Many machine learning algorithms perform better when numerical input variables are scaled to a standard range. This includes algorithms that use a weighted sum of the input, like linear regression, and algorithms that use distance measures, like k-nearest neighbors. The two most popular techniques for scaling numerical data prior to modeling are normalization and standardization.
One of the most common forms of pre-processing consists of a simple linear rescaling of the input variables.
Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step. Just to give you an example — if you have multiple independent variables like age, salary, and height; With their range as (18–100 Years), (25,000–75,000 Euros), and (1–2 Meters) respectively, feature scaling would help them all to be in the same range, for example- centered around 0 or in the range (0,1) depending on the scaling technique.
The million-dollar question: Normalization or Standardization? While there is no obvious answer to this question, it really depends on the application, there are still a few generalizations that can be drawn. 
When to scale your data? 	 Gradient Descent Based Algorithms: such as NN		 PCA 
 Distance-Based Algorithms: such as KNN, K-means, and SVM 		 Regression Algorithms
When scaling your data is NOT necessary? 	Tree-based algorithms




































































































P.51) Discrete Probability Distributions(DPDs): The probability for a discrete random variable can be summarized with a DPD. DPDs are used in machine learning, most notably in the modeling of binary and multiclass classification problems, but also in evaluating the performance for binary classification models, such as the calculation of confidence intervals, and in the modeling of the distribution of words in text for natural language processing. Knowledge of DPDs is also required in the choice of activation functions in the output layer of deep learning neural networks for classification tasks and selecting an appropriate loss function. 
P.60) Continuous Probability Distributions(CPDs): The probability for a continuous random variable can be summarized with a CPD. CPDs are encountered in machine learning, most notably in the distribution of numerical input and output variables for models and in the distribution of errors made by models. Knowledge of the normal CPD is also required more generally in the density and parameter estimation performed by many machine learning models.
Unlike a discrete random variable, the probability for a given continuous random variable cannot be specified directly; instead, it is calculated as an integral (area under the curve) for a tiny interval around the specific outcome.
There are many common continuous probability distributions. The most common is the normal probability distribution. Practically all continuous probability distributions of interest belong to the so-called exponential family of distributions, which are just a collection of parameterized probability distributions (e.g. distributions that change based on the values of parameters).
Probability Density Estimation: as we are using the observations in a random sample to estimate the general density of probabilities beyond just the sample of data we have available. There are a few steps in the process of density estimation for a random variable. The first step is to review the density of observations in the random sample with a simple histogram. From the histogram, we might be able to identify a common and well-understood probability distribution that can be used, such as a normal distribution. If not, we may have to fit a model to estimate the distribution.
Listing 10.2: Example of plotting a histogram with 10 bins.
from matplotlib import pyplot		# example of plotting a histogram of a random sample
from numpy.random import normal
sample = normal(size=1000)			# generate a sample
pyplot.hist(sample, bins=10)			# plot a histogram of the sample
pyplot.show()
P.77) Reviewing a histogram of a data sample with a range of different numbers of bins will help to identify whether the density looks like a common probability distribution or not. In most cases, you will see a unimodal distribution, such as the familiar bell shape of the normal, the flat shape of the uniform, or the descending or ascending shape of an exponential or Pareto distribution. You might also see complex distributions, such as two peaks that don’t disappear with different numbers of bins, referred to as a bimodal distribution, or multiple peaks, referred to as a multimodal distribution. You might also see a large spike in density for a given value or small range of values indicating outliers, often occurring on the tail of a distribution far away from the rest of the density.
Get familiar with the common probability distributions as it will help you to identify a given distribution from a histogram. Once identified, you can attempt to estimate the density of the random variable with a chosen probability distribution. This can be achieved by estimating the parameters of the distribution from a random sample of data.
It is possible that the data does match a common probability distribution, but requires a transformation before parametric density estimation. For example, you may have outlier values that are far from the mean or center of mass of the distribution. This may have the effect of giving incorrect estimates of the distribution parameters and, in turn, causing a poor fit to the data. These outliers should be removed prior to estimating the distribution parameters. Another example is the data may have a skew or be shifted left or right. In this case, you might need to transform the data prior to estimating the parameters, such as taking the log or square root, or more generally, using a power transform like the Box-Cox transform. These types of modifications to the data may not be obvious and effective parametric density estimation may require an iterative process of:
 Loop Until Fit of Distribution to Data is Good Enough:
	1. Estimating distribution parameters
	2. Reviewing the resulting PDF against the data
	3. Transforming the data to better fit the distribution
Listing 10.10: Example of generating and plotting a bimodal data sample.
# example of a bimodal data sample
from matplotlib import pyplot
from numpy.random import normal
from numpy  hstack
# generate a sample
sample1 = normal(loc=20, scale=5, size=300)
sample2 = normal(loc=40, scale=5, size=700)
sample = hstack((sample1, sample2))
pyplot.hist(sample, bins=50)		# plot the histogram
pyplot.show()
Listing 10.14: Example of kernel density estimation for a bimodal data sample.
from matplotlib import pyplot
from numpy.random import normal
from numpy import hstack
from numpy import asarray
from numpy import exp
from sklearn.neighbors import KernelDensity
# generate a sample
sample1 = normal(loc=20, scale=5, size=300)
sample2 = normal(loc=40, scale=5, size=700)
sample = hstack((sample1, sample2))
model = KernelDensity(bandwidth=2, kernel= ' gaussian ' )	# fit density
sample = sample.reshape((len(sample), 1))
model.fit(sample)
values = asarray([value for value in range(1, 60)])		# sample probabilities for a range of outcomes
values = values.reshape((len(values), 1))
probabilities = model.score_samples(values)
probabilities = exp(probabilities)
pyplot.hist(sample, bins=50, density=True)			# plot the histogram and pdf
pyplot.plot(values[:], probabilities)
pyplot.show()
P.88) Maximum Likelihood Estimation: involves defining a likelihood function for calculating the conditional probability of observing the data sample given a probability distribution and distribution parameters. This approach can be used to search a space of possible distributions and parameters. This flexible probabilistic framework also provides the foundation for many machine learning algorithms, including important methods such as linear regression and logistic regression for predicting numeric values and class labels respectively, but also more generally for deep learning artificial neural networks.
Under certain assumptions any learning algorithm that minimizes the squared error between the output hypothesis predictions and the training data will output a maximum likelihood hypothesis.
Linear regression fits the line to the data, which can be used to predict a new quantity, whereas logistic regression fits a line to best separate the two classes.
P.111) Many real-world problems have hidden variables (sometimes called latent variables), which are not observable in the data that are available for learning. Conventional maximum likelihood estimation does not work well in the presence of latent variables. A general technique for finding maximum likelihood estimators in latent variable models is the expectation-maximization (EM) algorithm.
The EM algorithm is an iterative approach that cycles between two modes. The first mode attempts to estimate the missing or latent variables, called the estimation-step or E-step. The second mode attempts to optimize the parameters of the model to best explain the data, called the maximization-step or M-step.
The EM algorithm can be applied quite widely, although is perhaps most well known in machine learning for use in unsupervised learning problems, such as density estimation and clustering. Perhaps the most discussed application of the EM algorithm is for clustering with a mixture model.
The Gaussian Mixture Model, or GMM for short, is a mixture model that uses a combination of Gaussian (Normal) probability distributions and requires the estimation of the mean and standard deviation parameters for each.
P.115) Listing 14.5: Example of fitting a Gaussian Mixture Model using the EM algorithm.
from numpy import hstack
from numpy.random import normal
from sklearn.mixture import GaussianMixture
X1 = normal(loc=20, scale=5, size=3000)		# generate a sample
X2 = normal(loc=40, scale=5, size=7000)
X = hstack((X1, X2))
X = X.reshape((len(X), 1))				# reshape into a table with one column
model = GaussianMixture(n_components=2, init_params= ' random ' )		# fit model
model.fit(X)
yhat = model.predict(X)				# predict latent values
print(yhat[:100])					# check latent value for first few points
print(yhat[-100:])					# check latent value for last few points 
P.119) Model selection: estimating the performance of different models in order to choose the best one.
There are many common approaches that may be used for model selection. For example, in the case of supervised learning, the three most common approaches are:
	 Train, Validation, and Test datasets.	 Resampling Methods.	 Probabilistic Statistics.
The simplest reliable method of model selection involves fitting candidate models on a training set, tuning them on the validation dataset, and selecting a model that performs the best on the test dataset according to a chosen metric, such as accuracy or error. A problem with this approach is that it requires a lot of data. Resampling techniques attempt to achieve the same as the train/val/test approach to model selection, although using a small dataset. An example is k-fold cross-validation where a training set is split into many train/test pairs and a model is fit and evaluated on each. This is repeated for each model and a model is selected with the best average score across the k-folds. A problem with this and the prior approach is that only model performance is assessed, regardless of model complexity.
A third approach to model selection attempts to combine the complexity of the model with the performance of the model into a score, then select the model that minimizes or maximizes the score. We can refer to this approach as statistical or probabilistic model selection as the scoring method uses a probabilistic framework.
P.131) Bayes Theorem provides a principled way for calculating a conditional probability. It is a deceptively simple calculation, providing a method that is easy to calculate for scenarios where our intuition often fails.
				P (A|B)  =  ( P (B|A) × P (A) ) / P (B)
	 P (A|B): Posterior probability.			 P (A): Prior probability.
	 P (B|A): Likelihood.					 P (B): Evidence.
This allows Bayes Theorem to be restated as: 	Posterior  = ( Likelihood × Prior ) /  Evidence
We can also think about the calculation in the terms of a binary classifier. For example, P (B|A) may be referred to as the True Positive Rate (TPR) or the sensitivity, P (B|not A) may be referred to as the False Positive Rate (FPR), the complement P (not B|not A) may be referred to as the True Negative Rate (TNR) or specificity, and the value we are calculating P (A|B) may be referred to as the Positive Predictive Value (PPV) or the precision.		 P (not B|not A): True Negative Rate or TNR (specificity).
	 P (B|not A): False Positive Rate or FPR.		 P (not B|A): False Negative Rate or FNR.
	 P (B|A): True Positive Rate or TPR (sensitivity or recall).
	 P (A|B): Positive Predictive Value or PPV (precision).
For example, we may re-state the calculation using these terms as follows:
			PPV  =  ( T P R × P (A) ) / ( T P R × P (A) + F P R × P (not A) )
P.134) Example: Elderly Fall and Death
Consider the case where an elderly person (over 80 years of age) falls, what is the probability that they will die from the fall? Let’s assume that the base rate of someone elderly dying P (A) is 10%, and the base rate for elderly people falling P (B) is 5%, and from all elderly people, 7% of those that die had a fall P (B|A). Let’s plug what we know into the theorem:
	P (A|B)  =  ( P (B|A) × P (A) ) / P (B)  ==>	P (Die|Fall) = (P (Fall|Die) × P (Die)) / P (Fall)
 	P (A|B)  =  ( 0.07 × 0.10 ) /  0.05         ==>	P (Die|Fall) = 0.14 
That is if an elderly person falls then there is a 14% probability that they will die from the fall.
P,144) Both Maximum Likelihood Estimation (MLE) and Maximum a Posterior (MAP) often converge to
the same optimization problem for many machine learning algorithms. One framework is not better than another, and as mentioned, in many cases, both frameworks frame the same optimization problem from different perspectives. Instead, MAP is appropriate for those problems where there is some prior information, e.g. where a meaningful prior can be set to weight the choice of different distributions and parameters or model parameters. MLE is more appropriate where there is no such prior.
Any system that classifies new instances according to [the equation] is called a Bayes optimal classifier, or Bayes optimal learner. No other classification method using the same hypothesis space and same prior knowledge can outperform this method on average.
Despite the fact that it is a very simple approach, KNN can often produce classifiers that are surprisingly close to the optimal Bayes classifier.
P.155) In practice, it is a good idea to use optimized implementations of the Naive Bayes algorithm. The scikit-learn library provides three implementations, one for each of the three main probability distributions; for example, BernoulliNB, MultinomialNB, and GaussianNB for binomial, multinomial and Gaussian distributed input variables respectively. To use a s scikit-learn Naive Bayes model, first the model is defined, then it is fit on the training dataset. Once fit, probabilities can be predicted via the predict proba() function and class labels can be predicted directly via the predict() function.
Listing 18.13: Example of the Naive Bayes using the scikit-learn library.
from sklearn.datasets import make_blobs		from sklearn.naive_bayes import GaussianNB
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
model = GaussianNB()				# define the model
model.fit(X, y)						# fit the model
Xsample, ysample = [X[0]], y[0]			# select a single sample
yhat_prob = model.predict_proba(Xsample)		# make a probabilistic prediction
print( ' Predicted Probabilities: ' , yhat_prob)
yhat_class = model.predict(Xsample)		# make a classification prediction
print( ' Predicted Class: ' , yhat_class)		print( ' Truth: y=%d ' % ysample)
P.156) 5 Tips When Using Naive Bayes
	1) Use a KDE for Complex Distributions
	If the probability distribution for a variable is complex or unknown, it can be a good idea to use a kernel density estimator or KDE to approximate the distribution directly from the data samples. A good example would be the Gaussian KDE.
	2) Decreased Performance With Increasing Variable Dependence
	By definition, Naive Bayes assumes the input variables are independent of each other. This works well most of the time, even when some or most of the variables are in fact dependent. Nevertheless, the performance of the algorithm degrades the more dependent the input variables happen to be.
	3) Avoid Numerical Underflow with Log
	The calculation of the independent conditional probability for one example for one class label involves multiplying many probabilities together, one for the class and one for each input variable. As such, the multiplication of many small numbers together can become numerically unstable, especially as the number of input variables increases. To overcome this problem, it is common to change the calculation from the product of probabilities to the sum of log probabilities.
	4) Update Probability Distributions
	As new data becomes available, it can be relatively straightforward to use this new data with the old data to update the estimates of the parameters for each variable’s probability distribution. This allows the model to easily make use of new data or the changing distributions of data over time.
	5) Use as a Generative Model
The probability distributions will summarize the conditional probability of each input variable value for each class label. These probability distributions can be useful more generally beyond use in a classification model. For example, the prepared probability distributions can be randomly sampled in order to create new plausible data instances. The conditional independence assumption may mean that the examples are more or less plausible based on how much actual interdependence exists between the input variables in the dataset.
P.159) Bayesian Optimization provides a principled technique based on Bayes Theorem to direct a search of a global optimization problem that is efficient and effective. It works by building a probabilistic model of the objective function, called the surrogate function, that is then searched efficiently with an acquisition function before candidate samples are chosen for evaluation on the real objective function. Bayesian Optimization is often used in applied machine learning to tune the hyperparameters of a given well-performing model on a validation dataset. Global function optimization, or function optimization for short, involves finding the minimum or maximum of an objective function. Samples are drawn from the domain and evaluated by the objective function to give a score or cost.
Summary of optimization in machine learning:
	 Algorithm Training: Optimization of model parameters.
	 Algorithm Tuning: Optimization of model hyperparameters.
	 Predictive Modeling: Optimization of data, data preparation, and algorithm selection.
P.175) Two popular libraries for Bayesian Optimization include Scikit-Optimize and HyperOpt. In machine learning, these libraries are often used to tune the hyperparameters of algorithms. Hyperparameter tuning is a good fit for Bayesian Optimization because the evaluation function is computationally expensive (e.g. training models for each set of hyperparameters) and noisy (e.g. noise in training data and stochastic learning algorithms).
P.177) Listing 19.27: Examples of Bayesian Optimization for model hyperparameters.
from numpy import mean						from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score		from skopt import gp_minimize
from sklearn.neighbors import KneighborsClassifier		from skopt.space import Integer
from skopt.utils import use_named_args				
X, y = make_blobs(n_samples=500, centers=3, n_features=2)	# generate 2d classification dataset
model = KneighborsClassifier()					# define the model
# define the space of hyperparameters to search
search_space = [Integer(1, 5, name= ' n_neighbors ' ), Integer(1, 2, name= ' p ' )]
@use_named_args(search_space)		# define the function used to evaluate a given configuration
def evaluate_model(**params):
model.set_params(**params)					# something
# calculate 5-fold cross validation
result = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring= ' accuracy ' )
estimate = mean(result)						# calculate the mean of the scores
return 1.0 - estimate
result = gp_minimize(evaluate_model, search_space)		# perform optimization
print( ' Best Accuracy: %.3f ' % (1.0 – result.fun))			# summarizing finding:
print( ' Best Parameters: n_neighbors=%d, p=%d ' % (result.x[0], result.x[1]))
P.181) Bayesian belief networks provide an intermediate approach that is less constraining than the global assumption of conditional independence made by the naive Bayes classifier, but more tractable than avoiding conditional independence assumptions altogether.
A probabilistic graphical model (PGM), or simply graphical model for short, is a way of representing a probabilistic model with a graph structure. The nodes in the graph represent random variables and the edges that connect the nodes represent the relationships between the random variables.
P.189) Information Theory is concerned with representing data in a compact fashion (a task known as data compression or source coding), as well as with transmitting and storing it in a way that is robust to errors (a task known as error correction or channel coding).
The basic intuition behind information theory is that learning that an unlikely event has occurred is more informative than learning that a likely event has occurred.
P.208) Cross-entropy is a measure of the difference between two probability distributions for a given random variable or set of events. Specifically, it builds upon the idea of entropy from information theory and calculates the average number of bits required to represent or transmit an event from one distribution compared to the other distribution.
P.241) The scikit-learn machine learning library provides an implementation of the majority class naive classification algorithm that you can use on your next classification predictive modeling project. It is provided as part of the DummyClassifier class. To use the naive classifier, the class must be defined and the strategy argument must be set and can take three values:
	 Random Guess: Set the strategy argument to ‘uniform’.
	 Select Random Class: Set the strategy argument to ‘stratified’.
	 Majority Class: Set the strategy argument to ‘most_frequent’.. 
P.241) Listing 25.14: Example of evaluating majority class classification strategy with scikit-learn.
from numpy import asarray
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
X = asarray([0 for _ in range(100)])				# define dataset
class0 = [0 for _ in range(25)]
class1 = [1 for _ in range(75)]
y = asarray(class0 + class1)
X = X.reshape((len(X), 1))						# reshape data for sklearn
model = DummyClassifier(strategy= ' most_frequent ' )		# define model
model.fit(X, y)								# fit model
yhat = model.predict(X)						# make predictions
accuracy = accuracy_score(y, yhat)					# calculate accuracy
print( ' Accuracy: %.3f ' % accuracy)
