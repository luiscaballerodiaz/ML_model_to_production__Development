import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import os


# MAIN
pd.set_option('display.max_columns', None)  # Enable option to display all dataframe columns
pd.set_option('display.max_rows', None)  # Enable option to display all dataframe rows
pd.set_option('display.max_colwidth', None)  # Enable printing the whole column content
pd.set_option('display.max_seq_items', None)  # Enable printing the whole sequence content

df = pd.read_csv('cardio_train.csv')
print('Original shape {}\n'.format(df.shape))
print('NA assessment: \n{}\n'.format(df.isna().sum()))
print('Dataframe information: \n{}\n'.format(df.info()))
print('Feature description: \n{}\n'.format(df.describe()))

# Conclusions
# - No need to impute
# - Id feature is not relevant to prediction --> drop it
# - Cholesterol and gluc are categorical features --> apply onehot
# - Smoke, alco, gender and active are binary features --> apply onehot with drop if binary
# - Age, height and weight are numerical features --> apply standardization
# - Ap_hi and ap_lo are numerical features with outliers --> collapse outliers and apply standardization
# - Cardio is the categorical target --> no change

# Remove id feature
df.drop('id', inplace=True, axis=1)
# Split X features and y target
X = df.drop('cardio', axis=1)
y = df['cardio']
# Convert features in the proper type
fcat = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
fnum = [x for x in X.columns.values.tolist() if x not in fcat]
X = X.astype('string')
X[fcat] = X[fcat].astype('category')
X[fnum] = X[fnum].astype('float64')
# Collapse outliers for ap_hi and ap_lo at 1.5 x IQ range
feats = ['ap_hi', 'ap_lo']
for feat in feats:
    max_limit = X[feat].quantile(0.75) + (X[feat].quantile(0.75) - X[feat].quantile(0.25)) * 1.5
    min_limit = X[feat].quantile(0.25) - (X[feat].quantile(0.75) - X[feat].quantile(0.25)) * 1.5
    X[feat] = X[feat].apply(lambda row: min(max_limit, max(row, min_limit)))
# Split between training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Define transformers per each feature
preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), fnum),
                                               ('cat', OneHotEncoder(drop='if_binary'), fcat)],
                                 remainder='passthrough')
pipeline = Pipeline([('preprocessor', preprocessor),
                     ('classifier', LogisticRegression(random_state=0))])
pipeline.fit(X_train, y_train)

print('EXAMPLE 1:\nTest example inputs: \n{}'.format(X_test.iloc[[0]]))
print('Test example true target: {}'.format(y_test.iloc[0]))
print('Test example prediction: {}\n'.format(pipeline.predict_proba(X_test.iloc[[0]])[0]))

print('EXAMPLE 2:\nTest example inputs: \n{}'.format(X_test.iloc[[1800]]))
print('Test example true target: {}'.format(y_test.iloc[1800]))
print('Test example prediction: {}\n'.format(pipeline.predict_proba(X_test.iloc[[1800]])[0]))

print('EXAMPLE 3:\nTest example inputs: \n{}'.format(X_test.iloc[[6000]]))
print('Test example true target: {}'.format(y_test.iloc[6000]))
print('Test example prediction: {}\n'.format(pipeline.predict_proba(X_test.iloc[[6000]])[0]))

print('TRAIN SCORE: {}'.format(pipeline.score(X_train, y_train)))
print('TEST SCORE: {}\n'.format(pipeline.score(X_test, y_test)))

# Save pipeline in the parent folder to use it in production
mycwd = os.getcwd()
os.chdir('..')
pickle.dump(pipeline, open(os.getcwd() + '\\model.pkl', 'wb'))
os.chdir(mycwd)
