import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
# np.random.seed(42)
DATA_PATH = "ds.dataset_raw.csv"
labels = ['user_id', 'total_stream', 'normal_clips', 'downloaded_clips', 'shared_clips', 'edited_clips', 'avg_waiting_time', 'days_active', 'total_spent_minutes', 'join_via', 'days_in_premium', 'churn_status']
def load_data_from_csv(path = DATA_PATH):
    return pd.read_csv(path)

raw_data = load_data_from_csv()
split = StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=5)
for train_index, val_index in split.split(raw_data, raw_data["churn_status"]):
    strat_train_set = pd.DataFrame(raw_data.loc[train_index])
    strat_test_set = pd.DataFrame(raw_data.loc[val_index])

categorical_columns = ['join_via']
numerical_columns =['user_id', 'total_stream', 'normal_clips', 'downloaded_clips', 'shared_clips', 'edited_clips', 'avg_waiting_time', 'days_active', 'total_spent_minutes', 'days_in_premium']

columns_to_drop = ['user_id','shared_clips','downloaded_clips','edited_clips',"normal_clips",'total_spent_minutes','days_active','total_stream'] #we wont use user id because it has no value in predict the churn status
# columns_to_drop = ['user_id'] #we wont use user id because it has no value in predict the churn status
for dropcol in columns_to_drop:
    numerical_columns.remove(dropcol) 
    
    
bin_columns = ['shared_clips','normal_clips','downloaded_clips','edited_clips']
clip_columns = ['shared_clips','normal_clips','downloaded_clips','edited_clips']
avg_columns = ['total_spent_minutes','total_stream','days_active']
class CombinedAttributesAdder(BaseEstimator,TransformerMixin): #this transformer is just a template
    def __init__(self,add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        return self

class NumerizeCategoryPipe(BaseEstimator, TransformerMixin):#this transformer is used to change categoty to binary cols (if is target, =1, else = 0)
    def __init__(self, target_column='join_via', target_value='Desktop'):
        self.target_column = target_column
        self.target_value = target_value
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()  # Avoid modifying the original data
        X_copy[self.target_column+'_'+self.target_value] = (X_copy[self.target_column] == self.target_value).astype(int)
        return X_copy[[self.target_column+'_'+self.target_value]]
    def get_feature_names_out(self,input_features=None):
        return [self.target_column+'_'+self.target_value]
        
class IsActivePipe(BaseEstimator, TransformerMixin): #this transfomer is used to check is a value in cols > a threshold (default 0) or not 
    def __init__(self, target_column = None, threshold=0):
        self.target_column = target_column
        self.threshold = threshold
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        self.feature_names = []
        X_copy = X.copy()  # Avoid modifying the original data
        if self.target_column is not None:
            X_copy[self.target_column+"_bin"] = (X_copy[self.target_column] > self.threshold).astype(int)
            return X_copy[self.target_column+"_bin"]
        else:
            for col in X_copy.select_dtypes(include='number'):  
                X_copy[col+"_bin"] = (X_copy[col] > self.threshold).astype(int)
                self.feature_names += [col + "_bin"]
        return X_copy[self.feature_names]
    def get_feature_names_out(self,input_features=None):
        if self.target_column is not None: return [self.target_column+"_bin"] 
        else: return self.feature_names
    
class SumarizeClipPipe(BaseEstimator,TransformerMixin): #this transfomer is used to calculate all the clips cols to one
    def __init__(self,added_clips = None):
        if added_clips is None: self.added_clips = ['shared_clips','normal_clips','downloaded_clips','edited_clips']
        else : self.added_clips = added_clips
        
    def fit(self, X, y=None):
        return self
    def transform(self, X:pd.DataFrame, y=None):
        X['clips'] = X[self.added_clips].sum(axis=1)
        return X[['clips']]
    def get_feature_names_out(self,input_features=None):
        return ["clips"]

class PerDayPipe(BaseEstimator, TransformerMixin): #this transfomer is used to calculate average value per days active
    def __init__(self, perday_cols=['total_spent_minutes', 'total_stream']):
        self.perday_cols = perday_cols
        self.cols_name = [col + '_avg' for col in perday_cols]  # List comprehension

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        for col, new_col in zip(self.perday_cols, self.cols_name):
            X[new_col] = X[col] / X["days_active"]  # Calculate the average directly
            X[new_col] = X[new_col].replace([np.inf, -np.inf], np.nan) 
            X[new_col].fillna(0, inplace=True)  # Handle potential division by zero

        return X[self.cols_name]

    def get_feature_names_out(self, input_features=None):
        return self.cols_name

num_pipeline = Pipeline([
    ("std_scaler",StandardScaler())
])
clip_pipeline = Pipeline([
    ('summarize_clips', SumarizeClipPipe()),
    ('standardize', StandardScaler())  # Add a StandardScaler step
])
average_pipeline = Pipeline([
    ('perday',PerDayPipe()),
    ('standardize',StandardScaler()),
])
full_pipeline = ColumnTransformer([
    ('avg',average_pipeline, avg_columns),
    ('clip',clip_pipeline,clip_columns),
    ('num', num_pipeline, numerical_columns),
    # ('bin', IsActivePipe(), bin_columns),
    # ('cat', NumerizeCategoryPipe('join_via','Desktop'), ['join_via']),
])
borderline = BorderlineSMOTE(sampling_strategy = 1,random_state=5)
test_data = strat_test_set.drop("churn_status",axis=1)
test_data_labels = strat_test_set["churn_status"].copy()
train_data = strat_train_set.drop("churn_status",axis=1)
train_data_labels = strat_train_set["churn_status"].copy()
rfc = RandomForestClassifier(bootstrap=True, max_depth=15, min_samples_leaf=2, min_samples_split=10, n_estimators= 1)
X = full_pipeline.fit_transform(train_data,train_data_labels)
y = train_data_labels
X1 = full_pipeline.fit_transform(test_data)
y1 = test_data_labels
X, y = borderline.fit_resample(X,y)
rfc.fit(X,y)
y_pred = rfc.predict(X1)
y_scores = rfc.predict_proba(X1)[:, 1]
auc_score = roc_auc_score(y1, y_scores)        
f1_scr = f1_score(y1, y_pred) 
print(f"{auc_score = }")
print(f"{f1_scr = }")