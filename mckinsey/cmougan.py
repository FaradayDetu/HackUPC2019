
# coding: utf-8

# Carlos Mougan
import pandas as pd
import numpy as np
import scipy
import sklearn
import category_encoders
from sklearn.base import BaseEstimator, TransformerMixin
from matplotlib import pyplot as plt
from sklearn.feature_selection import chi2 as chi2
from sklearn import ensemble
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import scipy

from sklearn.model_selection import train_test_split

def missing_data(data):
    '''
    Receives a dataframe and return the type of the column, the total NaNs of the column and the percentage they represent
    '''
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))

def resumetable(df):
    '''
    Provides a short summary of a given dataframe
    '''
    
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(scipy.stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary.sort_values('Entropy',ascending=False)

def contains_nan(df_col):
    '''
    This functions checks if a certain column has nans
    '''
    return df_col.isna().any()

def create_cols_for_cols_with_nans(df, inplace=False):
    '''
    This function applied to a dataframe returns a list with the columns with NaNs
    and also returns a data frame with a column with 1 if value is NaN else 0 for all the columns with nans.
    '''
    if inplace:
        cols_with_nan = []
        for c in df.columns:
            if contains_nan(df[c]):
                cols_with_nan.append(c)
                df[c + "_nan"] = df[c].isna().values        
        return cols_with_nan
    else:
        df_copy = df.copy(deep=True)
        cols_with_nan = []
        for c in df.columns:
            if contains_nan(df[c]):
                cols_with_nan.append(c)
                df_copy[c + "_nan"] = df[c].isna().values
        return cols_with_nan, df_copy


def create_statististical_columns_for_nans(df,do_mean=True,do_median=True,do_mode=True,
                                           do_skew=True,do_kurtosis=True,do_std=True):
    '''
    This function applied to a dataframe returns  a data frame with a column with different statistical values
    if value is NaN else 0 for all the columns with nans
    '''
    
    df_copy = df.copy(deep=True)
    cols_with_nan = []
    for c in df.columns:
        if contains_nan(df[c]):
            if do_mean and df[c].dtype !='object':
                media = df[c].mean()
                df_copy[c+'_nan_mean'] = df[c].apply(lambda x: media if np.isnan(x) else 0)
            if do_median and df[c].dtype !='object':
                mediana = df[c].median()
                df_copy[c+'_nan_median'] = df[c].apply(lambda x: mediana if np.isnan(x) else 0)
            if do_mode:
                moda = df[c].mode()
                print(c)
                print(moda)
                #import pdb;pdb.set_trace()
                #df_copy[c + '_nan_mode'] = df[c].apply(lambda x: moda[0] if np.isnan(x) else 0)
            
            if do_std and df[c].dtype !='object':
                deviation = df[c].std()
                df_copy[c+'_nan_std'] = df[c].apply(lambda x: deviation if np.isnan(x) else 0)
            
            
            if do_skew and df[c].dtype !='object':
                skew = df[c].skew()
                df_copy[c+'_nan_skew'] = df[c].apply(lambda x: skew if np.isnan(x) else 0)
                
            if do_kurtosis and df[c].dtype !='object':
                kurtosis = scipy.stats.kurtosis(df[c].dropna())
                df_copy[c+'_nan_kurtosis'] = df[c].apply(lambda x: kurtosis if np.isnan(x) else 0)
                
                
    return df_copy
class TypeSelector(BaseEstimator, TransformerMixin):
    '''
    Transformer that filters a type of columns of a given data frame.
    '''
    def __init__(self, dtype):
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        #print("Type Selector out shape {}".format(X.select_dtypes(include=[self.dtype]).shape))
        #print(X.select_dtypes(include=[self.dtype]).dtypes)
        return X.select_dtypes(include=[self.dtype])


class Encodings(BaseEstimator, TransformerMixin):
    '''
    This class implements fit and transform methods that allows to encode categorical features in different ways.
    
    '''
    
    def __init__(self, encoding_type="TargetEncoder",columns="All",return_categorical=True):
        #cols: list -> a list of columns to encode, if All, all string columns will be encoded.
        
        self._allowed_encodings = ["TargetEncoder","WOEEncoder","CatBoostEncoder","OneHotEncoder"]           
        assert encoding_type in self._allowed_encodings, "the encoding type introduced {} is not valid. Please use one in {}".format(encoding_type, self._allowed_encodings)
        self.encoding_type = encoding_type
        
        self.columns = columns
        self.return_categorical = return_categorical
        
        
    def fit(self,X,y):
        """
        This method learns encodings for categorical variables/values.
        """
        
        #import pdb;pdb.set_trace()
        
        # Obtain a list of categorical variables
        if self.columns == "All":
            self.categorical_cols = X.columns[X.dtypes==object].tolist() +  X.columns[X.dtypes=="category"].tolist()
        else:
            self.categorical_cols = self.columns
        
    
        # Split the data into categorical and numerical
        self.data_encode = X[self.categorical_cols]

        
        # Select the type of encoder
        if self.encoding_type == "TargetEncoder":
            self.enc = category_encoders.target_encoder.TargetEncoder()
            
        if self.encoding_type == "WOEEncoder":
            self.enc = category_encoders.woe.WOEEncoder()
            
        if self.encoding_type == "CatBoostEncoder":
            #This is very similar to leave-one-out encoding, 
            #but calculates the values “on-the-fly”.
            #Consequently, the values naturally vary during the training phase and it is not necessary to add random noise.
            # Needs to be randomly permuted
            # Random permutation
            perm = np.random.permutation(len(X))
            self.data_encode = self.data_encode.iloc[perm].reset_index(drop=True)
            y = y.iloc[perm].reset_index(drop=True)
            self.enc = category_encoders.cat_boost.CatBoostEncoder()
            
        if self.encoding_type == "OneHotEncoder":
            self.enc = category_encoders.one_hot.OneHotEncoder()
            
            # Check if all columns have certain number of elements bf OHE
            self.new_list=[]
            for col in self.data_encode.columns:
                if len(self.data_encode[col].unique())<50:
                    self.new_list.append(col)
                    
            self.data_encode = self.data_encode[self.new_list]
        
        # Fit the encoder
        self.enc.fit(self.data_encode,y)
        return self

    def transform(self, X):
        
        
        if self.columns == "All":
            self.categorical_cols = X.columns[X.dtypes==object].tolist() +  X.columns[X.dtypes=="category"].tolist()
        else:
            self.categorical_cols = self.columns
        
       
    
        # Split the data into categorical and numerical
        
        self.data_encode = X[self.categorical_cols]
        
        # Transform the data
        self.transformed = self.enc.transform(self.data_encode)
        
        # Modify the names of the columns with the proper suffix
        self.new_names = []
        for c in self.transformed.columns:
            self.new_names.append(c+'_'+self.encoding_type)
        self.transformed.columns = self.new_names
         
        if self.return_categorical:
            #print('The encoding {} has made {} columns, the input was {} and the output shape{}'.
             #     format(self.encoding_type,self.transformed.shape, X.shape,self.transformed.join(X).shape))
            #print(self.transformed.join(X).dtypes)

            return self.transformed.join(X)
        else:
            return self.transformed.join(X)._get_numeric_data()

class NaNtreatment(BaseEstimator, TransformerMixin):
    '''
    This class implements a fit and transform methods that enables to implace NaNs in different ways.
    '''
    def __init__(self, treatment="mean"):
        self._allowed_treatments = ["fixed_value", "mean",'median','mode','None']     
        assert treatment in self._allowed_treatments or isinstance(treatment,(int,float)),  "the treatment introduced {} is not valid. Please use one in {}".format(treatment, self._allowed_treatments)
        self.treatment = treatment
    
    def fit(self, X, y):
        """
        Learns statistics to impute nans.
        """
        
        if self.treatment == "mean" or self.treatment==None:
            self.treatment_method = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
        elif self.treatment == "median":
            self.treatment_method = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='median')
        elif self.treatment == "most_frequent":
            self.treatment_method = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        elif isinstance(self.treatment, (int,float)):
            self.treatment_method = sklearn.impute.SimpleImputer(missing_values=np.nan,
                                                                 strategy="constant",fill_value=self.treatment)       
        

        self.treatment_method.fit(X.values)
        return self

    def transform(self, X):
        if self.treatment==None:
            return X
        return self.treatment_method.transform(X)


def reduce_mem_usage(df):
    """ 
    iterate through all the columns of a dataframe and modify the data type to reduce memory usage.  
    From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    WARNING! THIS CAN DAMAGE THE DATA 
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
    
def auc_score(y_true, preds):
    return sklearn.metrics.roc_auc_score(y_true, preds[:,1])


def fit_cv_subsample (pipe_cv, X, y, n_max = 10_000):
    '''
    This function fits a CV in a subsample of the first n_max rows
    returns the trained pipe and the best estimator
    '''
    X_sub = X[0:n_max]
    y_sub = y[0:n_max]
    pipe_cv.fit(X_sub,y_sub)
    #pipe_cv.best_estimator_.fit(X,y)
    return pipe_cv, pipe_cv.best_estimator_

##############################################
            # Feature Engineering #
##############################################
def plot_boxplot(data,feature):
    '''
    Plots a boxplot of the feature given in the dataframe
    '''
    fig, ax = plt.subplots(figsize=(12,4))
    sns.boxplot(x = feature, data =data, orient = 'h', width = 0.8, 
                     fliersize = 3, showmeans=True, ax = ax)
    plt.show()
def winsorize(data,feature,limits=0.01,inplace=True):
    if inplace:   
        data[feature]=scipy.stats.mstats.winsorize(data[feature], limits=limits)
    else:
        var = feature + '_w'
        data[var] = scipy.stats.mstats.winsorize(data[feature], limits=limits)
    return data

def group_statistics(data,group_col,aggregated_col,statistic):
    '''
    This method receives a dataset, a column to group by, an a column where we want to apply certain statistics.
    It returns the original dataframe with one more column with the aggregated value
    
    '''
    renamed_col = group_col+'_'+aggregated_col+'_'+statistic
    while renamed_col in data.columns:
        renamed_col = renamed_col+'_dup'

    temp = data.groupby(group_col)[aggregated_col].agg([statistic]).rename({statistic:renamed_col},axis=1)
    data = pd.merge(data,temp,on=group_col,how='left')
    return data
def frecuency_encodings(data,col):
    '''
    Receives a column to apply a frecuency encoding, it returns the dataframe with the FE
    
    Frequency encoding is a powerful technique that allows LGBM to see whether column values are rare or common. 
    For example, if you want LGBM to "see" which credit cards are used infrequently
    '''
    temp = data[col].value_counts().to_dict()
    var_name = col + '_counts'
    data[var_name] = data[col].map(temp)
    return data
def one_hot_simple(df,columns_encoding,erase=False):
    '''
    Extremely basic method and non optimized method for One Hot Encoding, 
    receives a dataframe and a list of columns to encode.
    Returns the dataframe with or without the original columns (erase)
    
    '''
    for col in columns_encoding:
        variables = df[col].unique()
        for v in variables:
            df[col + str(v)] = [1 if row == v else 0 for row in df[col].values]
    if erase:
        return df.drop(columns=columns_encoding)
    return df
##############################################
            # Feature Importance #
##############################################

def plot_feature_importance(columnas,model_features,columns_ploted=10,model_name='Catboost'):
    '''
    This method is yet non-tested
    
    This function receives a set of columns feeded to a model, and the importance of each of feature.
    Returns a graphical visualization
    
    Call it fot catboost pipe example:
    plot_feature_importance(pipe_best_estimator[:-1].transform(X_tr).columns,pipe_best_estimator.named_steps['cb'].get_feature_importance(),20)
    
    Call it for lasso pipe example:
    plot_feature_importance(pipe_best_estimator[:-1].transform(X_tr).columns,np.array(pipe_best_estimator.named_steps['clf'].coef_.squeeze()),20)

    '''

    feature_importance = pd.Series(index = columnas, data = np.abs(model_features))
    n_selected_features = (feature_importance>0).sum()
    print('{0:d} features, reduction of {1:2.2f}%'.format(n_selected_features,(1-n_selected_features/len(feature_importance))*100))
    plt.figure()
    feature_importance.sort_values().tail(columns_ploted).plot(kind = 'bar', figsize = (18,6))
    plt.title('Feature Importance for {}'.format(model_name))
    plt.show()


#############################################
            # Feature Select #
##############################################


class ColSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self,percent=1., feature_selector_type="f_classif"):
        
        self._allowed_featureselectors = ["chi2", "f_classif"]       
       
        assert percent<=1,            "the percent introduced {} is not valid. Please a number in 0<percent<=1"
        
        self.percent = percent
        
        assert feature_selector_type in self._allowed_featureselectors,            "the featureselector introduced {} is not valid. Please use one in {}".format(featureselector, self._allowed_featureselectors)
        self.feature_selector_type = feature_selector_type
    
    def fit(self,X,y):
        n_cols = X.shape[1]
        
        self.n_features_selected = int(self.percent * n_cols)
        
        #import pdb;pdb.set_trace()
        
        if self.feature_selector_type == "chi2":
            self.featureselector = sklearn.feature_selection.SelectKBest(chi2,k= self.n_features_selected)
                
        if self.feature_selector_type == "f_classif":
            self.featureselector = sklearn.feature_selection.SelectKBest(f_classif,k= self.n_features_selected)
    
        self.featureselector.fit(X,y)
        return self

    def transform(self, X):
        return self.featureselector.transform(X)

        
        

#############################################
            # XGB Feature Generator #
##############################################

class GradientBoostingFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Feature generator from a gradient boosting
    """
    
    def __init__(self,
                 stack_to_X=True,
                 sparse_feat=True,
                 add_probs=True,
                 criterion='friedman_mse',
                 init=None,
                 learning_rate=0.1,
                 loss='deviance',
                 max_depth=3,
                 max_features=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 min_samples_leaf=1,
                 min_samples_split=2,
                 min_weight_fraction_leaf=0.0,
                 n_estimators=50,
                 n_iter_no_change=None,
                 presort='auto',
                 random_state=None,
                 subsample=1.0,
                 tol=0.0001,
                 validation_fraction=0.1,
                 verbose=0,
                 warm_start=False):
        
        # Deciding wheather to append features or simply return generated features
        self.stack_to_X  = stack_to_X
        self.sparse_feat = sparse_feat  
        self.add_probs   = add_probs   

        # GBM hyperparameters
        self.criterion = criterion
        self.init = init
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.n_estimators = n_estimators
        self.n_iter_no_change = n_iter_no_change
        self.presort = presort
        self.random_state = random_state
        self.subsample = subsample
        self.tol = tol
        self.validation_fraction = validation_fraction
        self.verbose = verbose
        self.warm_start = warm_start
        
        
    def _get_leaves(self, X):
        X_leaves = self.gbm.apply(X)
        n_rows, n_cols, _ = X_leaves.shape
        X_leaves = X_leaves.reshape(n_rows, n_cols)
        
        return X_leaves
    
    def _decode_leaves(self, X):

        if self.sparse_feat:
            #float_eltype = np.float32
            #return scipy.sparse.csr.csr_matrix(self.encoder.transform(X), dtype=float_eltype)
            return scipy.sparse.csr.csr_matrix(self.encoder.transform(X))
        else:
            return self.encoder.transform(X).todense()
        
    
    def fit(self, X, y):
        
        self.gbm = sklearn.ensemble.gradient_boosting.GradientBoostingClassifier(criterion = self.criterion,
                            init = self.init,
                            learning_rate = self.learning_rate,
                            loss = self.loss,
                            max_depth = self.max_depth,
                            max_features = self.max_features,
                            max_leaf_nodes = self.max_leaf_nodes,
                            min_impurity_decrease = self.min_impurity_decrease,
                            min_impurity_split = self.min_impurity_split,
                            min_samples_leaf = self.min_samples_leaf,
                            min_samples_split = self.min_samples_split,
                            min_weight_fraction_leaf = self.min_weight_fraction_leaf,
                            n_estimators = self.n_estimators,
                            n_iter_no_change = self.n_iter_no_change,
                            presort = self.presort,
                            random_state = self.random_state,
                            subsample = self.subsample,
                            tol = self.tol,
                            validation_fraction = self.validation_fraction,
                            verbose = self.verbose,
                            warm_start = self.warm_start)
        
        self.gbm.fit(X,y)
        self.encoder = sklearn.preprocessing.OneHotEncoder(categories='auto')
        X_leaves = self._get_leaves(X)
        self.encoder.fit(X_leaves)
        return self
        
    def transform(self, X):
        """
        Generates leaves features using the fitted self.gbm and saves them in R.

        If 'self.stack_to_X==True' then '.transform' returns the original features with 'R' appended as columns.
        If 'self.stack_to_X==False' then  '.transform' returns only the leaves features from 'R'
        Ìf 'self.sparse_feat==True' then the input matrix from 'X' is cast as a sparse matrix as well as the 'R' matrix.
        """
        R = self._decode_leaves(self._get_leaves(X))
        
        if self.sparse_feat:
            if self.add_probs:
                P = self.gbm.predict_proba(X)
                X_new =  scipy.sparse.hstack((scipy.sparse.csr.csr_matrix(X), R, scipy.sparse.csr.csr_matrix(P))) if self.stack_to_X==True else R
            else:
                X_new =  scipy.sparse.hstack((scipy.sparse.csr.csr_matrix(X), R)) if self.stack_to_X==True else R

        else:

            if self.add_probs:
                P = self.gbm.predict_proba(X)
                X_new =  scipy.sparse.hstack((scipy.sparse.csr.csr_matrix(X), R, scipy.sparse.csr.csr_matrix(P))) if self.stack_to_X==True else R
            else:
                X_new =  np.hstack((X, R)) if self.stack_to_X==True else R

        return X_new


