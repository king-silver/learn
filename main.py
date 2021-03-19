import os
import gc
import math
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from sklearn.decomposition import PCA 
warnings.filterwarnings('ignore')


class solution(object):
    def reduce_mem_usage(self,df):
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

    def find_outliers_by_zero(self,data,fea):
        rate_zero = sum([1 for x in data[fea] if x ==0])/len(data[fea])
        if rate_zero<0.75:
            return True
        else:
            return False


    def abs_sum(self,y_pre,y_tru):
        y_pre=np.array(y_pre)
        y_tru=np.array(y_tru)
        loss=sum(sum(abs(y_pre-y_tru)))
        return loss

    def cv_model(self,clf, train_x, train_y, test_x, clf_name):
        folds = 5
        seed = 2021
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        test = np.zeros((test_x.shape[0],4))

        cv_scores = []
        onehot_encoder = OneHotEncoder(sparse=False)
        for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
            print('************************************ {} ************************************'.format(str(i+1)))
            trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]
        
            if clf_name == "lgb":
                train_matrix = clf.Dataset(trn_x, label=trn_y)
                valid_matrix = clf.Dataset(val_x, label=val_y)

                params = {
                    'boosting_type': 'gbdt',
                    'objective': 'multiclass',
                    #'max_depth': 10,
                    'num_class': 4,
                    'num_leaves': 2 ** 6,
                    'feature_fraction': 0.75,
                    'bagging_fraction': 0.75,
                    'bagging_freq': 4,
                    'learning_rate': 0.05,
                    'seed': seed,
                    'nthread': 28,
                    'n_jobs':24,
                    'verbose': -1,
                }

                model = clf.train(params, 
                          train_set=train_matrix, 
                          valid_sets=valid_matrix, 
                          num_boost_round=2000, 
                          verbose_eval=100, 
                          early_stopping_rounds=200)
                val_pred = model.predict(val_x, num_iteration=model.best_iteration) #验证集
                test_pred = model.predict(test_x, num_iteration=model.best_iteration)  #测试集
            
            
            if clf_name == "cat":
                train_matrix = clf.Dataset(trn_x, label=trn_y)
                valid_matrix = clf.Dataset(val_x, label=val_y)

                params = {'learning_rate': 0.05, 'depth': 5, 'l2_leaf_reg': 5, 'bootstrap_type': 'Bernoulli',
                          'od_type': 'Iter', 'od_wait': 50, 'random_seed': 11, 'allow_writing_files': False}

                model = clf.train(params, 
                          train_set=train_matrix, 
                          valid_sets=valid_matrix, 
                          num_boost_round=2000, 
                          verbose_eval=100, 
                          early_stopping_rounds=200)
                val_pred = model.predict(val_x, num_iteration=model.best_iteration)
                test_pred = model.predict(test_x, num_iteration=model.best_iteration)
            
            val_y=np.array(val_y).reshape(-1, 1)
            val_y = onehot_encoder.fit_transform(val_y)
            print('预测的概率矩阵为：')
            print(test_pred)
            test += test_pred
            score=self.abs_sum(val_y, val_pred)
            cv_scores.append(score)
            print(cv_scores)
        print("%s_scotrainre_list:" % clf_name, cv_scores)
        print("%s_score_mean:" % clf_name, np.mean(cv_scores))
        print("%s_score_std:" % clf_name, np.std(cv_scores))
        test=test/kf.n_splits

        return test

    def lgb_model(self,x_train, y_train, x_test):
        lgb_test = self.cv_model(lgb, x_train, y_train, x_test, "lgb")
        return lgb_test



if __name__ == '__main__':

    train = pd.read_csv("E:/competdata/train.csv")
    test = pd.read_csv("E:/competdata/testA.csv")
    train_list = []
    for items in train.values:
        train_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [items[2]])
    train = pd.DataFrame(np.array(train_list))
    train.columns = ['id'] + ['s_'+str(i) for i in range(len(train_list[0])-2)] + ['label']
    train = solution().reduce_mem_usage(train)

    test_list=[]
    for items in test.values:
        test_list.append([items[0]] + [float(i) for i in items[1].split(',')])

    test = pd.DataFrame(np.array(test_list))
    test.columns = ['id'] + ['s_'+str(i) for i in range(len(test_list[0])-1)]
    test = solution().reduce_mem_usage(test)

    x_train = train.drop(['id','label'], axis=1)
    #x_train.to_csv('E:/competdata/x_train.csv',index=False)
    y_train = train['label']
    x_test=test.drop(['id'], axis=1)

    numerical_fea = list(x_train.select_dtypes(exclude=['object']).columns)
    deltzero_fea = []
    for fea in numerical_fea:
        if solution().find_outliers_by_zero(x_train,fea):
            deltzero_fea.append(fea)

    x_train_new = x_train[deltzero_fea]
    x_test_new = x_test[deltzero_fea] 
    #pca = PCA(n_components=50, copy=True, whiten=False)
    #x_train_pca = pd.DataFrame(pca.fit_transform(x_train_new))
    #x_test_pca = pd.DataFrame(pca.fit_transform(x_test_new))

    lgb_test = solution().lgb_model(x_train_new, y_train, x_test_new)
    temp=pd.DataFrame(lgb_test)
    result=pd.read_csv('E:/competdata/sample_submit.csv')
    result['label_0']=temp[0]
    result['label_1']=temp[1]
    result['label_2']=temp[2]
    result['label_3']=temp[3]
    result.to_csv('E:/competdata/submit.csv',index=False)
