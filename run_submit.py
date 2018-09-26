# test
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.cross_validation import train_test_split
import gc
from sklearn.preprocessing import OneHotEncoder

folder = '../input/'

train_data_ = pd.read_csv(folder +'/train_data_2gram_.csv')
test_data_ = pd.read_csv(folder + '/test_data_2gram_.csv')

train_features = [col for col in train_data_.columns if col in test_data_.columns and col!='file_id' and col!='label']
train_label = 'label'

def lgb_logloss(preds,data):
    labels_ = data.get_label()
    classes_ = np.unique(labels_) 
    preds_prob = []
    for i in range(len(classes_)):
        preds_prob.append(preds[i*len(labels_):(i+1) * len(labels_)])
    preds_prob_ = np.vstack(preds_prob) 
    
    loss = []
    for i in range(preds_prob_.shape[1]):  # 样本个数
        sum_ = 0
        for j in range(preds_prob_.shape[0]): #类别个数
            pred = preds_prob_[j,i] # 第i个样本预测为第j类的概率
            if  j == labels_[i]:
                sum_ += np.log(pred)
            else:
                sum_ += np.log(1 - pred)
        loss.append(sum_)       
    return 'loss is: ',-1 * (np.sum(loss) / preds_prob_.shape[1]),False

preds = []

# 10次4折
for i in range(10): 
    print(i)
    train_X, test_X, train_Y, test_Y = train_test_split(train_data_[train_features],train_data_[train_label].values, test_size = 0.25, random_state = i) 
    dtrain = lgb.Dataset(train_X,train_Y) 
    dval   = lgb.Dataset(test_X,test_Y, reference = dtrain)   
    params = {
            'task':'train', 
            'num_leaves': 255,
            'objective': 'multiclass',
            'num_class':6,
            'min_data_in_leaf': 50,
            'learning_rate': 0.05,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5, 
            'max_bin':128,
            'num_threads': 64,
            'random_state':100
        }  
    model = lgb.train(params, dtrain, num_boost_round=1000,valid_sets=[dtrain,dval],verbose_eval=10, early_stopping_rounds=30, feval=lgb_logloss)  
    preds.append(model.predict(test_data_[train_features]))

    
        
# 10次3折
for i in range(10): 
    print(i)
    train_X, test_X, train_Y, test_Y = train_test_split(train_data_[train_features],train_data_[train_label].values, test_size = 0.333, random_state = i) 
    dtrain = lgb.Dataset(train_X,train_Y) 
    dval   = lgb.Dataset(test_X,test_Y, reference = dtrain)   
    params = {
            'task':'train', 
            'num_leaves': 255,
            'objective': 'multiclass',
            'num_class':6,
            'min_data_in_leaf': 50,
            'learning_rate': 0.05,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.85,
            'bagging_freq': 5, 
            'max_bin':128,
            'num_threads': 64,
            'random_state':100
        }  
    model = lgb.train(params, dtrain, num_boost_round=1000,valid_sets=[dtrain,dval],verbose_eval=10, early_stopping_rounds=30, feval=lgb_logloss)  
    preds.append(model.predict(test_data_[train_features]))
    
test_prob = np.mean(preds,axis=0)
for i in range(6):
    test_data_['prob'+str(i)] =  test_prob[:,i] 
test_data_[['file_id','prob0','prob1','prob2','prob3','prob4','prob5']].to_csv('Gram_2.csv',index = None)

