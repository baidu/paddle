#!/public/home/wcx_wangyang/bin/python2.7.5/bin/python
#_*_ coding:utf-8 _*_
import pandas as pd
import sys
import numpy as np
import datetime
import lightgbm as lgb
from lightgbm import LGBMRegressor
import random 
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

from sklearn import linear_model
from numpy import array
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV 

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor 
import xgboost as xgb
from xgboost import XGBRegressor 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer

def load_basic_data(filename):
	df = pd.read_csv(filename)
	pd.set_option('display.max_columns', 100)
	print "==original data information=="
	print df.info()
	return df

def normalization_data(df,columns_list):
	for cl in columns_list:
		df[cl] = (df[cl].max() - df[cl]) / (df[cl].max() - df[cl].min())
	return

def load_train_test_to_csv(filename):
	df = load_basic_data(sys.argv[1])
	print "data information: ", df.shape
	#onehot encode
	print "start to onehot some columns..."
	df = pd.get_dummies(df,columns=['district_id']) 
	df = pd.get_dummies(df,columns=['area_id']) 
	df = pd.get_dummies(df,columns=['poi_id']) 
	df = pd.get_dummies(df,columns=['is_hot_hour'])
	#df = pd.get_dummies(df,columns=['day'])
	df = pd.get_dummies(df,columns=['quarter'])
	#df = pd.get_dummies(df,columns=['hour'])
	df = pd.get_dummies(df,columns=['customer_district_id'])
	df = pd.get_dummies(df,columns=['weekday'])

	print('constructing training data...')
	cols = df.columns.tolist()
	df_train = df[(df['date'] < 20170731) &(df['is_hot_hour_0'] == 0)]
	df_train = df_train[(df_train['delivery_duration']<4654) & (df_train['delivery_duration']>663)];
	df_test= df[(df['date'] >= 20170731)&(df['is_hot_hour_0'] == 0)]

	#train_to_drop = ['customer_district_id','poi_id','weekday','minute','order_unix_time','order_id','time_id','date','day','distance_cus_poi_to_delivery_dis','poi_lng','poi_lat','customer_longitude','customer_latitude']
	train_to_drop = ['hour_avg_duration','poi_avg_duration','wind','hour','day','is_hot_hour_0','is_hot_hour_2','order_unix_time','order_id','time_id','date','temperature']
	select_features = list(np.setdiff1d(cols,train_to_drop))
	#print "features of train data: "
	#print select_features
	#print"the remain features for train: ", select_features
	df_train = df_train[select_features]
	#print "start to save train data to final_train_data.csv..."
	#df_train.to_csv('final_train_data.csv',index=False)

	test_to_drop = ['hour_avg_duration','poi_avg_duration','wind','hour','day','is_hot_hour_0','is_hot_hour_2','order_unix_time','time_id','date','temperature']
	select_features = list(np.setdiff1d(cols,test_to_drop))
	#print "features of test data: "
	#print select_features
	#print"the remain features for train: ", select_features
	df_test = df_test[select_features]
	#df_test.to_csv('final_test_data.csv',index=False)
	#print "train and test data have been stored to file."
	return df_train,df_test

def model_SGD(x_train,y_train):

	kfold = KFold(n_splits=10,shuffle=True,random_state = 0)
	print "start to training SGD..."
	clf = linear_model.SGDRegressor(learning_rate='optimal',eta0=0.001,verbose=10,max_iter=20000)
	cross_val_score(clf, x_train, y_train, cv=kfold,scoring='mean_absolute_error')

	print "feature weights: "
	print clf.coef_
	return clf

def model_dp(x_train,y_train):
	model = Sequential()
	
	model.add(Dense(units=x_train.shape[0],input_dim = x_train.shape[1],activation = "relu"))		
	model.add(Dense(units=10,activation = "relu"))		
	model.add(Dense(units=1,activation = "linear"))		
	sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error',optimizer=sgd,metrics=['mae'])
	print "training deep network....."
	model.fit(x_train,y_train,epochs=100,verbose=5)
	
	return model

def model_lightGBM(x_train,y_train,x_val,y_val):
	lgb_train = lgb.Dataset(x_train.values, y_train.values)
	lgb_eval = lgb.Dataset(x_val.values, y_val.values,reference=lgb_train)
	params = {
	'boosting_type': 'gbdt',
	'objective': 'regression_l1',
	'metric': 'mae',
	'num_leaves': 35,
	'learning_rate': 0.001,
	'feature_fraction': 0.85,
	'bagging_fraction': 0.85,
	'bagging_freq': 5,
	'seed': 20171023,
	'verbose': 0
	}
	print 'Start training...'
	early_stopping_callback = lgb.callback.early_stopping(200)
	gbm = lgb.train(params,lgb_train,num_boost_round=20000,valid_sets=lgb_eval,callbacks=[early_stopping_callback],verbose_eval=10)
	print "training finish!!"

	return gbm

def model_xgb_gridsearch(x_train,y_train):
	#params
	model = XGBRegressor(n_estimators = 109,booster='gbtree',objective='reg:linear',learning_rate=0.13,subsample=0.8,colsample_bytree=0.7,max_depth=9,n_jobs=5)
	kfold = KFold(n_splits=10,shuffle=True,random_state = 0)
	params_grid={
			'learning_rate':[i/10.0 for i in range(1,10,2)]
			}
	#create model
	grid_search = GridSearchCV(model,params_grid,scoring='neg_mean_absolute_error',cv=kfold,verbose=5)
	print "start to grid search on xgb...."
	grid_result = grid_search.fit(x_train,y_train)
	print "best socre:"
	print grid_result.best_score_
	print "best estimator:"
	print grid_result.best_estimator_
	return grid_result

def model_sklaern_gbm(x_train,y_train):
	lgb_train = lgb.Dataset(x_train.values, y_train.values)
	best_gbm = LGBMRegressor(objective='regression_l1',num_leaves=20,learning_rate = 0.0015)
	params = {
	'boosting_type': 'gbdt',
	'objective': 'regression_l1',
	'metric': 'mae',
	'num_leaves': 31,
	'learning_rate': 0.0015,
	'feature_fraction': 0.75,
	'bagging_fraction': 0.8,
	'bagging_freq': 5,
	'seed': 20171023,
	}
	print "gbm cv start...."
	cv_output =lgb.cv(params, lgb_train, num_boost_round=20000, early_stopping_rounds=100,verbose_eval=5, show_stdv=True,metrics='mae',nfold=10)
		
	print 'cv output: ',cv_output
	key[0] = cv_output.keys()
	print "best number estimators: %d" % len(cv_output[key[0]])
	print "lightGBM train ok!!"
	return best_gbm

def model_xgb(x_train,y_train,x_val,y_val):
	dtrain = xgb.DMatrix(x_train,y_train)
	dval= xgb.DMatrix(x_val,y_val)
	print "training xgb model...." 
	watchlist = [(dval,'val')]
	param = {
			'booster' : 'gbtree',
			'objective' : 'reg:linear',
			'eval_metric' : 'mae',
			'eta' : 0.1,
			'num_round': 20000,
			'lambda' : 0.1,
			'colsample': 0.8,
			'subsample': 0.8,
			'max_depth': 9,
			'nthread' : -1,
			'seed' : 20171001,
			'silent' : 1,	
			}
	#best_xgb = XGBRegressor(n_estimators = 109,booster='gbtree',objective='reg:linear',learning_rate=0.13,subsample=0.8,colsample_bytree=0.7,max_depth=9,n_jobs=5)
	#kfold = KFold(n_splits=10,shuffle=True,random_state = 0)
	#cv_output =xgb.cv(best_xgb.get_xgb_params(), dtrain, num_boost_round=20000, early_stopping_rounds=100,verbose_eval=5, show_stdv=True,metrics='mae',folds=kfold)
	bst = xgb.train(param,dtrain,param['num_round'],watchlist,verbose_eval = 5,early_stopping_rounds=100)
	#best_xgb.fit(x_train,y_train)
	#best_xgb.set_params(n_estimators=cv_output.shape[0])
	#print 'cv output: ',cv_output
	#print "best number estimators: %d" % cv_output.shape[0]
	print "xgb train ok!!"
	return bst 

def get_xgb_fea_imp(bst, feat_names):
	imp_vals = bst.get_fscore()
	imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
	total = array(imp_dict.values()).sum()
	fea_importance = {k:v/total for k,v in imp_dict.items()}
	for key in fea_importance:
		print "Feature: %s\t%.4f" % (key,fea_importance[key])
	return
	
def model_sklearn_mlp_regressor(x_train,y_train):
	mpl = RandomForestRegressor(verbose=0,random_state=1,criterion='mae')
	print "neural networking model training...."
	mpl.fit(x_train,y_train)
	print "neural network params: "
	print mpl.get_params()
	print "score: %.4f" % mpl.loss_

	return mpl
def model_randomForestRegressor(x_train,y_train):
	#x_train = Imputer().fit_transform(x_train)

	#设置K折交叉参数
	kfold = KFold(n_splits=5,shuffle=True,random_state = 0)

	#创建随机森林分类器
	regr = RandomForestRegressor(n_estimators = 100,criterion='mae',max_depth=15,min_samples_leaf = 3,min_samples_split = 10,max_features = 'log2',random_state=0,verbose=10)
	#regr = RandomForestRegressor(criterion='mae',verbose=10)
	print "start training with random forest regress...."
	#scores = cross_val_score(regr, x_train, y_train, cv=kfold,scoring='mean_absolute_error')
	regr.fit(x_train,y_train)
	print "random Forest Regress mean scores: ",scores.mean()
	return regr
	
def split_to_train_data(df,rate=0.7):
	train_len = int(len(df.index)*rate)
	print "train data number: %d" % train_len
	print "val data number: %d" % (len(df.index) - train_len)
	rows = random.sample(df.index,train_len)
	train_data = df.ix[rows]
	val_data = df.drop(rows)
	return train_data,val_data



def main():
	#load train and test data from file
	if len(sys.argv) < 3:
		print "usage: ./predic_model.py [train data filename] [result filename]"
		return
	print "start to load train and test data from file...."
	#1. load original data
	df_train,df_test = load_train_test_to_csv(sys.argv[1])
	print "df_train shape: " ,df_train.shape
	print "df_test shape: ",df_test.shape
	pd.set_option('display.max_columns', 100)
	
	#normalization all data
	normal_columns = ['minute','ratio_waiting_rider_same_areaid_date_hothour_quarter','ratio_waiting_rider_same_poiid_date_hothour_quarter','cusid_food_num_sum','poiid_areaid_waiting_order_size','ratio_waiting_rider_same_poiid_areaid_date_hothour_quarter','avg_same_areaid_hothour_duration','avg_same_cusid_weekday_hour_quarter_duration','avg_same_poiid_weekday_hour_quarter_duration','avg_same_areaid_weekday_hothour_quarter_duration','avg_same_areaid_weekday_hothour_quarter_busy_rider_duration','area_avg_duration','area_hour_avg_duration','customer_poi_distance','delivery_distance','deliverying_order_num','food_total_value','level_2_district_customer_num','mean_price_food','not_fetched_order_num','notbusy_working_rider_num','poi_hour_avg_duration','same_time_cus_to_poi_rank','weekday_avg_duration','working_rider_num','customer_poi_distance','level_2_district_customer_num','delivery_distance','poi_total_orders']
	#print "normalizing features: ",normal_columns
	print "normalizing the features...."
	normalization_data(df_train,normal_columns)
	normalization_data(df_test,normal_columns)
	print "df_train shape: " ,df_train.shape
	print "df_test shape: ",df_test.shape

	#2. split train data: 80% train and 20% val data
	train_data,val_data = split_to_train_data(df_train,0.8)
	
	train_cols = df_train.columns.tolist()
	train_label = ['delivery_duration']	
	train_features = list(np.setdiff1d(train_cols,train_label))
	x_train = train_data[train_features] 
	y_train = train_data['delivery_duration'] 
	x_val = val_data[train_features] 
	y_val = val_data['delivery_duration'] 
	print "features for training: ",x_train.columns.tolist()

	test_cols = df_test.columns.tolist()
	test_label = ['order_id','delivery_duration']	
	test_features = list(np.setdiff1d(test_cols,test_label))
	x_test = df_test[test_features]
	id_test =  df_test['order_id']
	print "training data x test shape: ",x_test.shape

	#3. train the model
	gbm = model_lightGBM(x_train,y_train,x_val,y_val)
	x = df_train[train_features]
	y = df_train['delivery_duration']
	print "training data x shape: ",x.shape
	print "training data y shape: ",y.shape


	# feature importance ranking
	print("Feature ranking:")
	feature_names = train_features
	feature_importances = gbm.feature_importance()
	#feature_importances = regf.feature_importances_
	indices = np.argsort(feature_importances)[::-1]
	for f in indices:
		print("feature %s (%f)" % (feature_names[f], feature_importances[f]))

	#4. predict test data
	print "start to predict the test data...."
	predict_label = gbm.predict(x_test.values)
	result_df = pd.DataFrame({'order_id':id_test, 'delivery_duration':predict_label}, columns=['order_id','delivery_duration'])
	print result_df.head()

	result_df.to_csv(sys.argv[2],index=False)
	print "result has been saved to file: ",sys.argv[2]
	return
	
if __name__ == "__main__":
	main()
