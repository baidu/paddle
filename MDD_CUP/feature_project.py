#!/usr/bin/python
#_*_ coding:utf-8 _*_
import pandas as pd
import sys
import numpy as np
import datetime
import sys
from math import radians,cos,sin,asin,sqrt
def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
	"""
		Calculate the great circle distance between two points 
		on the earth (specified in decimal degrees)
	"""
	#将十进制度数转化为弧度
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
	#haversine公式
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
	c = 2 * asin(sqrt(a)) 
	r = 6371 # 地球平均半径，单位为公里
	return c * r * 1000

def load_order_data(file_name, split=','):
	df = pd.read_csv(file_name, sep=split)
	if ('test_a.csv' in file_name):
		df.columns = [u'order_id', u'poi_id',u'area_id', u'food_total_value',u'box_total_value', u'food_num',u'delivery_distance',u'order_unix_time',u'arriveshop_unix_time',u'fetch_unix_time',u'finish_unix_time',u'customer_longitude',u'customer_latitude',u'poi_lng', u'poi_lat',u'waiting_order_num', u'delivery_duration']
	c = 'order_unix_time'
	mask = pd.notnull(df[c])
	df.loc[mask, c] = df.loc[mask, c].apply(lambda x: datetime.datetime.fromtimestamp(x))
	df.loc[mask, 'date'] = df.loc[mask, c].apply(lambda x: x.strftime('%Y%m%d'))
	df.loc[mask, 'weekday'] = df.loc[mask, c].apply(lambda x: x.strftime('%u'))
	df.loc[mask, 'is_weekend'] = df.loc[mask, 'weekday'].apply(lambda x: 1 if (x == 6 or x == 7) else 0)
	df.loc[mask, 'hour'] = df.loc[mask, c].apply(lambda x: x.hour)
	#df.loc[mask, 'is_hot_hour'] = df.loc[mask, 'hour'].apply(lambda x: 1 if x in (11, 17) else 0)
	df.loc[mask, 'minute'] = df.loc[mask, c].apply(lambda x: x.minute)
	df.loc[mask, 'quarter'] = df.loc[mask, 'minute'].apply(lambda x: int(x) / 5)
	df['date'] = df['date'].apply(lambda x : int(x));
	df['weekday'] = df['weekday'].apply(lambda x : int(x));
	df['date_diff'] = df['date'].apply(lambda x : x%100 if x<20170801 else 31+x%100);
	df['time_id'] = (df['date_diff']*24*60 + df['hour']*60 + df['minute']) / 15;
	df['time_id'] = df['time_id'].apply(lambda x : int(x));
	#df['time_id_day'] = (df['hour']*60 + df['minute']) / 15;
	#df['time_id_day'] = df['time_id_day'].apply(lambda x : int(x));
	return df

def load_area_data(file_name):
	df = pd.read_csv(file_name)
	mask = pd.notnull(df['time'])
	df.loc[mask, 'hour'] = df.loc[mask, 'time'].apply(lambda x: x/100)
	df.loc[mask, 'minute'] = df.loc[mask, 'time'].apply(lambda x: x%100)
	df.drop(['log_unix_time', 'time'], axis=1, inplace=True)
	df['not_fetched_order_num'] = df['not_fetched_order_num'].apply(lambda x : 0 if x<0 else x);
	df['deliverying_order_num'] = df['deliverying_order_num'].apply(lambda x : 0 if x<0 else x);
	return df

def load_weather_data(file_name):
	df = pd.read_csv(file_name)
	df = df.fillna(method='ffill')
	df = df.fillna(method='bfill')
	mask = pd.notnull(df['time'])
	df.loc[mask, 'hour'] = df.loc[mask, 'time'].apply(lambda x: x/100)
	df.loc[mask, 'minute'] = df.loc[mask, 'time'].apply(lambda x: x%100)
	df.loc[mask, 'quarter'] = df.loc[mask, 'minute'].apply(lambda x: int(x) / 5)
	df['date_diff'] = df['date'].apply(lambda x : x%100 if x<20170801 else 31+x%100);
	df['time_id'] = (df['date_diff']*24*60 + df['hour']*60 + df['minute']) / 15;
	df['time_id'] = df['time_id'].apply(lambda x : x+1);
	df['time_id'] = df['time_id'].apply(lambda x : int(x));
	df.drop(['log_unix_time', 'time', 'minute'], axis=1, inplace=True)
	df = df.groupby(['time_id', 'area_id']).agg({'temperature': 'mean', 'wind': 'mean', 'rain': 'mean'}).reset_index()
	return df

def load_basic_data(file_name):
	df = pd.read_csv(file_name, sep="\t")
	return df

def is_holiday(data):
	holiday_of_year_2017 = {1:[1,2,5,27,28,29,30,31],2:[1,2,11,14],3:[8],4:[2,3,4,29,30],5:[1,28,29],6:[1],8:[28],9:[10],10:[1,2,3,4,5,6,7,8],12:[22]} 
	newdata = data - 20170000
	month = newdata / 100
	day = newdata % 100
	if month in [3,7,8,11]:
		return 0
	if day in holiday_of_year_2017[month]:
		return 1
	else:
		return 0

def load_original_data(filename):
	df = pd.read_csv(filename)
	return df
	
def merge_all_date(path):
	train_order_file = path + "waybill_info.csv" 
	test_order_file_a = path + "waybill_info_test_a.csv"
	test_order_file_b = path + "waybill_info_test_b.csv"
	train_weather_file = path + "weather_realtime.csv"
	test_weather_file = path + "weather_realtime_test.csv"
	train_area_realtime_file = path + "area_realtime.csv"
	test_area_realtime_file = path + "area_realtime_test.csv"

	print "load train order data from %s" % train_order_file 
	df_tr_order = load_order_data(train_order_file)
	#mask = ((df_tr_order.delivery_duration < 4654.0) & (df_tr_order.delivery_duration > 663.0))
	#df_tr_order = df_tr_order.loc[mask]
	df_tr_order.drop(['arriveshop_unix_time', 'fetch_unix_time','finish_unix_time'], axis=1, inplace=True)
	df_tr_order = df_tr_order.dropna()
	print"train shape:" ,df_tr_order.info()

	df_test_a = load_order_data(test_order_file_a,split='\t')
	#df_test_a = df_test_a.loc[mask]
	df_test_a.drop(['arriveshop_unix_time', 'fetch_unix_time','finish_unix_time'], axis=1, inplace=True)
	df_test_a = df_test_a.dropna()
	print"test a shape:" ,df_test_a.info()
	#print df_test_a['date'].value_counts() 
	#print df_test_a['hour'].value_counts() 

	df_test_b = load_order_data(test_order_file_b)
	#df_test_b = df_test_b.loc[mask]
	df_test_b['delivery_duration'] = 0
	print"test b shape:" ,df_test_b.info()
	#print df_test_b['date'].value_counts() 
	#print df_test_b['hour'].value_counts() 

	print "concat.."
	df_train_val = pd.concat([df_tr_order,df_test_a])
	print "train + test a shape: ",df_train_val.shape
	df_all = pd.concat([df_tr_order,df_test_a,df_test_b])
	print"train + a + b shape: ", df_all.shape
	#print df_all[(df_all['date'] > 20170731)]['hour'].value_counts()


	print "load weather information..."
	df_tr_weather = load_weather_data(train_weather_file)
	print "train weather info: ",df_tr_weather.info()
	df_te_weather = load_weather_data(test_weather_file)
	print "test weather info: ",df_te_weather.info()
	df_weather = pd.concat([df_tr_weather,df_te_weather])
	#print df_weather['time_id'].value_counts()
	print "load weather information...ok"
	#print df_weather.shape

	print "load area information...."
	df_tr_area = load_area_data(train_area_realtime_file)
	print "train area info: ",df_tr_area.info()
	df_te_area = load_area_data(test_area_realtime_file)
	print "test area info: ",df_te_area.info()
	df_area = pd.concat([df_tr_area,df_te_area])
	#print df_area.shape
	print "load area information....ok"

	print "merge weather..."
	print "befoer merge weather df all info",df_all.info()
	df_all = pd.merge(df_all,df_weather,on=["time_id","area_id"],how="left")
	df_all = df_all.fillna(method='bfill')
	print "after merge weather df all info",df_all.info()
	print "merge area..."
	print "befoer merge area df all info",df_all.info()
	df_all = pd.merge(df_all,df_area,on=["date","hour","minute","area_id"],how="left")
	print "after merge area df all info",df_all.info()

	print "before drop null shape: ",
	print df_all.shape
	print "a shape: ",
	print df_all[(df_all['date'] >= 20170731) &(df_all['hour'] != 11) &(df_all['hour'] != 17)].info()
	print "b shape: ",
	print df_all[(df_all['date'] >= 20170731) &((df_all['hour'] == 11) |(df_all['hour'] == 17))].info()
	print "drop null..."
	df_all = df_all.dropna()
	print "afger drop null shape: ",
	print df_all.shape
	print "a shape: ",
	print df_all[(df_all['date'] >= 20170731) &(df_all['hour'] != 11) &(df_all['hour'] != 17)].shape
	print "b shape: ",
	print df_all[(df_all['date'] >= 20170731) &((df_all['hour'] == 11) |(df_all['hour'] == 17))].shape
	cols = df_all.columns.tolist()
	to_drop = ['time_id','date_diff']
	features = list(np.setdiff1d(cols, to_drop))
	df_all = df_all[features]
	print "final df_all shape: ",
	print df_all.shape
	print "a shape: ",
	print df_all[(df_all['date'] >= 20170731) &(df_all['hour'] != 11) &(df_all['hour'] != 17)].shape
	print "b shape: ",
	print df_all[(df_all['date'] >= 20170731) &((df_all['hour'] == 11) |(df_all['hour'] == 17))].shape
	return df_all,df_train_val

poi_id_dic = {}
def feature_project(df):

	#df.columns = [u'order_id', u'poi_id',u'area_id', u'food_total_value',u'box_total_value', u'food_num',u'delivery_distance',u'order_unix_time',u'arriveshop_unix_time',u'fetch_unix_time',u'finish_unix_time',u'customer_longitude',u'customer_latitude',u'poi_lng', u'poi_lat',u'waiting_order_num', u'delivery_duration']
	#该月的日期
	df['day'] = df['date'] % 100
	df['is_weekend'] = map(lambda x: 1 if x >= 6 else 0 ,df['weekday'])

	#高峰时刻
	#df.loc[:, 'poi_id'] =	map(lambda x,y : map_poi_id(x,y), df['poi_lng'], df['poi_lat'])
	df.loc[:,'is_hot_hour'] = map(lambda x: map_hot_hour(x),df['hour']) 
	df['is_hot_hour'].value_counts()

	#根据相同的商店位置得到相同的商店,并对商店进行编号,相同位置并不代表是同一个商店,商店编号只是用来计算该相同位置的热度，即订单量
	#poi_groups = df[['order_id','poi_lng','poi_lat']].groupby(['poi_lng','poi_lat']).groups
	#poi_index = 0
	#global poi_id_dic
	#for key in poi_groups:
	#	poi_id_dic[key] = [poi_index,len(poi_groups[key])]
	#	poi_index += 1
	#	
	##商户ID，相同位置商店的ID相同
	#df.loc[:, 'poi_id'] =	map(lambda x,y : map_poi_id(x,y), df['poi_lng'], df['poi_lat'])

	#相同商户的历史订单量
	poi_orders = df['poi_id'].value_counts().reset_index()
	poi_orders.columns = ['poi_id','poi_total_orders']
	df = pd.merge(df,poi_orders,on=['poi_id'],how='left')
	print"poi total orders",df[['poi_total_orders']].describe()

	#df.loc[:, 'poi_hot_degree'] =	map(lambda x,y : map_poi_id_degree(x,y), df['poi_lng'], df['poi_lat'])
	

	#以15分钟为间隔分割所有时间
	#增加用户在所在相同配送区域相同时间段的下单时间排名
	df['quarter_rank'] = df[['area_id','date','hour','quarter','minute']].groupby(['area_id','date','hour','quarter']).rank(method = 'dense')

	#增加用户在所在时间段的下单时间差
	df['quarter_distance_to_peak'] = df['minute'] - df['quarter'] * 5 

	#距离高峰时刻的时间差:distance_to_peak_minute
	#df['distance_to_peak_minute'] = (df['hour'] - 11)*60 + df['minute'] 
	#df['distance_to_peak_minute'] = map(lambda x: (x - 10)*60 + df['minute'] if x	else , df['hour'] 
	#df['distance_to_peak_minute'] = map(lambda x: (x - 16)*60 + df['minute'] if x >= 16 and x =< 18, df['hour'] 
	#print df[['order_id','distance_to_peak_minute']][:15].sort_values(by='distance_to_peak_minute')
	#print df['distance_to_peak_minute'].value_counts()

	#下单时刻用户订单的菜品数量与下单时刻商户未完成单量比例: ratio_food_waiting_num
	df['ratio_food_waiting_num'] = df['food_num'] / (df['waiting_order_num'] + 1) 

	#是否节假日
	#print df['date'].value_counts()
	#df['is_holiday'] = map(is_holiday,df['date'])
	#print df['is_holiday'].value_counts()

	#下单时商户未完成订单数在该配送区域内所有未完成订单数的排名: rank_waiting_num_area
	#print df[['area_id','waiting_order_num']][:15]
	#df['rank_waiting_num_area'] = df[['area_id','waiting_order_num']].groupby('area_id').rank(method='dense')['waiting_order_num']
	#print df[['area_id','rank_waiting_num_area','waiting_order_num']][:15]

	#用户位置与商家位置的欧氏距离
	df['customer_poi_distance'] = map(haversine,df['customer_longitude'], df['customer_latitude'], df['poi_lng'], df['poi_lat'])
	#用户位置与商家位置之间的距离在相同时间段相同配送区域的排名
	df['same_time_cus_to_poi_rank'] = df[['area_id','date','hour','quarter','customer_poi_distance']].groupby(['area_id','date','hour','quarter']).rank(method = 'dense')
	#print df['customer_poi_distance'].value_counts()

	#用户位置与商家为止的欧式距离与配送导航距离的比例
	df['distance_cus_poi_to_delivery_dis'] = df['delivery_distance'] - df['customer_poi_distance']
	#print"distance_cus_poi_to_delivery_dis value counts: ", df['distance_cus_poi_to_delivery_dis'].value_counts()

	#用户每个菜品的平均价格
	df['mean_price_food'] = df['food_total_value'] / (df['food_num'] + 1)

	#菜品数量与餐盒数的比值
	df['ratio_food_num_to_box'] = df['food_num'] / (df['box_total_value'] + 1)

	#未取餐数量与空闲骑士数量的比值
	df['notbusy_rider_load_ratio'] = df['not_fetched_order_num'] / (df['notbusy_working_rider_num'] + 1)
	#未取餐数量与忙碌骑士数量的比值
	df['busy_rider_load_ratio'] = df['not_fetched_order_num'] / (df['working_rider_num'] + 1)
	
	#忙碌骑士与空闲骑士的比例
	df['ratio_busy_to_notbusy'] = df['working_rider_num'] / (df['notbusy_working_rider_num'] + 1)

	#区域取餐未送达单量与忙碌骑士的比例
	df['ratio_deliverying_num_to_busy_rider'] = df['deliverying_order_num'] / (df['working_rider_num'] + 1)

	#商店未完成单量与区域未取餐单量比例
	df['ratio_waiting_order_to_not_fetch'] = df['waiting_order_num'] / (df['not_fetched_order_num'] + 1)
	
	#按照每个配送区域的数量分成7个等级
	area_id_dic = {}
	area_id_dic[1004986] = 1
	area_id_dic[1002570] = 2
	area_id_dic[1003209] = 3
	area_id_dic[1002568] = 4
	area_id_dic[1002471] = 4
	area_id_dic[1002487] = 4
	area_id_dic[1002435] = 5
	area_id_dic[1002430] = 5
	area_id_dic[1003189] = 5
	area_id_dic[1004285] = 6
	area_id_dic[1002483] = 7 

	#df['district_id'] = map(area_id_dic[x],df['area_id'])
	df.loc[:,'district_id'] = df['area_id'].apply(lambda x:area_id_dic[x])
	#print df['district_id'].value_counts()
	
	#1. construct the area of customer
	#district_range = df[['area_id', 'customer_longitude', 'customer_latitude']].groupby('area_id').agg(['max', 'min'])
	#district_range.columns = ['customer_longitude_max', 'customer_longitude_min', 'customer_latitude_max', 'customer_latitude_min']
	#district_range.reset_index(inplace=True)
	#print district_range.head()
	#split data to train data and test data
	#df_test = df[(df['date'] > 20170731)]
	#df_train = df[(df['date'] <= 20170731)]
	return df

district_id_dic = {}
def divide_customer_of_area(df):
	district_range = df[['district_id', 'customer_longitude', 'customer_latitude']].groupby('district_id').agg(['max', 'min'])
	district_range.columns = ['customer_longitude_max', 'customer_longitude_min', 'customer_latitude_max', 'customer_latitude_min']
	district_range.reset_index(inplace=True)

	district_range['customer_longitude_range'] = district_range['customer_longitude_max'] - district_range['customer_longitude_min']

	#订单量在30k以上的划分为20*20的小块,其余区域划分为10*10的小块
	district_range['customer_longitude_range_step'] = map(lambda x,y : x/10.0 if 3<y else x/5.0, district_range['customer_longitude_range'], district_range['district_id'])
	district_range['customer_latitude_range'] = district_range['customer_latitude_max'] - district_range['customer_latitude_min']
	district_range['customer_latitude_range_step'] = map(lambda x,y : x/10.0 if 3<y else x/5.0, district_range['customer_latitude_range'], district_range['district_id'])
	
	#确定每个用户所在的区域id:district_id_x_y
	global district_id_dic
	for district in district_range[['district_id', 'customer_longitude_min', 'customer_latitude_min', 'customer_longitude_range_step', 'customer_latitude_range_step']].values:
		district_id, customer_longitude_min, customer_latitude_min, customer_longitude_range_step, customer_latitude_range_step = list(district)
		district_id_dic[district_id] = {'customer_longitude_min':customer_longitude_min, 'customer_latitude_min':customer_latitude_min, 'customer_longitude_range_step':customer_longitude_range_step, 'customer_latitude_range_step':customer_latitude_range_step}

	df.loc[:, 'customer_district_id'] =  map(lambda x,y,z : map_district_id(x,y,z), df['district_id'], df['customer_longitude'], df['customer_latitude'])
	#print df[(df['district_id'] == 7)]['customer_district_id'].value_counts()


	#相同二级区域中同一天相同时间段相同配送区域的订单总量(用户个数)
	same_time_day = df.groupby(['date','hour','district_id','customer_district_id']).size()
	same_time_day = pd.DataFrame(same_time_day)
	same_time_day.columns = ['level_2_district_customer_num']
	same_time_day = same_time_day.reset_index()
	df = pd.merge(df,same_time_day,on=['date','hour','district_id','customer_district_id'],how='left')

	#配送区域所有时间段的早晚高峰的平均时间
	print "calculate area hot hour avg duration..."
	features_for_hothour_area = ['area_id','is_hot_hour','delivery_duration']
	condition = (df['date'] <20170731) | ((df['date'] >= 20170731) & (df['is_hot_hour'] == 0))

	same_areaid_hot_hour = df[condition][features_for_hothour_area].groupby(['area_id','is_hot_hour'])
	same_areaid_hot_hour_size = pd.DataFrame(same_areaid_hot_hour.size())  
	same_areaid_hot_hour_size.columns=['size'] 
	same_areaid_hot_hour_sum= pd.DataFrame(same_areaid_hot_hour.sum())  
	same_areaid_hot_hour_sum.columns=['sum'] 

	same_areaid_hot_hour_sum['avg_same_areaid_hothour_duration']= same_areaid_hot_hour_sum['sum'] / same_areaid_hot_hour_size['size']
	same_areaid_hot_hour_sum = same_areaid_hot_hour_sum.reset_index()
	same_areaid_hot_hour_sum.drop(['sum'], axis=1, inplace=True)
	df = pd.merge(df,same_areaid_hot_hour_sum,on=['area_id','is_hot_hour'],how='left')
	test_condition = ((df['date'] >= 20170731) & (df['is_hot_hour'] > 0))
	print df[test_condition][['avg_same_areaid_hothour_duration']].info()
	print "calculate area hot hour avg duration...ok"

	#相同二级区域中相同日期相同时间段的平均订单送达时间
	print "calculate same day cusid hour quarter avg duration.."
	features_for_cusid = ['customer_district_id','weekday','is_hot_hour','delivery_duration']
	condition = (df['date'] <20170731) | ((df['date'] >= 20170731) & (df['is_hot_hour'] == 0))
	test_condition = ((df['date'] >= 20170731) & (df['is_hot_hour'] > 0))
	same_cusid_date_hour_quarter = df[condition][features_for_cusid].groupby(['customer_district_id','weekday','is_hot_hour'])
	same_cusid_date_hour_quarter_size = pd.DataFrame(same_cusid_date_hour_quarter.size())  
	same_cusid_date_hour_quarter_size.columns=['size'] 
	same_cusid_date_hour_quarter_sum = pd.DataFrame(same_cusid_date_hour_quarter.sum())  
	same_cusid_date_hour_quarter_sum.columns=['sum'] 

	same_cusid_date_hour_quarter_sum['avg_same_cusid_weekday_hour_quarter_duration']= same_cusid_date_hour_quarter_sum['sum'] / same_cusid_date_hour_quarter_size['size']
	same_cusid_date_hour_quarter_sum = same_cusid_date_hour_quarter_sum.reset_index()
	same_cusid_date_hour_quarter_sum.drop(['sum'], axis=1, inplace=True)
	df = pd.merge(df,same_cusid_date_hour_quarter_sum,on=['customer_district_id','weekday','is_hot_hour'],how='left')

	print df[test_condition][['avg_same_cusid_weekday_hour_quarter_duration']].info()
	print "calculate same weekday cusid hour quarter avg duration..ok"


	#同一个商家在早高峰和晚高峰在相同日期的平均送达时间
	print "calculate same weekday poiid hour quarter avg duration.."
	features_for_poiid = ['poi_id','weekday','is_hot_hour','delivery_duration']
	same_poiid_date_hour_quarter = df[condition][features_for_poiid].groupby(['poi_id','weekday','is_hot_hour'])
	same_poiid_date_hour_quarter_size = pd.DataFrame(same_poiid_date_hour_quarter.size())  
	same_poiid_date_hour_quarter_size.columns=['size'] 
	same_poiid_date_hour_quarter_sum = pd.DataFrame(same_poiid_date_hour_quarter.sum())  
	same_poiid_date_hour_quarter_sum.columns=['sum'] 

	same_poiid_date_hour_quarter_sum['avg_same_poiid_weekday_hour_quarter_duration']= same_poiid_date_hour_quarter_sum['sum'] / same_poiid_date_hour_quarter_size['size']
	same_poiid_date_hour_quarter_sum = same_poiid_date_hour_quarter_sum.reset_index()
	same_poiid_date_hour_quarter_sum.drop(['sum'], axis=1, inplace=True)
	df = pd.merge(df,same_poiid_date_hour_quarter_sum,on=['poi_id','weekday','is_hot_hour'],how='left')
	print df[test_condition][['avg_same_poiid_weekday_hour_quarter_duration']].info()
	print "calculate same weekday poiid hour quarter avg duration..ok"

	#相同二级区域未取订单与忙碌骑士和空闲骑士的比例
	print "calculate same weekday cusid hour quarter avg duration.."
	features_for_cusid_rider = ['customer_district_id','date','is_weekend','is_hot_hour','quarter','not_fetched_order_num']
	same_cusid_rider = df[features_for_cusid_rider].groupby(['customer_district_id','is_weekend','is_hot_hour','quarter','date'])
	same_cusid_rider_sum = pd.DataFrame(same_cusid_rider.sum())  
	same_cusid_rider_sum.columns=['cusid_food_num_sum'] 

	#same_cusid_rider_sum['avg_same_cusid_food_num_avg']= same_cusid_rider_sum['sum'] / same_areaid_date_hour_quarter_size['size']
	same_cusid_rider_sum = same_cusid_rider_sum.reset_index()
	df = pd.merge(df,same_cusid_rider_sum,on=['is_weekend','is_hot_hour','customer_district_id','date','quarter'],how='left')
	df['same_cusid_rider_food_num_avg'] = df['cusid_food_num_sum'] / df['working_rider_num']
	print df[test_condition][['same_cusid_rider_food_num_avg']].info()
	print "calculate same cusid food num avg... ok"

	#利用areaid_hot_hour_avg_duration填充缺失值
	#相同日期，相同配送区域相同时间段送达平均时间
	print "calculate same weekday areaid hour quarter avg duration.."
	features_for_rider = ['area_id','weekday','is_hot_hour','quarter','delivery_duration']
	same_areaid_date_hour_quarter = df[condition][features_for_rider].groupby(['area_id','weekday','is_hot_hour','quarter'])
	same_areaid_date_hour_quarter_size = pd.DataFrame(same_areaid_date_hour_quarter.size())  
	same_areaid_date_hour_quarter_size.columns=['size'] 
	same_areaid_date_hour_quarter_sum = pd.DataFrame(same_areaid_date_hour_quarter.sum())  
	same_areaid_date_hour_quarter_sum.columns=['sum'] 

	same_areaid_date_hour_quarter_sum['avg_same_areaid_weekday_hothour_quarter_duration']= same_areaid_date_hour_quarter_sum['sum'] / same_areaid_date_hour_quarter_size['size']
	same_areaid_date_hour_quarter_sum = same_areaid_date_hour_quarter_sum.reset_index()
	same_areaid_date_hour_quarter_sum.drop(['sum'], axis=1, inplace=True)
	df = pd.merge(df,same_areaid_date_hour_quarter_sum,on=['weekday','is_hot_hour','area_id','quarter'],how='left')

	print df[test_condition][['avg_same_areaid_weekday_hothour_quarter_duration']].info()
	print "calculate same weekday  areaid hour quarter avg duration.. ok"

	#相同日期相同商家相同配送地区相同高峰时刻未完成订单量与忙碌骑士的比例来衡量骑士负载o
	print "calculate same date areaid poi_id hot hoour quarter load degree.."
	features_for_poiid_areaid_rider = ['area_id','poi_id','date','is_weekend','is_hot_hour','quarter','waiting_order_num']
	same_poiid_areaid_date_hothour_quarter_rider = df[features_for_poiid_areaid_rider].groupby(['area_id','poi_id','date','is_hot_hour','is_weekend','quarter'])
	same_poiid_areaid_date_hothour_quarter_rider_size = pd.DataFrame(same_poiid_areaid_date_hothour_quarter_rider.size())  
	same_poiid_areaid_date_hothour_quarter_rider_size.columns=['poiid_areaid_waiting_order_size'] 
	same_poiid_areaid_date_hothour_quarter_rider_size = same_poiid_areaid_date_hothour_quarter_rider_size.reset_index()
	df = pd.merge(df,same_poiid_areaid_date_hothour_quarter_rider_size,on=['date','is_hot_hour','area_id','poi_id','quarter','is_weekend'],how='left')

	df['ratio_waiting_rider_same_poiid_areaid_date_hothour_quarter'] = df['poiid_areaid_waiting_order_size'] / (df['working_rider_num'] + 1) 
	print df[test_condition][['ratio_waiting_rider_same_poiid_areaid_date_hothour_quarter']].info()
	print "calculate same date areaid poi_id hot hoour quarter load degree...ok"

	#相同日期相同高峰期相同配送区域内未完成订单数与忙碌骑士的比例衡量骑士忙碌程度
	print "calculate same date areaid hot hoour quarter load degree..."
	features_for_areaid_rider = ['area_id','date','is_hot_hour','quarter','waiting_order_num']
	same_areaid_date_hothour_quarter_rider = df[features_for_areaid_rider].groupby(['area_id','date','is_hot_hour','quarter'])
	same_areaid_date_hothour_quarter_rider_size = pd.DataFrame(same_areaid_date_hothour_quarter_rider.size())  
	same_areaid_date_hothour_quarter_rider_size.columns=['areaid_waiting_order_size'] 
	same_areaid_date_hothour_quarter_rider_size = same_areaid_date_hothour_quarter_rider_size.reset_index()
	df = pd.merge(df,same_areaid_date_hothour_quarter_rider_size,on=['date','is_hot_hour','area_id','quarter'],how='left')

	df['ratio_waiting_rider_same_areaid_date_hothour_quarter'] = df['areaid_waiting_order_size'] / (df['working_rider_num'] + 1) 
	print df[test_condition][['ratio_waiting_rider_same_areaid_date_hothour_quarter']].info()
	print "calculate same date area_id hot hoour quarter load degree...ok"



	#相同日期相同配送地区相同高峰时刻未完成订单量与忙碌骑士的比例来衡量骑士负载o
	print "calculate same date poi_id hot hoour quarter load degree.."
	features_for_poiid_rider = ['poi_id','date','is_hot_hour','quarter','waiting_order_num']
	same_poiid_date_hothour_quarter_rider = df[features_for_poiid_rider].groupby(['poi_id','date','is_hot_hour','quarter'])
	same_poiid_date_hothour_quarter_rider_size = pd.DataFrame(same_poiid_date_hothour_quarter_rider.size())  
	same_poiid_date_hothour_quarter_rider_size.columns=['poiid_waiting_order_size'] 
	same_poiid_date_hothour_quarter_rider_size = same_poiid_date_hothour_quarter_rider_size.reset_index()
	df = pd.merge(df,same_poiid_date_hothour_quarter_rider_size,on=['date','is_hot_hour','poi_id','quarter'],how='left')

	df['ratio_waiting_rider_same_poiid_date_hothour_quarter'] = df['poiid_waiting_order_size'] / (df['working_rider_num'] + 1) 
	print df[test_condition][['ratio_waiting_rider_same_poiid_date_hothour_quarter']].info()
	print "calculate same date poi_id hot hoour quarter load degree...ok"


	#相同区域相同时间段每个忙碌骑士平均送达时间
	df['avg_same_areaid_weekday_hothour_quarter_busy_rider_duration'] = df['avg_same_areaid_weekday_hothour_quarter_duration'] / (df['working_rider_num'] + 1)
	print df[test_condition][['avg_same_areaid_weekday_hothour_quarter_busy_rider_duration']].info()

	#相同二级区域中同一天相同时间段相同配送区域的订单总量与同一天相同时间段相同配送区域总骑士数量比例
	df['ratio_level_2_district_customer_to_rider_num'] = df['level_2_district_customer_num'] / (df['working_rider_num'] + df['notbusy_working_rider_num'] + 1)	

	#相同二级区域中同一天相同时间段相同配送区域的订单总量与同一天相同时间段相同配送区域与(区域未取餐单量)比例
	df['ratio_level_2_distict_customer_to_not_fetched_num'] = df['level_2_district_customer_num'] /(df['not_fetched_order_num'] + 1)

	#相同二级区域中同一天相同时间段相同配送区域的订单总量与同一天相同时间段相同配送区域与(区域未送达单量)比例
	df['ratio_level_2_distict_customer_to_deliverying_num'] = df['level_2_district_customer_num'] /(df['deliverying_order_num'] + 1)

	#相同二级区域中同一天相同时间段用户下单时间在相同条件下所有用户的下单时间的排名
	df['level_2_order_time_rank'] = df[['date','hour','district_id','customer_district_id','minute']].groupby(['date','hour','district_id','customer_district_id']).rank(method='dense') 
	print "fill Nan by using aread avg durations..."
	dest_columns = ['avg_same_poiid_weekday_hour_quarter_duration','avg_same_cusid_weekday_hour_quarter_duration']
	print "before fillna shape: ",df[dest_columns].info()

	tmp = df[df[dest_columns[0]].isnull()]	
	tmp[dest_columns[0]] = tmp['avg_same_areaid_hothour_duration']
	df = pd.concat([df,tmp])
	tmp = df[df[dest_columns[1]].isnull()]	
	tmp[dest_columns[1]] = tmp['avg_same_areaid_hothour_duration']
	df = pd.concat([df,tmp])
	print "after fillna shape: ",df[dest_columns].info()
	print "fill Nan by using aread avg durations...ok"
	
	return df



def map_hot_hour(hour):
	flag = 0
	if hour >= 11 and hour <= 12:
		flag = 1
	elif hour >= 17 and hour <= 18:
		flag = 2
	return flag
def map_district_id(district_id, longitude, latitude):
	return str(district_id)+"_"+str(int((longitude-district_id_dic[district_id]['customer_longitude_min'])/district_id_dic[district_id]['customer_longitude_range_step']))+"_"+str(int((latitude-district_id_dic[district_id]['customer_latitude_min'])/district_id_dic[district_id]['customer_latitude_range_step']))

def map_poi_id(longitude, latitude):
	return poi_id_dic[(longitude,latitude)][0] 
def map_poi_id_degree(longitude, latitude):
	degree = poi_id_dic[(longitude,latitude)][1]/100
	return degree

# poi_id  avg_duration
def poi_avg_duration_generator(df):
	unq_date = df['date'].unique();
	unq_date.sort();
	avg_duration = None;
	for i in range(1, len(unq_date)):
		s, e = unq_date[0], unq_date[i];
		duration = df[(df['date']>=s) & (df['date']<e)].groupby(['poi_id']).agg({'delivery_duration':'mean'}).reset_index();
		duration = duration.rename(index=str, columns={'delivery_duration':'poi_avg_duration'});
		duration['date'] = unq_date[i];
		avg_duration = pd.concat([avg_duration, duration]) if avg_duration is not None else duration;
	return avg_duration;

#hour_poi_id avg_duration
def poi_hour_avg_duration_generator(df):
	unq_date = df['date'].unique();
	unq_date.sort();
	avg_duration = None;
	for i in range(1, len(unq_date)):
		s, e = unq_date[0], unq_date[i];
		duration = df[(df['date']>=s) & (df['date']<e)].groupby(['hour', 'poi_id']).agg({'delivery_duration':'mean'}).reset_index();
		duration = duration.rename(index=str, columns={'delivery_duration':'poi_hour_avg_duration'});
		duration['date'] = unq_date[i];
		avg_duration = pd.concat([avg_duration, duration]) if avg_duration is not None else duration;
	return avg_duration;

#area_id avg_duration
def area_avg_duration_generator(df):
	unq_date = df['date'].unique();
	unq_date.sort();
	avg_duration = None;
	for i in range(1, len(unq_date)):
		s, e = unq_date[0], unq_date[i];
		duration = df[(df['date']>=s) & (df['date']<e)].groupby(['area_id']).agg({'delivery_duration':'mean'}).reset_index();
		duration = duration.rename(index=str, columns={'delivery_duration':'area_avg_duration'});
		duration['date'] = unq_date[i];
		avg_duration = pd.concat([avg_duration, duration]) if avg_duration is not None else duration;
	return avg_duration;

#hour_area_id avg_duration
def area_hour_avg_duration_generator(df):
	unq_date = df['date'].unique();
	unq_date.sort();
	avg_duration = None;
	for i in range(1, len(unq_date)):
		s, e = unq_date[0], unq_date[i];
		duration = df[(df['date']>=s) & (df['date']<e)].groupby(['hour', 'area_id']).agg({'delivery_duration':'mean'}).reset_index();
		duration = duration.rename(index=str, columns={'delivery_duration':'area_hour_avg_duration'});
		duration['date'] = unq_date[i];
		avg_duration = pd.concat([avg_duration, duration]) if avg_duration is not None else duration;
	return avg_duration;

#hour avg duration
def hour_avg_duration_generator(df):
	unq_date = df['date'].unique();
	unq_date.sort();
	avg_duration = None;
	for i in range(1, len(unq_date)):
		s, e = unq_date[0], unq_date[i];
		duration = df[(df['date']>=s) & (df['date']<e)].groupby(['hour']).agg({'delivery_duration':'mean'}).reset_index();
		duration = duration.rename(index=str, columns={'delivery_duration':'hour_avg_duration'});
		duration['date'] = unq_date[i];
		avg_duration = pd.concat([avg_duration, duration]) if avg_duration is not None else duration;
	return avg_duration;

# weekday avg duration
def weekday_avg_duration_generator(df):
	unq_date = df['date'].unique();
	unq_date.sort();
	avg_duration = None;
	for i in range(1, len(unq_date)):
		s, e = unq_date[0], unq_date[i];
		duration = df[(df['date']>=s) & (df['date']<e)].groupby(['weekday']).agg({'delivery_duration':'mean'}).reset_index();
		duration = duration.rename(index=str, columns={'delivery_duration':'weekday_avg_duration'});
		duration['date'] = unq_date[i];
		avg_duration = pd.concat([avg_duration, duration]) if avg_duration is not None else duration;
	return avg_duration;

# poi_id 和 customer_district_id 的小格子pair的统计相关信息
def delivery_id_avg_duration_generator(df):
	unq_date = df['date'].unique();
	unq_date.sort();
	avg_duration = None;
	for i in range(1, len(unq_date)):
		s, e = unq_date[0], unq_date[i];
		duration = df[(df['date']>=s) & (df['date']<e)].groupby(['poi_id','customer_district_id']).agg({'delivery_duration':'mean'}).reset_index();
		duration = duration.rename(index=str, columns={'delivery_duration':'delivery_id_avg_duration'});
		duration['date'] = unq_date[i];
		avg_duration = pd.concat([avg_duration, duration]) if avg_duration is not None else duration;
	return avg_duration;



def main():
    if(len(sys.argv) != 3):
        print"usage: ./feature_project.py [all_data/] [generate train dataset filename]"
        return -1

	#整理原始数据
	df_all,df_train_val = merge_all_date(sys.argv[1])
	poi_avg_duration = poi_avg_duration_generator(df_all)
	df_all = pd.merge(df_all,poi_avg_duration,on=['poi_id','date'],how='left')
	#print "b shape: ",
	#print df_all[(df_all['date'] >= 20170731) &((df_all['hour'] == 11) |(df_all['hour'] == 17))].info()
	df_all['poi_avg_duration'].fillna(2031.35, inplace=True);

	print('poi hour avg duration...')
	poi_hour_avg_duration = poi_hour_avg_duration_generator(df_all);
	df_all = pd.merge(df_all, poi_hour_avg_duration, on = ['date', 'hour', 'poi_id'], how='left');
	df_all['poi_hour_avg_duration'].fillna(2031.35, inplace=True);
	#print "b shape: ",
	#print df_all[(df_all['date'] >= 20170731) &((df_all['hour'] == 11) |(df_all['hour'] == 17))].info()

	print('area duration...')
	area_avg_duration = area_avg_duration_generator(df_all);
	df_all = pd.merge(df_all, area_avg_duration, on = ['date', 'area_id'], how='left') ;
	#print "b shape: ",
	#print df_all[(df_all['date'] >= 20170731) &((df_all['hour'] == 11) |(df_all['hour'] == 17))].info()

	print('area hour duration...')
	area_hour_avg_duration = area_hour_avg_duration_generator(df_all);
	df_all = pd.merge(df_all, area_hour_avg_duration, on = ['date', 'hour', 'area_id'], how='left')
	#print "b shape: ",
	#print df_all[(df_all['date'] >= 20170731) &((df_all['hour'] == 11) |(df_all['hour'] == 17))].info()

	print('hour duration...')
	hour_avg_duration = hour_avg_duration_generator(df_all);
	df_all = pd.merge(df_all, hour_avg_duration, on = ['date', 'hour'], how='left');
	#print "b shape: ",
	#print df_all[(df_all['date'] >= 20170731) &((df_all['hour'] == 11) |(df_all['hour'] == 17))].info()

	print('weekday duration...')
	weekday_avg_duration = weekday_avg_duration_generator(df_all);
	df_all = pd.merge(df_all, weekday_avg_duration, on = ['date', 'weekday'], how='left');
	#print "b shape: ",
	#print df_all[(df_all['date'] >= 20170731) &((df_all['hour'] == 11) |(df_all['hour'] == 17))].info()
	df_all = feature_project(df_all)
	#2.1 divide area by customer position
	df_all = divide_customer_of_area(df_all)

	print "b shape: ",
	print df_all[(df_all['date'] >= 20170731) &((df_all['hour'] == 11) |(df_all['hour'] == 17))].info()
	print "drop Nan..."
	df_all = df_all.dropna()
	#df_all = df_all.fillna(0)
	print "b shape: ",
	print df_all[(df_all['date'] >= 20170731) &((df_all['hour'] == 11) |(df_all['hour'] == 17))].info()
	#print df_all.info()
	print "start to save data to file %s "%sys.argv[2]
	df_all.to_csv(sys.argv[2],index=False)
	print 'save to %s ok' % sys.argv[2]
	return

if __name__ == "__main__":
	main()
