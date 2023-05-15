import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

def dataset(data_file):
    data = pd.read_excel(data_file)
    #使用平均值填充空值
    data.fillna(data.mean(), inplace=True)
    data = outlier(data)
    #读取所使用的特征
    X_data = X_data = data.loc[:, ['Area', 'Perimeter','Major_Axis', 'Minor_Axis', 'Eccentricity', 'Eqdiasq',
                            'Solidity', 'Convex_Area', 'Extent', 'Aspect_Ratio','Roundness', 'Compactness',
                           'Shapefactor_1', 'Shapefactor_2','Shapefactor_3', 'Shapefactor_4']]
    #读取label
    Y_data = data.loc[:, ['Class']]
    #print(X_data)
    #print(Y_data)
    #数据归一化处理
    X_minmax = preprocessing.minmax_scale(X_data)
    #print(X_minmax)
    #对标签进行编码，有两类，则标签为0，1
    enc = preprocessing.LabelEncoder()
    enc = enc.fit(['Kirmizi_Pistachio', 'Siirt_Pistachio'])
    Y_encoder = enc.transform(Y_data.values.ravel())
    #print(Y_encoder)
    #划分训练集，测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X_minmax, Y_encoder, test_size=0.3, random_state=321)
    #print(X_train)
    #print(X_test)
    #print(Y_train)
    #print(Y_test)
    return X_train, Y_train, X_test, Y_test

def outlier(data):
    iso = IsolationForest(contamination=0.01)
    scaler = preprocessing.MinMaxScaler()
    o_data = scaler.fit_transform(data.iloc[:, :-1])
    outliers = iso.fit_predict(o_data)
    #print(outliers)
    data = data[~np.isnan(outliers)]
    #print(data)
    return data