# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 06:32:11 2020

@author: Aditya

"""
import numpy as np
import pandas as pd
import pickle
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer

def predict_sales(data, model, features):
    with open('train_dataset.pkl', 'rb') as handle:
        train = pickle.load(handle)
        handle.close()
    
    test = pd.DataFrame.from_dict(data.items()).set_index(0).T
    test['Item_Outlet_Sales'] = 256
    
    train['df_train'] = 1
    test['df_train'] = 0
    
    #Interpolating numeric variables
    test['Item_Weight'] = np.interp(test['Item_Weight'],[0.01,25.0], 
                                    [train['Item_Weight'].min(),train['Item_Weight'].max()])
    
    test['Item_Visibility'] = int(np.interp(test['Item_Visibility'],[0.01,35.00], 
                                    [train['Item_Visibility'].min(),train['Item_Visibility'].max()]))
    
    test['Item_MRP'] = int(np.interp(test['Item_MRP'],[0.01,270.00], 
                                    [train['Item_MRP'].min(),train['Item_MRP'].max()]))
    
    combined = pd.concat([train,test])
    
    #Transforming target
    combined['Item_Outlet_Sales'] = np.sqrt(combined['Item_Outlet_Sales'])
    
    # Transforming Item_Weight and Item_MRP
    for i, c in enumerate(['Item_Weight','Item_MRP']):
        qt = QuantileTransformer(n_quantiles=4000, output_distribution='normal')
        scaled = qt.fit_transform(combined[[c,'Item_Outlet_Sales']])[:,0]
        combined[c] = scaled
        
    # Transforming Item Visibility
    for i, c in enumerate(['Item_Visibility']):
        scaled = PowerTransformer().fit_transform(combined[[c,'Item_Outlet_Sales']])[:,0]
        combined[c] = scaled

    # Mapping Categorical Variables : Label Encoding
    df_Fat_Mapper          = {'Low Fat':0, 'low fat':0,'LF':0,'Regular':1,'reg':1}
    df_Outlet_size_mapper  = {'Small':0, 'Medium':1, 'High':2}
    combined['Item_Fat_Content'] = combined['Item_Fat_Content'].map(df_Fat_Mapper)
    combined['Outlet_Size']      = combined['Outlet_Size'].map(df_Outlet_size_mapper)

    #Categorical Variables : OHE
    combined = pd.get_dummies(combined, columns = ['Item_Type','Outlet_Location_Type','Outlet_Type',
                                       'Outlet_Identifier','Outlet_Establishment_Year'])
    
    #Dropping unnecessary features
    combined = combined.drop(['Item_Outlet_Sales','df_train'],axis=1)
    
    # We want the scaled data to be in DataFrame instead of a numpy array
    # Because XGboost gives feature mismatch error, so we use a dataframe to ensure that model is provided
    # same features ..in the same order as train using 'features' 
    mapper = DataFrameMapper([(combined.columns, StandardScaler())])
    scaled_features = mapper.fit_transform(combined.copy(), len(combined.columns))
    scaled_features_df = pd.DataFrame(scaled_features, index=combined.index, columns=combined.columns)
    scaled_features_df = scaled_features_df.tail(1)
    
    prediction = model.predict(scaled_features_df[features])
    
    return round(np.power(prediction[0],2),5)
