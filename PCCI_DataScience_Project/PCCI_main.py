#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:08:16 2020

@author: sahanaasokan
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns 

df = pd.read_excel('PCCI_Data.xlsx')
y = df['readmit30']
x = df.drop(columns=['readmit30'], axis=1)

missing_count= (x.isnull().sum()/(1460)).sort_values(ascending=False)
missing_data= pd.concat([missing_count],axis=1,keys=['Missing Count'])

# Get Numeric Variables/Features
numerical_features =df._get_numeric_data()

#Encoding Categorical Variables (There should be 8)
encoded_x=pd.get_dummies(x, columns=['sex','weekday','month','ethnic_group_c'
 ,'marital_status_c','insurance_provider', 'tobacco_user','Condition','care_plan_following_discharge'])



# Correlation Matrix
correlations= df.corr()
fig = plot.subplots(figsize=(25,25))
figure=sns.heatmap(correlations,vmax= 0.9,cmap='Blues',square=True)

#0.54 correlation between cost of readmission and readmission30



test_data=df[['readmit30','care_plan_costs']]
correlation_test= test_data.corr()
# 0.078












