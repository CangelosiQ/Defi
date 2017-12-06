# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:29:19 2017

@author: Quentin
"""

import numpy as np
import pandas as pd
import datetime as dt
from os import listdir
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split

def open_and_transform(file):
    path='./../data_meteo/'
    #input_file='./../data_meteo/train_1.csv'
    df = pd.read_csv(path+file, header=0, delimiter=";",decimal=",")
    #print("Dimensions:",np.shape(df))
    df['date']=df['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
    df['insee'] = df['insee'].astype('category')
    df['mois'] = df['mois'].astype('category')
    df['ddH10_rose4'] = df['ddH10_rose4'].astype('category')
    df['ech'] = df['ech'].astype('category')
    df['flvis1SOL0'] = df['flvis1SOL0'].astype('float64')
    #print("Fichier",file," ouvert.")
    return df

def get_data_imputed():
    path='./../data_meteo/'
    file='imputer_final.csv'
    df = pd.read_csv(path+file, header=None, delimiter=";",decimal=".")
    return df
    
def get_data_raw(scale, add_dummies,var_dummies,TrainTestSplit=True,sz_test=0.3,impute_method='drop',convert_month2int=False,date_method='drop'):
    print('We are addressing your request.')
    if impute_method is 'imputed':
        df=get_data_imputed()
        print('Data has been imported. Size:',df.shape)    

        if TrainTestSplit:
            Y=df.iloc[:,-1]
            X=df.iloc[:,:-1]
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=sz_test,random_state=11)
            print('Train size: %d, Test size: %d'%(X_train.shape[0],X_test.shape[0]))

    else:
        listdir('./../data_meteo/')
        list_files=np.empty(36, dtype='|U12')
        i=0
        for fichier in listdir('./../data_meteo/'):
            if 'train' in fichier:
                list_files[i]=fichier
                i=i+1
        
        df=pd.DataFrame()
        for file in list_files:
            df=pd.concat([df,open_and_transform(file)])

        df=df.sort_values(by=['ech','date'],ascending=True)
        print('Data has been imported. Size:',df.shape)    
        
        if convert_month2int:
            df=convert_month_to_int(df)
            print('Months converted to int.')
        
        if add_dummies:
            df_dummies=pd.get_dummies(df[var_dummies])
            df=pd.concat([df,df_dummies],axis=1)
            df=df.drop(var_dummies,axis=1)
            print('Dummies added.')
        
        if date_method=='drop':
            df=df.drop(['date'],axis=1)
            print('Date dropped.')

        if impute_method=='drop':
            N_before=df.shape[0]
            df=df.dropna(axis=0)
            N_after=df.shape[0]
            print("%d data points deleted. %0.2f %s"%(N_before-N_after,(N_before-N_after)/N_before*100,'%'))
        
        if TrainTestSplit:
            Y=df['tH2_obs']
            X=df
            X=X.drop(['tH2_obs'],axis=1) ## !!! Date?
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=sz_test,random_state=11)
            print('Train size: %d, Test size: %d'%(X_train.shape[0],X_test.shape[0]))

        
        
    if scale:
        scaler = StandardScaler()  
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)  
        # Meme transformation sur le test
        X_test = scaler.transform(X_test)
        print('Data scaled')
            
    return X_train,X_test,Y_train,Y_test,scaler


    
    
    
def get_data_tidied():
    df_tot=get_data_raw()
    meteo_quant=df_tot[["tH2", "capeinsSOL0", "ciwcH20","clwcH20", "ffH10","flir1SOL0",
                    "fllat1SOL0","flsen1SOL0","flvis1SOL0","hcoulimSOL0","huH2","iwcSOL0",
                    "nbSOL0_HMoy","nH20","ntSOL0_HMoy","pMER0","rr1SOL0","rrH20",
                    "tH2_VGrad_2.100","tH2_XGrad","tH2_YGrad","tpwHPA850","ux1H10",
                    "vapcSOL0","vx1H10","ech"]]
    meteo_qual = df_tot[["insee","ddH10_rose4","mois"]]
    meteo_date = df_tot[["date"]]
    meteo_y = df_tot[["tH2_obs"]]    
    return meteo_quant, meteo_qual, meteo_date, meteo_y

def load_test_set():
    file='./../data_meteo/test.csv'
    df = pd.read_csv(file, header=0, delimiter=";")
    #print("Dimensions:",np.shape(df))
    df['date']=df['date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
    df['insee'] = df['insee'].astype('category')
    df['mois'] = df['mois'].astype('category')
    df['ddH10_rose4'] = df['ddH10_rose4'].astype('category')
    df['ech'] = df['ech'].astype('category')
    df['flvis1SOL0'] = np.float64(df['flvis1SOL0'])
    return df

def convert_month_to_int(df):
    df.mois=df.mois.astype('str')
    df.mois[df.mois=='janvier']='1'
    df.mois[df.mois=='février']='2'
    df.mois[df.mois=='mars']='3'
    df.mois[df.mois=='avril']='4'
    df.mois[df.mois=='mai']='5'
    df.mois[df.mois=='juin']='6'
    df.mois[df.mois=='juillet']='7'
    df.mois[df.mois=='août']='8'
    df.mois[df.mois=='septembre']='9'
    df.mois[df.mois=='octobre']='10'
    df.mois[df.mois=='novembre']='11'
    df.mois[df.mois=='décembre']='12'
    df.mois=df.mois.astype('int')
    return df

def generate_submission_file(name, model, scaler, fillna_method):
    df_TEST=Annex.load_test_set()
    df_TEST=Annex.convert_month_to_int(df_TEST)
    df_dummies=pd.get_dummies(df_TEST[['insee']])
    df_TEST_full_qtt=pd.concat([df_TEST,df_dummies],axis=1)
    if fillna_method==True:
        df_TEST_full_qtt.flir1SOL0=df_TEST_full_qtt.flir1SOL0.fillna(0)
        df_TEST_full_qtt.fllat1SOL0=df_TEST_full_qtt.fllat1SOL0.fillna(0)
        df_TEST_full_qtt.flsen1SOL0=df_TEST_full_qtt.flsen1SOL0.fillna(0)
        df_TEST_full_qtt.flvis1SOL0=df_TEST_full_qtt.flvis1SOL0.fillna(0)
        df_TEST_full_qtt.rr1SOL0=df_TEST_full_qtt.rr1SOL0.fillna(0)

    df_TEST_full_qtt=df_TEST_full_qtt.drop(['insee','date'],axis=1)
    X_TEST = scaler.transform(df_TEST_full_qtt)  
    Y_PRED = model.predict(X_TEST)
    
    path='./../data_meteo/'
    df_template=pd.read_csv('./../data_meteo/test_answer_template.csv', header=0, delimiter=";",decimal=",")
    df_template.tH2_obs=Y_PRED
    df_template.to_csv(path+name,sep=';',decimal=',',index=False)
    return 'File %s generated.' %(path+name)