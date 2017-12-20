# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:29:19 2017

Defi
"""

import numpy as np
import pandas as pd
import datetime as dt
from os import listdir
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split
from predictive_imputer import predictive_imputer

pd.options.mode.chained_assignment = None

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

def get_data_imputed(file):
    path='./../data_meteo/'
    df = pd.read_csv(path+file+'.csv', header=None, delimiter=";",decimal=".")
    df=df.iloc[:,1:]
    return df
    
def get_data_raw(scale, add_dummies,var_dummies,TrainTestSplit=True,sz_test=0.3,impute_method='drop',convert_month2int=False,date_method='drop'):
    print('We are addressing your request.')
    if impute_method is not 'drop':
        df=get_data_imputed(impute_method)
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
        
        df=data_preprocessing(df, convert_month2int, add_dummies, var_dummies, date_method)
            
        if impute_method=='drop':
            N_before=df.shape[0]
            df=df.dropna(axis=0)
            N_after=df.shape[0]
            print("%d data points deleted. %0.2f %s"%(N_before-N_after,(N_before-N_after)/N_before*100,'%'))
        
        Y=df['tH2_obs']
        X=df
        X=X.drop(['tH2_obs'],axis=1) ## !!! Date?
        if TrainTestSplit:    
            X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=sz_test,random_state=11)
            X_train.columns=X.columns
            X_test.columns=X.columns
            print('Train size: %d, Test size: %d'%(X_train.shape[0],X_test.shape[0]))
        else:
            X_train=X
            Y_train=Y
            X_test=None
            Y_test=None
        
        
    if scale:
        scaler = StandardScaler()  
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)  
        # Meme transformation sur le test
        X_test = scaler.transform(X_test)
        print('Data scaled')
    else:
        scaler=None
            
    return X_train,X_test,Y_train,Y_test, X, Y, scaler


def data_preprocessing(df, convert_month2int, add_dummies, var_dummies, date_method):
    if convert_month2int:
       df=convert_month_to_int(df)
       print('Months converted to int.')
    
    if add_dummies: 
        for var in var_dummies:
                if df[var].dtypes.type is not pd.core.dtypes.dtypes.CategoricalDtypeType:
                #    print(df[var].dtypes.type)
                    df[var]=df[var].astype('category')
                    print('Feature %s converted into categorical type.'%var)

        df_dummies=pd.get_dummies(df[var_dummies])
        df=pd.concat([df,df_dummies],axis=1)
        df=df.drop(var_dummies,axis=1)
        print('Dummies added.')
    
    if date_method=='drop':
        df=df.drop(['date'],axis=1)
        print('Date dropped.')
    else: 
        if date_method in ['week_number','week_circle']:
            df.date=df.date.apply(lambda x: dt.datetime.strftime(x,"%U"))
            df.date=df.date.astype('int64')
            if date_method is 'week_circle':
                df['cosdate']=np.cos(2*np.pi*df.date/52)
                df['sindate']=np.sin(2*np.pi*df.date/52)
                df=df.drop(['date'],axis=1)
                print('Date transformed in a projection of the week number on a circle.')
            else:
                print('Date transformed in week number.')
    return df
    
    
    
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

def generate_submission_file(name, model, scaler, add_dummies, var_dummies, convert_month2int=False, date_method='drop', fillna_method='zeros' ):
    df_TEST=load_test_set()
    df_TEST=data_preprocessing(df_TEST, convert_month2int, add_dummies, var_dummies, date_method)
    
    if fillna_method=='zeros':
        df_TEST.flir1SOL0=df_TEST.flir1SOL0.fillna(0)
        df_TEST.fllat1SOL0=df_TEST.fllat1SOL0.fillna(0)
        df_TEST.flsen1SOL0=df_TEST.flsen1SOL0.fillna(0)
        df_TEST.flvis1SOL0=df_TEST.flvis1SOL0.fillna(0)
        df_TEST.rr1SOL0=df_TEST.rr1SOL0.fillna(0)
    if fillna_method=='imputer':
        # imputer = predictive_imputer.PredictiveImputer(f_model="RandomForest")
        # df_TEST = imputer.fit(df_TEST).transform(df_TEST)
        # np.savetxt('./../data_meteo/test_data_imputed_weekcircle.csv',df_TEST, delimiter=';')
        df_TEST=np.loadtxt('./../data_meteo/test_data_imputed_weekcircle.csv', delimiter=';')
    if scaler is None:
        X_TEST = df_TEST  
    else:
        X_TEST = scaler.transform(df_TEST)  
        
    if type(model) is list:
        n_models=len(model)-1
        ypreds=[]
        for m in model[:-1]:
            ypred=m.predict(X_TEST)
            ypreds.append(ypred.reshape(len(ypred),1))
        L=len(ypreds[0])
        X_super_TEST=np.concatenate(ypreds,axis=1)
        if model[-1] is None:
            Y_PRED=X_super_TEST.mean(axis=1)
        else:    
            Y_PRED=model[-1].predict(X_super_TEST)
    else:
        Y_PRED = model.predict(X_TEST)
    print(Y_PRED.shape)
    path='./../data_meteo/'
    df_template=pd.read_csv('./../data_meteo/test_answer_template.csv', header=0, delimiter=";")
    df_template.tH2_obs=Y_PRED
    df_template.to_csv(path+name,sep=';',decimal=',',index=False)
    return 'File %s generated.' %(path+name)
    
def combine_submission_files(name, names):
    path='./../data_meteo/'
    dfs=[]
    for n in names:
        df=pd.read_csv(path+n, header=0, delimiter=";",decimal=",")
        dfs.append(df.iloc[:,-1])
        print(dfs[-1].shape)
    prevision=pd.DataFrame(np.mean(dfs,axis=0))
    df.tH2_obs=prevision
    print(df.shape)
    df.to_csv(path+name,sep=';',decimal=',',index=False)
    return 'File %s generated.' %(path+name)