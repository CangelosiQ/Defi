# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:29:19 2017

@author: Quentin
"""

import numpy as np
import pandas as pd
import datetime as dt
from os import listdir

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

def get_data_raw():
    listdir('./../data_meteo/')
    list_files=np.empty(36, dtype='|U12')
    i=0
    for fichier in listdir('./../data_meteo/'):
        if 'train' in fichier:
            list_files[i]=fichier
            i=i+1

    df_tot=pd.DataFrame()
    for file in list_files:
        df_tot=pd.concat([df_tot,open_and_transform(file)])

    df_tot=df_tot.sort_values(by=['ech','date'],ascending=True)

    return df_tot

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
    #df['flvis1SOL0'] = np.float64(df['flvis1SOL0'])
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

def generate_submission_file(name, Y_PRED):
    path='./../data_meteo/'
    df_template=pd.read_csv('./../data_meteo/test_answer_template.csv', header=0, delimiter=";",decimal=",")
    df_template.tH2_obs=Y_PRED
    df_template.to_csv(path+name,sep=';',decimal=',',index=False)
    return 'File %s generated.' %(path+name)