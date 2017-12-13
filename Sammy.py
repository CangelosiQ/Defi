# -*- coding: utf-8 -*-
"""
Created on Dec 2 10:53:27 2017

MACHINE LEARNING OPTIMIZER

@author: Quentin


"""

import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import sys
import copy
import copy

Methods_names={LinearSVC:"LinearSVC",
         SVC:"SVC",
         LogisticRegression:"LogisticRegression",
         XGBClassifier:"XGBClassifier",
         MLPClassifier:"MLPClassifier",
         MLPRegressor:"MLPRegressor",
         RandomForestClassifier:"RandomForestClassifier",
         AdaBoostClassifier:"AdaBoostClassifier", 
         GradientBoostingClassifier:"GradientBoostingClassifier",
         KNeighborsClassifier:"KNeighborsClassifier"
         }

Methods={"LinearSVC":LinearSVC,
         "SVC":SVC,
         "LogisticRegression":LogisticRegression,
         "XGBClassifier":XGBClassifier,
         "MLPClassifier":MLPClassifier,
         "MLPRegressor":MLPRegressor,
         "RandomForestClassifier":RandomForestClassifier,
         "AdaBoostClassifier":AdaBoostClassifier, 
         "GradientBoostingClassifier":GradientBoostingClassifier,
         "KNeighborsClassifier":KNeighborsClassifier
         }
         
"""
################################################################################
							CLASS result
################################################################################
"""
class result:
    def __init__(self, method, params, cv_score):
        self.method=method
        self.cv_score=cv_score
        self.params=params
        
    def __str__(self):
        print('%s:\nParameters:'%Methods_names[self.method])
        print(self.params)
        print('CV Score:   %0.3e' %self.cv_score)
        return ''
        
    def __eq__(self, other):
        return (self.__dict__ == other.__dict__)



"""
################################################################################
							CLASS Results
################################################################################
"""
class Results:

## INIT
## ===============================
    def __init__(self,list=[]):
        self.nb_results=len(list)
        self.list_results=list
        self.total_time=0
        self.wrong_configs=[]


## ADD_RESULT
## ===============================
    def add_result(self,result):
        #print("Adding result %s\n"%result)
        self.list_results.append(result)
        self.nb_results+=1

        
## STR
## ===============================
    def __str__(self):
        print('%d Results:' %self.nb_results)
        for r in self.list_results:
            print(r)
        return '\n'


## SORT
## ===============================
    def sort(self,reverse=False):
        #print("sorting %d results"%self.nb_results)
        results_sorted=sorted(self.list_results, key=operator.attrgetter('cv_score'), reverse=reverse)
        #print("%d results are sorted \n"%len(results_sorted))
        return Results(results_sorted)


## BEST_SCORE
## ===============================
    def best_score(self,max=True):
        return sorted(self.list_results, key=operator.attrgetter('cv_score'), reverse=max)[0].cv_score


## BESTS
## ===============================
    def bests(self,max=True,nb=1):
        return sorted(self.list_results, key=operator.attrgetter('cv_score'), reverse=max)[:nb]

        
## ISNAN
## ===============================
    def isnan(self,obj):
        if type(obj) is Results:
            l=[]
            for r in obj.list_results:
                l.append(np.isnan(r.cv_score))
        return l

        
## SAY_HELLO
## ===============================
    def say_hello(self):
        if self.nb_results==0:
            print('Welcome, I am Sammy Nimizz.\nMay I help you?')
        else:
            print('Welcome back, I am Sammy Nimizz.\nI have already run %d models and got a best score of %0.3e.\n May I help you? (I am %d secondes old)'%(self.nb_results,self.best_score(),self.total_time))

            
## RUN
## ===============================
    # def run(self,Xtrain,ytrain,method,param,time_limit=3600):
        # start=time.time()
        # for parm in ParameterGrid(param):
            # if self.check_isnew(method,parm):
                # ts = time.time()
                # scores=cross_val_score(method(**parm), Xtrain, ytrain, cv=10)
                # te = time.time()
                # self.total_time+=(te-ts)
                # self.add_result(result(method,parm,scores.mean()))
                # if (time.time()-start)>(time_limit):
                    # print('Your time limit was reached.\n See you soon!')
                    # break
            # else:
                # print('This configuration was already ran.',parm)

            
## CHECK_ISNEW
## ===============================
    def check_isnew(self,method,param):
        res=True
        for r in self.list_results:
            if method==r.method:
                if param==r.params:
                    res= False
                    break
        for p in self.wrong_configs:
                if p==param:
                    res=False
                    break
        #print('Method is new? %s \n'%res,method,param)
        return res
                  
                                
## ADD_WRONG_CONFIG
## ===============================
    def add_wrong_config(self,param):
        #print("Adding wrong config\n")
        self.wrong_configs.append(param)
        return 0

        
## RANDOM_START
## ===============================
    def random_start(self, Xtrain, ytrain, nb_run=10, methods='all'):
        params={}
            
        for run in range(nb_run):
            print("Run %d/%d"%(run+1,nb_run))
            if methods is not 'all':
                if type(methods) is not list:
                    method=methods
                else:
                    method=methods[np.random.randint(0,len(methods)-1)]
                if method not in params.keys():
                    params.update({method:get_param_choices(method)})
                ind=np.random.randint(0,len(ParameterGrid(params[method]))-1)
                new_param=ParameterGrid(params[method])[ind]
            else:
                ind_method=np.random.randint(0,len(Methods)-1)
                method=Methods[list(Methods.keys())[ind_method]]
                print(method)
                if method not in params.keys():
                    params.update({method:get_param_choices(method)})
                ind=np.random.randint(0,len(ParameterGrid(params[method]))-1)
                new_param=ParameterGrid(params[method])[ind]
            if self.check_isnew(method,new_param):
                self.run_cross_val(Xtrain,ytrain, method,new_param)
    
    
## OPTIMIZE
## ===============================
    def optimize(self,Xtrain,ytrain, nb_epoch=5,method='all',goal='max',coef_selection=0.2):
        print('I want to improve myself!')
        if self.nb_results<10:
            print("I am too young to be optimized, let me try some randomness first.\n")
            self.random_start(Xtrain,ytrain,Xtest,ytest, nb_run=10,methods=method)
        for epoch in range(nb_epoch):
            print('\nEpoch',epoch+1)
            pop=copy.deepcopy(self.select(coef_selection,method,goal))
            pop=pop.breed()
            pop=pop.mutate()
            self.update(pop,Xtrain,ytrain)


## SELECT
## ===============================
    def select(self,coef_selection,method,goal):
        nb_selected=round(self.nb_results*coef_selection)
        if goal=='max':
            bests=self.sort(True)
        else:
            bests=self.sort(False)
        #if method!='all':
            #bests=bests.restrict_to_methods(method)
            #nb_selected=min(bests.nb_results,nb_selected)
        print("I selected the %d best solutions I know."%(nb_selected))
        selection=bests.list_results[:nb_selected]
        out=Results(selection)
        # print("Selection: %s \n"%out)
        return out

    
## BREED
## ===============================
    def breed(self):
        return self

        
## MUTATE
## ===============================
    def mutate(self):
        print('Mutation!')
        params={}
        for r in self.list_results:
            if r.method not in params.keys():
                params.update({r.method:get_param_choices(r.method)})
            try:
                ind=list(ParameterGrid(params[r.method])).index(r.params)
                new_ind=np.random.randint(max(0,ind-3),min(ind+3,len(list(ParameterGrid(params[r.method])))))
                new_param=ParameterGrid(params[r.method])[new_ind]
                
            except:
                print("An error during mutation.", sys.exc_info()[0])
                raise
                new_ind=np.random.randint(0,len(ParameterGrid(params[r.method]))-1)
                new_param=ParameterGrid(params[r.method])[new_ind]
            r.params=new_param
            r.cv_score=0
        return self

        
## UPDATE
## ===============================
    def update(self,pop, Xtrain,ytrain):
        #print('Evaluation...')
        for r in pop.list_results:
            if self.check_isnew(r.method,r.params):
                self.run_cross_val(Xtrain,ytrain, r.method,r.params)

    
## RUN_CROSS_VAL
## ===============================
    def run_cross_val(self,Xtrain,ytrain, method,param):
        print('Running %s with parameters: %s'%(Methods_names[method],param))            
        ts = time.time()

        try:
            scores=cross_val_score(method(**param), Xtrain, ytrain, cv=10)
            score=scores.mean()
            print('Score: %0.3e\n'%score)
            self.add_result(result(method,param,score))
        
        except TypeError as err:
            print("Type error: {0}".format(err))
        except:
            print("An error occured with this configuration.%s \n"% sys.exc_info()[0])
            self.add_wrong_config(param)
        te = time.time()
        self.total_time+=(te-ts)  

            
## RESTRICT_TO_METHODS
## ===============================
    def restrict_to_methods(self,methods):
        if type(methods) is not list:
            methods=[methods]
        New_Results=Results()
        for r in self.list_results:
            if r.method in methods:
                New_Results.add_result(r)
        return New_Results


## ELIMINATE_DOUBLES
## ===============================
    def eliminate_doubles(self):
        for r in self.list_results:
            pass

            
## TEST
## ===============================
    def test(self, Xtrain, ytrain, Xtest, ytest, nb_test=5):
        nb_test=min(nb_test,self.nb_results)
        l_bests=self.bests(nb=nb_test)
        for r in l_bests:
            print('Testing %s with parameters: %s'%(Methods_names[r.method],r.params))
            model=r.method(**r.params)
            predictor=model.fit(Xtrain,ytrain)
            score = predictor.score(Xtest, ytest)
            print("Test Score=",score)

## BASIC_TOUR
## ===============================
    def basic_tour(self, Xtrain, ytrain):
        for method in Methods_names:
            self.run_cross_val(Xtrain,ytrain, method,{})

## BASIC_TOUR
## ===============================

            
## ANALYSE
## ===============================
    def analyse(self,method='all'):
        if method is not 'all':
            # convert list_results to data frame so that we can run a LogisticRegression on it
            #list=[]
            names_param=list(*(self.list_results[0].param))
            for r in self.list_results:
                print(r.method)
                print(**(r.params))
                print(r.cv_score)
            
            #df=pd.DataFrame(np.array(list).reshape(self.nb_results,len(list[0]), columns = ['method',names_param,'score'])
            #print(df)
    
"""
################################################################################
							Other Functions
################################################################################
"""    
def get_param_choices(method):
    if method is LinearSVC:
        param=[{"penalty":['l2','l1'],
        "loss":['squared_hinge','hinge'],
        "dual":[True,False],
        "tol":[0.0001],
        "C":np.concatenate([np.linspace(0,5,21),np.linspace(5.5,10,10)]),
        "multi_class":['ovr'],#'crammer_singer'
        "fit_intercept":[True,False],
        "intercept_scaling":[1],
        "class_weight":[None]
        }] 
        
    if method is SVC:
        param=[{"C":np.concatenate([np.linspace(0,1,21),np.linspace(1.1,100,21)]),
            "kernel":['rbf','poly', 'sigmoid', 'precomputed'],
            "degree":range(2,11),
            "gamma":['auto'], 
            "coef0":[0.0], 
            "probability":[False], 
            "shrinking":[True,False], 
            "tol":[0.001],
            "cache_size":[200], 
            "class_weight":[None],
          }]
    if method is LogisticRegression:
        param=[{"penalty":['l2','l1'],
        "dual":[True,False],
        "tol":[0.0001],
        "C":np.linspace(0.5,10,20),
        "fit_intercept":[True],#,False
        "intercept_scaling":[1],
        "class_weight":[None],
        "random_state":[None],
        "solver":['liblinear','newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        "max_iter":[100],
        "multi_class":['ovr'],#'multinomial'
          }]
    
    if method is XGBClassifier:
        param=[{"max_depth":range(2,15),
        "learning_rate":[0.1], 
        "n_estimators":[100], #np.linspace(50,500,10)
        "silent":[True], 
        "objective":['multi:softmax'], 
        "booster":['gbtree'], 
        "n_jobs":[4], 
        "nthread":[None], 
        "gamma":[0], 
        "min_child_weight":[1], 
        "max_delta_step":[0], 
        "subsample":[1], 
        "colsample_bytree":[1], 
        "colsample_bylevel":[1], 
        "reg_alpha":[0], 
        "reg_lambda":[1], 
        "scale_pos_weight":[1], 
        "base_score":[0.5]
        }]

    if method is MLPClassifier:
        param=[{"hidden_layer_sizes":nn_hidden_space(5),
            "activation":['relu'],
            "solver":['adam'], 
            "alpha":[0.0001], 
            "batch_size":['auto'], 
            "learning_rate":['constant'], 
            "learning_rate_init":[0.001], 
            "power_t":[0.5],    
            "max_iter":[200], 
            "shuffle":[True], 
            "random_state":[None], 
            "tol":[0.0001], 
            "verbose":[False], 
            "warm_start":[False], 
            "momentum":[0.9], 
            "nesterovs_momentum":[True], 
            "early_stopping":[False], 
            "validation_fraction":[0.1], 
            "beta_1":[0.9], 
            "beta_2":[0.999], 
            "epsilon":[1e-08]
        }]
    
    if method is MLPRegressor:
        param=[{"hidden_layer_sizes":nn_hidden_space(5),
            "activation":['relu','logistic','tanh'],
            "solver":['adam'], 
            "alpha":[0.0001], 
            "batch_size":['auto'], 
            "learning_rate":['constant'], 
            "learning_rate_init":[0.001,0.01], 
            "power_t":[0.5],    
            "max_iter":[200], 
            "shuffle":[True], 
            "random_state":[None], 
            "tol":[0.0001], 
            "verbose":[False], 
            "warm_start":[False], 
            "momentum":[0.9,0.8,0.95], 
            "nesterovs_momentum":[True], 
            "early_stopping":[False,True], 
            "validation_fraction":[0.1], 
            "beta_1":[0.9], 
            "beta_2":[0.999], 
            "epsilon":[1e-08]
        }]
    
    if method is RandomForestClassifier:
        param=[{"n_estimators":range(10,200,20),
            "criterion":['gini','entropy'],
            "max_depth":range(2,15),
            "min_samples_split":[1e-2,1e-3,5e-2],
            "min_samples_leaf":[1],
            "min_weight_fraction_leaf":[0.0],
            "max_features":['auto'], 
            "max_leaf_nodes":[None], 
            "min_impurity_decrease":[0.0], 
            "min_impurity_split":[None],
            "bootstrap":[True], 
            "oob_score":[False],
            "n_jobs":[1], 
            "random_state":[None], 
            "verbose":[0], 
            "warm_start":[False], 
            "class_weight":[None]
        }]
    
    if method is AdaBoostClassifier:
        param=[{"base_estimator":[None],
            "n_estimators":range(10,200,20), 
            "learning_rate":[1.0], 
            "algorithm":['SAMME.R'], 
            "random_state":[None]
        }]
        
    if method is GradientBoostingClassifier:
        param=[{"loss":['deviance'], 
            "learning_rate":[0.1], 
            "n_estimators":[50,100,150,200], 
            "subsample":[1.0], 
            "criterion":['friedman_mse'], 
            "min_samples_split":[1e-2,1e-3,5e-2], 
            "min_samples_leaf":[1], 
            "min_weight_fraction_leaf":[0.0], 
            "max_depth":range(2,15), 
            "min_impurity_decrease":[0.0], 
            "min_impurity_split":[None], 
            "init":[None], 
            "random_state":[None], 
            "max_features":[None], 
            "verbose":[0], 
            "max_leaf_nodes":[None], 
            "warm_start":[False], 
            "presort":['auto']
        }]
        
    if method is KNeighborsClassifier:
        param=[{"n_neighbors":range(2,15), 
            "weights":['uniform','distance'], 
            "algorithm":['auto'], 
            "leaf_size":[30], 
            "p":[2,1], 
            "metric":['minkowski'], 
            "metric_params":[None], 
            "n_jobs":[1]
        }]
        
        
    if method not in list(Methods_names.keys()):
        print("Method %s not implemented."%method,type(method))

    return param

def nn_hidden_space(max_layers=10,max_node_per_layer=200):
    space=[]
    for n_layer in range(max_layers):
        for n_node in np.concatenate([range(10,50,10),range(50,200,50)]):
            space.append(tuple(np.repeat(n_node,n_layer+1)))
    return space


    
