# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 10:53:27 2017

GENETIC ALGORITHM Library

@author: Quentin Cangelosi

Library intended at first to be generic, meaning that one can just create the class and needed functions to its specific problem
and then use the evolutionary algorithm without any modification to make. This is partially done, some functions specific to the
PRessure Drop Prediction in Coriolis Flow Meter are implemented in the core of the evolutionary algorithm. However this will be updated
when needed.

What can still be improved:
    - learn the impact of adding/removing a node/layer on the cost function and make it easier to evolve to better solutions
    - The mutation threshold is able to go above the maximum possible improvement (ex: threshold=1e-3 eval=1e-4)
    - Learn the hyperparameters (coefficients, threshold, when to stop) by itself
        Mutation coef: add weight to orders of value, for example if coef=0.1 leads to consistent good improvement while 0.5 leads to periodic improv. only, prefer 0.1
    - semi-random?

"""
## Common imports and config
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import time
plt.close("all")
start=time.time()
# plt.rc('text',usetex=True)

## Specific imports
import NNet
import presentation as pr
import Annex
from math import ceil
import pandas as pd
# final_layer='linear_pos'
final_layer='linear'

def generate_rand_array(dim1,dim2):
    return np.random.randn(dim1,dim2) / np.sqrt(dim1)


def evaluation_NN_hybrid(neural_net,X,y,loss_func,args_theo):
    return loss_func(neural_net,X,y,0,args_theo)

def default_loss_func(neural_net,X,y,weight_decay,extra_args):
    predictions=NNet.predict(neural_net,X)
    # return np.linalg.norm(predictions-y,axis=1)
    return np.mean((predictions-y)**2,axis=1)

"""
================================================================================
                            CLASS INDIVIDUAL
================================================================================
The class individual is the class one has to implement for each different problem
considered. It has to follow a certain template depicted below which will allow the
upper classes to access the required information.
Along with the definition of the class is need a function "breeding" and "generate
population"

Template:
class Individual:
    def __init__(self,dim_model,type_layers,max_nodes,max_layers, X,y):

    ## Function __str__ --------------------------------------------------------
    def __str__(self):
        print('')
        return ''

    def evaluate(self):
        self.evaluation=...
        return 0

    def mutate(self,mutation_coef):
        return 0

"""
## Function breed ----------------------------------------------------------
# return a breeded generation which consist of a selection of neural nets (individuals) that we multiply to the portion of population given by the breeding coefficient. The remaining part of the new generation are brand new individuals. Usually, in evolutionary algorithms you would concretely "breed" the individuals by mixing their features together like in real life (with the genes). Here I don't think it makes sense to breed neural nets by taking half of the sum of their layers or nodes or by taking one layer from each parents because I think that mixing two good models with different structures does not lead to a better model. This is especially true for the weigths which are dependent of the structure. Still, it is my personal opinion that I've not proved.
def breed_NN(generation,sz_pop,breeding_coef,selection,args_pop,args_ind,echo_level=1):
    n=0
    N=len(selection)
    # looking for how many times (n) we can multiply the selection of individuals
    while n*N<sz_pop*breeding_coef and n*N<sz_pop:
        n+=1
    # if you don't make a deepcopy there will be parts of the generation and new_generation which will be linked in the memory (thought that as I only keep one generation now it might not be necessary anymore,
    ## To Check)
    new_generation=copy.deepcopy(generation)

    new_generation.population[:n*N]=copy.deepcopy(np.tile(selection,n))
    new_generation.population[n*N:]=generate_population_NN(sz_pop-n*N,*args_pop,args_ind)

    new_generation.nb_ind=sz_pop # because we can change the size of the pop during the iterations

    if echo_level>0:
        print(sz_pop-n*N,'new specimens inserted in population')
    return new_generation


## Function Generate Pop -------------------------------------------------------
# generate a population of neural nets. these neural nets have random weights, number of layers and  nodes per layers within the limits given by the user (max_nodes,max_layers). For more information about the class Neural Net, go to libs/NNet.py
def generate_population_NN(nb_ind, type_layers, max_nodes, max_layers, n_input,n_output,X,y,loss_func,eval_func,eval_args):
    pop=[]
    for i in range(nb_ind):
        nb_layers=np.random.randint(1,max_layers)
        dim_hid=np.random.randint(2,max_nodes,(nb_layers,))
        dim_model=np.concatenate([[n_input],dim_hid,[n_output]])
        if len(type_layers)==1:
            t=np.tile(type_layers,nb_layers)
            t=np.append(t,final_layer)
        pop.append(Ind_Neural_Net(dim_model,t, max_nodes, max_layers, X,y,loss_func,eval_func, eval_args))
    return pop

## Function Similar Struct NN -------------------------------------------------------
# At one point, I wanted to get only different structures and make them evolve together. This avoided a single structure to take the lead and make every solution with a different structure impossible. This is because when you start to have fewer and fewer different structure selected for breeding, the chances for a non selected structure to evolve disappear while the chances for a structure with a single representant in the selection are low compared to the structure which has multiple representations in the selection. Note that this function goes with a features call "doublons". "doublons" is a boolean that you can turn to False to avoid any structure to have multiple representants in the selection of the best nets (again to give a chance to different structures to grow, otherwise a single structure will have at first 1 set of weights that is very good, then 10 differents sets of weights being better than other solutions, and then this structure will just have so much more chances of getting better that it will quickly monopolyse the selection of all future iterations.) after this feature "doublons", I created the function below and a feature called "similar" which, when set False, does not allow very close structures (with only tol [=1 by default] node of difference per layer). The goal of this features was to combine the evolutionary approach with the back propagation approach: while the back prop. approach is much faster at getting to a minimun, it needs to be initialized with the good structure which has many possibilities. By making some iterations with the evolutionary approach, one can get the different structures that seem to work the better and just send these to the back prop approach (with already initialized weights)
def similar_struct_NN(struct,list_structs,tol=1):
    sim=0
    similar=False
    for S in list_structs:
        if len(S)==len(struct):
            for i in range(len(S)):
                if struct[i] in range(-tol,tol+1)+S[i]:
                    sim+=1
            if sim==len(S):
                similar=True
                break
            sim=0
    return similar

## Class Ind_Neural_Net
# this is a class which defines neural nets as individuals for the evolutionary algorithm
# it uses the class NN of the library NNLib/NNet and adds members and functions necessary for the evolutionary classes (evaluation, evolution, mutate(), evaluate() ) and specific to the neural nets
class Ind_Neural_Net:
    def __init__(self,dim_model,type_layers,max_nodes,max_layers,# X,y,
                 loss_func=default_loss_func,
                 eval_func=evaluation_NN_hybrid,
                 eval_args=None): # p_com,DN,moodyapprox
        self.neural_net=NNet.NN(dim_model,type_layers)
        self.neural_net=NNet.initial_W(self.neural_net,True)
        self.neural_net=NNet.initial_b(self.neural_net,True)
        self.evaluation=np.Inf

#        if type(X)==pd.core.frame.DataFrame:
#            X=X.values
#        if type(y)==pd.core.frame.DataFrame:
#            y=y.values
#
        #self.X=X
        #self.y=y
        self.max_nodes=max_nodes
        self.max_layers=max_layers
        self.evolution=0
        # self.moodyapprox=moodyapprox
        # self.DN=DN
        # self.p_com=p_com
        self.loss_func=loss_func
        self.evaluation_function=eval_func
        self.evaluation_args=eval_args

    ## Function __str__ --------------------------------------------------------
    def __str__(self):
        return 'Individual: %d layers, %s( %d nodes) eval=%0.4e, evol=%0.2e'% (len(self.neural_net.dim_model),self.neural_net.dim_model,np.sum(self.neural_net.dim_model),self.evaluation,self.evolution)

    ## Function Evaluate --------------------------------------------------------
    def evaluate(self,X,y):
        ## Half Mean square loss defined in Annex with weight_decay coefficient=0
        self.evaluation=self.evaluation_function(self.neural_net,X,y,self.loss_func,self.evaluation_args)
        self.evaluation=np.mean(self.evaluation) ## /!\
        return 0

    ## Function Mutate --------------------------------------------------------
    # As the breeding phase of neural nets is only multiplying promising neural nets, the mutation phase is entirely responsible for the improvment of the solutions. It is divided in three parts, each responsible for the mutation of one of the features of neural nets: nodes, layers and weights.
    # Here I also switched to an approach of mutation functions that might not be conventional. To accelerate the whole algorithm, I did not wanted to settle for bad mutation too easily so I make a loop that try new mutations until the evolution is greater than a defined threshold (seuil) or after 500 iterations. I later went further by taking the best evolution of the 500 iterations instead of the 500th evolution which can be much worse than a good evolution which was just not under the threshold
    def mutate(self,mutation_coef,seuil):
        evolution=0
        saved=copy.deepcopy(self) # we save the initial state of the neural net
        i=1
        self.evaluate()
        before=self.evaluation
        best_net=[]
        best_so_far=0
        while evolution>=-seuil and i<500:
            i+=1
            self=copy.deepcopy(saved)

            ## MUTATION Weights
            # we update the weights by taking a random number from a normal distribution
            # centered on the current value and with a standard deviation proportional to the mutation coef
            theta=NNet.model_to_theta(self.neural_net.W)
            ##### HERE we divide the mutation coef by 10 because small changes can have very big impacts (not yet proven)
            self.neural_net.W=NNet.theta_to_model(np.random.normal(theta,np.abs(theta*mutation_coef*1e-1)+1e-9,theta.shape),self.neural_net.dim_model)
            theta_b=NNet.model_to_theta(self.neural_net.b)
            self.neural_net.b=NNet.theta_to_b(np.random.normal(theta_b,np.abs(theta_b*mutation_coef*1e-1)+1e-9,theta_b.shape),self.neural_net.dim_model)

            ## MUTATION NB NODES
            # loop over the hidden layers and add/remove one node in the layer or do nothing depending on the mutation coefficient
            for l in range(1,len(self.neural_net.dim_model)-1):
                mutation_nodes=np.random.binomial(1,mutation_coef)*np.sign(np.random.uniform(-1,1))
                # why not a random node in the layer?? ;-)
                if mutation_nodes==1: # Adding one node
                    if self.neural_net.dim_model[l]>=self.max_nodes:
                        continue
                    self.neural_net.dim_model[l]+=mutation_nodes
                    col=generate_rand_array(self.neural_net.dim_model[l+1],1)
                    row=generate_rand_array(1,self.neural_net.dim_model[l-1])

                    self.neural_net.W[l]=np.append(self.neural_net.W[l],col,axis=1)
                    self.neural_net.W[l-1]=np.append(self.neural_net.W[l-1],row,axis=0)
                    self.neural_net.b[l-1]=np.append(self.neural_net.b[l-1],np.reshape(generate_rand_array(1,1),(1,)))

                if mutation_nodes==-1: # Removing one node
                    if self.neural_net.dim_model[l]<=2:
                        continue
                    self.neural_net.dim_model[l]+=mutation_nodes
                    self.neural_net.W[l]=self.neural_net.W[l][:,:-1]
                    self.neural_net.W[l-1]=self.neural_net.W[l-1][:-1,:]
                    self.neural_net.b[l-1]=self.neural_net.b[l-1][:-1]


            ## MUTATION NB LAYERS
            mutation_layer=np.random.binomial(1,mutation_coef)*np.sign(np.random.uniform(-1,1))

            if mutation_layer==1: # Adding one layer
                if len(self.neural_net.dim_model)<self.max_layers:
                    new_layer_sz=np.random.randint(2,self.max_nodes)
                    cut=np.random.randint(1,len(self.neural_net.dim_model)-1)
                    self.neural_net.dim_model=np.concatenate([self.neural_net.dim_model[:cut+1],[new_layer_sz],self.neural_net.dim_model[cut+1:]])

                    new_layer_W= generate_rand_array(self.neural_net.dim_model[cut+1], self.neural_net.dim_model[cut])
                    self.neural_net.W.insert(cut,new_layer_W)
                    self.neural_net.W[cut+1]= generate_rand_array(self.neural_net.dim_model[cut+2],self.neural_net.dim_model[cut+1])
                    new_layer_b=np.zeros(new_layer_sz)
                    self.neural_net.b.insert(cut,new_layer_b)

                    self.neural_net.type_layer=np.tile(self.neural_net.type_layer[0],len(self.neural_net.dim_model)-2)
                    self.neural_net.type_layer=np.append(self.neural_net.type_layer,final_layer)

            if mutation_layer==-1: # Removing one layer
                if len(self.neural_net.dim_model)>3:
                    cut=np.random.randint(1,len(self.neural_net.dim_model)-1)
                    self.neural_net.W[cut-1]=generate_rand_array(self.neural_net.dim_model[cut+1],self.neural_net.dim_model[cut-1])
                    self.neural_net.dim_model=np.concatenate([self.neural_net.dim_model[:cut],self.neural_net.dim_model[cut+1:]])
                    del self.neural_net.W[cut]
                    del self.neural_net.b[cut-1]
                    self.neural_net.type_layer=np.tile(self.neural_net.type_layer[0],len(self.neural_net.dim_model)-2)
                    self.neural_net.type_layer=np.append(self.neural_net.type_layer,final_layer)



            self.evaluate()
            evolution=self.evaluation-before
            if evolution<best_so_far:
                best_so_far=evolution
                best_net=copy.deepcopy(self)
        if evolution<best_so_far:
           self=best_net
           self.evolution=best_so_far
        else:
            self.evolution=evolution
        return self



"""
================================================================================
                            CLASS POPULATION
================================================================================
Containing individuals and functions used by the evolutionary algorithm
"""
class Population:
    def __init__(self,nb_ind,func_generation,args_pop, args_ind):
        self.population=func_generation(nb_ind,*args_pop, args_ind)
        self.nb_ind=nb_ind

    ## Function best -----------------------------------------------------------
    def best(self):
        if self.population==[]:
            sys.exit('population empty')
        best_score=np.Inf
        if self.population[0].evaluation==np.Inf:
            sys.exit('population not evaluated')
        for ind in self.population:
            if ind.evaluation<best_score:
                best_score=ind.evaluation
                best=ind
        return best

    ## Function average --------------------------------------------------------
    def average(self):
        if self.population==[]:
            sys.exit('population empty')
        average_score=0
        average_nodes=0
        average_layers=0
        if self.population[0].evaluation==np.Inf:
            sys.exit('population not evaluated')
        for ind in self.population:
            average_score+=ind.evaluation
            average_layers+=len(ind.neural_net.dim_model)
            average_nodes+=np.sum(ind.neural_net.dim_model)
        return average_score/self.nb_ind,average_layers/self.nb_ind,average_nodes/self.nb_ind

    ## Function __str__ --------------------------------------------------------
    def __str__(self):
        print('Population:')
        print(' - Size:',self.nb_ind)
        print(' - Best:',self.best())
        average_score,average_layers,average_nodes=self.average()
        print(' - Average:',average_layers,'layers (',average_nodes,'nodes), Score:',average_score)
        return '\n'

"""
================================================================================
                            CLASS EVOLUTION
================================================================================
THis is the class containing the evolutionary algorithm. It has to be initialized and then executed via the function execute() which does everything
"""
class Evolution:

    def __init__(self,
                 nb_epoch=10,
                 sz_pop=[100, 50, 25],
                 mutation_coef=[0.5, 0.5, 0.5],
                 func_breed=breed_NN,
                 breeding_coef=[0.5, 0.95, 1],
                 selection_coef=[0.5, 0.1, 0.2],
                 func_generate_Pop=generate_population_NN,
                 args_pop=None,
                 args_ind=None):

        self.nb_epoch=nb_epoch              # Number of epochs to run
        self.sz_population=sz_pop           # Size of the population
        self.mutation_coef=mutation_coef    # Mutation coefficient
        self.selection_coef=selection_coef  # Selection coefficient
        self.breeding_coef=breeding_coef    # Breeding coefficient
        self.generation=[]                  # Generation (one population of individuals)
        if type(self.selection_coef)==list:
            sz_popu=sz_pop[0]
        first_pop=Population(sz_popu, func_generate_Pop, args_pop, args_ind) # Creation of the first generation/population
        self.generation.append(first_pop)
        self.selections=[]                 # Selections are lists of selected individuals
        self.current_gen=0 ### First generation is index 0
        self.args_pop=args_pop             # Arguments to send to the class Population
        self.args_ind=args_ind             # Arguments to send to the class Individual
        self.func_breed=func_breed         # Breeding function specific to the class Individual
        self.bests=[]                      # lists of bests individuals (the difference with selections is that this is cross-generational)
        self.bests_gen=[]                  # Information about the generations where the bests appeared (paired with self.bests)
        self.mut_seuil=1e-3                # Threshold for the evolution wanted during mutation. It is not given by user because it has an automatic evolution
        self.echo_level=1

    ## Function execute --------------------------------------------------------
    # This is the Evolutionary Algorithm.
    # Inputs:
    #   - nb_bests: Number of bests you want to be printed during each epoch
    #   - evolve: Boolean that allow the coefficients (mutation, selection, breeding) to evolve or not
    #   - nb_changes: Number of changes of the coefficients (if evolve==True)
    #   - doublons: Boolean controlling the presence of individuals with the same structure (nodes+layers) but different weights
    #   - similar: Boolean controlling the presence of individuals with a similar structure (nodes+layers)
    def execute(self,nb_bests=5,evolve=True,nb_changes=1,doublons=True,similar=True,loss_threshold=1e-3,plots=False,echo_level=1):
        self.echo_level=echo_level
        if echo_level>0:
            pr.title('Evolutionary Algorithm')
        if type(self.selection_coef)==list: # Useful only if evolve==True
            mode_list=True
            list_sel_coef=self.selection_coef
            list_mut_coef=self.mutation_coef
            list_breed_coef=self.breeding_coef
            list_sz_pop=self.sz_population
            self.selection_coef=list_sel_coef[0]
            self.mutation_coef=list_mut_coef[0]
            self.breeding_coef=list_breed_coef[0]
            self.sz_population=list_sz_pop[0]
            change=1
        else:
            mode_list=False
        tooslow=0
        trialsErrors=[]
        for g in range(self.nb_epoch):
            if echo_level>0:
                pr.big_iter(g+1,'Sel. coef: %0.4f Mut. coef: %0.4f Breed. coef:%0.4f   %d %0.2e' % (self.selection_coef,self.mutation_coef,self.breeding_coef,self.sz_population,self.mut_seuil))
            else:
                if g==0:
                    print('Epoch %d /%d [%s%s]'%(g,self.nb_epoch,'='*g,' '*(self.nb_epoch-g)),end='\r')
                else:
                    print('Epoch %d /%d [%s%s] %0.3e'%(g,self.nb_epoch,'='*g,' '*(self.nb_epoch-g),Bests[0].evaluation),end='\r')

            self.evaluate()
            self.select()

            [Bests,bests_gen]=self.best(nb_bests,doublons,similar)
            trialsErrors.append(float(Bests[0].evaluation))
            if Bests[0].evaluation<loss_threshold: # Last update, since results can be very good very fast when we consider only one model of flow meter and one model
                pr.line()
                if echo_level>0:
                    print('RESULTS UNDER THRESHOLD :',loss_threshold)
                pr.line()
                nb_iters=g
                break
            i=0
            for b in Bests:
                if echo_level>0:
                    print('  - From generation',int(bests_gen[i]),b)
                i+=1
            if evolve and self.current_gen-bests_gen[0]>5: # If no improvment has been made since 5 epochs, we divide the mutation threshold and coefficient, assuming there exist very small changes in the model that lead to improvment
                self.mut_seuil=self.mut_seuil/2
                self.mutation_coef=self.mutation_coef*0.75
                if echo_level>0:
                    print('===== New threshold: %0.2e ====='%self.mut_seuil)
            if evolve and self.current_gen-bests_gen[0]<=1 and Bests[0].evolution<=-self.mut_seuil: # On the opposite, if the last epoch led to improvement of the best solution greater than the current threshold, we try to challenge the evolutions a bit more by increasing the mutation threshold by 20%
                self.mut_seuil=self.mut_seuil*1.2
                if echo_level>0:
                    print('===== New threshold: %0.2e ====='%self.mut_seuil)
            if evolve and self.current_gen-bests_gen[0]<=1: # if the last epoch led to improvement but under the mutation threshold (even if it was greater) maybe it is because the mutation coefficient is too small and don't allow bigger changes/improvements. We record that and by the 5th time in a row that this situation persist, we interact and increase the mutation coefficient by 20%
                tooslow+=1
                if tooslow>5:
                    self.mutation_coef=min(1,self.mutation_coef*1.2)
            else:
                tooslow=0

            self.breed(doublons,similar)
            self.mutate()


            if evolve: # Evolution of the coefficients either passively (when the given coefficients were single numbers we change them the number of times defined by nb_changes and with a constant change coef (divided by 2, or 10% increase for the breeding coef) or actively (we defined when the changes will occur and what the new coefs will be)
                if mode_list and g+1 in nb_changes:
                    if echo_level>0:
                        print('\n------------\n------------\nCHANGING THE EVOLUTION PARAMETERS\n------------\n------------\n')
                    self.selection_coef=list_sel_coef[change]
                    self.mutation_coef=list_mut_coef[change]
                    self.breeding_coef=list_breed_coef[change]
                    self.sz_population=list_sz_pop[change]
                    change+=1
                else:
                    if mode_list==False and (g%round(self.nb_epoch/nb_changes))==(round(self.nb_epoch/nb_changes)-1):
                        if echo_level>0:
                            print('\n------------\n------------\nCHANGING THE EVOLUTION PARAMETERS\n------------\n------------\n')
                        self.selection_coef=self.selection_coef/2
                        self.mutation_coef=self.mutation_coef/2
                        self.breeding_coef=min(self.breeding_coef*1.1,1)
                if g+1==10:
                    doublons=True
                    similar=True
            nb_iters=g+1


        self.evaluate()
        self.select()
        [Bests,bests_gen]=self.best(5,doublons,similar)
        if plots:
            self.plot()
        if echo_level>0:
            print('----------- END ------------')
        print('')
        return trialsErrors # the nb of epochs ran is returned for information (though they are all printed and there is not option for not prints yet but if you run multiple simulations you won't be able to know when the previous stopped)

    ## Function evaluate -------------------------------------------------------
    def evaluate(self):
        for ind in self.generation[0].population:
            ind.evaluate()
        return 0

    ## Function mutate ---------------------------------------------------------
    def mutate(self):
        evolution=0
        i=0
        N=self.generation[0].nb_ind
        nb_failed=0
        for ind in self.generation[0].population:
            if self.echo_level>0:
                print('Mutation %d/%d, %d failed\r'%(i+1,N,nb_failed),end='')
            self.generation[0].population[i]=ind.mutate(self.mutation_coef,self.mut_seuil)
            evolution+=self.generation[0].population[i].evolution
            if self.generation[0].population[i].evolution>=0:
                nb_failed+=1
            i+=1
        if nb_failed>N*0.75: # Pareto law, if 25% of the population succeeded to improve over the threshold, we do not touch anything and continue with the same configuration. If less than 25% succeeded, we consider changing first the mutation coefficient by reducing it of 25%. Then if the mutation coef gets too low, we divide the mutation threshold by 2 and reset the mutation coef
            self.mutation_coef=self.mutation_coef*0.75
            if self.mutation_coef<1e-3:
                self.mut_seuil=self.mut_seuil/2
                self.mutation_coef=0.5
        if nb_failed<N/10: # if 90% if the population suceeded, we increase the mutation coefficient to get even greater improvements. If it continues to be like that and the mutation coefficient is big, we also increase the mutation threshold
            self.mutation_coef=min(self.mutation_coef*1.1,1)
            if self.mutation_coef>0.9:
                self.mut_seuil=self.mut_seuil*2
                if self.echo_level>0:
                    print('------ New threshold: %0.2e -----'%self.mut_seuil)
        return 0

    ## Function select ---------------------------------------------------------
    # We sort the population by the evaluation values of its individuals and take the best part of size defined by the selection coefficient
    def select(self):
        def getKey(item):
            return item.evaluation
        sorted_selection=sorted(self.generation[0].population,key=getKey)
        self.selections.append(sorted_selection[:ceil(self.selection_coef*self.sz_population)])
        if self.echo_level>0:
            print(len(self.selections[-1]),' selected individuals (best:',self.selections[-1][0],') average:',self.average_in_selection(self.current_gen)[0])
        return 0

    ## Function breed ----------------------------------------------------------
    # Normally in evolutionary algorithm you breed the individuals of the selection of the previous generation. Here I changed it a bit by breeding instead the best individuals of all generations. Remember that you mutate all the individuals and as they seem to be sensitive to small changes (which can lead to big changes in the evaluation) it would be possible to have a very good individual which after mutating never get as good as it was. Thus we make very good individual "immortal", they will be reintroduced until they improve or get beaten by better ones
    def breed(self,doublons,similar):
        # selection=self.selections[self.current_gen]
        [selection,no_use]=self.best(len(self.selections[self.current_gen]),doublons,similar)

        self.generation[0]=self.func_breed(self.generation[0],self.sz_population,self.breeding_coef,selection, self.args_pop, self.args_ind, self.echo_level)

        self.current_gen+=1
        return 0

    ## Function best -----------------------------------------------------------
    # Manage to get the bests individuals with or without the constraint that there should be no doublons or no similar structures.
    def best(self,nb,doublons=True,similar=True):

        list_sel=[]
        list_gen=[]
        def getKey(item):
            return item[0].evaluation

        if self.bests==[] or len(self.bests)<=self.current_gen or len(self.bests[self.current_gen])<nb:
            if self.bests==[] or nb>len(self.bests[-1]):
                gen=1
                for sel in self.bests:
                    list_sel=np.concatenate([list_sel,sel])
                    list_gen=np.concatenate([list_gen,np.ones(len(sel))*gen])
                    gen+=1
            else:
                list_sel=self.bests[-1]
                list_gen=self.bests_gen[-1]

            list_sel=np.concatenate([list_sel,self.selections[self.current_gen]])
            list_gen=np.concatenate([list_gen,np.ones(len(self.selections[self.current_gen]))*self.current_gen+1])

            list_sel=np.reshape(list_sel,(len(list_sel),1))
            list_gen=np.reshape(list_gen,(len(list_gen),1))
            lists=np.concatenate([list_sel,list_gen],axis=1)
            lists=sorted(lists,key=getKey)
            lists=np.reshape(lists,(len(list_sel),2))


            if not doublons or not similar:
                i=0
                list_ind=[]
                list_dims=[]
                for [ind,g] in lists:
                    if list(ind.neural_net.dim_model) in list_dims:
                        list_ind.append(i)
                    else:
                        list_dims.append(list(ind.neural_net.dim_model))
                    i+=1
                lists=np.delete(lists,list_ind,axis=0)

            if not similar:
                i=0
                list_ind=[]
                list_dims=[]
                for [ind,g] in lists:
                    if similar_struct_NN(ind.neural_net.dim_model,list_dims):
                        list_ind.append(i)
                    else:
                        list_dims.append(list(ind.neural_net.dim_model))
                    i+=1
                lists=np.delete(lists,list_ind,axis=0)

            best_nn=lists[:min(nb,len(lists)),0]
            best_gen=lists[:min(nb,len(lists)),1]

            if len(self.bests)>self.current_gen:
                del self.bests[-1]
                del self.bests_gen[-1]
            self.bests.append(best_nn)
            self.bests_gen.append(best_gen)
        else:
            best_nn=self.bests[self.current_gen][:min(nb,len(self.bests[self.current_gen]))]
            best_gen=self.bests_gen[:min(nb,len(self.bests[self.current_gen]))]
        return [best_nn,best_gen]


    ## Function average_in_selection -------------------------------------------
    def average_in_selection(self,gen):
        av_s=0
        av_l=0
        av_n=0
        sz=len(self.selections[gen])
        for ind in self.selections[gen]:
            av_s+=ind.evaluation
            av_l+=len(ind.neural_net.dim_model)
            av_n+=np.sum(ind.neural_net.dim_model)
        return av_s/sz,av_l/sz,av_n/sz

    ## Function plot -----------------------------------------------------------
    def plot(self):
        bests=[]
        av_score=[]
        av_layers=[]

        av_nodes_per_l=[]
        gen=0
        for sel in self.selections:
            bests.append(sel[0].evaluation)
            av_s,av_l,av_n=self.average_in_selection(gen)
            av_score.append(av_s)
            av_layers.append(av_l)
            av_nodes_per_l.append(av_n/av_l)
            gen+=1
        plt.figure()

        plt.subplot(3,1,1)
        plt.semilogy(bests,'r-+')
        plt.semilogy(av_score,'k')
        plt.semilogy(np.argmin(bests),min(bests),'go')
        plt.legend(['lowest in selection','average in selection','lowest over all'])
        plt.title('Loss per generation')
        plt.xlabel('generations')
        plt.ylabel('loss J(X,t) (log scale)')

        plt.subplot(3,1,2)
        plt.plot(av_layers,'b')
        plt.title('Average nb of Layers in selection per generation')
        plt.xlabel('generations')
        plt.ylabel('layers')

        plt.subplot(3,1,3)
#        plt.plot(av_nodes,'b')
        plt.plot(av_nodes_per_l,'b')
        plt.title('Average nb of nodes per layer in selection')
        plt.xlabel('generations')
        plt.ylabel('nodes')
        # plt.show()
        return 0

    ## Function __str__ --------------------------------------------------------
    def __str__(self):
        print('Evolution')
        print(' -',self.nb_epoch,'generations of',self.sz_population,'individuals')
        print(' - Mutation coefficient',self.mutation_coef)
        print(' - Breeding coefficient',self.breeding_coef)
        print(' - Selection coefficient',self.selection_coef)
        print(' - Generations:')
        k=1
        for i in self.generation:
            print('      ',k,'th Generation:\n',i)
            k+=1
        return '\n'
