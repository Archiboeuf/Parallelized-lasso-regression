import math
import random as rd
import numpy as np
import pandas as pd
import scipy
import threading
import time

############ Simulation des données ##########################
def simul_normal(mu=0,b=1): #simulation d'une loi normale
    return(np.random.normal(mu,b))

def simul_parametres(n,prop,loi): #simule un paramètre beta avec des coefficients nuls
    parametres=[0]*n
    for i in range(n):
        u=rd.random()
        if u<prop:
            parametres[i]=loi()
    return(pd.Series(parametres))
#parametres=simul_parametres(1000,0.05,simul_normal)

def simul_var_explic(N,n): #simule des variables explicatives
    var=pd.DataFrame(np.random.randn(N,n)) # N observations de taille n
    return(var)
#var=simul_var_explic(10000,1000)

    
def simul_Y(var,parametres): #simule la variable expliquée Y
    non_null_param=parametres[parametres>0]
    non_null_index=non_null_param.index
    N=len(var)
    Y=[0]*N
    for i in range(N):
        Y[i]=np.random.randn(1)
        for j in non_null_index:
            Y[i]+=var.iloc[i][j]*parametres[j]
    return(pd.Series(np.sign(Y)))

def simul_Y2(var,parametres):  #simule Y mais plus rapidement
    N=len(var)
    Y=pd.Series(np.random.randn(N))
    Y+=var.dot(parametres)
    return(np.sign(Y))
#Y=simul_Y2(var,parametres)

def simul_donnees(n,prop,loi,N): #les 3 en une fonction
    parametres=simul_parametres(n,prop,loi)
    var=simul_var_explic(N,n)
    Y=simul_Y2(var,parametres)
    return(Y,var,parametres)

Y,var,parametres=simul_donnees(150,0.05,simul_normal,2000)
beta=parametres.copy()

#################################### Calculs ###################################

##########Log vraisemblance pénalisée et coordinate descent ####################
def T(x,a): # T défini dans l'article (eq. 6)
    return(np.sign(x)*max(abs(x)-a,0))

def calculs_intermediaires(beta,var,Y): # p,w,z définis dans l'article (eq. 4)
    prod_scal=var.dot(beta)
    p=prod_scal.apply(lambda x : 1/(1+math.exp(-x)))
    w=p*(1-p)
    z=(((Y+1)/2)-p)/w
    return(p,w,z)
p,w,z=calculs_intermediaires(beta,var,Y)

def calculs_intermediaires_robustes (beta,var,Y): #robust to math overflow in exponentials
    prod_scal= var.dot(beta)
    def f (prod_scal):
        if prod_scal <35 :
            return(1/(1+math.exp(prod_scal)))
        else :
            return(0.000000000000000000000001)
    p=(-prod_scal).apply(f)
    w=p*(1-p)
    z=(((Y+1)/2)-p)/w
    return(p,w,z)

def q(z,var,beta,delta_beta,j): # q défini dans l'article (eq. 6)
    return(z-var.dot(delta_beta)+(beta[j]+delta_beta[j])*var[j])
    
def L_q (beta, delta_beta, var, Y): # L_q defini dans l'article (eq. 4)
    p, w, z = calculs_intermediaires_robustes(beta, var, Y)
    return(np.sum(w*(z-np.dot(var, delta_beta))**2))
L_q(beta, np.ones(len(beta)), var,    Y)
def L_q_penalized (beta, delta_beta, var, Y,L): # eq. 5
    
    return(L_q(beta, delta_beta, var, Y)+L*np.sum(np.absolute(delta_beta+beta)))
L_q_penalized(beta, np.ones(len(beta)), var,    Y,200)

def delta_betaj_star(beta, delta_beta,var,Y,j,L): # eq. 6
    p,w,z=calculs_intermediaires_robustes(beta,var,Y)
    return(T((var[j]*w*q(z,var,beta,delta_beta,j)).sum(), L)/(var[j]*var[j]*w).sum() -beta[j])

############# Hessienne (eq. 7) ##################
def block_hessian(partition,beta,var,Y,nu=10**-4): #exemple de partition [[i for i in range(int(j*p/5),int((j+1)*p/5))] for j in range(5)]
    def f (prod_scal):
        if prod_scal <35 :
            return(1/(1+math.exp(prod_scal)))
        else :
            return(0.000000000000000000000001)
    p=len(beta)
    hess=pd.DataFrame(np.zeros((p,p)))
    exp=(-Y*var.dot(beta)).apply(f)
    denom=(1+exp)**2
    k = (exp/denom).sum()
    def un_bloc(partie,beta,k,lock): #fonction qui sera exécutée en parallèle
        nonlocal hess #variable commune qui sera modifiée par chaque thread
        copcop=hess.copy()
        hess.loc[partie,partie]=np.outer(beta[partie],beta[partie])*k 
        lock.acquire() #besoin du verrou pour modifier la variable
        hess.select(lambda x : x in partie)[partie]=copcop.select(lambda x : x in partie)[partie]
        lock.release()
    threads=[]
    lock=threading.Lock() #création d'un verrou
    
    for partie in partition : 
        threads.append(threading.Thread(target=un_bloc,
                                        args=(partie,beta,k,lock))) # on crée les threads
    for thread in threads:
        thread.start() #on exécute les threads
    for thread in threads:
        thread.join() #on attend que tous les threads aient fini
    return(hess+nu*np.identity(len(beta)))

beta_test=np.random.rand(len(beta))         
partition=[[i for i in range(int(j*len(beta)/30),int((j+1)*len(beta)/30))] for j in range(30)]
hess=block_hessian(partition,beta_test,var,Y)

############# Linesearch (algorithme 3 de l'article) ##################
def anti_overflow (prod_scal):
    if prod_scal <35 :
        return(1/(1+math.exp(prod_scal)))
    else :
        return(0.000000000000000000000001)
        
def log_vrais(beta,var,Y): # log vraisemblance négative (eq. 3)
    return(np.sum((-Y*var.dot(beta)).apply(anti_overflow)))
    
def f(beta,var,Y,L): # fonction objectif (log vraissemblance négative pénalisée eq.2)
    return(log_vrais(beta,var,Y)+L*np.sum(np.absolute(beta)))

def nabla_vrais(beta,var,Y): #nabla de la log vraisemblance négative
    exp=(-Y*var.dot(beta)).apply(anti_overflow)
    nab=pd.Series([(-Y*beta[i]*exp/(1+exp)).sum() for i in range(len(beta))])
    return(nab)
    

def linesearch(beta,delta_beta,var,Y,L,sufficient_decrease,delta,b,sigma,gamma,partition): #algo 3
    if 1-f(beta+delta_beta,var,Y,L)/f(beta,var,Y,L)<sufficient_decrease: # 1.
        return(1)
    def f_a_minimiser(alpha): #2.
        return(f(beta+alpha*delta_beta,var,Y,L))
    alpha=scipy.optimize.fminbound(f_a_minimiser,delta,1) 
#    hess=np.identity(len(beta)) # éventuellement à remplacer par la Hessienne plus tard

    hess=block_hessian(partition,beta,var,Y)
    nab=nabla_vrais(beta,var,Y)
    D=(np.dot(nab,delta_beta)+gamma*np.dot(beta,np.dot(hess,beta))
        +L*(np.sum(np.absolute(beta+delta_beta))-np.sum(np.absolute(beta))))
    test=False
    while not test:
        if f(beta+delta_beta,var,Y,L)<=f(beta,var,Y,L)+alpha*sigma*D:
            test=True
        else:
            alpha=alpha*b
    return(alpha)
sufficient_decrease=0.05
b=0.95
sigma=0.5
gamma=0.5
delta=0.2
beta = np.zeros(np.shape(var)[1])
delta_beta = np.ones(np.shape(var)[1])
linesearch(beta,delta_beta,var,Y,30,sufficient_decrease,delta,b,sigma,gamma,partition)


##################################Critère de convergence################################
#critère L1
def convergence_indicator (beta_t, beta_t_plus_1, seuil) : 
    return(np.sum(np.abs(beta_t-beta_t_plus_1))>seuil)


###############################parallélisation de la coordinate descent#################################

def coordinate_descent_parallel(partition, beta,var,Y,L,seuil,max_iter): #parallélisation du calcul de delta beta
    delta_beta = pd.Series(np.zeros(np.shape(var) [1]))
    
    def coordinate_descent_choice(beta,var,Y,L, idx,lock,seuil=10**-3): # calcul d'une partie de delta betaj
        nonlocal delta_beta #variable commune qui sera modifiée par chaque thread
        copcop=delta_beta.copy()
        for i in idx :
            copcop[i] = delta_betaj_star(beta, copcop,var,Y,i,L)
            lock.acquire() #besoin du verrou (lock) pour modifier delta_beta (écritures concurrentes sinon)
            delta_beta.loc[[i for i in idx]]=copcop.loc[[i for i in idx]]
            lock.release()
            
    test=True
    count=0   
    lock=threading.Lock() #création du verrou
    while test and count<max_iter: #test de convergence
        delta_beta_t1=delta_beta.copy()
        count+=1
        
        threads=[]
        for partie in partition : 
            threads.append(threading.Thread(target=coordinate_descent_choice,
                                            args=(beta, var, Y, L,partie,lock))) # on crée les threads
        for thread in threads:
            thread.start() #on exécute les threads
        for thread in threads:
            thread.join() #on attend que tous les threads aient fini
        test=convergence_indicator(delta_beta, delta_beta_t1, seuil)
        
    return(delta_beta) 

    
Y,var,parametres=simul_donnees(1000,0.05,simul_normal,2000)
idxs = len(parametres)/5 # définition des groupes d'indices pour les différents threads
idxs = np.around(idxs*(np.array(list(range(5+1))))).astype(int)
idxs=[[i for i in range(idxs[j],idxs[j+1])] for j in range(len(idxs)-1)]

coordinate_descent_parallel(idxs, np.ones(1000),var,Y,200,seuil=10**-2,max_iter=100) # essai


##################################Looped parallel coordinate descent#######################


def coordinate_descent_parallel_loop (variables,Y,L, n_threads, 
                                      sufficient_decrease = 0.5, b = 0.95, sigma = .5, gamma = .5,
                                      delta = .2, seuil=None, max_iter=20) : 
    #on ajoute un intercept
    var = variables.copy()
    var[np.shape(var)[1]]=1

    beta_t = pd.Series(np.ones(np.shape(var) [1]))
    beta_t_plus_1 = pd.Series(0.001*np.ones(np.shape(var) [1]))
    #variable pour éviter le ping_pong

    breaks = len(beta_t_plus_1)/n_threads # définition des groupes d'indices pour les différents threads
    breaks = np.around(breaks*(np.array(list(range(n_threads+1))))).astype(int)
    idxs = [i for i in range(len(beta_t))]
    idxs = [[idxs[i] for i in range(breaks[j],breaks[j+1])] for j in range(len(breaks)-1)]
    
    count=0
    if seuil==None:
        seuil=10**-4*math.sqrt(len(beta_t))
    while convergence_indicator(beta_t, beta_t_plus_1, seuil) and count<max_iter : 

        beta_t = beta_t_plus_1
        

        #coordinate descent
        delta_beta = coordinate_descent_parallel(idxs, beta_t_plus_1,var,Y,L,seuil,15)
        #line search
        alpha=linesearch(beta_t_plus_1,delta_beta,var,Y,L,sufficient_decrease,delta,b,sigma,gamma, idxs)
        #update
        beta_t_plus_1 = beta_t_plus_1 + alpha * delta_beta
        count +=1
        #message
        print("norme L1 de beta : \n {} \n norme L1 de l'écart entre t et t-1 : \n {} \n \n".format(
            sum(abs(beta_t_plus_1)),sum(abs(beta_t_plus_1 - beta_t))))

    return(beta_t_plus_1)
        

#fonctions pour tester

def pred(var,beta):
    return(np.sign(var.dot(beta)))

def variable_selection(beta_estimated, true_parameters) : 
    print(("faux positifs", np.count_nonzero(beta_estimated + true_parameters)-np.count_nonzero(true_parameters)))
    print(("non detectes",  np.count_nonzero(beta_estimated + true_parameters)-np.count_nonzero(beta_estimated)))
    print(("nombre de betas non nuls",  np.count_nonzero(true_parameters)))
    print(("nombre de betas estimés non nuls",   np.count_nonzero(beta_estimated)))


#####################test pas sympathique : 200 individus, 1000 variables

Y1,var1,parametres1=simul_donnees(1000,0.05,simul_normal,200)
sum(abs(parametres1))

#test avec différents lambda
test_1_50 = coordinate_descent_parallel_loop(variables = var1,Y = Y1,L = 50,n_threads =  5)
test_1_20 = coordinate_descent_parallel_loop(variables = var1,Y = Y1,L = 20,n_threads =  5)
test_1_10 = coordinate_descent_parallel_loop(variables = var1,Y = Y1,L = 10,n_threads =  5)

#échantillon de test
vart = simul_var_explic(10000, 1000)
Yt = simul_Y2(vart, parametres1)
vart[np.shape(vart)[1]]=1

#test
Yt_50_pred=pred(vart,test_1_20)
Yt_20_pred=pred(vart,test_1_20)
Yt_10_pred=pred(vart,test_1_10)

#prédiction avec les vrais paramètres
parametres = parametres1.copy()
parametres[1000] = 0
Yt_best_pred=pred(vart,parametres)

#résultats
#On met /2 car une erreur compte 
sum(abs(Yt))
sum(abs(Yt-Yt_50_pred)/2)
sum(abs(Yt-Yt_20_pred)/2)
sum(abs(Yt-Yt_10_pred)/2)
sum(abs(Yt-Yt_best_pred)/2)

variable_selection(test_1_20, parametres1)
variable_selection(test_1_10, parametres1)


#####################test plus syùmpathique : 200 individus, 200 variables
Y2,var2,parametres2=simul_donnees(200,0.05,simul_normal,200)
sum(abs(parametres2))

#test avec différents lambda
test_2_50 = coordinate_descent_parallel_loop(variables = var2,Y = Y1,L = 50,n_threads =  5)
test_2_20 = coordinate_descent_parallel_loop(variables = var2,Y = Y1,L = 20,n_threads =  5)
test_2_10 = coordinate_descent_parallel_loop(variables = var2,Y = Y1,L = 10,n_threads =  5)
test_2_5 = coordinate_descent_parallel_loop(variables = var2,Y = Y1,L = 5,n_threads =  5)
test_2_45 = coordinate_descent_parallel_loop(variables = var2,Y = Y1,L = 4.5,n_threads =  5)

#échantillon de test
vart = simul_var_explic(10000, 200)
Yt = simul_Y2(vart, parametres1)
vart[np.shape(vart)[1]]=1

#test
Yt_50_pred=pred(vart,test_2_20)
Yt_20_pred=pred(vart,test_2_20)
Yt_10_pred=pred(vart,test_2_10)
Yt_5_pred=pred(vart,test_2_5)
Yt_45_pred=pred(vart,test_2_45)

#prédiction avec les vrais paramètres
parametres = parametres1.copy()
parametres[1000] = 0
Yt_best_pred=pred(vart,parametres)

#résultats
sum(abs(Yt))
sum(abs(Yt-Yt_50_pred)/2)
sum(abs(Yt-Yt_20_pred)/2)
sum(abs(Yt-Yt_10_pred)/2)
sum(abs(Yt-Yt_5_pred)/2)
sum(abs(Yt-Yt_45_pred)/2)
sum(abs(Yt-Yt_best_pred)/2)

variable_selection(test_2_20, parametres2)
variable_selection(test_2_10, parametres2)
variable_selection(test_2_5, parametres2)
variable_selection(test_2_45, parametres2)

#test de vitese
start5=time.time() 
test_1_20 = coordinate_descent_parallel_loop(variables = var1,Y = Y1,L = 20,n_threads =  5)
end5=time.time()

start1=time.time()
test_1_20 = coordinate_descent_parallel_loop(variables = var1,Y = Y1,L = 20,n_threads =  1)
end1=time.time()

print(end5-start5)
print(end1-start1)






# Lasso avec sklearn

from sklearn.linear_model import Lasso

model = Lasso(alpha=0.1,normalize=False, max_iter=1e5) 
#à noter que la pénalité L1 se fait sur l'erreur quadratique (paramètres alpha/lambda non comparables)
model.fit(var1,Y1)

vart = simul_var_explic(10000, 1000)
Yt = simul_Y2(vart, parametres1)
y_pred = np.sign(model.predict(vart))

beta_hat=model.coef_
intercept=model.intercept_

print((Yt!=y_pred).sum()) #environ 25% d'erreur





