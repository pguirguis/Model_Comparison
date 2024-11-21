# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 18:35:05 2023

@author: peter
"""

import numpy as np
import pandas as pd
# from lmfit import Model
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
# from scipy.optimize import minimize
# from sklearn.model_selection import train_test_split
# from concurrent.futures import ProcessPoolExecutor
# import concurrent.futures
# from iapws import IAPWS97

data = pd.read_excel(r'Molec.xlsx', "Fitting", engine= 'openpyxl', header= 1)

x = np.vstack((data["Other Carbs wt%"].to_numpy(),data["Cellulose wt%"].to_numpy(),data["Starch wt%"].to_numpy(),data["Hemicellulose wt%"].to_numpy(),data["Protein wt%"].to_numpy(),data["Lipids wt%"].to_numpy(),data["Lignin wt%"].to_numpy(),data["Sa wt%"].to_numpy(),data["AA wt%"].to_numpy(),data["FA wt%"].to_numpy(),data["Ph wt%"].to_numpy(),data["Ash wt%"].to_numpy(),data["Solid content (w/w) %"].to_numpy(),data["Total time (min)"].to_numpy(),data["Temperature (C)"].to_numpy(),data["b"].to_numpy()))

X = np.transpose(x)

y = np.vstack((data["Solids wt%"].to_numpy(),data["Biocrude wt%"].to_numpy(),data["Aquous wt%"].to_numpy(),data["Gas + Loss"].to_numpy()))

y = np.transpose(y)

data_pred = pd.read_excel(r'Molec.xlsx', "Predict", engine= 'openpyxl', header= 1)

# data_pred = data_pred[data_pred["Total Lignin wt%"] == 0]

x_pred = np.vstack((data_pred["Other Carbs wt%"].to_numpy(),data_pred["Cellulose wt%"].to_numpy(),data_pred["Starch wt%"].to_numpy(),data_pred["Hemicellulose wt%"].to_numpy(),data_pred["Protein wt%"].to_numpy(),data_pred["Lipids wt%"].to_numpy(),data_pred["Lignin wt%"].to_numpy(),data_pred["Sa wt%"].to_numpy(),data_pred["AA wt%"].to_numpy(),data_pred["FA wt%"].to_numpy(),data_pred["Ph wt%"].to_numpy(),data_pred["Ash wt%"].to_numpy(),data_pred["Solid content (w/w) %"].to_numpy(),data_pred["Total time (min)"].to_numpy(),data_pred["Temperature (C)"].to_numpy(),data_pred["b"].to_numpy()))

X_pred = np.transpose(x_pred)

y_pred = np.vstack((data_pred["Solids wt%"].to_numpy(),data_pred["Biocrude wt%"].to_numpy(),data_pred["Aquous wt%"].to_numpy(),data_pred["Gas + Loss"].to_numpy()))

y_pred = np.transpose(y_pred)

global res_min



def r_sq(observed, predicted):
    # Calculate the mean of observed data`
    mean_observed = np.nanmean(observed)

    # Total sum of squares (proportional to the variance of the observed)
    ss_tot = np.nansum((observed - mean_observed) ** 2)

    # Residual sum of squares (sum of squares of residuals)
    ss_res = np.nansum((observed - predicted) ** 2)

    # R-squared
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared


def analysis(exp, pred, N_ks,mod_name,suffix,count):
    exp = np.reshape(exp,-1)
    pred = np.reshape(pred,-1)
    plt.scatter(pred, exp)
    plt.plot([0, 100], [0, 100], color='black')
    plt.plot([0, 100], [10, 110], '--', color='red')
    plt.plot([0, 100], [-10, 90], '--', color='red')
    plt.xlabel('Calculated value')
    plt.ylabel('Experimental data')
    plt.title(mod_name+" "+suffix)
    plt.savefig(mod_name+"_"+suffix+'.png')
    plt.clf()
    Res = np.subtract(exp, pred)
    rel_abs_Res = np.abs(np.divide(Res,exp))
    RSS = sum([item*item for item in Res if str(item) != "nan"  and str(item) != "inf" and type(item) != bool ])
    res_clean = [item for item in Res if str(item) != "nan"  and str(item) != "inf"  and type(item) != bool]
    rel_clean = [item for item in rel_abs_Res if str(item) != "nan"  and str(item) != "inf"  and type(item) != bool]
    abs_res = [abs(item) for item in res_clean ]
    n_point = len(res_clean)
    med_Res = np.nanmedian(res_clean)
    # mean_Res = np.nanmean(res_clean)
    std_Res = np.std(res_clean)
    mean_abs_Res = np.nanmean(abs_res)
    mape = np.nanmean(rel_clean)*100
    med_abs_res = np.nanmedian(abs_res)
    med_rel = np.nanmedian(rel_clean)*100
    AIC = 2*N_ks + np.log(RSS/n_point)*n_point
    BIC = N_ks*np.log(n_point) +np.log(RSS/n_point)*n_point
    num_5 = len([item for item in abs_res  if item <= 5])/n_point*100
    num_10 = len([item for item in abs_res if item <= 10])/n_point*100
    num_75 = len([item for item in abs_res if item >= 75])/n_point*100
    num_100_rel = len([item for item in rel_clean if item >= 1])/n_point*100
    med_Res, std_Res, mean_abs_Res, mape, med_abs_res, med_rel, BIC, AIC, n_point, num_5, num_10, num_75, num_100_rel = np.round(med_Res, decimals=5), np.round(std_Res, decimals=5), np.round(mean_abs_Res, decimals=5),  np.round(mape, decimals=5), np.round(med_abs_res, decimals=5),np.round(med_rel, decimals=5), np.round(BIC, decimals=5) , np.round(AIC, decimals=5) , n_point, np.round(num_5, decimals=5), np.round(num_10, decimals=5), np.round(num_75, decimals=5), np.round(num_100_rel, decimals=5)
    # df = pd.DataFrame([mod_name+"_"+suffix,mean_Res, std_Res, mean_abs_Res, med_abs_res, mape, med_rel, BIC, AIC, n_point, num_5, num_10, num_75, num_100_rel])
    # df = df.T
    # # df_loc = pd.DataFrame(index=range(count + 1))
    # # df_loc.loc[count] = [mod_name+suffix,mean_Res, std_Res, mean_abs_Res, med_abs_res, mape, med_rel, BIC, AIC, n_point, num_5, num_10, num_75, num_100_rel]
    # df.to_excel(out,startrow=count, index=False, header=False)
    return [float(value) if isinstance(value, np.float64) else value for value in [n_point, N_ks, med_Res, mean_abs_Res , med_abs_res, mape, AIC,  num_5, num_10]]


def Morse(x,b,T):
    T = T -273
    f = T*(1-np.exp(-b*x+ np.log(1-np.sqrt(25/T))))**2
    return f +273

def Sheehan2017(x):
    A  = [-0.10662476,10.16427521,7.545085171,13.75877438,7.416747047,-2.820341747,0.369509635,1.73066761,10.14800696,14.33989239,9.178858482,-2.051355516,6.300935346,2.106228102,-2.86488537,14.25245459,40.55628384,10.92027356,20.64089359,7.875008282,4.729692341,45.34296379,15.35651219,9.616742591,4.47078445,15.97806511,19.11114904,25.72904721]
    Ea = [0.008600918,33.45323623,47.86596824,88.12698254,40.51769781,4.351328796,5.176955812,3.391731868,69.90082099,86.8447291,58.72222259,13.93860869,214.1736667,48.88914766,10.89103371,106.5446397,202.6120022,104.8163444,85.49384411,25.24006005,1.328515017,248.2801371,54.5086251,49.70165647,17.9440547,57.43335072,36.74063995,227.1923693]

    
    count = 0
    all_pred = np.zeros((len(x),4))
    # try:
    for i in x:
    
        other,cell,starch,hemi, Prot, Lip, Lig, Sa, AA, FA, Ph, ash, solid_wt, time, T, b = i
        
        Temp = T + 273
        
        other,cell,starch,hemi, Prot, Lip, Lig, Sa, AA, FA, Ph = [item/100 for item in [other,cell,starch,hemi, Prot, Lip, Lig, Sa, AA, FA, Ph]]

    
        def my_ls_func(time, Temp):
            """definition of function for LS fit
                x gives evaluation points,
                teta is an array of parameters to be varied for fit"""
            
            
            x0 = [Prot, FA+Lip, other+cell,hemi,starch, Lig, Ph*0.4+Sa+AA*0.7, Ph*0.6+AA*0.3, 0]
            #print(sum(x0))
    
            def ode(x, t, Tempf):
    
                if np.isfinite(b):
                    Temp = Morse(t, b, Tempf)
                else:
                    Temp =Tempf
    
                    
    
                R = 8.314/1000
    
                xp = x[0]
                xl = x[1]
                xc = x[2]
                xh = x[3]
                xs = x[4]
                xlg= x[5]
                x2 = x[6]
                x3 = x[7]
                
                tot_carb = xc + xh + xs
    
                k1p  = math.exp(A[0]-(Ea[0]/(R*Temp)))
                k1l  = math.exp(A[1]-(Ea[1]/(R*Temp)))
                k1c  = math.exp(A[2]-(Ea[2]/(R*Temp)))
                k1h  = math.exp(A[3]-(Ea[3]/(R*Temp)))
                k1s  = math.exp(A[4]-(Ea[4]/(R*Temp)))
                k1lg = math.exp(A[5]-(Ea[5]/(R*Temp)))
                k2p  = math.exp(A[6]-(Ea[6]/(R*Temp)))
                k2l  = math.exp(A[7]-(Ea[7]/(R*Temp)))
                k2c  = math.exp(A[8]-(Ea[8]/(R*Temp)))
                k2h  = math.exp(A[9]-(Ea[9]/(R*Temp)))
                k2s  = math.exp(A[10]-(Ea[10]/(R*Temp)))
                k2lg = math.exp(A[11]-(Ea[11]/(R*Temp)))
                k3   = math.exp(A[12]-(Ea[12]/(R*Temp)))
                k4   = math.exp(A[13]-(Ea[13]/(R*Temp)))
                k5   = math.exp(A[14]-(Ea[14]/(R*Temp)))
                k6   = math.exp(A[15]-(Ea[15]/(R*Temp)))
                k1pl = math.exp(A[16]-(Ea[16]/(R*Temp)))
                k1pc = math.exp(A[17]-(Ea[17]/(R*Temp)))
                k1plg= math.exp(A[18]-(Ea[18]/(R*Temp)))
                k1lc = math.exp(A[19]-(Ea[19]/(R*Temp)))
                k1llg= math.exp(A[20]-(Ea[20]/(R*Temp)))
                k1clg= math.exp(A[21]-(Ea[21]/(R*Temp)))
                k2pl = math.exp(A[22]-(Ea[22]/(R*Temp)))
                k2pc = math.exp(A[23]-(Ea[25]/(R*Temp)))
                k2plg= math.exp(A[24]-(Ea[24]/(R*Temp)))
                k2lc = math.exp(A[25]-(Ea[25]/(R*Temp)))
                k2llg= math.exp(A[26]-(Ea[26]/(R*Temp)))
                k2clg= math.exp(A[27]-(Ea[27]/(R*Temp)))
    
                sol = [-(k1p+k2p)*xp -k1pl*xp*xl -k2pl*xp*xl -k1pc*xp*tot_carb -k2pc*xp*tot_carb -k1plg*xp*xlg -k2plg*xp*xlg,
                       -(k1l+k2l)*xl -k1pl*xp*xl -k2pl*xp*xl -k1lc*xl*tot_carb -k2lc*xl*tot_carb -k1llg*xl*xlg -k2llg*xl*xlg,
                       -(k1c+k2c)*xc -(k1pc*xp*xc +k2pc*xp*xc +k1lc*xl*xc +k2lc*xl*xc +k1clg*xc*xlg +k2clg*xc*xlg),
                       -(k1h+k2h)*xh -(k1pc*xp*xh +k2pc*xp*xh +k1lc*xl*xh +k2lc*xl*xh +k1clg*xh*xlg +k2clg*xh*xlg),
                       -(k1s+k2s)*xs -(k1pc*xp*xs +k2pc*xp*xs +k1lc*xl*xs +k2lc*xl*xs +k1clg*xs*xlg +k2clg*xs*xlg),
                       -(k1lg+k2lg)*xlg -k1plg*xp*xlg -k2plg*xp*xlg -k1llg*xl*xlg -k2llg*xl*xlg -k1clg*tot_carb*xlg -k2clg*tot_carb*xlg,
                       -(k4+k5)*x2 + k1p*xp + k1l*xl +k1c*xc +k1h*xh +k1s*xs + k1lg*xlg + k3*x3 +2*k1pl*xp*xl +2*k1pc*xp*tot_carb +2*k1plg*xp*xlg +2*k1lc*xl*tot_carb +2*k1llg*xl*xlg +2*k1clg*tot_carb*xlg,
                       -(k3+k6)*x3 + k2p*xp + k2l*xl +k2c*xc +k2h*xh +k2s*xs + k2lg*xlg + k4*x2 +2*k2pl*xp*xl +2*k2pc*xp*tot_carb +2*k2plg*xp*xlg +2*k2lc*xl*tot_carb +2*k2llg*xl*xlg +2*k2clg*tot_carb*xlg,
                       k5*x2 + k6*x3]
        
                #print(sol)
    
                return sol
            # create an alias to f which passes the optional params
            def f2(t, y): return ode(y, t, Temp)
            # calculate ode solution, retuen values for each entry of "x"
            #try:
            r = solve_ivp(f2, (0, time), x0, t_eval=[time], method='LSODA')

            pred = np.transpose(r.y)


            return pred[0]
            
            #except:
                
            #    return np.ones(9)*10000
            
    
        if time == 0:
            pred = [Prot, FA+Lip, other+cell,hemi,starch, Lig, Ph*0.4+Sa+AA*0.7, Ph*0.6+AA*0.3, 0]
        else:
            pred = my_ls_func(time, Temp)

    
        Y_Char = pred[0]*0.54 + pred[2] + pred[3]*0.97 + pred[4]*0.89 + pred[5]
    
        Y_Crude = pred[1] + pred[7]
        
        Y_Aq = pred[0]*.46 + pred[3]*0.03 + pred[4]*0.11 + pred[6]
    
        Y_Gas = pred[8]
        
        all_pred[count,:] = [Y_Char*100,Y_Crude*100,Y_Aq*100,Y_Gas*100]
        # print(sum([Y_Char,Y_Crude,Y_Aq,Y_Gas]))
        count +=1
    

    return all_pred 

# min_res = 10e25
# for n in range(100):
# A = np.random.uniform(8,9,24)
# Ea = np.random.uniform(60,70,24)
# fin_g = [9.34156,12.2539,-0.538086,-1.93603,14.0574,10.3279,3.04274,9.36841,11.5364,13.4372,1.26313,11.2373,16.6855,14.3155,18.3285,13.6574,17.6421,26.8242,16.7257,12.7154,14.9822,19.3183,24.1535,13.331,154.856,69.9969,197.85,168.716,86.853,78.0108,152.111,87.5155,75.6199,86.0091,154.692,87.0406,79.7706,83.4051,158.778,59.0416,59.5571,173.051,75.6801,79.8977,99.6773,84.3686,48.2836,185.646]
    # y_pred = Sheehan2017_1(guess,X)
    # if y_pred < min_res:
    #     min_res = y_pred
    #     fin_g = guess
# fin_g = [15.3168,21.8028,14.8395,4.00894,10.6248,17.9934,18.8737,18.9257,15.4764,6.88557,4.59033,6.33143,12.9544,2.85208,2.30252,9.04403,14.7214,21.0321,15.1307,20.765,9.52905,24.6679,24.491,14.7886,66.3151,92.9994,50.2688,116.95,54.7596,74.8027,75.8753,121.13,108.695,64.1776,122.845,70.9859,83.9793,64.6478,32.1669,116.338,50.2617,123.673,82.6655,102.107,61.7841,133.616,76.9232,64.1036]    


# A = np.random.uniform(11,12,24)
# Ea = np.random.uniform(70,80,24)

# fin_g = np.append(A,Ea)

# fin_g = [18.0414076,8.704835721,9.650408471,0.030890325,19.37646896,20.25540653,5.210694567,10.256654,13.09434775,23.55043685,0.276538402,36.34199338,18.48145665,8.755595944,26.93479743,3.5571731,40,14.32880416,25.43478213,2.406009826,8.61973276,4.050857701,26.73336124,27.22703909,1.199521463,48.96415,62.57485801,5.473544449,9.226760888,122.8354562,29.07949172,47.24137313,103.6864573,250,27.56898324,250,48.60928945,119.0388458,56.98639684,43.49773591,250,167.1089405,120.7175507,58.11754018,249.8227675,41.51267065,111.0525968,138.9957416]



pred = Sheehan2017(X)

print(analysis(y, pred, 28*2,"Sheehan_2017","_Fitting_full",2))

print(analysis(y[:,1], pred[:,1], 28*2,"New Model","Fitted Biocrude",1))

res = np.subtract(y[:,1], pred[:,1])

pred1 = Sheehan2017(X_pred)

print(analysis(y_pred, pred1, 28*2,"Sheehan_2017","_Fitting_full",1))

print(analysis(y_pred[:,1], pred1[:,1], 28*2,"New Model","Biocrude Predictions",1))

res1 = np.subtract(y_pred[:,1], pred1[:,1])