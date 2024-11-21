# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:43:15 2024

@author: peter
"""

import numpy as np
from scipy.optimize import minimize
import pandas as pd 
import math
from scipy.integrate import quad


data = pd.read_excel(r'Molec.xlsx', "Fitting", engine= 'openpyxl', header= 1)

SI_data = np.vstack((data["Total time (min)"].to_numpy(),data["Temperature (C)"].to_numpy(),data["b"].to_numpy(),data["Solid content (w/w) %"].to_numpy()))

SI_data = np.transpose(SI_data)

x = np.vstack((data["Total Carbs wt%"].to_numpy(),data["Other Carbs wt%"].to_numpy(),data["Starch wt%"].to_numpy(),data["Cellulose wt%"].to_numpy(),data["Hemicellulose wt%"].to_numpy(),data["Protein wt%"].to_numpy(),data["Lipids wt%"].to_numpy(),data["Lignin wt%"].to_numpy(),data["Sa wt%"].to_numpy(),data["AA wt%"].to_numpy(),data["FA wt%"].to_numpy(),data["Ph wt%"].to_numpy(),data["Solid content (w/w) %"].to_numpy()))

X = np.transpose(x)

def Morse(x,b,T):
    if x == 0:
        f =25
    else:
        f = T * (1 - np.exp(-b * x + np.log(1 - np.sqrt(25 / T)))) ** 2
    return f
def Integrand(x,b,T):
    Ea = 83
    R = 8.314/1000
    if np.isfinite(b):
        return math.exp(-Ea/R*((1/(Morse(x,b,T)+273)-1/(700))))
    else:
        return math.exp(-Ea/R*((1/(T+273)-1/(700))))
        
y = data["Biocrude wt%"].to_numpy()


# plt.plot(lnSI,y,  label='Data Set 1', color='blue', linestyle='None', marker='o')

lnSI = []

for i in SI_data:
    if i[0] == 0:
        I = quad(Integrand,0,0.01,args =(i[2],i[1]))
    else:
        I = quad(Integrand,0,i[0],args =(i[2],i[1]))
    # print(I)

    SI = math.log(I[0])
    lnSI.append(SI)

count = 0



def f(x, L, k1, k2, x0, x1, C):
    term1 = L / (1 + np.exp(-abs(k1) * (x - x0)))  # Increasing part
    term2 = L / (1 + np.exp(-abs(k2) * (x - (abs(x1) + x0) )))  # Decreasing part
    return term1 - term2 + C  # Combining terms with an offset


def syst_9(carb,other,starch,cell,hemi,prot,lip,lig,Sa,AA,FA,Ph,solid,lnSI):

    carb,other,starch,cell,hemi,prot,lip,lig,Sa,AA,FA,Ph = carb/100,other/100,starch/100,cell/100,hemi/100,prot/100,lip/100,lig/100,Sa/100,AA/100,FA/100,Ph/100


    c = np.array([-0.564928,0.483768,2.2376,-0.082129,1.2611,0.679054,-0.103365,-1.61523,1.40838,1.91103,3.78708,6.44045,-5.08042,-24.1639,-10.9946,-0.00215947,0.738281,1.78526,0.000146593,-0.143409,3.95983,-10.8021,17.9819,-0.215694,-0.157966,0.322851,0.174539,2.62665e-05,-0.000453458,0.554452,1.73748,0.147375,-0.904569,550.624,0.94887,26.3634,78.8807,0.910595,18.3894,-0.810106,-1.97446,5.20127,67.0229,0.266541,-0.220565,-0.880906,-1.40119,0.10052,-0.796925,4.63414,0.543906,-3.9166,6.74528,-7.15621,-3.14326,0.892804,-6.51644,-5.05874,-0.377212,-9.53429,-0.00524675,-2.13967,1.35153,-1.20732,2.12951e-05,2.53847,6.42139,-0.00969975,9.5068,5.68419,2.27109,5.42597,-11.0099,0.00463512,9.24964,43.6394,11.5521,18.046,41.8618,65.8278,3.25978,-5.90463e-05,8.03129,38.3236,21.6073,-53.96,-62.6043,-0.359772,329.723,-4.36487,5.49777,2.22231,0.577807,-2.40979,0.146469,0.165292,-3.78574e-05,0.0624836,4.03881,0.874352,0.0453981,1.9135,-3.206,-0.217839,0.0420713,-0.202809,0.0921019,-0.138144,1.37912,0.191843,0.00201082,-0.0788401,-0.0064798,0.0102251,-0.000256194,14.488,-0.252748,-0.0299722,-32.0487,3.89651,-0.881552,-0.0543056,-2.53544,10.1467,-1.07744,2.59229,-1.18793,0.672652,2.49992,2.4982,1.73535,2.40841,8.44924,2.98966,5.21319,0.886304,8.90231,11.2383,-85.9187,40.5794,-10.0667,-0.297242,0.171008,0.0167001,6.42764,-17.1697,-0.000205167,-0.184531,7.77836,-0.622317])

    n = 15

    # print(c)
    L1 = c[:n]*100
    k1 = c[n:n*2]
    k2 = c[n*2:n*3]
    x0 = c[n*3:n*4]
    x1 = c[n*4:n*5]
    C0 = c[n*5:n*6]
    L3 = np.ones(n)
    k3 = c[n*6:n*7]
    k4 = c[n*7:n*8]
    x2 = c[n*8:n*9]
    x3 = c[n*9:n*10]
    C1 = np.zeros(n)
    
    # c1  = f(lnSI, abs(L1[0]),  k1[0],  k2[0],  x0[0],   x1[0],   abs(C0[0]))
    # c2  = f(lnSI, abs(L1[1]),  k1[1],  k2[1],  x0[1],   x1[1],   abs(C0[1]))
    # c3  = f(lnSI, abs(L1[2]),  k1[2],  k2[2],  x0[2],   x1[2],   abs(C0[2]))
    # c4  = f(lnSI, abs(L1[3]),  k1[3],  k2[3],  x0[3],   x1[3],   abs(C0[3]))
    # c5  = f(lnSI, abs(L1[4]),  k1[4],  k2[4],  x0[4],   x1[4],   abs(C0[4]))
    # c6  = f(lnSI, abs(L1[5]),  k1[5],  k2[5],  x0[5],   x1[5],   abs(C0[5]))
    # c7  = f(lnSI, abs(L1[6]),  k1[6],  k2[6],  x0[6],   x1[6],   abs(C0[6]))
    # c8  = f(lnSI, abs(L1[7]),  k1[7],  k2[7],  x0[7],   x1[7],   abs(C0[7]))
    # c9  = f(lnSI, abs(L1[8]),  k1[8],  k2[8],  x0[8],   x1[8],   abs(C0[8]))
    # c10 = f(lnSI, abs(L1[9]),  k1[9],  k2[9],  x0[9],   x1[9],   abs(C0[9]))
    # c11 = f(lnSI, L1[10], k1[10], k2[10], x0[10],  x1[10],  C0[10])
    # c12 = f(lnSI, L1[11], k1[11], k2[11], x0[11],  x1[11],  C0[11])
    # c13 = f(lnSI, L1[12], k1[12], k2[12], x0[12],  x1[12],  C0[12])
    # c14 = f(lnSI, L1[13], k1[13], k2[13], x0[13],  x1[13],  C0[13])
    # c15 = f(lnSI, L1[14], k1[14], k2[14], x0[14],  x1[14],  C0[14])


    c1  = f(lnSI,  L1[0],   k1[0],  k2[0],  x0[0],   x1[0],   C0[0])* f(solid,L3[0],  k3[0],   k4[0],  x2[0],  x3[0],  C1[0])
    c2  = f(lnSI,  L1[1],   k1[1],  k2[1],  x0[1],   x1[1],   C0[1])* f(solid,L3[1],  k3[1],   k4[1],  x2[1],  x3[1],  C1[1])
    c3  = f(lnSI,  L1[2],   k1[2],  k2[2],  x0[2],   x1[2],   C0[2])* f(solid,L3[2],  k3[2],   k4[2],  x2[2],  x3[2],  C1[2])
    c4  = f(lnSI,  L1[3],   k1[3],  k2[3],  x0[3],   x1[3],   C0[3])* f(solid,L3[3],  k3[3],   k4[3],  x2[3],  x3[3],  C1[3])
    c5  = f(lnSI,  L1[4],   k1[4],  k2[4],  x0[4],   x1[4],   C0[4])* f(solid,L3[4],  k3[4],   k4[4],  x2[4],  x3[4],  C1[4])
    c6  = f(lnSI,  L1[5],   k1[5],  k2[5],  x0[5],   x1[5],   C0[5])* f(solid,L3[5],  k3[5],   k4[5],  x2[5],  x3[5],  C1[5])
    c7  = f(lnSI,  L1[6],   k1[6],  k2[6],  x0[6],   x1[6],   C0[6])* f(solid,L3[6],  k3[6],   k4[6],  x2[6],  x3[6],  C1[6])
    c8  = f(lnSI,  L1[7],   k1[7],  k2[7],  x0[7],   x1[7],   C0[7])* f(solid,L3[7],  k3[7],   k4[7],  x2[7],  x3[7],  C1[7])
    c9  = f(lnSI,  L1[8],   k1[8],  k2[8],  x0[8],   x1[8],   C0[8])* f(solid,L3[8],  k3[8],   k4[8],  x2[8],  x3[8],  C1[8])
    c10 = f(lnSI,  L1[9],   k1[9],  k2[9],  x0[9],   x1[9],   C0[9])* f(solid,L3[9],  k3[9],   k4[9],  x2[9],  x3[9],  C1[9])
    c11 = f(lnSI,  L1[10],  k1[10], k2[10], x0[10],  x1[10],  C0[10])* f(solid,L3[10], k3[10],  k4[10], x2[10], x3[10], C1[10])
    c12 = f(lnSI,  L1[11],  k1[11], k2[11], x0[11],  x1[11],  C0[11])* f(solid,L3[11], k3[11],  k4[11], x2[11], x3[11], C1[11])
    c13 = f(lnSI,  L1[12],  k1[12], k2[12], x0[12],  x1[12],  C0[12])* f(solid,L3[12], k3[12],  k4[12], x2[12], x3[12], C1[12])
    c14 = f(lnSI,  L1[13],  k1[13], k2[13], x0[13],  x1[13],  C0[13])* f(solid,L3[13], k3[13],  k4[13], x2[13], x3[13], C1[13])
    c15 = f(lnSI,  L1[14],  k1[14], k2[14], x0[14],  x1[14],  C0[14])* f(solid,L3[14], k3[14],  k4[14], x2[14], x3[14], C1[14])

    # c_values = [c1, c2, c3, c4, c6, c7, c8, c9, c10]
    # c_values = [np.where(item > 0, item, 0) for item in c_values]
    # c1, c2, c3, c4, c6, c7, c8, c9, c10 = c_values


    Crude = c1*starch + c2*(cell + other) + c3*hemi + c4*prot + c5*lip + c6*lig + c7*Sa + c8*AA + c9*FA + c10*Ph + c11*(carb+Sa)*(prot+AA) +c12*(carb+Sa)*(lip+FA) + c13*(prot+AA)*(lip+FA) + c14*(carb+Sa)*(prot+AA)*(lip+FA) + c15*(carb+Sa)*(lip+FA)*(lig+Ph) 
    
    
    return Crude



def fit_fun2(X,lnSI):
     
    crude = syst_9(X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5],X[:,6],X[:,7],X[:,8],X[:,9],X[:,10],X[:,11],X[:,12],lnSI)

    
    return crude





def analysis(exp, pred, N_ks,mod_name,suffix,count):
    print(mod_name)
    # print(len(pred),len(exp))
    # plt.scatter(pred, exp)
    # plt.plot([0, 100], [0, 100], color='black')
    # plt.plot([0, 100], [10, 110], '--', color='red')
    # plt.plot([0, 100], [-10, 90], '--', color='red')
    # plt.xlabel('Predictions')
    # plt.ylabel('Experimental data')
    # plt.title(mod_name+" "+suffix+" "+'Predictions vs Experimental data')
    # plt.savefig(mod_name+"_"+suffix+'.png')
    # plt.clf()
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
    mean_abs_Res = np.mean(abs_res)
    mape = np.nanmean(rel_clean)*100
    med_abs_res = np.nanmedian(abs_res)
    med_rel = np.nanmedian(rel_clean)*100
    AIC = 2*N_ks + np.log(RSS/n_point)*n_point
    BIC = N_ks*np.log(n_point) +np.log(RSS/n_point)*n_point
    num_5 = len([item for item in abs_res  if item <= 5])/n_point*100
    num_10 = len([item for item in abs_res if item <= 10])/n_point*100
    num_75 = len([item for item in abs_res if item >= 75])/n_point*100
    num_100_rel = len([item for item in rel_clean if item >= 1])/n_point*100
    med_Res, std_Res, mean_abs_Res, mape, med_abs_res, med_rel, BIC, AIC, n_point, num_5, num_10, num_75, num_100_rel = np.round(med_Res, decimals=8), np.round(std_Res, decimals=8), np.round(mean_abs_Res, decimals=8),  np.round(mape, decimals=8), np.round(med_abs_res, decimals=8),np.round(med_rel, decimals=8), np.round(BIC, decimals=8) , np.round(AIC, decimals=8) , n_point, np.round(num_5, decimals=8), np.round(num_10, decimals=8), np.round(num_75, decimals=8), np.round(num_100_rel, decimals=8)
    # df = pd.DataFrame([mod_name+"_"+suffix,n_point, N_ks, med_Res, mean_abs_Res, mape, AIC, num_5, num_10])
    # df = df.T
    # df_loc = pd.DataFrame(index=range(count + 1))
    # df_loc.loc[count] = [mod_name+suffix,mean_Res, std_Res, mean_abs_Res, med_abs_res, mape, med_rel, BIC, AIC, n_point, num_5, num_10, num_75, num_100_rel]
    # Save the DataFrame to an Excel file
    # df.to_excel(out,startrow=count, index=False, header=False)
    return [float(value) if isinstance(value, np.float64) else value for value in [n_point, N_ks,  med_Res, mean_abs_Res, med_abs_res, mape, AIC, num_5, num_10]]

syst_eval1 = fit_fun2(X,lnSI)

res1 = np.subtract(y,syst_eval1)

print(analysis(y, syst_eval1,150,"CAM_SI_cont","_Predict",count))

data = pd.read_excel(r'Molec.xlsx', "Predict", engine= 'openpyxl', header= 1)

data = data[data["Total Lignin wt%"] == 0]

x = np.vstack((data["Carbs wt%"].to_numpy(),data["Other Carbs wt%"].to_numpy(),data["Starch wt%"].to_numpy(),data["Cellulose wt%"].to_numpy(),data["Hemicellulose wt%"].to_numpy(),data["Protein wt%"].to_numpy(),data["Lipids wt%"].to_numpy(),data["Lignin wt%"].to_numpy(),data["Sa wt%"].to_numpy(),data["AA wt%"].to_numpy(),data["FA wt%"].to_numpy(),data["Ph wt%"].to_numpy(),data["Solid content (w/w) %"].to_numpy()))

X = np.transpose(x)

y = data["Biocrude wt%"].to_numpy()

SI_data = np.vstack((data["Total time (min)"].to_numpy(),data["Temperature (C)"].to_numpy(),data["b"].to_numpy(),data["Solid content (w/w) %"].to_numpy()))

SI_data = np.transpose(SI_data)

lnSI = []

for i in SI_data:
    if i[0] == 0:
        I = quad(Integrand,0,0.01,args =(i[2],i[1]))
    else:
        I = quad(Integrand,0,i[0],args =(i[2],i[1]))
    # print(I)

    SI = math.log(I[0])
    lnSI.append(SI)

syst_eval = fit_fun2(X,lnSI)

print(analysis(y, syst_eval,150,"CAM_SI_cont","_Predict",count))

res = np.subtract(y,syst_eval)

