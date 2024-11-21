# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:02:23 2024

@author: peter
"""

import math
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.colors import Normalize

Un = 27.5
Ce = 0
He = 0
St = 0
Sa = 0
Ps = Un + Ce + He + St
Ct = Un + Ce + He + St + Sa
Pe = 35.3
AA = 0
Pt = Pe + AA
Lp = 9.7
FA = 0
Ft = Lp + FA
Lg = 10.8
Ph = 0
Lt = Lg + Ph
Ash = 0

w = 10  #Loading mass 



optimize_data = np.array([Un,St,Ce,Pe,He,Lp,Lg,Ash,Sa,AA,FA,Ph,w])

data = pd.read_excel(r'Molec.xlsx', "Fitting", engine= 'openpyxl', header= 1)
x = np.vstack((data["Other Carbs wt%"].to_numpy(),data["Starch wt%"].to_numpy(),data["Cellulose wt%"].to_numpy(),data["Protein wt%"].to_numpy(),data["Hemicellulose wt%"].to_numpy(),data["Lipids wt%"].to_numpy(),data["Lignin wt%"].to_numpy(),data["Ash wt%"].to_numpy(),data["Sa wt%"].to_numpy(),data["AA wt%"].to_numpy(),data["FA wt%"].to_numpy(),data["Ph wt%"].to_numpy(),data["Solid content (w/w) %"].to_numpy(),data["Total time (min)"].to_numpy(),data["Temperature (C)"].to_numpy()))

X = np.transpose(x)

y = data["Biocrude wt%"].to_numpy()

reg = DecisionTreeRegressor(criterion='absolute_error',random_state=3,max_depth=24)
reg.fit(X, y)


# Define the range for time and temperature
time_range = np.linspace(0, 30, 1000)  # Time in minutes
temp_range = np.linspace(150, 650, 1000)  # Temperature in Celsius

# Create a meshgrid for time and temperature
temp_grid, time_grid = np.meshgrid(temp_range, time_range)

# Initialize the output grid for lnSI and predicted biocrude yield
lnSI_grid = np.zeros_like(time_grid)
y_pred_grid = np.zeros_like(time_grid)

# Optimize data (fixed values for other features)
# optimize_data = np.array([27.5, 0, 0, 34.3, 0, 9.7, 10.8, 0])

# Calculate lnSI and predictions for the grid
for i in range(time_grid.shape[0]):
    for j in range(time_grid.shape[1]):
        g = np.hstack((optimize_data, [time_grid[i, j], temp_grid[i, j]]))
        y_pred_grid[i, j] = reg.predict([g])


 # Plot the contour plot with increased font sizes
norm = Normalize(vmin=0, vmax=66, clip=True)
plt.figure(figsize=(12, 8), dpi=300)
contour = plt.contourf(time_grid, temp_grid, y_pred_grid, levels=100, cmap='Spectral_r', norm=norm)


# Set axis labels and title with font size 16
plt.xlabel('Time (minutes)', fontsize=24)
plt.ylabel('Temperature (°C)', fontsize=24)
# plt.title('c)', fontsize=24)

# Set tick label size for x and y axes
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# Display the plot
plt.show()

def f(x, L, k1, k2, x0, x1, C):
    term1 = L / (1 + np.exp(-abs(k1) * (x - x0)))  # Increasing part
    term2 = L / (1 + np.exp(-abs(k2) * (x - (abs(x1) + x0) )))  # Decreasing part
    return term1 - term2 + C  # Combining terms with an offset

    

def syst_9(g,x):
    

    other,starch,cell,prot,hemi, lip, lig, Ash, Sa, AA, FA, Ph, solid = x/100
    
    carb = other + starch + cell
    
    lnSI = g
    
    c = np.array([-0.564928,0.483768,2.2376,-0.082129,1.2611,0.679054,-0.103365,-1.61523,1.40838,1.91103,3.78708,6.44045,-5.08042,-24.1639,-10.9946,-0.00215947,0.738281,1.78526,0.000146593,-0.143409,3.95983,-10.8021,17.9819,-0.215694,-0.157966,0.322851,0.174539,2.62665e-05,-0.000453458,0.554452,1.73748,0.147375,-0.904569,550.624,0.94887,26.3634,78.8807,0.910595,18.3894,-0.810106,-1.97446,5.20127,67.0229,0.266541,-0.220565,-0.880906,-1.40119,0.10052,-0.796925,4.63414,0.543906,-3.9166,6.74528,-7.15621,-3.14326,0.892804,-6.51644,-5.05874,-0.377212,-9.53429,-0.00524675,-2.13967,1.35153,-1.20732,2.12951e-05,2.53847,6.42139,-0.00969975,9.5068,5.68419,2.27109,5.42597,-11.0099,0.00463512,9.24964,43.6394,11.5521,18.046,41.8618,65.8278,3.25978,-5.90463e-05,8.03129,38.3236,21.6073,-53.96,-62.6043,-0.359772,329.723,-4.36487,5.49777,2.22231,0.577807,-2.40979,0.146469,0.165292,-3.78574e-05,0.0624836,4.03881,0.874352,0.0453981,1.9135,-3.206,-0.217839,0.0420713,-0.202809,0.0921019,-0.138144,1.37912,0.191843,0.00201082,-0.0788401,-0.0064798,0.0102251,-0.000256194,14.488,-0.252748,-0.0299722,-32.0487,3.89651,-0.881552,-0.0543056,-2.53544,10.1467,-1.07744,2.59229,-1.18793,0.672652,2.49992,2.4982,1.73535,2.40841,8.44924,2.98966,5.21319,0.886304,8.90231,11.2383,-85.9187,40.5794,-10.0667,-0.297242,0.171008,0.0167001,6.42764,-17.1697,-0.000205167,-0.184531,7.77836,-0.622317])
    
    n = 15
    
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

# Define the range of solid and lnSI values for the contour plot
# Constants
Ea = 83  # kJ/mol
R = 8.314 / 1000  # kJ/mol*K

# Define the Morse potential function
def Morse(x, b, T):
    if x == 0:
        f = 25
    else:
        f = T * (1 - np.exp(-b * x + np.log(1 - np.sqrt(25 / T)))) ** 2
    return f

# Define the integrand function
def Integrand(x, b, T):
    if np.isfinite(b):
        return math.exp(-Ea / R * ((1 / (Morse(x, b, T) + 273)) - (1 / 700)))
    else:
        return math.exp(-Ea / R * ((1 / (T + 273)) - (1 / 700)))

# Initialize the grid for the contour plot
time_range = np.linspace(0, 30, 100)
temp_range = np.linspace(150, 650, 100)
time_grid, temp_grid = np.meshgrid(time_range, temp_range)
lnSI_grid = np.zeros_like(time_grid)

# Calculate lnSI for each point in the grid
for i in range(time_grid.shape[0]):
    for j in range(time_grid.shape[1]):
        if time_grid[i, j] == 0:
            I = quad(Integrand, 0, 0.01, args=(10000, temp_grid[i, j]))
        else:
            I = quad(Integrand, 0, time_grid[i, j], args=(10000, temp_grid[i, j]))
        lnSI_grid[i, j] = math.log(I[0])

# Initialize the grid for the contour plot
time_grid, temp_grid = np.meshgrid(time_range, temp_range)
Crude_grid = np.zeros_like(time_grid)


# Calculate Crude values for the grid
for i in range(time_grid.shape[0]):
    for j in range(time_grid.shape[1]):
        g = lnSI_grid[i, j]
        Crude_grid[i, j] = syst_9(g, optimize_data)

# Plot the contour plot with increased font sizes
plt.figure(figsize=(12, 8), dpi=300)
contour = plt.contourf(time_grid, temp_grid, Crude_grid, levels=100, cmap='Spectral_r',norm=norm)

# # Add text in the top-right corner with font size 16
# plt.text(27, 640, '18.43 g/g wt%, Isothermal', fontsize=24, color='white',
#          ha='right', va='top', bbox=dict(facecolor='black', alpha=0.5))

# Set axis labels and title with font size 16
plt.xlabel('Time (minutes)', fontsize=24)
plt.ylabel('Temperature (°C)', fontsize=24)
# plt.title('a)', fontsize=24)

# Set tick label size for x and y axes
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# Display the plot
plt.show()


fitted = pd.read_excel(r'James_fit.xlsx', engine='openpyxl', header=None).to_numpy()

# Define the Morse function
def Morse2(x, b, T):
    T = T - 273
    return T * (1 - np.exp(-b * x + np.log(1 - np.sqrt(25 / T)))) ** 2 + 273

# Define Sheehan2017 function
def Lunp_kinetic(g,x):
    
    A  = [-0.10662476,10.16427521,7.545085171,13.75877438,7.416747047,-2.820341747,0.369509635,1.73066761,10.14800696,14.33989239,9.178858482,-2.051355516,6.300935346,2.106228102,-2.86488537,14.25245459,35.95111365,6.315103376,16.0357234,3.269838096,0.124522155,40.7377936,10.751342,5.011572405,-0.134385736,11.37289492,14.50597886,21.12387703]

    Ea = [0.008600918,33.45323623,47.86596824,88.12698254,40.51769781,4.351328796,5.176955812,3.391731868,69.90082099,86.8447291,58.72222259,13.93860869,214.1736667,48.88914766,10.89103371,106.5446397,202.6120022,104.8163444,85.49384411,25.24006005,1.328515017,248.2801371,54.5086251,49.70165647,17.9440547,57.43335072,36.74063995,227.1923693]

    
    other,starch,cell,Prot,hemi, Lip, Lig, Ash, Sa, AA, FA, Ph, w = x
    
    time, T = g
    
    Temp = T + 273
    

    def my_ls_func(time, Temp):
        """definition of function for LS fit
            x gives evaluation points,
            teta is an array of parameters to be varied for fit"""
        
        
        x0 = [Prot, FA+Lip, other+cell,hemi,starch, Lig, Ph*0.4+Sa+AA*0.7, Ph*0.6+AA*0.3, 0]
        #print(sum(x0))

        def ode(x, t, Tempf):

            Temp = Morse(t, 1000, Tempf)

                

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


    Y_Crude = pred[1] + pred[7]
    

    return Y_Crude 

# Define the range for time and temperature
time_range = np.linspace(0, 30, 100)
temp_range = np.linspace(150, 650, 100)

# Create a meshgrid
time_grid, temp_grid = np.meshgrid(time_range, temp_range)
Y_Crude_grid = np.zeros_like(time_grid)


# Calculate Y_Crude for the grid
for i in range(time_grid.shape[0]):
    for j in range(time_grid.shape[1]):
        g = [time_grid[i, j], temp_grid[i, j]]
        Y_Crude_grid[i, j] = Lunp_kinetic(g, optimize_data)

# Plot the contour plot with increased font sizes
plt.figure(figsize=(12, 8), dpi=300)
contour = plt.contourf(time_grid, temp_grid, Y_Crude_grid, levels=100,  cmap='Spectral_r',norm=norm)

# Add text in the top-right corner with font size 16
# plt.text(27, 640, 'Isothermal', fontsize=24, color='white',
         # ha='right', va='top', bbox=dict(facecolor='black', alpha=0.5))

# Set axis labels and title with font size 16
plt.xlabel('Time (minutes)', fontsize=24)
plt.ylabel('Temperature (°C)', fontsize=24)
# plt.title('b)', fontsize=24)
# plt.yticks([])  # Removes the y-axis ticks and labels


# Set tick label size for x and y axes
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
# plt.grid(axis='x', linestyle='--', alpha=0.7)

# Display the plot
plt.show()
