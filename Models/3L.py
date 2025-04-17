import os
import numpy as np
import gsw

'''Create time array'''
dt = 6e6/2                    #~1/5 years
ti = 0
tf = 3e12 #3e11                   #~10,000 years
t = np.arange(ti,tf,dt)
sizet = len(t)

'''Fixed Parameters'''
Ly = 1e6#0.7e6                  #Meridional extent of southern ocean outcropping [m]
g = 9.81                    #acceleration due to gravity [ms-2]
fS = 1e-4                   #Absolute value of the coriolis parameter [s-1]
rho0 = 1027                 #Reference density [kgm-3]
S0 = 35                     #Reference salinity [gkg-1]

'''Basin Geometry'''
D = 5e3                     #Depth of Ocean Basin [m]
#A = 1e14                    #Horizontal area of ocean [m2]
Lx = 2.6e7                  #Zonal extent of southern ocean [m]
Vdn = 9e15                  #Volume of dn box [m3] UNCERTAIN
Vn = 3e15                   #Volume of n box [m3]
Vsd = 8e15                  #Volume of sd box [m3]
Vsb = 1e15                  #Volume of sb box [m3]
hn = 500                    #Depth of hn box [m3]
hsd = 500                   #Depth of sd box [m3]
hsb = 500                   #Depth of sb box [m3]
harray = np.array([hn,hsd,hsb])

'''Temperature of Surface Boxes'''
Tt = 21                     #Temperature of t box [C]
Tn = 5                      #Temperature of n box [C]
Tsd = 5                     #Temperature of sm box [C]
Tsb = 0                    #Temperature of sb box [C]

'''Equation of State'''
density = gsw.rho

'''Fluxes'''
#Ekman flux is held constant
#qEk = tau*Lx/rho0/fS

#Eddy return-flow scales as H1
def q_Eddy(K_GM,H1):
    #qEd = Lx*K_GM*H1/Ly
    return Lx*K_GM*H1/Ly

def q_AABW(eta,S,T):
    #return 8e6
    return eta*(density(S[6],T[6],g*rho0*hsb/1e4)-density(S[1],T[1],g*rho0*hsb/1e4))*np.heaviside(density(S[6],T[6],g*rho0*hsb/1e4)-density(S[1],T[1],g*rho0*hsb/1e4),1)/rho0

def q_South(qEk,qEd,qAABW):
    return np.array([qEk-qEd,-qAABW])

def q_Up(A,k,h):
    #qU1 = A*k[0]*(1/h[0]-1/h[1])
    #qU2 = A*k[1]*(1/h[1]-1/h[2])
    #qU = np.array([qU1,qU2])]
    return A * k * (1/h[:2]-1/h[1:])

def q_North(epsilon,H1,S,T):
    #rhon = density(S[4],T[4],g*rho0*hn/1e4)
    #rhomn = density(S[3],T[3],g*rho0*hn/1e4)
    #rhomt = density(S[3],T[3],g*rho0*H1/2/1e4)
    #rhot = density(S[0],T[0],g*rho0*H1/2/1e4)
    #m = np.heaviside(rhon - rhomn,1)
    #qN = np.array([m*(rhomt-rhot)*H[0]**2/rho0/epsilon,0])
    return np.array([np.heaviside(density(S[4],T[4],g*rho0*hn/1e4)-density(S[3],T[3],g*rho0*hn/1e4),1)*(density(S[1],T[1],g*rho0*H1/2/1e4)-density(S[0],T[0],g*rho0*H1/2/1e4))*H1**2/rho0/epsilon,0])

def SaltChange(r,E,I,S,qS,qU,qN):

    qS1 = np.heaviside(qS[0],1)
    qU1 = np.heaviside(qU[0],1)
    qU2 = np.heaviside(qU[1],1)
    
    dSt = ((-r[0]-r[1]-qN+qS[0]*(1-qS1)+qU[0]*(1-qU1))*S[0]+(qU[0]*qU1)*S[1]+r[0]*S[4]+(r[1]+qS[0]*qS1)*S[5]+np.sum(E)*S0)
    dSd = ((-qU[0]*(1-qU1))*S[0]+(-qU[0]*qU1+qU[1]*(1-qU2)+qS[1]-qS[0]*qS1)*S[1]+(qU[1]*qU2)*S[2]+qN*S[3]-qS[0]*(1-qS1)*S[5]+r[3]*(S[3]-S[1]))
    dSb = ((-qU[1]*(1-qU2))*S[1]+(-qU[1]*qU2)*S[2]+(-qS[1])*S[6])
    dSdn = (qN*(S[4]-S[3])+r[3]*(S[1]-S[3]))
    dSn = ((qN+r[0])*(S[0]-S[4])) - E[0]*S0
    dSsd = ((-qS[0]*(1-qS1)+r[1])*S[0]+(qS[0]*qS1)*S[1]+(qS[0]*(1-2*qS1)-r[1]-r[2])*S[5]+(r[2])*S[6]) - (E[1]+I)*S0
    dSsb = (-qS[1]*(S[1]-S[6])+r[2]*(S[5]-S[6])) + (I-E[2])*S0
    dS = np.array([dSt,dSd,dSb,dSdn,dSn,dSsd,dSsb])
    
    
    return dS

def HeatChange(r,T,qS,qU,qN):

    qS1 = np.heaviside(qS[0],1)
    qU1 = np.heaviside(qU[0],1)
    qU2 = np.heaviside(qU[1],1)
    
    dTd = ((-qU[0]*(1-qU1))*T[0]+(-qU[0]*qU1+qU[1]*(1-qU2)+qS[1]-qS[0]*qS1)*T[1]+(qU[1]*qU2)*T[2]+qN*T[3]-qS[0]*(1-qS1)*T[5]+r[3]*(T[3]-T[1]))
    dTb = ((-qU[1]*(1-qU2))*T[1]+(-qU[1]*qU2)*T[2]+(-qS[1])*T[6])
    dTdn = (qN*(T[4]-T[3])+r[3]*(T[1]-T[3]))
    
    dT = np.array([dTd,dTb,dTdn])
    return dT

'''def SalinityChange(S,h,dh,qS,qU,qN):

    qS1 = np.heaviside(qS[0],1)
    qU1 = np.heaviside(qU[0],1)
    qU2 = np.heaviside(qU[1],1)

    dSt = ((-r[0]-r[1]-qN[0]+qS[0]*(1-qS1)+qU[0]*(1-qU1)-A*dh[0])*S[0]+(qU[0]*qU1)*S[1]+r[0]*S[4]+(r[1]+qS[0]*qS1)*S[5]+np.sum(E)*S0)/A/h[0]
    dSd = ((-qU[0]*(1-qU1))*S[0]+(-qU[0]*qU1+qU[1]*(1-qU2)+qS[1]-qS[0]*qS1-A*dh[1])*S[1]+(qU[1]*qU2)*S[2]+qN[0]*S[3]-qS[0]*(1-qS1)*S[5]+r[3]*(S[3]-S[1]))/A/h[1]
    dSb = ((-qU[1]*(1-qU2))*S[1]+(-qU[1]*qU2-A*dh[2])*S[2]+(-qS[1])*S[6])/A/h[2]
    dSdn = (qN[0]*(S[4]-S[3])+r[3]*(S[1]-S[3]))/Vdn
    dSn = ((qN[0]+r[0])*(S[0]-S[4]))/Vn - E[0]*S0/Vn
    dSsd = ((-qS[0]*(1-qS1)+r[1])*S[0]+(qS[0]*qS1)*S[1]+(qS[0]*(1-2*qS1)-r[1]-r[2])*S[5]+(r[2])*S[6])/Vsd - (E[1]+I)*S0/Vsd
    dSsb = (-qS[1]*(S[1]-S[6])+r[2]*(S[5]-S[6]))/Vsb + (I-E[2])*S0/Vsb
    dS = np.array([dSt,dSd,dSb,dSdn,dSn,dSsd,dSsb])
    return dS

def TempChange(T,h,dh,qS,qU,qN):

    qS1 = np.heaviside(qS[0],1)
    qU1 = np.heaviside(qU[0],1)
    qU2 = np.heaviside(qU[1],1)
    dTd = ((-qU[0]*(1-qU1))*T[0]+(-qU[0]*qU1+qU[1]*(1-qU2)+qS[1]-qS[0]*qS1-A*dh[1])*T[1]+(qU[1]*qU2)*T[2]+qN[0]*T[3]-qS[0]*(1-qS1)*T[5]+r[3]*(T[3]-T[1]))/A/h[1]
    dTb = ((-qU[1]*(1-qU2))*T[1]+(-qU[1]*qU2-A*dh[2])*T[2]+(-qS[1])*T[6])/A/h[2]
    dTdn = (qN[0]*(T[4]-T[3])+r[3]*(T[1]-T[3]))/Vdn
    dT = np.array([dTd,dTb,dTdn])
    return dT'''


'''
7-box model of NADW and AABW formation. 
===================================

Requires: numpy, gsw (non-linear equation of state), os

Uses AB3 (Adam-Bashforth 3rd order) to solve the coupled ODEs governing the circulation of 
water, salt, and heat in a 7-box Gnandesikan-style model of meridional overturning circulation 
in the Atlantic.

This is based on Johnson et. al. Clim. Dyn. 2007 paper (henceforth J07), but extended to represent 
the formation of AABW. This model adds 3 new boxes:
    1. An additional southern surface box loosely corresponding to the Wendall sea – the site of 
        AABW formation.
    2. A bottom box that lies below the Deep box to represent AABW.
    3. A northern deep box to the north of the Deep box and below the northern box to account
        for bathymetry.

There are 7 boxes, each denoted by an abbreviation:
    t:     Thermocline
    d:     Deep
    b:     Bottom
    dn:    Deep - Northern
    n:     Northern
    sd:    Southern - Deep
    sb:    Southern - Bottom

There are 12 ODEs:
    2 for the interface depths (total volume is conserved). 
    7 for the salinity in all 7 boxes.
    3 for the temperature in the d, dn, and b boxes.

Fluxes are parameterised mostly as in J07. 

Inputs:
    Parameters (7):
        tau      ~ 1e-1              Southern wind stress [Nm-2] (float)
        K_GM     ~ 1e3               GM Eddy Diffusion Coefficient [m2s-1] (float)
        k        ~ [2e-5,2e-4]       Diapycnal upwelling diffusivity [Nm-2] ((2,) numpy.ndarray)
                                     The index specifies the interface (i.e., k[0] is the t, m 
                                     interface).
        epsilon  ~ 6e-5              Northern Sinking Parameter [s-1] (float)
        eta      ~ 1e7               AABW Formation Parameter [m6 kg-1 s-1]
        E        ~ [0.25,0.3,0]e6    Freshwater evaporative fluxes from __ box into into t box 
                                     [m3s-1] ((3,) numpy.ndarray)
                                     E[0] = n
                                     E[1] = sd
                                     E[2] = sb
        r        ~ [5,1,0,1]e6       Latering mixing [m3s-1] ((4,) numpy.ndarray)
                                     r[0] = n <-> t
                                     r[1] = t <-> sd
                                     r[2] = sd <-> sb
        I        ~ 0.05e6            Freshwater Ice Fluxes from sb->sd [m3s-1] (float)
    Initial Conditions:
        Hi       ~ [500,1500]        Initial interface depths [m] ((2,) numpy.ndarray)
                                     Hi[0] = t-d
                                     Hi[1] = d-b
        Ti       ~ [5,5,0]           Initial temperatures [C] ((3,) numpy.ndarray)
                                     Ti[0] = d
                                     Ti[1] = b
                                     Ti[2] = dn
        Si       ~ 34*np.ones(7)     Initial salinities
                                     Si[0] = t
                                     Si[1] = d
                                     Si[2] = b
                                     Si[3] = dn
                                     Si[4] = n
                                     Si[5] = sd
                                     Si[6] = sb
Outputs: (all numpy arrays)

'''

def Run_Model(A,tau,K_GM,k12,k23,epsilon,eta,E,r,I,Hi,Ti,Si):
#def BoxModel(tau,K_GM,k12,k23,epsilon,eta,En,Esd,Esb,rn,rsd,rsb,I,Hi,Ti,Si):
#Initial parameter arrays:
    k = np.array([k12,k23])
    #E = np.array([En,Esd,Esb])
    #r = np.array([rn,rsd,rsb])  
    
    '''Create folder to hold data (don't worry about this part)'''
    inputs = locals()
    path = ''
    nump = 9 #number of parameters
    tmodule = type(os)
    tarray = type(np.array([]))
    parameters = {}
    IC = {}
    j = 1
    for i in inputs:
        if type(inputs[i]) != tmodule:
            if type(inputs[i]) == tarray:
                path += str(i)+'='+str([format(item,'.5g') for item in inputs[i]])
            else:
                path += str(i)+'='+str(format(inputs[i],'.5g'))+' '
            if j<nump:
                parameters.update({i:inputs[i]})
            elif j == nump:
                parameters.update({i:inputs[i]})
                path += '/'
            else:
                IC.update({i:inputs[i]}) 
            j+=1
            
    
    #print('Running: '+path+'\n\n')
    
    '''Set Initial/Boundary Conditions'''
    
    #Interface depth
    H = np.zeros((sizet,4))
    H[0] = [0,Hi[0],Hi[1],D]
    H[:,-1] = D
    #Layer thicknesses
    h = np.zeros((sizet,3))
    h[0] = np.ediff1d(H[0])
    #Temperature
    T = np.zeros((sizet,7))
    T[:,0],T[:,4],T[:,5],T[:,6] = Tt, Tn, Tsd, Tsb
    T[0,1:4] = Ti
    #Salinity
    S = np.zeros((sizet,7))
    S[0] = Si
    #Density
    rho = np.zeros((sizet,7))
    
    #Volume
    V = np.zeros((sizet,7))
    V[0,:3] = h[0] * A
    V[:,3], V[:,4], V[:,5], V[:,6] = Vdn,Vn,Vsd,Vsb
    Stot = np.zeros((sizet,7))
    Ttot = np.zeros((sizet,7))
    Stot[0] = S[0] * V[0]
    Ttot[0] = T[0] * V[0] 
    Ttot[:,0],Ttot[:,4],Ttot[:,5],Ttot[:,6] = Ttot[0,0],Ttot[0,4],Ttot[0,5],Ttot[0,6]
    
    
    '''Fluxes'''
    qEk = tau*Lx/rho0/fS
    qEd = np.zeros((sizet))
    qS = np.zeros((sizet,2))
    qAABW = np.zeros((sizet))
    qU = np.zeros((sizet,2))
    qN = np.zeros((sizet,2))
        
    '''Changes per unit of time'''
    dH = np.zeros((sizet,4))
    dS = np.zeros((sizet,7))
    dT = np.zeros((sizet,3))
    
    '''Euler Time-step forward for 3 times'''
    for i in range(2):

        z = np.concatenate((H[i,:3]+h[i]/2,H[i,1:2]+h[i,1:2]/2,harray/2))
        rho[i] = density(S[i],T[i],g*rho0*z/1e4)
        
        #Compute fluxes
        qEd[i] = q_Eddy(K_GM,H[i,1])
        qAABW[i] = q_AABW(eta,S[i],T[i])
        qS[i] = q_South(qEk,qEd[i],qAABW[i])
        qU[i] = q_Up(A,k,h[i])
        qN[i] = q_North(epsilon,H[i,1],S[i],T[i])
        
        dH[i,1:-1] = (qS[i] + qU[i] - qN[i])/A
        dh = np.ediff1d(dH[i])
        
        dS[i] = SaltChange(r,E,I,S[i],qS[i],qU[i],qN[i,0])
        dT[i] = HeatChange(r,T[i],qS[i],qU[i],qN[i,0])
        
        H[i+1,1:-1] = H[i,1:-1] + dt*dH[i,1:-1]
        Stot[i+1] = Stot[i] + dt*dS[i]
        Ttot[i+1,1:4] = Ttot[i,1:4] + dt*dT[i]
        
        h[i+1] = np.ediff1d(H[i+1])
        V[i+1,:3] = h[i+1]*A
        S[i+1] = Stot[i+1]/V[i+1]
        T[i+1,1:4] = Ttot[i+1,1:4]/V[i+1,1:4]
        
    for i in range(2,sizet-1):
        z = np.concatenate((H[i,:3]+h[i]/2,H[i,1:2]+h[i,1:2]/2,harray/2))
        rho[i] = density(S[i],T[i],g*rho0*z/1e4)
        
        #Compute fluxes
        qEd[i] = q_Eddy(K_GM,H[i,1])
        qAABW[i] = q_AABW(eta,S[i],T[i])
        qS[i] = q_South(qEk,qEd[i],qAABW[i])
        qU[i] = q_Up(A,k,h[i])
        qN[i] = q_North(epsilon,H[i,1],S[i],T[i])
        
        dH[i,1:-1] = (qS[i] + qU[i] - qN[i])/A
        dh = np.ediff1d(dH[i])
        
        dS[i] = SaltChange(r,E,I,S[i],qS[i],qU[i],qN[i,0])
        dT[i] = HeatChange(r,T[i],qS[i],qU[i],qN[i,0])
        
        H[i+1,1:-1] = H[i,1:-1] + dt/12*(23*dH[i,1:-1]-16*dH[i-1,1:-1]+5*dH[i-2,1:-1])
        Stot[i+1] = Stot[i] + dt/12*(23*dS[i]-16*dS[i-1]+5*dS[i-2])
        Ttot[i+1,1:4] = Ttot[i,1:4] + dt/12*(23*dT[i]-16*dT[i]+5*dT[i])
        
        h[i+1] = np.ediff1d(H[i+1])
        V[i+1,:3] = h[i+1]*A
        S[i+1] = Stot[i+1]/V[i+1]
        T[i+1,1:4] = Ttot[i+1,1:4]/V[i+1,1:4]
        
    z = np.concatenate((H[-1,:3]+h[-1]/2,H[-1,1:2]+h[-1,1:2]/2,harray/2))
    rho[-1] = density(S[-1],T[-1],g*rho0*z/1e4)
    
    #Compute fluxes
    qEd[-1] = q_Eddy(K_GM,H[-1,1])
    qAABW[-1] = q_AABW(eta,S[-1],T[-1])
    qS[-1] = q_South(qEk,qEd[-1],qAABW[-1])
    qU[-1] = q_Up(A,k,h[-1])
    qN[-1] = q_North(epsilon,H[-1,1],S[-1],T[-1])
    
    '''Compress Data Points and Save'''
    
    #print('Saving: '+path+'\n\n')
    
    if not os.path.exists(path):
        os.makedirs(path)

    #ndp = 400 #Number of data points
    ndp = 1000
    entries = 1#round(sizet/ndp)
    
    np.save(path+'/parameters.npy',parameters)
    
    np.save(path+'/IC.npy',IC)

    np.save(path+'/t',t[::entries])
    
    H = (H[::entries]/1000).astype(np.float16)
    h = (h[::entries]/1000).astype(np.float16)
    S = (S[::entries].astype(np.float16))
    T = (T[::entries].astype(np.float16))
    rho = ((rho[::entries]-1000).astype(np.float16))
    
    qEk = (qEk*np.ones(len(t[::entries]))/1e6).astype(np.float16)
    qEd = (qEd[::entries]/1e6).astype(np.float16)
    qU = (qU[::entries]/1e6).astype(np.float16)
    qS = (qS[::entries]/1e6).astype(np.float16)
    qN = (qN[::entries]/1e6).astype(np.float16)

    np.save(path+'/H',H)
    
    np.save(path+'/thickness',h)
    
    np.save(path+'/S',S)
    
    #Called 'Temp' rather than t    
    np.save(path+'/Temp',T)
    
    np.save(path+'/rho',rho)

    np.save(path+'/qEk',qEk)
    np.save(path+'/qEd',qEd)
    np.save(path+'/qS',qS)
    np.save(path+'/qU',qU)
    np.save(path+'/qN',qN)

        
'''A function that plots using the control values'''
def Control_Run():
    tau = 0.1
    K_GM = 1000
    k = np.array([1e-5,1e-4])
    epsilon = 6e-5
    eta = 1
    E = np.array([0.3,0.3,0])*1e6
    r = np.array([5,0,0])*1e6
    I = 0.25e6 
    H0 = np.array([500,1500])
    T0 = np.array([6,0,6])
    S0 = 34*np.ones(7)
    Run_Model(tau,K_GM,k,epsilon,eta,E,r,I,H0,T0,S0)