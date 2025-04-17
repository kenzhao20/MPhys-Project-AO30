import os
import numpy as np
import gsw
import shutil



'''Create time array'''
dt = 6e6                    #~1/5 years
ti = 0
tf = 3e11 #3e11                   #~10,000 years
t = np.arange(ti,tf,dt)
sizet = len(t)

'''Fixed Parameters'''
g = 9.81                    #acceleration due to gravity [ms-2]
fS = 1e-4                   #Absolute value of the coriolis parameter [s-1]
fP = 7e-5                   #Absolute value of the coriolis parameter [s-1]
rho0 = 1027                 #Reference density [kgm-3]
S0 = 35                     #Reference salinity [gkg-1]

'''Atlantic Basin Geometry'''
Da = 3.4e3                     #Depth of Ocean Basin [m]
Aa = 1e14                    #Horizontal area of ocean [m2]
#Lx = 3e7 
Lxa = 1e7                    #Zonal extent of southern ocean [m]
Lya = 1.2e6                  #Meridional extent of southern ocean outcropping [m]
Vdn = 9e15                   #Volume of dn box [m3] UNCERTAIN
Vn = 3e15                    #Volume of n box [m3]
Vsda = 5e15                  #Volume of sd box [m3]
Vsba = 1e15                  #Volume of sb box [m3] Weddell sea
hn = 500                     #Depth of hn box [m3]
hsda = 500                   #Depth of sd box [m3]
hsba = 500                   #Depth of sb box [m3]
harraya = np.array([hn,hsda,hsba])

'''Pacific Basin Geometry'''
Dp = 4e3                     #Depth of Ocean Basin [m]
Ap = 2e14                  #Horizontal area of ocean [m2]
Lxp = 1.5e7                  #Zonal extent of southern ocean [m]
Lyp = 0.8e6
Vsdp = 5e15                  #Volume of sd box [m3]
Vsbp = 8e14                  #Volume of sb box [m3] Ross sea
hsdp = 500                   #Depth of sd box [m3]
hsbp = 500                   #Depth of sb box [m3]
harrayp = np.array([hsdp,hsbp])

'''Temperature of Surface Boxes'''
Tta = 21                     #Temperature of t box [C]
Ttp = 21                     #Temperature of t box [C]
Tn = 5                      #Temperature of n box [C]
Tsda = 5                     #Temperature of sm box [C]
Tsba = 0                    #Temperature of sb box [C]
Tsdp = 5                     #Temperature of sm box [C]
Tsbp = 0                    #Temperature of sb box [C]

'''Equation of State'''
density = gsw.rho

def LinearEoS(S,T,p):
    return rho0 * (1-0.0002*(T-5)+0.0008*(S-35))

def NonLinEoS(S,T,p):
    return rho0 * (1-1.67e-4*(1+0)*(T-10) - 1e-5/2*(T-10)**2 + 0.78e-3*(S-35))

'''Functions'''
'''Fluxes'''
#Ekman flux is held constant

#Eddy return-flow scales as H1
def q_Eddy(Lx,Ly,K_GM,H1):
    #qEd = Lx*K_GM*H1/Ly
    return Lx*K_GM*H1/Ly

def q_AABWa(eta,S,T):
    return eta*(density(S[6],T[6],g*rho0*hsba/1e4)-density(S[1],T[1],g*rho0*hsba/1e4))*np.heaviside(density(S[6],T[6],g*rho0*hsba/1e4)-density(S[1],T[1],g*rho0*hsba/1e4),1)/rho0

def q_AABWp(eta,S,T):
    return eta*(density(S[4],T[4],g*rho0*hsbp/1e4)-density(S[1],T[1],g*rho0*hsbp/1e4))*np.heaviside(density(S[4],T[4],g*rho0*hsbp/1e4)-density(S[1],T[1],g*rho0*hsbp/1e4),1)/rho0

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
    return np.array([np.heaviside(density(S[4],T[4],g*rho0*hn/1e4)-density(S[3],T[3],g*rho0*hn/1e4),1)*(density(S[3],T[3],g*rho0*H1/2/1e4)-density(S[0],T[0],g*rho0*H1/2/1e4))*H1**2/rho0/epsilon,0])

def q_Zonal(mu,Sp,Sa,Tp,Ta,Hp,Ha):
    qZ = g/2/fP * (Hp[2]**2-Ha[2]**2) * (density(Sa[1],Ta[1],g*rho0*Ha[1]/2/1e4)-density(Sp[1],Tp[1],g*rho0*Hp[1]/2/1e4))/rho0
    #qZ>0 corresponds to Pacific->Atlantic in top 2 layers and Atlantic->Pacific in bottom layer
    return np.array([qZ*mu,qZ])

def q_Zonal_Gnanadesikan(Sp,Sa,Tp,Ta,Hp,Ha):
    qZ = g/0.7e-4 * np.min([Hp[1],Hp[0]]) * ((density(Sp[2],Tp[2],g*rho0*Hp[1]/2/1e4)/density(Sp[0],Tp[0],g*rho0*Hp[1]/2/1e4)-1)*Hp[1]-(density(Sa[2],Ta[2],g*rho0*Ha[1]/2/1e4)/density(Sa[0],Ta[0],g*rho0*Ha[1]/2/1e4)-1)*Ha[1])
    return np.array([qZ*mu,qZ])

def SaltChangeA(mu,r,E,I,S,qS,qU,qN,qZ,Sp):
    
    qS1 = np.heaviside(qS[0],1)
    qU1 = np.heaviside(qU[0],1)
    qU2 = np.heaviside(qU[1],1)
    qZ1 = np.heaviside(qZ,1)
    
    dSt = (qS[0]*(qS1*S[5]+(1-qS1)*S[0])
           +qU[0]*(qU1*S[1]+(1-qU1)*S[0])
           +mu*qZ*(qZ1*Sp[0]+(1-qZ1)*S[0])
           -qN*S[0]
           +r[0]*(S[4]-S[0])
           +r[1]*(S[5]-S[0])
           +(E[0]+E[1]+E[2])*S0)
    
    dSd = (-qS[0]*(qS1*S[1]+(1-qS1)*S[5])
           +qS[1]*S[1]
           -qU[0]*(qU1*S[1]+(1-qU1)*S[0])
           +qU[1]*(qU2*S[2]+(1-qU2)*S[1])
           +(1-mu)*qZ*(qZ1*Sp[1]+(1-qZ1)*S[1])
           +qN*S[3]
           +r[3]*(S[3]-S[1]))
    
    dSb = (-qS[1]*S[6]
           -qU[1]*(qU2*S[2]+(1-qU2)*S[1])
           -qZ*(qZ1*S[2]+(1-qZ1)*Sp[2]))
    
    dSdn = (qN*(S[4]-S[3])
            +r[3]*(S[1]-S[3]))
    
    dSn = ((qN+r[0])*(S[0]-S[4])
           -E[0]*S0)
    
    dSsd = (qS[0]*(qS1*(S[1]-S[5])+(1-qS1)*(S[5]-S[0]))
            +r[1]*(S[0]-S[5])
            +r[2]*(S[6]-S[5])
            -(I+E[1])*S0)
    
    dSsb = (qS[1]*(S[6]-S[1])
            +r[2]*(S[5]-S[6])
            +I*S0)
    
    return np.array([dSt,dSd,dSb,dSdn,dSn,dSsd,dSsb])

def SaltChangeP(mu,r,E,I,S,qS,qU,qZ,Sa):
    
    qS1 = np.heaviside(qS[0],1)
    qU1 = np.heaviside(qU[0],1)
    qU2 = np.heaviside(qU[1],1)
    qZ1 = np.heaviside(qZ,1)
    
    dSt = (qS[0]*(qS1*S[3]+(1-qS1)*S[0])
           +qU[0]*(qU1*S[1]+(1-qU1)*S[0])
           -mu*qZ*(qZ1*S[0]+(1-qZ1)*Sa[0])
           +r[1]*(S[3]-S[0])
           +(E[1]-E[2])*S0)
    
    dSd = (-qS[0]*(qS1*S[1]+(1-qS1)*S[3])
           +qS[1]*S[1]
           -qU[0]*(qU1*S[1]+(1-qU1)*S[0])
           +qU[1]*(qU2*S[2]+(1-qU2)*S[1])
           -(1-mu)*qZ*(qZ1*S[1]+(1-qZ1)*Sa[1]))
    
    dSb = (-qS[1]*S[4]
           -qU[1]*(qU2*S[2]+(1-qU2)*S[1])
           +qZ*(qZ1*Sa[2]+(1-qZ1)*S[2]))
    
    dSsd = (qS[0]*(qS1*(S[1]-S[3])+(1-qS1)*(S[3]-S[0]))
            +r[1]*(S[0]-S[3])
            +r[2]*(S[4]-S[3])
            -(I+E[1])*S0)
    
    dSsb = (qS[1]*(S[4]-S[1])
            +r[2]*(S[3]-S[4])
            +I*S0)
    
    return np.array([dSt,dSd,dSb,dSsd,dSsb])

def HeatChangeA(mu,r,T,qS,qU,qN,qZ,Tp):
    
    qS1 = np.heaviside(qS[0],1)
    qU1 = np.heaviside(qU[0],1)
    qU2 = np.heaviside(qU[1],1)
    qZ1 = np.heaviside(qZ,1)
    

    dTd = (-qS[0]*(qS1*T[1]+(1-qS1)*T[5])
           +qS[1]*T[1]
           -qU[0]*(qU1*T[1]+(1-qU1)*T[0])
           +qU[1]*(qU2*T[2]+(1-qU2)*T[1])
           +(1-mu)*qZ*(qZ1*Tp[1]+(1-qZ1)*T[1])
           +qN*T[3]
           +r[3]*(T[3]-T[1]))
    
    dTb = (-qS[1]*T[6]
           -qU[1]*(qU2*T[2]+(1-qU2)*T[1])
           -qZ*(qZ1*T[2]+(1-qZ1)*Tp[2]))
    
    dTdn = (qN*(T[4]-T[3])
            +r[3]*(T[1]-T[3]))
    
    return np.array([dTd,dTb,dTdn])

def HeatChangeP(mu,T,qS,qU,qZ,Ta):
    
    qS1 = np.heaviside(qS[0],1)
    qU1 = np.heaviside(qU[0],1)
    qU2 = np.heaviside(qU[1],1)
    qZ1 = np.heaviside(qZ,1)
    
    dTd = (-qS[0]*(qS1*T[1]+(1-qS1)*T[3])
           +qS[1]*T[1]
           -qU[0]*(qU1*T[1]+(1-qU1)*T[0])
           +qU[1]*(qU2*T[2]+(1-qU2)*T[1])
           -(1-mu)*qZ*(qZ1*T[1]+(1-qZ1)*Ta[1]))
    
    dTb = (-qS[1]*T[4]
           -qU[1]*(qU2*T[2]+(1-qU2)*T[1])
           +qZ*(qZ1*Ta[2]+(1-qZ1)*T[2]))
    
    return np.array([dTd,dTb])


'''
2 Basin model

7-box model of NADW and AABW formation in Atlantic
5-box model in Pacific
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
        tau      ~ 0.1               Southern wind stress [Nm-2] (float)
        K_GM     ~ 1e3               GM Eddy Diffusion Coefficient [m2s-1] (float)
        k        ~ [1e-5,1e-4]       Diapycnal upwelling diffusivity [Nm-2] ((2,) numpy.ndarray)
                                     The index specifies the interface (i.e., k[0] is the t, m 
                                     interface).
        epsilon  ~ 6e-5              Northern Sinking Parameter [s-1] (float)
        eta      ~ 1e7               AABW Formation Parameter [m6 kg-1 s-1]
        mu       ~ 0.5               Partitioning of zonal transport between the top two layers [dimensionless] (float)
        E        ~ [0.2,0.2,0]e6    Freshwater evaporative fluxes from __ box into into t box 
                                     [m3s-1] ((3,) numpy.ndarray)
                                     E[0] = n
                                     E[1] = sd
                                     E[2] = sb
        r        ~ [5,1,0,1]e6       Latering mixing [m3s-1] ((4,) numpy.ndarray)
                                     r[0] = n <-> t
                                     r[1] = t <-> sd
                                     r[2] = sd <-> sb
                                     r[3] = dn <-> d
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

def Run_Model(tau,K_GM,k,epsilon,eta,mu,E,r,I,Hia,Hip,Tia,Tip,Sia,Sip):
#def BoxModel(tau,K_GM,k12,k23,epsilon,eta,En,Esd,Esb,rn,rsd,rsb,I,Hi,Ti,Si):
#Initial parameter arrays:
    #k = np.array([k12,k23])
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
    '''Atlantic'''
    #Interface depth
    Ha = np.zeros((sizet,4))
    Ha[0] = [0,Hia[0],Hia[1],Da]
    Ha[:,-1] = Da
    #Layer thicknesses
    ha = np.zeros((sizet,3))
    ha[0] = np.ediff1d(Ha[0])
    #Temperature
    Ta = np.zeros((sizet,7))
    Ta[:,0],Ta[:,4],Ta[:,5],Ta[:,6] = Tta, Tn, Tsda, Tsba
    Ta[0,1:4] = Tia
    #Salinity
    Sa = np.zeros((sizet,7))
    Sa[0] = Sia    
    #Density
    rhoa = np.zeros((sizet,7))
    
    Va = np.zeros((sizet,7))
    Va[0,0:3] = Aa * ha[0]
    Va[:,3], Va[:,4], Va[:,5], Va[:,6] = Vdn, Vn, Vsda, Vsba
    Satot = np.zeros((sizet,7))
    Satot[0] = Sia*Va[0]
    Tatot = np.zeros((sizet,7))
    Tatot[0] = Ta[0]*Va[0]
    
    '''Pacific'''
    #Interface depth
    Hp = np.zeros((sizet,4))
    Hp[0] = [0,Hip[0],Hip[1],Dp]
    Hp[:,-1] = Dp
    #Layer thicknesses
    hp = np.zeros((sizet,3))
    hp[0] = np.ediff1d(Hp[0])
    #Temperature
    Tp = np.zeros((sizet,5))
    Tp[:,0],Tp[:,3],Tp[:,4] = Ttp, Tsdp, Tsbp
    Tp[0,1:3] = Tip
    #Salinity
    Sp = np.zeros((sizet,5))
    Sp[0] = Sip
    #Density
    rhop = np.zeros((sizet,5))
    
    Vp = np.zeros((sizet,5))
    Vp[0,0:3] = Ap * hp[0]
    Vp[:,3], Vp[:,4] = Vsdp, Vsbp
    Sptot = np.zeros((sizet,5))
    Sptot[0] = Sip*Vp[0]
    Tptot = np.zeros((sizet,5))
    Tptot[0] = Tp[0]*Vp[0]
    
    
    '''Fluxes'''
    #qEk = tau*Lx/rho0/fS
    #qEka = qEk/3*2
    #qEkp = qEk-qEka
    
    qEka = tau*Lxa/rho0/fS
    qEkp = tau*Lxp/rho0/fS
    qEda = np.zeros((sizet))
    qEdp = np.zeros((sizet))
    qSa = np.zeros((sizet,2))
    qSp = np.zeros((sizet,2))
    qAABWa = np.zeros((sizet))
    qAABWp = np.zeros((sizet))
    qUa = np.zeros((sizet,2))
    qUp = np.zeros((sizet,2))
    qN = np.zeros((sizet,2))
    qZ = np.zeros((sizet,2))
        
    '''Changes per unit of time'''
    dHa = np.zeros((sizet,4))
    dSa = np.zeros((sizet,7))
    dTa = np.zeros((sizet,3))
    dHp = np.zeros((sizet,4))
    dSp = np.zeros((sizet,5))
    dTp = np.zeros((sizet,2))
    
    '''Euler Time-step forward for 3 times'''
    for i in range(2):
        
        za = np.concatenate((Ha[i,:3]+ha[i]/2,Ha[i,1:2]+ha[i,1:2]/2,harraya/2))
        zp = np.concatenate((Hp[i,:3]+hp[i]/2,harrayp/2))
        
        rhoa[i] = density(Sa[i],Ta[i],g*rho0*za/1e4)
        rhop[i] = density(Sp[i],Tp[i],g*rho0*zp/1e4)
        
        #Compute fluxes
        #Atlantic
        qEda[i] = q_Eddy(Lxa,Lya,K_GM,Ha[i,1])
        qAABWa[i] = q_AABWa(eta,Sa[i],Ta[i])
        qSa[i] = q_South(qEka,qEda[i],qAABWa[i])
        qUa[i] = q_Up(Aa,k,ha[i])
        qN[i] = q_North(epsilon,Ha[i,1],Sa[i],Ta[i])
        #Pacific
        qEdp[i] = q_Eddy(Lxp,Lyp,K_GM,Hp[i,1])
        qAABWp[i] = q_AABWp(eta,Sp[i],Tp[i])
        qSp[i] = q_South(qEkp,qEdp[i],qAABWp[i])
        qUp[i] = q_Up(Ap,k,hp[i])
        #Zonal
        qZ[i] = q_Zonal(mu,Sp[i],Sa[i],Tp[i],Ta[i],Hp[i],Ha[i])
        
        #Timestep forward
        dHa[i,1:-1] = (qSa[i] + qUa[i] - qN[i] + qZ[i])/Aa
        dha = np.ediff1d(dHa[i])
        
        dHp[i,1:-1] = (qSp[i] + qUp[i] - qZ[i])/Ap
        dhp = np.ediff1d(dHp[i])
        
        dTa[i] = HeatChangeA(mu,r,Ta[i],qSa[i],qUa[i],qN[i,0],qZ[i,1],Tp[i])
        dTp[i] = HeatChangeP(mu,Tp[i],qSp[i],qUp[i],qZ[i,1],Ta[i])
        
        dSa[i] = SaltChangeA(mu,r,E,I,Sa[i],qSa[i],qUa[i],qN[i,0],qZ[i,1],Sp[i])
        dSp[i] = SaltChangeP(mu,r,E,I,Sp[i],qSp[i],qUp[i],qZ[i,1],Sa[i])
        
        Ha[i+1,1:-1] = Ha[i,1:-1] + dt*dHa[i,1:-1]
        Tatot[i+1,1:4] = Tatot[i,1:4] + dt*dTa[i]
        Satot[i+1] = Satot[i] + dt*dSa[i]
        ha[i+1] = np.ediff1d(Ha[i+1])
        
        Hp[i+1,1:-1] = Hp[i,1:-1] + dt*dHp[i,1:-1]
        Tptot[i+1,1:3] = Tptot[i,1:3] + dt*dTp[i]
        Sptot[i+1] = Sptot[i] + dt*dSp[i]
        hp[i+1] = np.ediff1d(Hp[i+1])
        
        #calculate related quantities
        Va[i+1,:3] = Aa*ha[i+1]
        Sa[i+1] = Satot[i+1]/Va[i+1]
        Ta[i+1,1:4] = Tatot[i+1,1:4]/Va[i+1,1:4]
        
        Vp[i+1,:3] = Ap*hp[i+1]
        Sp[i+1] = Sptot[i+1]/Vp[i+1]
        Tp[i+1,1:3] = Tptot[i+1,1:3]/Vp[i+1,1:3]
        
    for i in range(2,sizet-1):
        za = np.concatenate((Ha[i,:3]+ha[i]/2,Ha[i,1:2]+ha[i,1:2]/2,harraya/2))
        zp = np.concatenate((Hp[i,:3]+hp[i]/2,harrayp/2))
        
        rhoa[i] = density(Sa[i],Ta[i],g*rho0*za/1e4)
        rhop[i] = density(Sp[i],Tp[i],g*rho0*zp/1e4)
        
        #Compute fluxes
        #Atlantic
        qEda[i] = q_Eddy(Lxa,Lya,K_GM,Ha[i,1])
        qAABWa[i] = q_AABWa(eta,Sa[i],Ta[i])
        qSa[i] = q_South(qEka,qEda[i],qAABWa[i])
        qUa[i] = q_Up(Aa,k,ha[i])
        qN[i] = q_North(epsilon,Ha[i,1],Sa[i],Ta[i])
        #Pacific
        qEdp[i] = q_Eddy(Lxp,Lyp,K_GM,Hp[i,1])
        qAABWp[i] = q_AABWp(eta,Sp[i],Tp[i])
        qSp[i] = q_South(qEkp,qEdp[i],qAABWp[i])
        qUp[i] = q_Up(Ap,k,hp[i])
        #Zonal
        qZ[i] = q_Zonal(mu,Sp[i],Sa[i],Tp[i],Ta[i],Hp[i],Ha[i])
        
        #Timestep forward
        dHa[i,1:-1] = (qSa[i] + qUa[i] - qN[i] + qZ[i])/Aa
        dha = np.ediff1d(dHa[i])
        
        dHp[i,1:-1] = (qSp[i] + qUp[i] - qZ[i])/Ap
        dhp = np.ediff1d(dHp[i])
        
        dTa[i] = HeatChangeA(mu,r,Ta[i],qSa[i],qUa[i],qN[i,0],qZ[i,1],Tp[i])
        dTp[i] = HeatChangeP(mu,Tp[i],qSp[i],qUp[i],qZ[i,1],Ta[i])
        
        dSa[i] = SaltChangeA(mu,r,E,I,Sa[i],qSa[i],qUa[i],qN[i,0],qZ[i,1],Sp[i])
        dSp[i] = SaltChangeP(mu,r,E,I,Sp[i],qSp[i],qUp[i],qZ[i,1],Sa[i])
        
        Ha[i+1,1:-1] = Ha[i,1:-1] + dt/12*(23*dHa[i,1:-1]-16*dHa[i-1,1:-1]+5*dHa[i-2,1:-1])
        Tatot[i+1,1:4] = Tatot[i,1:4] + dt/12*(23*dTa[i]-16*dTa[i]+5*dTa[i])
        Satot[i+1] = Satot[i] + dt/12*(23*dSa[i]-16*dSa[i-1]+5*dSa[i-2])
        ha[i+1] = np.ediff1d(Ha[i+1])
        
        Hp[i+1,1:-1] = Hp[i,1:-1] + dt/12*(23*dHp[i,1:-1]-16*dHp[i-1,1:-1]+5*dHp[i-2,1:-1])
        Tptot[i+1,1:3] = Tptot[i,1:3] + dt/12*(23*dTp[i]-16*dTp[i]+5*dTp[i])
        Sptot[i+1] = Sptot[i] + dt/12*(23*dSp[i]-16*dSp[i-1]+5*dSp[i-2])
        hp[i+1] = np.ediff1d(Hp[i+1])
        
        Va[i+1,:3] = Aa*ha[i+1]
        Sa[i+1] = Satot[i+1]/Va[i+1]
        Ta[i+1,1:4] = Tatot[i+1,1:4]/Va[i+1,1:4]
        
        Vp[i+1,:3] = Ap*hp[i+1]
        Sp[i+1] = Sptot[i+1]/Vp[i+1]
        Tp[i+1,1:3] = Tptot[i+1,1:3]/Vp[i+1,1:3]
        
    
    '''Compress Data Points and Save'''
    
    #print('Saving: '+path+'\n\n')
    
    if not os.path.exists(path):
        os.makedirs(path)

    #ndp = 400 #Number of data points
    ndp = 1000
    entries = round(sizet/ndp)
    
    #t = t[::entries].astype(np.float32)
    
    #f = gzip.GzipFile(path+"/t.npy.gz", "w")
    #np.save(file=f, arr=t[::entries].astype(np.float32))
    #f.close()

    np.save(path+'/t',t[::entries].astype(np.float32))

    Ha = Ha[::entries].astype(np.float32)
    Hp = Hp[::entries].astype(np.float32)
    
    #Called 'thickness' rather than h because os is not case-sensitive
    ha = ha[::entries].astype(np.float32)
    hp = hp[::entries].astype(np.float32)
    
    Sa = Sa[::entries].astype(np.float32)
    Sp = Sp[::entries].astype(np.float32)
    
    #Called 'Temp' rather than t because os is not case-sensitive
    Ta = Ta[::entries].astype(np.float32)
    Tp = Tp[::entries].astype(np.float32)
    
    rhoa = rhoa[::entries].astype(np.float32)
    rhop = rhop[::entries].astype(np.float32)

    qEka = qEka*np.ones(len(t[::entries]))
    qEkp = qEkp*np.ones(len(t[::entries]))
    qEda = qEda[::entries].astype(np.float32)
    qSa = qSa[::entries].astype(np.float32)
    qUa = qUa[::entries].astype(np.float32)
    qN = qN[::entries].astype(np.float32)
    qEdp = qEdp[::entries].astype(np.float32)
    qSp = qSp[::entries].astype(np.float32)
    qUp = qUp[::entries].astype(np.float32)
    qZ = qZ[::entries].astype(np.float32)
    
    dTa = dTa[::entries].astype(np.float32)
    dTp = dTp[::entries].astype(np.float32)
    dSa = dSa[::entries].astype(np.float32)
    dSp = dSp[::entries].astype(np.float32)
    
    print('saving...')
    
    np.save(path+'/parameters.npy',parameters)
    
    np.save(path+'/IC.npy',IC)
    
    #np.save(path+'/t',t)

    np.save(path+'/Ha',Ha)
    np.save(path+'/Hp',Hp)
    
    #Called 'thickness' rather than h because os is not case-sensitive
    np.save(path+'/thicknessa',ha)
    np.save(path+'/thicknessp',hp)
    
    np.save(path+'/Sa',Sa)
    np.save(path+'/Sp',Sp)
    
    #Called 'Temp' rather than t because os is not case-sensitive
    np.save(path+'/Tempa',Ta)
    np.save(path+'/Tempp',Tp)
    
    np.save(path+'/rhoa',rhoa)
    np.save(path+'/rhop',rhop)

    np.save(path+'/qEka',qEka)
    np.save(path+'/qEkp',qEkp)
    np.save(path+'/qEda',qEda)
    np.save(path+'/qSa',qSa)
    np.save(path+'/qUa',qUa)
    np.save(path+'/qN',qN)
    np.save(path+'/qEdp',qEdp)
    np.save(path+'/qSp',qSp)
    np.save(path+'/qUp',qUp)
    np.save(path+'/qZ',qZ)
    
    #shutil.make_archive(path, 'zip', path)
    
    print('saved!')
    
    #np.save(path+'/dTa',dTa)
    #np.save(path+'/dTp',dTp)
    #np.save(path+'/dSa',dSa)
    #np.save(path+'/dSp',dSp)
    

        
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
    BoxModel(tau,K_GM,k,epsilon,eta,E,r,I,H0,T0,S0)

    
'''Modifications'''


'''
Old flux functions that calculated salinity – now changed to calculate total salt
def SalinityChangeA(mu,r,E,I,S,h,dh,qS,qU,qN,qZ,Sp):
    
    qS1 = np.heaviside(qS[0],1)
    qU1 = np.heaviside(qU[0],1)
    qU2 = np.heaviside(qU[1],1)
    qZ1 = np.heaviside(qZ,1)
    
    dSt = (qS[0]*(qS1*S[5]+(1-qS1)*S[0])
           +qU[0]*(qU1*S[1]+(1-qU1)*S[0])
           +mu*qZ*(qZ1*Sp[0]+(1-qZ1)*S[0])
           -qN*S[0]
           +r[0]*(S[4]-S[0])
           +r[1]*(S[5]-S[0])
           +(E[0]+E[1]+E[2])*S0
           -Aa*dh[0]*S[0])/Aa/h[0]
    
    dSd = (-qS[0]*(qS1*S[1]+(1-qS1)*S[5])
           +qS[1]*S[1]
           -qU[0]*(qU1*S[1]+(1-qU1)*S[0])
           +qU[1]*(qU2*S[2]+(1-qU2)*S[1])
           +(1-mu)*qZ*(qZ1*Sp[1]+(1-qZ1)*S[1])
           +qN*S[3]
           +r[3]*(S[3]-S[1])
           -Aa*dh[1]*S[1])/Aa/h[1]
    
    dSb = (-qS[1]*S[6]
           -qU[1]*(qU2*S[2]+(1-qU2)*S[1])
           -qZ*(qZ1*S[2]+(1-qZ1)*Sp[2])
           -Aa*dh[2]*S[2])/Aa/h[2]
    
    dSdn = (qN*(S[4]-S[3])
            +r[3]*(S[1]-S[3]))/Vdn
    
    dSn = ((qN+r[0])*(S[0]-S[4])
           -E[0]*S0)/Vn
    
    dSsd = (qS[0]*(qS1*(S[1]-S[5])+(1-qS1)*(S[5]-S[0]))
            +r[1]*(S[0]-S[5])
            -(I+E[1])*S0)/Vsda
    
    dSsb = (qS[1]*(S[6]-S[1])
            +I*S0)/Vsba
    
    return np.array([dSt,dSd,dSb,dSdn,dSn,dSsd,dSsb])

def SalinityChangeP(mu,r,E,I,S,h,dh,qS,qU,qZ,Sa):
    
    qS1 = np.heaviside(qS[0],1)
    qU1 = np.heaviside(qU[0],1)
    qU2 = np.heaviside(qU[1],1)
    qZ1 = np.heaviside(qZ,1)
    
    dSt = (qS[0]*(qS1*S[3]+(1-qS1)*S[0])
           +qU[0]*(qU1*S[1]+(1-qU1)*S[0])
           -mu*qZ*(qZ1*S[0]+(1-qZ1)*Sa[0])
           +r[1]*(S[3]-S[0])
           +(E[1]-E[2])*S0
           -Ap*dh[0]*S[0])/Ap/h[0]
    
    dSd = (-qS[0]*(qS1*S[1]+(1-qS1)*S[3])
           +qS[1]*S[1]
           -qU[0]*(qU1*S[1]+(1-qU1)*S[0])
           +qU[1]*(qU2*S[2]+(1-qU2)*S[1])
           -(1-mu)*qZ*(qZ1*S[1]+(1-qZ1)*Sa[1])
           -Ap*dh[1]*S[1])/Ap/h[1]
    
    dSb = (-qS[1]*S[4]
           -qU[1]*(qU2*S[2]+(1-qU2)*S[1])
           +qZ*(qZ1*Sa[2]+(1-qZ1)*S[2])
           -Ap*dh[2]*S[2])/Ap/h[2]
    
    dSsd = (qS[0]*(qS1*(S[1]-S[3])+(1-qS1)*(S[3]-S[0]))
            +r[1]*(S[0]-S[3])
            -(I+E[1])*S0)/Vsdp
    
    dSsb = (qS[1]*(S[4]-S[1])
            +I*S0)/Vsbp
    
    return np.array([dSt,dSd,dSb,dSsd,dSsb])

def TempChangeA(mu,r,T,h,dh,qS,qU,qN,qZ,Tp):
    
    qS1 = np.heaviside(qS[0],1)
    qU1 = np.heaviside(qU[0],1)
    qU2 = np.heaviside(qU[1],1)
    qZ1 = np.heaviside(qZ,1)
    

    dTd = (-qS[0]*(qS1*T[1]+(1-qS1)*T[5])
           +qS[1]*T[1]
           -qU[0]*(qU1*T[1]+(1-qU1)*T[0])
           +qU[1]*(qU2*T[2]+(1-qU2)*T[1])
           +(1-mu)*qZ*(qZ1*Tp[1]+(1-qZ1)*T[1])
           +qN*T[3]
           +r[3]*(T[3]-T[1])
           -Aa*dh[1]*T[1])/Aa/h[1]
    
    dTb = (-qS[1]*T[6]
           -qU[1]*(qU2*T[2]+(1-qU2)*T[1])
           -qZ*(qZ1*T[2]+(1-qZ1)*Tp[2])
           -Aa*dh[2]*T[2])/Aa/h[2]
    
    dTdn = (qN*(T[4]-T[3])
            +r[3]*(T[1]-T[3]))/Vdn
    
    return np.array([dTd,dTb,dTdn])

def TempChangeP(mu,T,h,dh,qS,qU,qZ,Ta):
    
    qS1 = np.heaviside(qS[0],1)
    qU1 = np.heaviside(qU[0],1)
    qU2 = np.heaviside(qU[1],1)
    qZ1 = np.heaviside(qZ,1)
    
    dTd = (-qS[0]*(qS1*T[1]+(1-qS1)*T[3])
           +qS[1]*T[1]
           -qU[0]*(qU1*T[1]+(1-qU1)*T[0])
           +qU[1]*(qU2*T[2]+(1-qU2)*T[1])
           -(1-mu)*qZ*(qZ1*T[1]+(1-qZ1)*Ta[1])
           -Ap*dh[1]*T[1])/Ap/h[1]
    
    dTb = (-qS[1]*T[4]
           -qU[1]*(qU2*T[2]+(1-qU2)*T[1])
           +qZ*(qZ1*Ta[2]+(1-qZ1)*T[2])
           -Ap*dh[2]*T[2])/Ap/h[2]
    
    return np.array([dTd,dTb])'''