import os
import numpy as np
import gsw

#print('Running: '+path+'\n\n')

'''Create time array'''
dt = 1e7                    #~1/5 years
ti = 0
tf = 3e12 #3e11                   #~100,000 years
t = np.arange(ti,tf,dt)
sizet = len(t)

'''Fixed Parameters'''
Ly = 1e6                  #Meridional extent of southern ocean outcropping [m]
g = 9.81                    #acceleration due to gravity [ms-2]
fS = 1e-4                   #Absolute value of the coriolis parameter [s-1]
rho0 = 1027                 #Reference density [kgm-3]
S0 = 35                     #Reference salinity [gkg-1]

'''Basin Geometry'''
D = 4e3                     #Depth of Ocean Basin [m]
#A = 1.07e14    #for now
#A = 2.6e14                    #Horizontal area of ocean [m2]
Lx = 1e7 #3e7                 #Zonal extent of southern ocean [m]
Vn = 3e15                   #Volume of n box [m3]
Vs = 9e15                  #Volume of sd box [m3]
hn = 500                    #Depth of hn box [m3]
hs = 500                   #Depth of sd box [m3]
harray = np.array([hn,hs])

'''Temperature of Surface Boxes'''
Tt = 21                     #Temperature of t box [C]
Tn = 5                      #Temperature of n box [C]
Ts = 5                     #Temperature of sm box [C]

'''Fluxes'''
#Ekman flux is held constant
#qEk = tau*Lx/rho0/fS

#Eddy return-flow scales as H1
def q_Eddy(K_GM,H1):
    #qEd = Lx*K_GM*H1/Ly
    return Lx*K_GM*H1/Ly

def q_South(qEk,qEd):
    return np.array(qEk-qEd)

def q_Up(A,k,H):
    
    #qU1 = A*k[0]*(1/h[0]-1/h[1])
    #qU2 = A*k[1]*(1/h[1]-1/h[2])
    #qU = np.array([qU1,qU2])]
    #return A * k * (1/h[:2]-1/h[1:])
    return A*k/H

def q_North(epsilon,H1,S,T):
    
    '''
    rhon = gsw.rho(S[4],T[4],g*rho0*hn/1e4)
    rhomn = gsw.rho(S[3],T[3],g*rho0*hn/1e4)
    rhomt = gsw.rho(S[3],T[3],g*rho0*H1/2/1e4)
    rhot = gsw.rho(S[0],T[0],g*rho0*H1/2/1e4)
    m = np.heaviside(rhon - rhomn,1)
    qN = np.array([m*(rhomt-rhot)*H[0]**2/rho0/epsilon,0])'''
    
    #Heaviside function:
    #np.heaviside(gsw.rho(S[2],T[2],g*rho0*hn/1e4)-gsw.rho(S[1],T[1],g*rho0*hn/1e4),1)
    
    #a=0.1
    #Sin^2 function:
    #np.sin(np.pi/4/a*(a+np.clip(gsw.rho(S[2],T[2],g*rho0*hn/1e4)-gsw.rho(S[1],T[1],g*rho0*hn/1e4),-a,a)))**2
    
    #Heaviside with H(0)=0.5
    #np.heaviside(gsw.rho(S[2],T[2],g*rho0*hn/1e4)-gsw.rho(S[1],T[1],g*rho0*hn/1e4),0.5)
    
    #Smoothstep function:

    return np.heaviside(gsw.rho(S[2],T[2],g*rho0*hn/1e4)-gsw.rho(S[1],T[1],g*rho0*hn/1e4),1)*(gsw.rho(S[1],T[1],g*rho0*H1/2/1e4)-gsw.rho(S[0],T[0],g*rho0*H1/2/1e4))*H1**2/rho0/epsilon

def SaltChange(E,r,S,qS,qU,qN):
    
    qS1 = np.heaviside(qS,1)
    
    dSt = ((-r-qN+qS*(1-qS1))*S[0]+(qU)*S[1]+r*S[2]+(qS*qS1)*S[3]+2*E*S0)
    dSd = ((-qU-qS*qS1)*S[1]+qN*S[2]-qS*(1-qS1)*S[3])
    dSn = ((qN+r)*(S[0]-S[2])) - E*S0
    dSs = -dSt-dSd-dSn#((-qS*(1-qS1))*S[0]+(qS*qS1)*S[1]+(qS*(1-2*qS1))*S[3]) - (E)*S0
    dS = np.array([dSt,dSd,dSn,dSs])
    #print(np.sum(dS))
    return dS

def HeatChange(T,h,dh,qS,qU,qN):
    
    qS1 = np.heaviside(qS,1)
    
    dTd = ((-qU-qS*qS1)*T[1]+qN*T[2]-qS*(1-qS1)*T[3])
    
    return dTd

'''This is the 2007 2-layer model.'''
def Run_Model(A,tau,K_GM,k,epsilon,E,r,Hi,Ti,Si):
    
    '''Create folder to hold data (don't worry about this part)'''
    inputs = locals()
    path = ''
    nump = 7 #number of parameters
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
    
    
    '''Set Initial/Boundary Conditions'''
    
    #Interface depth
    H = np.zeros((sizet,3))
    H[0] = [0,Hi,D]
    H[:,-1] = D
    #Layer thicknesses
    h = np.zeros((sizet,2))
    h[0] = np.ediff1d(H[0])
    #Temperature
    T = np.zeros((sizet,4))
    T[:,0],T[0,1],T[:,2],T[:,3] = Tt, Ti, Tn, Ts
    #Salinity
    S = np.zeros((sizet,4))
    S[0] = Si
    #Volume of boxes
    V = np.zeros((sizet,4))
    V[0,0], V[0,1] = A*Hi, A*(D-Hi)
    V[:,2], V[:,3] = Vn, Vs
    #Total Salt
    Stot = np.zeros((sizet,4))
    Stot[0] = S[0] * V[0]
    #Total Heat
    Ttot = np.zeros(sizet)
    Ttot[0] = Ti*V[0,1]
    #Density
    rho = np.zeros((sizet,4))
    
    '''Fluxes'''
    qEk = tau*Lx/rho0/fS
    qEd = np.zeros(sizet)
    qS = np.zeros(sizet)
    qU = np.zeros(sizet)
    qN = np.zeros(sizet)
        
    '''Changes per unit of time'''
    dH = np.zeros((sizet,3))
    dS = np.zeros((sizet,4))
    dT = np.zeros(sizet)
    
    '''Euler Time-step forward for 3 times'''
    for i in range(2):
        
        z = np.concatenate((H[i,:2]+h[i]/2,harray/2))
        rho[i] = gsw.rho(S[i],T[i],g*rho0*z/1e4)
        
        #Compute fluxes
        qEd[i] = q_Eddy(K_GM,H[i,1])
        qS[i] = q_South(qEk,qEd[i])
        qU[i] = q_Up(A,k,H[i,1])
        qN[i] = q_North(epsilon,H[i,1],S[i],T[i])
        
        dH[i,1:-1] = (qS[i] + qU[i] - qN[i])/A
        dh = np.ediff1d(dH[i])
        
        dS[i] = SaltChange(E,r,S[i],qS[i],qU[i],qN[i])
        #dS[i] = SalinityChange(S[i],h[i],dh,qS[i],qU[i],qN[i])
        dT[i] = HeatChange(T[i],h[i],dh,qS[i],qU[i],qN[i])
        
        H[i+1,1:-1] = H[i,1:-1] + dt*dH[i,1:-1]
        Stot[i+1] = Stot[i] + dt*dS[i]
        Ttot[i+1] = Ttot[i] + dt*dT[i]
        h[i+1] = np.ediff1d(H[i+1])
        
        V[i+1,:2] = A * h[i+1]
        S[i+1] = Stot[i+1]/V[i+1]
        T[i+1,1] = Ttot[i+1]/V[i+1,1]
        
    for i in range(2,sizet-1):
        z = np.concatenate((H[i,:2]+h[i]/2,harray/2))
        rho[i] = gsw.rho(S[i],T[i],g*rho0*z/1e4)
        
        #Compute fluxes
        qEd[i] = q_Eddy(K_GM,H[i,1])
        qS[i] = q_South(qEk,qEd[i])
        qU[i] = q_Up(A,k,H[i,1])
        qN[i] = q_North(epsilon,H[i,1],S[i],T[i])
        
        dH[i,1:-1] = (qS[i] + qU[i] - qN[i])/A
        dh = np.ediff1d(dH[i])
        
        dS[i] = SaltChange(E,r,S[i],qS[i],qU[i],qN[i])
        #dS[i] = SalinityChange(S[i],h[i],dh,qS[i],qU[i],qN[i])
        dT[i] = HeatChange(T[i],h[i],dh,qS[i],qU[i],qN[i])
        
        H[i+1,1:-1] = H[i,1:-1] + dt/12*(23*dH[i,1:-1]-16*dH[i-1,1:-1]+5*dH[i-2,1:-1])
        Stot[i+1] = Stot[i] + dt/12*(23*dS[i]-16*dS[i-1]+5*dS[i-2])
        #S[i+1] = S[i] + dt/12*(23*dS[i]-16*dS[i-1]+5*dS[i-2])
        Ttot[i+1] = Ttot[i] + dt/12*(23*dT[i]-16*dT[i]+5*dT[i])
        h[i+1] = np.ediff1d(H[i+1])
        
        V[i+1,:2] = A * h[i+1]
        S[i+1] = Stot[i+1]/V[i+1]
        T[i+1,1] = Ttot[i+1]/V[i+1,1]

    qEd[-1] = q_Eddy(K_GM,H[-1,1])
    qS[-1] = q_South(qEk,qEd[-1])
    qU[-1] = q_Up(A,k,H[-1,1])
    qN[-1] = q_North(epsilon,H[-1,1],S[-1],T[-1])
    z = np.concatenate((H[-1,:2]+h[-1]/2,harray/2))
    rho[-1] = gsw.rho(S[-1],T[-1],g*rho0*z/1e4)
    
    '''Compress Data Points and Save'''
    
    #print('Saving: '+path+'\n\n')
    
    if not os.path.exists(path):
        os.makedirs(path)

    #ndp = 400 #Number of data points
    #ndp = 1000
    ndp = 5000
    entries = 1#round(sizet/ndp)

    H = (H[::entries]/1000).astype(np.float16)

    h = (h[::entries]/1000).astype(np.float16)

    S = (S[::entries]).astype(np.float16)

    T = (T[::entries]).astype(np.float16)

    rho = (rho[::entries]-1000).astype(np.float16)

    qEk = (qEk*np.ones(len(t[::entries]))/1e6).astype(np.float16)
    qEd = (qEd[::entries]/1e6).astype(np.float16)
    qS = (qS[::entries]/1e6).astype(np.float16)
    qU = (qU[::entries]/1e6).astype(np.float16)
    qN = (qN[::entries]/1e6).astype(np.float16)

    np.save(path+'/parameters.npy',parameters)
    
    np.save(path+'/IC.npy',IC)

    np.save(path+'/t',t[::entries].astype(np.float32))

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
    