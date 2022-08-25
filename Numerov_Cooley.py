import numpy as np
import matplotlib.pyplot as plt

#d is the tip-sample distance in nm
#V is the voltage bias in eV
#Vmin is the minimum potential of the substrate periodic potential in eV
#zm is the range of the potential set to a constant value near the sumple interface (nm)
def build_potential_no_dielectric(n,zmin,w,Vg,V0,d,phi_t,phi_s,V,zm):
    d*=1e-9 #convert nm to m
    zm*=1e-9 #convert nm to m
    zmin*=1e-9 #convert nm to m
    w*=1e-9 #convert nm to m
    Vg*=1.60218e-19 #eV to J
    V*=1.60218e-19 #eV to J
    V0*=1.60218e-19 #eV to J
    phi*=1.60218e-19 #eV to J
    e0=8.8541878128e-12 #F/m
    e=1.60217663e-19
    
    x=np.linspace(-zmin,d,n)
    field_pot=phi-V*(d-x)/d
    image_pot_sub=-e**2/4/x/e0/np.pi
    image_pot_tip=-e**2/4/abs(d-x)/e0/np.pi
    #image_pot_tip=np.zeros(n)
    pot=field_pot+image_pot_sub+image_pot_tip
    pot=np.nan_to_num(pot)
    
    pot*=np.heaviside(x-zm,1)
    Vmin=image_pot_sub[np.argmin(abs(x-zm))]
    pot+=np.heaviside((x-zm)*-1,0)*(-V0-Vg)
    for i in range(len(x)):
        if pot[i]<(V0-Vg):
            max_index=i
            break
    
    pot*=np.heaviside(x,1)
    bulk_pot=-Vg*np.cos(2*np.pi*x/w)-V0
    pot[:np.argmin(abs(x))+1]+=bulk_pot[:np.argmin(abs(x))+1]
    
    for i in range(len(x)):
        if pot[i]<(-V0-Vg):
            pot[i]=(-V0-Vg)
    
    #convert x back to nm
    x*=1e9
    
    return x,pot

def harmonic_test(npts,xmin,xmax,w):
    
    x=np.linspace(xmin,xmax,npts)
    m=9.11e-31
    pot=(1/2)*m*w**2*x**2
    
    return x,pot

def particle_in_a_box(n,L):
    
    x=np.linspace(-L/2,L/2,n)
    y=np.zeros(n)
    
    return x,y

class Numerov_Cooley():
    #x is in nm, pot is in J
    def __init__(self,x,pot,tol=0.0000001):
        h=6.626e-34/np.pi/2 #J*s
        m=9.11e-31 #kg
        self.k=2*m/h**2*1e-18 #1/nm**2/J
        
        self.npts=len(x)
        self.x=x
        self.pot=pot*self.k
        self.pot-=np.min(self.pot)
        self.dx=(np.max(self.x)-np.min(self.x))/len(self.x)
        self.wf=[]
        self.E=[]
        self.tol=tol
        self.nodes=[]
        self.nstates=0
        
    #E is the trial energy in eV
    def main(self,E):
        E*=self.k/6.242e18
        self.optimize_energy(E)
        
    def node_counter(self,R):
        counter=0
        for i in range(self.npts-1):
            if R[i]*R[i+1]<0:
                counter+=1
        
        return counter
        
    def optimize_energy(self,E):
        dE=self.tol+1
        counter=0
        while np.abs(dE)>self.tol:
            dE,R=self.integrator(E)
            E+=dE
            counter+=1
        print('energy converged after {} iterations'.format(counter))
        nodes=self.node_counter(R)
        if nodes not in self.nodes:
            self.nodes.append(nodes)
            self.nstates+=1
            self.E.append(E)
            self.wf.append(R)

    def integrator(self,E):
        Yin=np.zeros(self.npts)
        Rin=np.zeros(self.npts)
        Yout=np.zeros(self.npts)
        Rout=np.zeros(self.npts)
        U=self.pot-E
        counter=2
        for i in self.pot[::-1][:-2]:
            if i>E:
                xlim=self.npts-counter
                break
            counter+=1
        else:
            xlim=self.npts-2
        small_val=0.0000005
        Rout[1]=small_val
        Rin[-2]=small_val
        Yout[1]=small_val*(1-self.dx**2/12*U[1])
        Yin[-2]=small_val*(1-self.dx**2/12*U[-2])
        for i in range(2,xlim):
            tempvar=self.dx**2*U[i-1]*Rout[i-1]+2*Yout[i-1]-Yout[i-2]
            if np.isnan(tempvar):
                Yout/=1e100
                Rout/=1e100
                tempvar=self.dx**2*U[i-1]*Rout[i-1]+2*Yout[i-1]-Yout[i-2]
            Yout[i]=tempvar
            Rout[i]=Yout[i]/(1-self.dx**2/12*U[i])
            
            i=self.npts-i-1
            
            tempvar=self.dx**2*U[i+1]*Rin[i+1]+2*Yin[i+1]-Yin[i+2]
            if np.isnan(tempvar):
                Yin/=1e100
                Rin/=1e100
                tempvar=self.dx**2*U[i+1]*Rin[i+1]+2*Yin[i+1]-Yin[i+2]
            Yin[i]=tempvar
            Rin[i]=Yin[i]/(1-self.dx**2/12*U[i])
        maxima=self.find_maxima(Rout)
        mp=maxima[np.argmax(np.array([Rout[i] for i in maxima]))]
        Rin/=Rin[mp]
        Rout/=Rout[mp]
        R=np.zeros(self.npts)
        R[:mp]+=Rout[:mp]
        R[mp+1:]+=Rin[mp+1:]
        R[mp]=1.0
        Yout=R[mp-1]*(1-self.dx**2/12*U[mp-1])
        Yin=R[mp+1]*(1-self.dx**2/12*U[mp+1])
        Ym=R[mp]*(1-self.dx**2/12*U[mp])
        dE=((-Yout+2*Ym-Yin)/self.dx**2+U[mp])/(sum(R**2))
        
        return dE,R
    
    def find_maxima(self,R):
        maxima=[]
        for i in range(self.npts-1):
            if R[i+1]-R[i]<0.0:
                maxima.append(i)
                
        return maxima
                
    
    def cleanup_output(self):
        for i in range(self.nstates):
            self.E[i]*=6.242e18/self.k
        self.pot*=6.242e18
    
    def plot_output(self):
        plt.figure()
        plt.plot(self.x,self.pot,color='red',lw=2)
        for i in range(len(self.E)):
            plt.plot([self.x[0],self.x[-1]],[self.E[i] for j in range(2)],color='black',lw=2,linestyle='dashed')
            plt.plot(self.x,self.wf[i]+self.E[i],color='black',lw=2)
        plt.show()