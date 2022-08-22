import numpy as np

#d is the tip-sample distance in nm
#V is the voltage bias in eV
#Vmin is the minimum potential of the substrate periodic potential in eV
#zm is the range of the potential set to a constant value near the sumple interface (nm)
def build_potential_no_dielectric(n,zmin,w,Vg,V0,d,phi,V,zm):
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

class Numerov_Cooley_integration():
    def __init__(self,x,pot,tol=0.0001):
        self.npts=np.len(x)
        self.x=x
        self.pot=pot
        self.dx=(np.max(self.x)-np.min(self.x))/len(self.x)
        self.wf=[]
        self.E=[]
        self.tol=tol
        
    def optimize_energy(self,E):
        dE=self.tol+1
        counter=0
        while dE<self.tol:
            dE,R=self.integrator(E)
            counter+=1
        self.E.append(E)
        self.wf.append(R)
        
    def integrator(self,E):
        Yin=np.zeros(self.npts)
        Rin=np.zeros(self.npts)
        Yout=np.zeros(self.npts)
        Rout=np.zeros(self.npts)
        counter=0
        for i in self.pot[::-1]:
            if i>E:
                xlim=self.npts-counter
                break
            counter+=1
        for i in range(2,xlim):
            Rout[i]=self.dx**2*(self.pot[i-1]-E)*Rout[i-1]
            i=self.npts-i
            Rin[i]=self.dx**2*(self.pot[i+1]-E)*Rin[i+1]
        mp=self.argmax(Rout)
        Rin/=Rin[mp]
        Rout/=Rout[mp]
        R=Rin+Rout
        R[mp]=1.0
        Yout=R[mp-1]*(1-self.dx**2/12*(self.pot[mp-1]-E))
        Yin=R[mp+1]*(1-self.dx**2/12*(self.pot[mp+1]-E))
        Ym=R[mp]*(1-self.dx**2/12*(self.pot[mp]-E))
        dE=(-Yout+2*Ym-Yin)/self.dx**2+(self.pot[mp]-E)*R[mp]/(sum(R)*self.dx)
        
        return dE,R