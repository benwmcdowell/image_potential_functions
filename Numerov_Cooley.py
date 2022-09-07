import numpy as np
import matplotlib.pyplot as plt
import warnings

#d is the tip-sample distance in nm
#V is the voltage bias in eV
#Vmin is the minimum potential of the substrate periodic potential in eV
#zm is the range of the potential set to a constant value near the sumple interface (nm)
def build_potential_with_dielectric(n,zmin,w,Vg,V0,d,phis,phit,V,zm,t,e1,Vcbm):
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    d*=1e-9 #convert nm to m
    zm*=1e-9 #convert nm to m
    zmin*=1e-9 #convert nm to m
    w*=1e-9 #convert nm to m
    t*=1e-9 #convert nm to m
    Vg*=1.60218e-19 #eV to J
    V*=1.60218e-19 #eV to J
    V0*=1.60218e-19 #eV to J
    phis*=1.60218e-19 #eV to J
    phit*=1.60218e-19 #eV to J
    Vcbm*=1.60218e-19 #eV to J
    e0=8.8541878128e-12 #F/m
    e1*=e0
    e=1.60217663e-19 #C
    
    x=np.linspace(-zmin,d,n)
    edrop=V*(e0*(t))/(e1*(d-t)+e0*(t))
    field_pot=phis+(phit-phis)*x/d+((V-edrop)/(d-t)*(x-t)+edrop)*np.heaviside(x-t,1)+(edrop*x/t)*np.heaviside(t-x,0)
    
    image_pot_sub=-e**2/4/e0/np.pi/2/2
    image_pot_tip=-e**2/4/e0/np.pi/2/2
    image_pot_sub_sum=np.zeros(n)
    image_pot_tip_sum=np.zeros(n)
    sum_threshold=1/1000
    for i in range(np.argmin(abs(x)),n):
        dT=1
        dS=1
        B=(e1/e0-1)/(e1/e0+1)
        
        counter=1
        image_pot_sub_sum[i]+=B/(x[i]-t)
        k=-(1-B**2)/B
        while abs(image_pot_sub_sum[i]*sum_threshold)<abs(dS):
            dS=k*(-B)**counter/(x[i]-t+counter*d)
            image_pot_sub_sum[i]+=dS
            counter+=1
            
        counter=0
        while abs(image_pot_tip_sum[i]*sum_threshold)<abs(dT):
            dT=(-1)**counter/(abs(d-x[i])+counter*d)
            image_pot_tip_sum[i]+=dT
            counter+=1
            
    image_pot_sub*=image_pot_sub_sum
    image_pot_tip*=image_pot_tip_sum
    
    image_sum=image_pot_tip+image_pot_sub
    for i in range(len(x)):
        if x[i]>0 and x[i]<t:
            image_sum[i]=Vcbm
    
    pot=field_pot+image_sum
    pot=np.nan_to_num(pot)
    
    pot*=np.heaviside(x,1)
    tempvar=pot[np.argmin(abs(x-t-zm))]
    for i in range(len(x)):
        if x[i]>t and x[i]<t+zm:
            pot[i]=tempvar
    bulk_pot=-Vg*np.cos(2*np.pi*x/w)-V0
    pot[:np.argmin(abs(x))+1]+=bulk_pot[:np.argmin(abs(x))+1]
    
    for i in range(len(x)):
        if pot[i]<(-V0-Vg) and x[i]<d/2:
            pot[i]=(-V0-Vg)
        elif pot[i]<(-V0-Vg+V) and x[i]>d/2:
            pot[i]=(-V0-Vg+V)
    
    #convert x back to nm
    x*=1e9
    
    warnings.filterwarnings("default",category=RuntimeWarning)
    plt.figure()
    plt.plot(x,pot)
    plt.show()
    return x,pot

#d is the tip-sample distance in nm
#V is the voltage bias in eV
#Vmin is the minimum potential of the substrate periodic potential in eV
#zm is the range of the potential set to a constant value near the sumple interface (nm)
def build_potential_no_dielectric(n,zmin,w,Vg,V0,d,phis,phit,V,zm):
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    d*=1e-9 #convert nm to m
    zm*=1e-9 #convert nm to m
    zmin*=1e-9 #convert nm to m
    w*=1e-9 #convert nm to m
    Vg*=1.60218e-19 #eV to J
    V*=1.60218e-19 #eV to J
    V0*=1.60218e-19 #eV to J
    phis*=1.60218e-19 #eV to J
    phit*=1.60218e-19 #eV to J
    e0=8.8541878128e-12 #F/m
    e=1.60217663e-19 #C
    
    x=np.linspace(-zmin,d,n)
    field_pot=phis+(phit-phis+V)*x/d
    
    image_pot_sub=-e**2/4/e0/np.pi/2/2
    image_pot_tip=-e**2/4/e0/np.pi/2/2
    image_pot_sub_sum=np.zeros(n)
    image_pot_tip_sum=np.zeros(n)
    sum_threshold=1/10000
    for i in range(np.argmin(abs(x)),n):
        dT=1
        dS=1
        
        counter=0
        while image_pot_sub_sum[i]*sum_threshold<dS:
            dS=(-1)**counter/(x[i]+counter*d)
            image_pot_sub_sum[i]+=dS
            counter+=1
            
        counter=0
        while image_pot_tip_sum[i]*sum_threshold<dT:
            dT=(-1)**counter/(abs(d-x[i])+counter*d)
            image_pot_tip_sum[i]+=dT
            counter+=1
            
    image_pot_sub*=image_pot_sub_sum
    image_pot_tip*=image_pot_tip_sum
    
    pot=field_pot+image_pot_sub+image_pot_tip
    pot=np.nan_to_num(pot)
    
    pot*=np.heaviside(x-zm,1)
    pot+=np.heaviside((x-zm)*-1,0)*(-V0-Vg)
    
    pot*=np.heaviside(x,1)
    bulk_pot=-Vg*np.cos(2*np.pi*x/w)-V0
    pot[:np.argmin(abs(x))+1]+=bulk_pot[:np.argmin(abs(x))+1]
    
    for i in range(len(x)):
        if pot[i]<(-V0-Vg) and x[i]<d/2:
            pot[i]=(-V0-Vg)
        elif pot[i]<(-V0-Vg+V) and x[i]>d/2:
            pot[i]=(-V0-Vg+V)
    
    #convert x back to nm
    x*=1e9
    
    warnings.filterwarnings("default",category=RuntimeWarning)
    
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
    def __init__(self,x,pot,tol=0.0000001,pot_type='default',filter_mode='nodes',suppress_output=True):
        h=6.626e-34/np.pi/2 #J*s
        m=9.11e-31 #kg
        self.k=2*m/h**2*1e-18 #1/nm**2/J
        
        self.npts=len(x)
        self.x=x
        self.pot=pot*self.k
        self.pot_shift=np.min(self.pot)
        self.pot-=self.pot_shift
        self.dx=(np.max(self.x)-np.min(self.x))/len(self.x)
        self.wf=[]
        self.E=[]
        self.tol=tol
        self.nodes=[]
        self.xavg=[]
        self.nstates=0
        self.pot_type=pot_type
        self.filter_mode=filter_mode
        self.suppress_output=suppress_output
        
    #E is the trial energy in eV
    def main(self,E):
        E*=self.k/6.242e18
        E-=self.pot_shift
        self.optimize_energy(E)
        
    def node_counter(self,R):
        counter=0
        if self.pot_type=='default':
            search_range=range(np.argmin(abs(self.x)),np.argmax(self.pot))
        else:
            search_range=range(self.npts-1)
        for i in search_range:
            if R[i]*R[i+1]<0:
                counter+=1
        
        return counter
        
    def optimize_energy(self,E):
        dE=self.tol+1
        counter=0
        if not hasattr(self,'opt_fig'):
            self.opt_fig,self.opt_ax=plt.subplots(1,1)
            self.opt_ax.set(xlabel='# of energy corrections', ylabel='trial eigenvalue / eV')
        steps=[]
        trial_energies=[]
        while np.abs(dE)>self.tol:
            trial_energies.append(E)
            steps.append(counter)
            dE,R=self.integrator(E)
            E+=dE
            counter+=1
        self.opt_ax.plot(steps,(np.array(trial_energies)+self.pot_shift)*6.242e18/self.k,lw=2)
        self.opt_fig.canvas.draw()
        if not self.suppress_output:
            print('energy converged after {} iterations'.format(counter))
        nodes=self.node_counter(R)
        #R=self.normalize_wf(R)
        xavg=self.avg_pos(R)
        
        if self.filter_mode=='nodes':
            if nodes not in self.nodes:
                self.nodes.append(nodes)
                self.nstates+=1
                self.E.append(E)
                self.wf.append(R)
                self.xavg.append(xavg)
            if nodes in self.nodes:
                i=self.nodes.index(nodes)
                if abs(xavg)<abs(self.xavg[i]):
                    self.nodes[i]=nodes
                    self.E[i]=E
                    self.wf[i]=R
                    self.xavg[i]=xavg
                    
        elif self.filter_mode=='energy':
            if len(self.E)>0:
                energy_check=True
                new_val=False
            else:
                energy_check=False
                new_val=True
            counter=0
            while energy_check:
                if E>self.E[counter]-self.tol and E<self.E[counter]+self.tol:
                    energy_check=False
                counter+=1
                if counter==len(self.E):
                    new_val=True
                    energy_check=False
            if new_val:
                self.nodes.append(nodes)
                self.nstates+=1
                self.E.append(E)
                self.wf.append(R)
                self.xavg.append(xavg)
            else:
                i=np.argmin(abs(np.array(self.E)-E))
                if abs(xavg)<abs(self.xavg[i]):
                    self.nodes[i]=nodes
                    self.E[i]=E
                    self.wf[i]=R
                    self.xavg[i]=xavg
                    
        elif self.filter_mode=='none':
            self.nodes.append(nodes)
            self.nstates+=1
            self.E.append(E)
            self.wf.append(R)
            self.xavg.append(xavg)

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
        try:
            maxima=self.find_maxima(Rout)
            mp=maxima[np.argmax(np.array([Rout[i] for i in maxima]))]
        except ValueError:
            self.pot_type='temp'
            maxima=self.find_maxima(Rout)
            mp=maxima[np.argmax(np.array([Rout[i] for i in maxima]))]
            self.pot_type='default'
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
    
    def normalize_wf(self,R):
        R/=np.linalg.norm(R)
        
        return R
    
    def avg_pos(self,R):
        xavg=sum(self.x*R/np.linalg.norm(R))
        
        return xavg
    
    def find_maxima(self,R):
        if self.pot_type=='default':
            mpmin=np.argmin(abs(self.x-1))
        else:
            mpmin=0
            
        maxima=[]
        for i in range(mpmin,self.npts-1-mpmin):
            if R[i+1]-R[i]<0.0:
                maxima.append(i)
                
        return maxima
                
    def overlay_Fermi_levels(self,Vb):
        if not hasattr(self,'wf_fig'):
            self.wf_fig,self.wf_ax=plt.subplots(1,1)
            self.wf_ax.set(xlabel='position / nm', ylabel='energy / eV')
            
        mp=np.argmax(self.pot)
            
        counter=0
        for i in range(mp+1):
            if self.pot[i]<0.0:
                counter+=1
        self.wf_ax.plot(self.x[:counter+1],[0 for i in range(counter+1)],lw=2,color='blue')
        
        counter=0
        for i in range(mp,self.npts):
            if self.pot[i]<Vb:
                counter+=1
        self.wf_ax.plot(self.x[self.npts-counter-1:],[Vb for i in range(counter+1)],lw=2,color='blue',label='$E_F$')
        
        self.wf_fig.canvas.draw()
    
    def cleanup_output(self):
        self.pot_shift*=6.242e18/self.k
        for i in range(self.nstates):
            self.E[i]*=6.242e18/self.k
            self.E[i]+=self.pot_shift
        self.pot*=6.242e18/self.k
        self.pot+=self.pot_shift
    
    def plot_output(self):
        if not hasattr(self,'wf_fig'):
            self.wf_fig,self.wf_ax=plt.subplots(1,1)
            self.wf_ax.set(xlabel='position / nm', ylabel='energy / eV')
        self.wf_ax.plot(self.x,self.pot,color='red',lw=2,label='potential')
        for i in range(len(self.E)):
            self.wf_ax.plot([self.x[0],self.x[-1]],[self.E[i] for j in range(2)],color='black',lw=2,linestyle='dashed')
            self.wf_ax.plot(self.x,self.wf[i]+self.E[i],color='black',lw=2)
        self.wf_ax.legend()
        self.wf_fig.canvas.draw()