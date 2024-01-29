import matplotlib.pyplot as plt
import warnings
import math
import torch

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
    V=V*1.60218e-19 #eV to J
    V0*=1.60218e-19 #eV to J
    phis*=1.60218e-19 #eV to J
    phit*=1.60218e-19 #eV to J
    Vcbm*=1.60218e-19 #eV to J
    e0=8.8541878128e-12 #F/m
    e1*=e0
    e=1.60217663e-19 #C
    
    #reference Vcbm to field potential
    Vcbm-=phis    
    x=torch.linspace(-zmin,d,n)
    edrop=V*(e0*(t))/(e1*(d-t)+e0*(t))
    field_pot=phis+(phit-phis)*x/d+((V-edrop)/(d-t)*(x-t)+edrop)*torch.heaviside(x-t,1)+(edrop*x/t)*torch.heaviside(t-x,0)
    
    image_pot_sub=-e**2/4/e0/torch.pi/2/2
    image_pot_tip=-e**2/4/e0/torch.pi/2/2
    image_pot_sub_sum=torch.zeros(n)
    image_pot_tip_sum=torch.zeros(n)
    sum_threshold=1/1000
    for i in range(torch.argmin(abs(x)),n):
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
            
    tempvar=image_sum[torch.argmin(abs(x-t-zm))]
    for i in range(len(x)):
        if x[i]>t and x[i]<t+zm:
            image_sum[i]=tempvar
    
    pot=field_pot+image_sum
    pot=torch.nan_to_num(pot)
    
    pot*=torch.heaviside(x,1)
    bulk_pot=-Vg*torch.cos(2*math.pi*x/w)-V0
    pot[:torch.argmin(abs(x))+1]+=bulk_pot[:torch.argmin(abs(x))+1]
    
    for i in range(len(x)):
        if pot[i]<(-V0-Vg) and x[i]<d/2:
            pot[i]=(-V0-Vg)
        elif pot[i]<(-V0-Vg+V) and x[i]>d/2:
            pot[i]=(-V0-Vg+V)
    
    #convert x back to nm
    x*=1e9
    
    warnings.filterwarnings("default",category=RuntimeWarning)

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
    V=V*1.60218e-19 #eV to J
    V0*=1.60218e-19 #eV to J
    phis*=1.60218e-19 #eV to J
    #phit=phit*1.60218e-19 #eV to J
    e0=8.8541878128e-12 #F/m
    e=1.60217663e-19 #C
    
    x=torch.linspace(-zmin,d.item(),n)
    field_pot=phis+(phit*1.60218e-19-phis+V)*x/d
    
    image_pot_sub=-e**2/4/e0/torch.pi/2/2
    image_pot_tip=-e**2/4/e0/torch.pi/2/2
    image_pot_sub_sum=torch.zeros(n)
    image_pot_tip_sum=torch.zeros(n)
    sum_threshold=1/10000
    for i in range(torch.argmin(abs(x)),n):
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
    pot=torch.nan_to_num(pot)
    
    pot*=torch.heaviside(x-zm,torch.ones(1))
    pot+=torch.heaviside((x-zm)*-1,torch.zeros(1))*(-V0-Vg)
    
    pot*=torch.heaviside(x,torch.ones(1))
    bulk_pot=-Vg*torch.cos(2*math.pi*x/w)-V0
    pot[:torch.argmin(abs(x))+1]+=bulk_pot[:torch.argmin(abs(x))+1]
    
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
    
    x=torch.linspace(xmin,xmax,npts)
    m=9.11e-31
    pot=(1/2)*m*w**2*x**2
    
    return x,pot

def particle_in_a_box(n,L):
    
    x=torch.linspace(-L/2,L/2,n)
    y=torch.zeros(n)
    
    return x,y

class Numerov_Cooley():
    #x is in nm, pot is in J
    def __init__(self,x,pot,tol=0.0000001,pot_type='default',max_steps=100):
        h=6.626e-34/torch.pi/2 #J*s
        m=9.11e-31 #kg
        self.k=2*m/h**2*1e-18 #1/nm**2/J
        
        self.npts=len(x)
        self.x=x
        self.pot=pot*self.k
        self.pot_shift=torch.min(self.pot)
        self.pot-=self.pot_shift
        self.dx=(torch.max(self.x)-torch.min(self.x))/len(self.x)
        self.wf=[]
        self.E=[]
        self.tol=tol
        self.nstates=0
        self.pot_type=pot_type
        self.max_dE=1.0
        self.max_steps=max_steps
        
    #E is the trial energy in eV
    def main(self,E):
        E*=self.k/6.242e18
        E=E-self.pot_shift
        self.optimize_energy(E)
        
    #loops main function with trial eigenvalues ranging from initial to final in nsteps steps
    def loop_main(self,initial,final,nsteps,nprocs=1):
        counter=0
        for i in torch.linspace(initial.item(),final.item(),nsteps):
            self.main(i)
            counter+=1
        
    def node_counter(self,R):
        counter=0
        if self.pot_type=='default':
            search_range=range(torch.argmin(abs(self.x)),torch.argmax(self.pot))
        else:
            search_range=range(self.npts-1)
        for i in search_range:
            if R[i]*R[i+1]<0:
                counter+=1
        
        return counter
        
    def optimize_energy(self,E):
        dE=self.tol+1
        counter=0
        steps=[]
        trial_energies=[]
        while torch.abs(torch.Tensor([dE]))>self.tol:
            if counter>self.max_steps:
                break
            trial_energies.append(E)
            steps.append(counter)
            dE,R,mp=self.integrator(E)
            if dE>self.max_dE:
                dE=self.max_dE
            E+=dE
            counter+=1
        
        self.nstates+=1
        self.E.append(E)
        self.wf.append(R)

    def integrator(self,E):
        Yin=torch.zeros(self.npts)
        Rin=torch.zeros(self.npts)
        Yout=torch.zeros(self.npts)
        Rout=torch.zeros(self.npts)
        U=self.pot-E
        counter=2
        #sets right-hand integration limit to the rightmost potential maxima
        #for i in range(3,self.npts):
        #    counter+=1
        #    if self.pot[self.npts-i]>self.pot[self.npts-i-1]:
        for i in torch.flip(U,(0,))[2:]:
            counter+=1
            if i>0:
                xlim=self.npts-counter
                break
        else:
            xlim=self.npts-2
            
        small_val=0.0000005
        Rout[1]=small_val
        Yout[1]=small_val*(1-self.dx**2/12*U[1])
        Rin[xlim]=small_val
        Yin[xlim]=small_val*(1-self.dx**2/12*U[-2])
            
        for i in range(2,xlim):
            tempvar=self.dx**2*U[i-1]*Rout[i-1]+2*Yout[i-1]-Yout[i-2]
            if torch.isnan(tempvar):
                Yout/=1e100
                Rout/=1e100
                tempvar=self.dx**2*U[i-1]*Rout[i-1]+2*Yout[i-1]-Yout[i-2]
            Yout[i]=tempvar
            Rout[i]=Yout[i]/(1-self.dx**2/12*U[i])
            
            i=xlim+2-i-1
            
            tempvar=self.dx**2*U[i+1]*Rin[i+1]+2*Yin[i+1]-Yin[i+2]
            if torch.isnan(tempvar):
                Yin/=1e100
                Rin/=1e100
                tempvar=self.dx**2*U[i+1]*Rin[i+1]+2*Yin[i+1]-Yin[i+2]
            Yin[i]=tempvar
            Rin[i]=Yin[i]/(1-self.dx**2/12*U[i])
        
        #deals with error of taking argmax on tempy array/tensor
        #in numpy this returns an ValueError, in pytorch its an IndexError
        try:
            maxima=self.find_maxima(Rout)
            mp=maxima[torch.argmax(torch.Tensor([Rout[i] for i in maxima]))]
        except IndexError:
            self.pot_type='temp'
            maxima=self.find_maxima(Rout)
            mp=maxima[torch.argmax(torch.Tensor([Rout[i] for i in maxima]))]
            self.pot_type='default'
        Rin=Rin/Rin[mp]
        Rout=Rout/Rout[mp]
        self.Rin=Rin
        self.Rout=Rout
        R=torch.zeros(self.npts)
        R[:mp]=R[:mp]+Rout[:mp]
        R[mp+1:]+=Rin[mp+1:]
        R[mp]=1.0
        Yout=R[mp-1]*(1-self.dx**2/12*U[mp-1])
        Yin=R[mp+1]*(1-self.dx**2/12*U[mp+1])
        Ym=R[mp]*(1-self.dx**2/12*U[mp])
        dE=((-Yout+2*Ym-Yin)/self.dx**2+U[mp])/(sum(R**2))
        
        return dE,R,self.x[mp]
    
    def normalize_wf(self,R):
        R/=torch.norm(R)
        
        return R
    
    def find_maxima(self,R):
        if self.pot_type=='default':
            mpmax=torch.argmax(self.pot)
        else:
            mpmax=self.npts-1
            
        mpmin=1
            
        maxima=[]
        for i in range(mpmin,mpmax):
            if abs(R)[i+1]-abs(R)[i]<0.0 and abs(R)[i-1]-abs(R)[i]<0.0:
                maxima.append(i)
                
        return maxima
                
    def cleanup_output(self):
        self.pot_shift*=6.242e18/self.k
        for i in range(self.nstates):
            self.E[i]*=6.242e18/self.k
            self.E[i]+=self.pot_shift
        self.pot*=6.242e18/self.k
        self.pot+=self.pot_shift
        
class Numerov_Cooley_layer(torch.autograd.Function):
    @staticmethod
    def forward(ctx,z0,phit,nstates,npts,zmin,w,Vg,V0,phis,peak_energies,peak_heights,zm,loop_pts):
        ctx.save_for_backward(z0,phit)
        energy_min=0
        energy_tol=0.0001
        calc_energies=torch.ones(len(nstates))
        for i in range(len(nstates)):
            d_opt=z0+peak_heights[i]
            x,pot=build_potential_no_dielectric(npts,zmin,w,Vg,V0,d_opt,phis,phit,peak_energies[i],zm)
            tempvar=Numerov_Cooley(x,pot)
            tempvar.loop_main(peak_energies[i]-1,peak_energies[i]+1,loop_pts)
            tempvar.cleanup_output()
            
            temp_energies=[]
            counter=[]
            for j in tempvar.E:
                if j>energy_min:
                    if len(temp_energies)==0:
                        temp_energies.append(j)
                        counter.append(1)
                    else:
                        for k in range(len(temp_energies)):
                            if abs(j-temp_energies[k])<energy_tol:
                                #takes rolling average of energy value
                                temp_energies[k]=(temp_energies[k]*counter[k]+j)/(counter[k]+1)
                                counter[k]+=1
                                break
                        else:
                            temp_energies.append(j)
                            counter.append(1)
            
            if len(temp_energies)==0:
                temp_energies=torch.zeros(len(nstates))
            else:
                temp_energies=torch.Tensor(temp_energies)
            calc_energies[i]=temp_energies[torch.argmin(abs(temp_energies-peak_energies[i]))]
        return calc_energies
    
    @staticmethod
    def backward(ctx,grad_output):
        z0_in,phit_in=ctx.saved_tensors
        grad_sum=sum(grad_output)
        return -12000*grad_sum,10000*grad_sum,None,None,None,None,None,None,None,None,None,None,None
        
class optimize_parameters():
    def __init__(self,peak_energies,peak_heights,sigma=None,dielectric=False,loop_pts=100,npts=5000,suppress_plotting=False,nprocs=1,zmin=0.2402093333333333*5,w=0.2402093333333333,Vg=4.2,V0=4.633858138635734,z0=0,phis=4.59,phit=4.59,zm=0.015,t=0.249595,e1=5.688,vcbm=3.78):
    
        self.nstates=torch.tensor([i for i in range(len(peak_energies))])
        self.peak_energies=peak_energies
        self.peak_heights=peak_heights
        
        self.dielectric=dielectric
        self.loop_pts=loop_pts
        self.nprocs=nprocs
        
        self.npts=npts
        self.zmin=zmin
        self.w=w
        self.Vg=Vg
        self.V0=V0
        self.z0=z0
        self.phis=phis
        self.phit=phit
        self.zm=zm
        self.t=t
        self.e1=e1
        self.vcbm=vcbm
        
        self.loss_trajectory=[]
        
        class model_without_dielectric(torch.nn.Module):
            def __init__(self,z0,phit):
                super().__init__()
                self.z0=torch.nn.Parameter(torch.tensor(z0, dtype=torch.float32), requires_grad=True)
                self.phit=torch.nn.Parameter(torch.tensor(phit, dtype=torch.float32), requires_grad=True)
                
            def forward(self,nstates,npts,zmin,w,Vg,V0,phis,peak_energies,peak_heights,zm,loop_pts):

                calc_energies=Numerov_Cooley_layer.apply(self.z0,self.phit,nstates,npts,zmin,w,Vg,V0,phis,peak_energies,peak_heights,zm,loop_pts)
                
                return calc_energies
                
        
        model=model_without_dielectric(self.phit,self.z0)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1)
        for i in range(100):
            e_predicted=model(self.nstates,self.npts,self.zmin,self.w,self.Vg,self.V0,self.phis,self.peak_energies,self.peak_heights,self.zm,self.loop_pts)
            loss=criterion(e_predicted,self.peak_energies)
            self.loss_trajectory.append(loss.item())
            print(i, loss.item())
                
            optimizer.zero_grad()
            print(model.phit.grad,model.z0.grad)
            loss.backward()
            print(model.phit.grad,model.z0.grad)
            optimizer.step()