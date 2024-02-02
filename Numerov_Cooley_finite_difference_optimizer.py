import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import scipy
from pathos.multiprocessing import ProcessPool
import sys
import os
import getopt
from scipy.special import eval_legendre
import pyperclip

#d is the tip-sample distance in nm
#V is the voltage bias in eV
#Vmin is the minimum potential of the substrate periodic potential in eV
#zm is the range of the potential set to a constant value near the sumple interface (nm)
def build_potential_with_dielectric(n,zmin,w,Vg,V0,d,phis,phit,V,zm,t,e1,Vcbm,R=0.0):
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    d*=1e-9 #convert nm to m
    zm*=1e-9 #convert nm to m
    zmin*=1e-9 #convert nm to m
    w*=1e-9 #convert nm to m
    t*=1e-9 #convert nm to m
    R*=1e-9 #convert nm to m
    Vg*=1.60218e-19 #eV to J
    V*=1.60218e-19 #eV to J
    V0*=1.60218e-19 #eV to J
    phis*=1.60218e-19 #eV to J
    phit*=1.60218e-19 #eV to J
    Vcbm*=1.60218e-19 #eV to J
    e0=8.8541878128e-12 #F/m
    e1*=e0
    e=1.60217663e-19 #C
    
    #reference Vcbm to field potential
    Vcbm-=phis    
    x=np.linspace(-zmin,d,n)
    edrop=V*(e0*(t))/(e1*(d-t)+e0*(t))
    sum_threshold=1/1000
    if R==0:
        field_pot=phis+(phit-phis)*x/d+((V-edrop)/(d-t)*(x-t)+edrop)*np.heaviside(x-t,1)+(edrop*x/t)*np.heaviside(t-x,0)
    else:
        field_pot=-(V+phit-phis-edrop)*np.sqrt(2)
        a=np.sqrt((d+R)**2-R**2)
        tau0=np.arcsinh(a/R)
        tau=np.arcsinh(2*a*x/(x**2-a**2))
        field_pot*=np.sqrt(np.cosh(tau)-1)
        
        tempvar=0
        dt=1
        i=0
        while np.max(abs(dt))>np.max(abs(tempvar))*sum_threshold:
            dt=np.exp(-(i+1/2)*tau0)/np.sinh((i+1/2)*tau0)*np.sinh((i+1/2)*tau)*eval_legendre(i,1)
            tempvar+=dt
            i+=1
            
        field_pot*=tempvar
        field_pot+=edrop
        field_pot*=np.heaviside(x-t,1)
        field_pot+=phis
        field_pot+=((edrop*x/t)+(phit-phis)*x/d)*np.heaviside(t-x,0)
        
    
    image_pot_sub=-e**2/4/e0/np.pi/2/2
    image_pot_tip=-e**2/4/e0/np.pi/2/2
    image_pot_sub_sum=np.zeros(n)
    image_pot_tip_sum=np.zeros(n)
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
            
    tempvar=image_sum[np.argmin(abs(x-t-zm))]
    for i in range(len(x)):
        if x[i]>t and x[i]<t+zm:
            image_sum[i]=tempvar
    
    pot=field_pot+image_sum
    pot=np.nan_to_num(pot)
    
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

#d is the tip-sample distance in nm
#V is the voltage bias in eV
#Vmin is the minimum potential of the substrate periodic potential in eV
#zm is the range of the potential set to a constant value near the sumple interface (nm)
def build_potential_no_dielectric(n,zmin,w,Vg,V0,d,phis,phit,V,zm,R=0.0):
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    d*=1e-9 #convert nm to m
    zm*=1e-9 #convert nm to m
    zmin*=1e-9 #convert nm to m
    w*=1e-9 #convert nm to m
    R*=1e-9 #convert nm to m
    Vg*=1.60218e-19 #eV to J
    V*=1.60218e-19 #eV to J
    V0*=1.60218e-19 #eV to J
    phis*=1.60218e-19 #eV to J
    phit*=1.60218e-19 #eV to J
    e0=8.8541878128e-12 #F/m
    e=1.60217663e-19 #C
    
    #sets convergence threshold of infinte sums
    sum_threshold=1/10000
    
    x=np.linspace(-zmin,d,n)
    if R==0:
        field_pot=phis+(phit-phis+V)*x/d
    else:
        field_pot=-(V+phit-phis)*np.sqrt(2)
        a=np.sqrt((d+R)**2-R**2)
        tau0=np.arcsinh(a/R)
        tau=np.arcsinh(2*a*x/(x**2-a**2))
        field_pot*=np.sqrt(np.cosh(tau)-1)
        
        tempvar=0
        dt=1
        i=0
        while np.max(abs(dt))>np.max(abs(tempvar))*sum_threshold:
            dt=np.exp(-(i+1/2)*tau0)/np.sinh((i+1/2)*tau0)*np.sinh((i+1/2)*tau)*eval_legendre(i,1)
            tempvar+=dt
            i+=1
            
        field_pot*=tempvar
        field_pot+=phis
        
    image_pot_sub=-e**2/4/e0/np.pi/2/2
    image_pot_tip=-e**2/4/e0/np.pi/2/2
    image_pot_sub_sum=np.zeros(n)
    image_pot_tip_sum=np.zeros(n)
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
    def __init__(self,x,pot,tol=0.0000001,pot_type='default',filter_mode='none',suppress_output=True,max_steps=100,wf_height=1,overlay_stitch_point=False,localization_cutoff=1,suppress_timing_output=False):
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
        self.peak_slope=[]
        self.nstates=0
        self.pot_type=pot_type
        self.suppress_output=suppress_output
        self.max_dE=1.0
        self.max_steps=max_steps
        self.wf_height=wf_height
        self.stitch_points=[]
        self.overlay_stitch_point=overlay_stitch_point
        self.suppress_timing_output=suppress_timing_output
        
    #E is the trial energy in eV
    def main(self,E):
        E*=self.k/6.242e18
        E-=self.pot_shift
        self.optimize_energy(E)
        if self.nprocs>1:
            return self.E[0]
        
    #loops main function with trial eigenvalues ranging from initial to final in nsteps steps
    def loop_main(self,initial,final,nsteps,nprocs=1):
        counter=0
        start=time.time()
        percentage_counter=np.array([25,50,75,100])
        self.nprocs=nprocs
        
        if self.nprocs>1:
            with ProcessPool(self.nprocs) as pool:
                self.E=pool.map(self.main,np.linspace(initial,final,nsteps))
            
        else:
            percentage_counter=np.round(percentage_counter/100*(nsteps-1))
            for i in np.linspace(initial,final,nsteps):
                self.main(i)
                counter+=1
                
                if counter in percentage_counter and not self.suppress_timing_output:
                    print('{}% finished with range of trial eigenvalues. {} s elapsed so far'.format(round(counter/(nsteps-1)*100),time.time()-start))
        
    def node_counter(self,R):
        import numpy as np
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
        import numpy as np
        dE=self.tol+1
        counter=0
        if not hasattr(self,'opt_fig') and not self.suppress_output:
            self.opt_fig,self.opt_ax=plt.subplots(1,1)
            self.opt_ax.set(xlabel='# of energy corrections', ylabel='trial eigenvalue / eV')
        steps=[]
        trial_energies=[]
        while np.abs(dE)>self.tol:
            if counter>self.max_steps:
                if not self.suppress_output:
                    print('max iterations exceeded')
                break
            trial_energies.append(E)
            steps.append(counter)
            dE,R,mp=self.integrator(E)
            if dE>self.max_dE:
                dE=self.max_dE
            E+=dE
            counter+=1
        if not self.suppress_output:
            self.opt_ax.plot(steps,(np.array(trial_energies)+self.pot_shift)*6.242e18/self.k,lw=2)
            self.opt_fig.canvas.draw()
            print('energy converged after {} iterations'.format(counter))
        nodes=self.node_counter(R)
        #R=self.normalize_wf(R)
        
        self.nodes.append(nodes)
        self.nstates+=1
        self.E.append(E)
        self.wf.append(R)
        self.stitch_points.append(mp)

    def integrator(self,E):
        import numpy as np
        Yin=np.zeros(self.npts)
        Rin=np.zeros(self.npts)
        Yout=np.zeros(self.npts)
        Rout=np.zeros(self.npts)
        U=self.pot-E
        counter=2
        #sets right-hand integration limit to the rightmost potential maxima
        #for i in range(3,self.npts):
        #    counter+=1
        #    if self.pot[self.npts-i]>self.pot[self.npts-i-1]:
        for i in U[::-1][2:]:
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
            if np.isnan(tempvar):
                Yout/=1e100
                Rout/=1e100
                tempvar=self.dx**2*U[i-1]*Rout[i-1]+2*Yout[i-1]-Yout[i-2]
            Yout[i]=tempvar
            Rout[i]=Yout[i]/(1-self.dx**2/12*U[i])
            
            i=xlim+2-i-1
            
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
        self.Rin=Rin
        self.Rout=Rout
        R=np.zeros(self.npts)
        R[:mp]+=Rout[:mp]
        R[mp+1:]+=Rin[mp+1:]
        R[mp]=1.0
        Yout=R[mp-1]*(1-self.dx**2/12*U[mp-1])
        Yin=R[mp+1]*(1-self.dx**2/12*U[mp+1])
        Ym=R[mp]*(1-self.dx**2/12*U[mp])
        dE=((-Yout+2*Ym-Yin)/self.dx**2+U[mp])/(sum(R**2))
        
        return dE,R,self.x[mp]
    
    def normalize_wf(self,R):
        import numpy as np
        R/=np.linalg.norm(R)
        
        return R
    
    def avg_pos(self,R):
        import numpy as np
        xavg=sum(self.x*(R/np.linalg.norm(R))**2)
        
        return xavg
    
    def find_maxima(self,R):
        import numpy as np
        if self.pot_type=='default':
            mpmax=np.argmax(self.pot)
        else:
            mpmax=self.npts-1
            
        mpmin=1
            
        maxima=[]
        for i in range(mpmin,mpmax):
            if abs(R)[i+1]-abs(R)[i]<0.0 and abs(R)[i-1]-abs(R)[i]<0.0:
                maxima.append(i)
                
        return maxima
                
    def overlay_Fermi_levels(self,Vb):
        if not hasattr(self,'wf_fig'):
            self.wf_fig,self.wf_ax=plt.subplots(1,1)
            self.wf_ax.set(xlabel='position / nm', ylabel='energy / eV')
        
        if self.pot_type=='default':
            for i in range(2,self.npts-1):
                if self.pot[self.npts-i]>self.pot[self.npts-i+1] and self.pot[self.npts-i]>self.pot[self.npts-i-1]:
                    mp=i
        else:
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
        self.wf_ax.plot(self.x[-counter-1:],[Vb for i in range(counter+1)],lw=2,color='blue',label='$E_F$')
        
        self.wf_fig.canvas.draw()
        
        self.error=(self.E[np.argmin(abs(np.array(self.E)-Vb))]-Vb)/Vb*100
        
        print('closest eigenenergy to {} eV bias voltage is: {} eV with {} % error'.format(Vb,self.E[np.argmin(abs(np.array(self.E)-Vb))],self.error))
    
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
            self.wf_ax.plot(self.x,self.wf[i]/np.max(self.wf[i])*self.wf_height+self.E[i],color='black',lw=2)
            if self.overlay_stitch_point:
                self.wf_ax.scatter(self.stitch_points[i],self.E[i],color='green',s=100)
        self.wf_ax.legend()
        self.wf_fig.canvas.draw()
        
        
        
class optimize_parameters():
    def __init__(self,peak_energies,peak_heights,sigma=None,dielectric=False,loop_pts=100,npts=5000,nprocs=1,zmin=0.236965*5,w=0.236965,Vg=4.2,V0=4.633858138635734,z0=0,phis=4.59,phit=4.59,zm=0.015,t=0.249595,e1=5.688,vcbm=3.78,R=0.0,lr=1,e_threshold=1e-4):
    
        self.nstates=np.array([i for i in range(len(peak_energies))])
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
        self.R=R
            
        self.start=time.time()
        
        if not self.dielectric:
            z0_traj=[]
            phit_traj=[]
            R_traj=[]
            esum_traj=[]
            emag_traj=[]
            
            for i in range(100):
                z0_traj.append(self.z0)
                phit_traj.append(self.phit)
                R_traj.append(self.R)
                errors=self.model_no_dielectric(self.z0,self.phit,self.R)
                esum=sum(errors)
                emag=np.linalg.norm(errors)
                
                #uncomment to see print output of optimization progression
                print(self.z0,self.phit,self.R,emag)
                
                esum_traj.append(esum)
                emag_traj.append(emag)
                grad=self.calc_grad()
                self.z0-=lr*emag*np.average([grad[0,1]-self.z0,self.z0-grad[0,0]])
                self.phit-=lr*emag*np.average([grad[1,1]-self.phit,self.phit-grad[1,0]])
                self.R-=lr*emag*np.average([grad[2,1]-self.R,self.R-grad[2,0]])
                
                if len(emag_traj)>2:
                    if abs(emag_traj[-1]-emag_traj[-2])<e_threshold:
                        break
                
        elif self.dielectric:
            phis_traj=[]
            vcbm_traj=[]
            e1_traj=[]
            esum_traj=[]
            emag_traj=[]
            
            for i in range(100):
                phis_traj.append(self.phis)
                vcbm_traj.append(self.vcbm)
                e1_traj.append(self.e1)
                errors=self.model_with_dielectric(self.phis,self.vcbm,self.e1)
                esum=sum(errors)
                emag=np.linalg.norm(errors)
                
                #uncomment to see print output of optimization progression
                print(self.phis,self.vcbm,self.e1,emag)
                
                esum_traj.append(esum)
                emag_traj.append(emag)
                grad=self.calc_grad()
                self.phis-=lr*emag*np.average([grad[0,1]-self.phis,self.phis-grad[0,0]])
                self.vcbm-=lr*emag*np.average([grad[1,1]-self.vcbm,self.vcbm-grad[1,0]])
                self.e1-=lr*emag*np.average([grad[2,1]-self.e1,self.e1-grad[2,0]])
                
                if len(emag_traj)>2:
                    if abs(emag_traj[-1]-emag_traj[-2])<e_threshold:
                        break
                
        print('total # of optimization steps: {}'.format(i))
        print('total # of eigenvalue calculations: {}'.format(i*self.loop_pts))
        print('average time per eigenvalue calculation: {} s'.format((time.time()-self.start)/(i*self.loop_pts)))
        
        print('calculated energies:')
        for i in range(len(self.nstates)):
            print('{} eV with error of {} %'.format(self.calc_energies[i],(self.calc_energies[i]-self.peak_energies[i])/self.peak_energies[i]*100))
        
    def copy_energies(self):
        energy_text=''
        for i in self.calc_energies:
            if i!=self.calc_energies[0]:
                energy_text+='\n'
            energy_text+=str(i)
        pyperclip.copy(energy_text)
        
    def copy_params(self):
        if not self.dielectric:
            params=[self.z0,self.phit,self.R]
        elif self.dielectric:
            params=[self.phis,self.e1,self.vcbm]
            
        param_text=''
        for i,j in zip(params,range(3)):
            if j!=0:
                param_text+='\n'
            param_text+=str(i)
        pyperclip.copy(param_text)
                
    #function for fitting parameters in potential with no dielectric
    #the free parameters are the initial tip-sample distance and the tip work function
    def model_no_dielectric(self,z0,phit_opt,R_opt):
        energy_min=0
        energy_tol=0.0001
        
        self.calc_energies=np.zeros(len(self.nstates))
        for i in range(len(self.nstates)):
            d_opt=z0+self.peak_heights[i]
            x,pot=build_potential_no_dielectric(self.npts,self.zmin,self.w,self.Vg,self.V0,d_opt,self.phis,phit_opt,self.peak_energies[i],self.zm,R=R_opt)
            tempvar=Numerov_Cooley(x,pot,filter_mode='none',suppress_timing_output=True)
            tempvar.loop_main(self.peak_energies[i]-.5,self.peak_energies[i]+.5,self.loop_pts,nprocs=self.nprocs)
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
                            
            self.calc_energies[i]=temp_energies[np.argmin(abs(temp_energies-self.peak_energies[i]))]
            
        errors=self.calc_energies-self.peak_energies
        
        return errors
    
    #function for fitting parameters in potential with no dielectric
    #the free parameters are the initial tip-sample distance and the tip work function
    def model_with_dielectric(self,phis_opt,vcbm_opt,e1_opt):
        energy_min=0
        energy_tol=0.0001
        
        self.calc_energies=np.zeros(len(self.nstates))
        for i in range(len(self.nstates)):
            d=self.z0+self.peak_heights[i]
            x,pot=build_potential_with_dielectric(self.npts,self.zmin,self.w,self.Vg,self.V0,d,phis_opt,self.phit,self.peak_energies[i],self.zm,self.t,e1_opt,vcbm_opt,R=self.R)
            tempvar=Numerov_Cooley(x,pot,filter_mode='none',suppress_timing_output=True)
            tempvar.loop_main(self.peak_energies[i]-.5,self.peak_energies[i]+.5,self.loop_pts)
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
                            
            self.calc_energies[i]=temp_energies[np.argmin(abs(temp_energies-self.peak_energies[i]))]
            
        errors=self.calc_energies-self.peak_energies
        
        return errors
        
    def calc_grad(self,dx=0.01):
        grad_output=np.zeros((3,2))
        for i in range(3):
            for j,k in zip(range(2),[-abs(dx),abs(dx)]):
                if not self.dielectric:
                    inputs=[self.z0,self.phit,self.R]
                elif self.dielectric:
                    inputs=[self.phis,self.vcbm,self.e1]
                inputs[i]+=k
                if not self.dielectric:
                    grad_output[i,j]=np.linalg.norm(self.model_no_dielectric(inputs[0],inputs[1],inputs[2]))
                else:
                    grad_output[i,j]=np.linalg.norm(self.model_with_dielectric(inputs[0],inputs[1],inputs[2]))
                    
        return grad_output
            
if __name__=='__main__':
    if not os.path.exists('./params'):
        sys.exit()
    sys.path.append(os.getcwd())
    
    #default settings
    nprocs=1
    npts=5000,
    loop_pts=10
    zmin=0.2402093333333333*5
    w=0.2402093333333333
    Vg=4.2
    V0=4.633858138635734
    z0=0
    phis=4.59
    phit=4.59
    zm=0.015
    t=0.249595
    e1=5.688
    vcbm=3.7
    
    with open('./params') as fp:
        lines=fp.readlines()
        for i in lines:
            i=i.split('=')
            for j in range(2):
                i[j]=i[j].split('\n')[0]
            if i[0]=='w':
                w=float(i[1])
            if i[0]=='zmin':
                zmin=float(i[1])
            if i[0]=='Vg':
                Vg=float(i[1])
            if i[0]=='V0':
                V0=float(i[1])
            if i[0]=='z0':
                z0=float(i[1])
            if i[0]=='phis':
                phis=float(i[1])
            if i[0]=='phit':
                phit=float(i[1])
            if i[0]=='zm':
                zm=float(i[1])
            if i[0]=='t':
                t=float(i[1])
            if i[0]=='e1':
                e1=float(i[1])
            if i[0]=='vcbm':
                vcbm=float(i[1])
            if i[0]=='npts':
                npts=int(i[1])
            if i[0]=='loop_pts':
                loop_pts=int(i[1])
                
    try:
        opts,args=getopt.getopt(sys.argv[1:],'e:z:p:d:s:',['energies=','zvalues=','processors=','dielectric=','errors='])
    except getopt.GetoptError:
        print('error in command line syntax')
        sys.exit(2)
    for i,j in opts:
        if i in ['-e', '--energies']:
            energies=np.array([float(k) for k in j.split(',')])
        if i in ['-z', '--zvalues']:
            z=np.array([float(k) for k in j.split(',')])
        if i in ['-s', '--errors']:
            error=np.array([float(k) for k in j.split(',')])
        if i in ['-p', '--processors']:
            nprocs=int(j)
        if i in ['-d','--dielectric']:
            dielectric=bool(j)
    test=optimize_parameters(energies,z,nprocs=nprocs,sigma=error,zmin=w*zmin,w=w,Vg=Vg,V0=V0,z0=z0,phis=phis,phit=phit,zm=zm,t=t,e1=e1,vcbm=vcbm,suppress_plotting=True)
