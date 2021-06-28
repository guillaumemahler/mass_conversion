import argparse
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import sys
from astropy import units as u
from scipy.optimize import minimize
from scipy.optimize import fsolve


'''
          errl,errh=args.error.split(',')
          mu, sigma = M500c, float(errl)*1e14
          s = np.random.normal(mu, sigma, 100)
          smin=s[s<mu]
          sigma=float(errh)*1e14
          s = np.random.normal(mu, sigma, 100)
          smax=s[s>mu]
          stot=np.concatenate((smin, smax), axis=None)
          fin=[]
          for s_ind in stot:
            fin.append(M500ccylindrical_from_M500cspherical_tab(s_ind,z))
          #print(fin)
          print(len(fin))
'''

def sigma_M200(sigma,err_high=0,err_low=0): #from Evrard 2008 e6.   note that this is a M200[Msun/h]     #do M200 [Msun] = M200[Msun/h] * (h/0.7).
   alpha= 0.3361 # 0.3361+- 0.0026 BUT I use the error from the given standard deviation in the paper because the parameter might have some covariance that I do not know
   sigma_dm15=1082.9 #1082.9 +- 4.0 km.s-1  BUT same as above 
   log_scatter_dm=0.0426 #0.0426 +- 0.0015   



   M200 = (sigma/sigma_dm15)**(1/alpha)*(1e15)  #I used no bias: b=sigma_gal/sigma_dm=1
   M200err_high=np.power(10,np.log10(M200)+log_scatter_dm)-M200
   M200err_low=np.power(10,np.log10(M200)-log_scatter_dm)-M200
   return M200,M200err_high,M200err_low

def compute_rho_c(z): #Critical density of the universe
   cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
   rho_c=cosmo.critical_density(z).to(u.Msun/u.kpc**3)  
   return rho_c.value

def compute_R500c(M500c,z,verbose=False): #M500 should be in Msol
   rho_c=compute_rho_c(z)
   rho_c_500c=500.0*rho_c
   cst=3.0/(4.0*np.pi)
   R500c=np.cbrt(cst*M500c/rho_c_500c) #cubic root
   if verbose:
      print("R500c",R500c*u.kpc)
   return R500c

def compute_R200c(M200c,z,verbose=False): #M500 should be in Msol
   rho_c=compute_rho_c(z)
   rho_c_200c=200.0*rho_c
   cst=3.0/(4.0*np.pi)
   R200c=np.cbrt(cst*M200c/rho_c_200c) #cubic root
   if verbose:
      print("R200c",R200c*u.kpc)
   return R200c


#I have to turn the spherical mass from Bocquet+19 into a NFW prifle parameters.
#They used a YSZ-mass relation without assuming an NFM, that's why I first need to get the paramters.
#Using wikipedia equation, Maybe Sharon2015b.pdf have it to
#https://en.wikipedia.org/wiki/Navarro%E2%80%93Frenk%E2%80%93White_profile
#USing the equation for the total Mass M within R_vir but I change it for R_500 by changing the value of c
#with c if I use c_500 (=R_500/rs) this will give me M_500 
def M_NFW(rs,c,rho0,verbose=False):
   a=4.0*np.pi*rho0*rs**3.0
   b=np.log(1.0+c)-c/(1.0+c)
   M_NFW=a*b
   if verbose:
      print("M_NFW",M_NFW)
   return M_NFW

#I don't have rs, c and rho0 (rho0 and rs vary from halo to halo)
#I need c and rs, I can find rho if I have rs and M500c from the wikipedia formula.
#I need c_xxx and Mass_xxx to find rs.

#First let's find c
#I can use Outer simulation from Child2018
#But first I need to convert M500 -> M200 using Hu2003

#Eq 19 from Child+2018 
#Looking at the paper this should be valid from 0<z<1 but this might be double check 
def c200c_from_M200c(M200c,z,verbose=False):
  #Parameters took from table 2 in Child+2018 "indidual, all"
  A=75.4
  d=-0.422
  m=-0.089
  #c200c=A * (1.0+z)**d * M200c**m 
  c200c=A * np.power(1.0+z,d) * np.power(M200c,m) 
  if verbose:
     print("c200c",c200c) 
  return c200c

#Eq 19 from Child+2018 
#Looking at the paper this should be valid from 0<z<1 but this might be double check
def c200c_from_M200c_tabmcmc(M200c,z,verbose=False):
  #Parameters took from table 2 in Child+2018 "indidual, all"
  A=75.4
  d=-0.422
  m=-0.089
  #c200c=A * (1.0+z)**d * M200c**m 
  c200c_mu=A * np.power(1.0+z,d) * np.power(M200c,m) 
  c200c_sigma=c200c_mu/3.0
  c200c = np.random.normal(c200c_mu, c200c_sigma, 100)
  if verbose:
     print("c200c",c200c) 
     print("len(c200c)",len(c200c)) 
  return c200c

def Duffy2008_c200c_from_M200c(M200c,z):
  #Not complete
  #Mpivot=
  c200c=0
  return c200c

#C3 from Hu & Kravtsov 2003 
def f_x(x):
  f_x=x**3.0 * ( np.log(1.0+1.0/x) - 1.0/(1.0+x) )
  return f_x

#C3 from Hu & Kravtsov 2003 and C9 to create fh_x 
#If I read that correctly
def fh_x_200_to_500(x):
  fh_x=(500.0/200.0)*f_x(x)
  return fh_x

#C11 from Hu & Kravtsov 2003 x here represent c500c #This is very specific to turn 500 to 200
def x_fh(c200c):
   inv_c200c=1.0/c200c
   a1=0.5116
   a2=-0.4283
   a3=-3.13*1e-3
   a4=-3.52*1e-5
   p= a2 + a3*np.log(fh_x_200_to_500(inv_c200c)) + a4*(np.log(fh_x_200_to_500(inv_c200c)))**2.0
   x_fh=( a1 * fh_x_200_to_500(inv_c200c)**(2.0*p) + (3.0/4.0)**2 )**(-1.0/2.0) + 2.0*fh_x_200_to_500(inv_c200c)
   return x_fh

#I rewrite the equation above
def invc500c_from_c200c(c200c):
   return x_fh(c200c)

#C9 and C10 from Hu & Kravtsov 2003
#Plus Chid2018 Eq19 
def with_printsM500c_from_M200c(M200c,z):
   c200c=c200c_from_M200c(M200c,z) #Eq19
   rs_over_R500c=invc500c_from_c200c(c200c) #eq C9
   c500c=1.0/rs_over_R500c 
   M500c=M200c*500.0/200.0*(c500c/c200c)**3.0 #eq C10
   print("c200c",c200c,"c500c",c500c,"M500c",M500c,"M500c/1e14",M500c/1e14)
   return M500c


def rs_from_M200c(M200c,z,c):
   R200c = compute_R200c(M200c,z) #From the formula of volume:  M = rho * V_sphere
   rs=R200c/c
   print("rs",rs," kpc")
   return rs

def rs_from_M500c(M500c,z,c):
   R500c = compute_R500c(M500c,z) #From the formula of volume:  M = rho * V_sphere
   rs=R500c/c
   print("rs",rs," kpc")
   return rs

def rho0_from_M_rs_c(M,rs,c): #Wikipedia eq, comme from the actual formula of mass not for the Deltac*rhoc
   a=M/(4.0*np.pi*rs**3.0)
   b=np.log(1.0+c)-c/(1.0+c)
   rho0=a/b
   return rho0

#Golse 2002 page 2 function g(x)
def g(x):
  if x<1.0:
     aa=1.0/np.sqrt(1.0-x**2.0)
     bb=np.arccosh(1.0/x)
     val=np.log(x/2.0)+ aa*bb
  if x==1.0:
     val=1.0+np.log(1.0/2.0)
  if x>1.0:
     aa=1.0/np.sqrt(x**2.0-1.0)
     bb=np.arccos(1.0/x)
     val=np.log(x/2.0)+ aa*bb
  return val

#Golse 2002 eq5
#the mean surface density inside the dimensionless radius x
#rho0 is rho_c in eq5 
def Sigma_mean(rho0,rs,R): 
   x=R/rs
   gx=g(x)
   Sigma=4.0*rho0*rs*(gx/x**2.0)
   return Sigma

#Adaption of the above
def Sigma_mean_from_M200czc200c_atR200c(M200c,z,c):
   R200c = compute_R200c(M200c,z)
   rs=R200c/c
   rho0=rho0_from_M_rs_c(M200c,rs,c)
   x=R200c/rs
   gx=g(x)
   Sigma=4.0*rho0*rs*(gx/x**2.0)
   return Sigma

#Adaption of the above
def Sigma_mean_from_M200czc200c_anyR(M200c,z,c,R):
   R200c = compute_R200c(M200c,z)
   rs=R200c/c
   #print('M200c,rs,c',M200c,rs,c)
   rho0=rho0_from_M_rs_c(M200c,rs,c)
   x=R/rs
   #print(rho0)
   #x=R200c/rs
   gx=g(x)
   Sigma=4.0*rho0*rs*(gx/x**2.0)
   return Sigma

#Adaption of the above
def Sigma_mean_from_M500czc500c_anyR(M200c,z,c,R):
   R500c = compute_R500c(M500c,z)
   rs=R500c/c
   print('M500c,rs,c',M500c,rs,c)
   rho0=rho0_from_M_rs_c(M500c,rs,c)
   x=R/rs
   print(rho0)
   #x=R200c/rs
   gx=g(x)
   Sigma=4.0*rho0*rs*(gx/x**2.0)
   return Sigma
#FromGolse2002
def cylindrical_mass(rho0,rs,R):
   Sigma=Sigma_mean(rho0,rs,R)
   Mass=Sigma*np.pi*R**2.0
   return Mass

def M500ccylindrical_from_M500cspherical(M500c,z,verbose=False):
   M200c = M200c_from_M500c(M500c,z)
   c200c_tab=c200c_from_M200c(M200c,z)
   R500c=compute_R500c(M500c,z)
   Sigma=Sigma_mean_from_M200czc200c_anyR(M200c,z,c200c,R500c)
   Mass_cyl=Sigma*np.pi*R500c**2.0
   if verbose:
      print('Mass_cyl',Mass_cyl)
   return Mass_cyl

def M500ccylindrical_from_M500cspherical_tab(M500c,z,verbose=False):
   #b=(1e12,1e16) #boundaries 
   #bnds=((z,z),b,(M500c,M500c))
   #vect2=(z,guess_M200c,M500c)
   # I dont think that is counting as constraints in minimize con1= {'type': 'eq', 'fun':func_to_mini} #constraints 1 
   #sol = minimize(func_to_mini,vect2,method='SLSQP',bounds=bnds)
   #sol = minimize(func_to_mini,vect2,method='L-BFGS-B',bounds=bnds)
   #sol = fsolve(func_to_mini,guess_M200c,args=(M500c,z))
   M200c = M200c_from_M500c(M500c,z)
   #print(sol)
   #print(sol[0])
   #M200c=sol[0]
   c200c_tab=c200c_from_M200c_tabmcmc(M200c,z)
   R500c=compute_R500c(M500c,z)
   #R500=compute_R500(M500*1e14,z).value
   #errR500cmin=R500c-compute_R500(M500c-0.66*1e14,z).value
   #errR500cmax=R500c-compute_R500(M500+0.59*1e14,z).value
   Mass_cyl=[]
   for c200c in c200c_tab:
      Sigma=Sigma_mean_from_M200czc200c_anyR(M200c,z,c200c,R500c)
      Mass_cyl.append(Sigma*np.pi*R500c**2.0)
   #Sigma_500c=Sigma_mean_from_M500czc500c_anyR(M500c,z,c500c,R500c)
   #Mass_cyl_500c=Sigma_500c*np.pi*R500c**2.0
   #print('Mass_cyl_500c',Mass_cyl_500c)
   if verbose:
      print('Mass_cyl',Mass_cyl)
   return Mass_cyl   

def M200c_from_M500c(M500c,z,verbose=False):
   guess_M200c=5e15
   sol = fsolve(func_to_mini,guess_M200c,args=(M500c,z)) #I use a solver here because the relation only go from M500c to M200c
   if verbose:
      print('M200c',sol[0])
   return sol[0]
#def M500c_from_M200c(vect):
def M500c_from_M200c(M200c,z):
#   z=vect[0]
#   M200c=vect[1]
   c200c=c200c_from_M200c(M200c,z) #Eq19 Child+18
   rs_over_R500c=invc500c_from_c200c(c200c) #eq C9 Hu & Kravtsov 2003
   c500c=1.0/rs_over_R500c 
   M500c=M200c*500.0/200.0*(c500c/c200c)**3.0 #eq C10 Hu & Kravtsov 2003
   return M500c

def func_to_mini(M200c,M500c,z): #vect2 is vect,M500c that I want
   #vect=(vect2[0],vect2[1])
   #vect=(vect2[0],vect2[1])
   #test=vect2[2]-M500c_from_M200c(vect)
   test=M500c-M500c_from_M200c(M200c,z)
   return test

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1','on'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','off'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def define_args_from_parser():
        parser = argparse.ArgumentParser()
        #parser.add_argument("-v", "--verbosity", action="count", default=True,help="increase output verbosity")
        #parser.add_argument("--nice", type=str2bool, nargs='?',const=True, default=True,help="De-activate verbose mode."
        parser.add_argument('-m', '--mass', dest='mass', help='Input mass')
        parser.add_argument('-e', '--error', dest='error', help='Input error:  err_low,err_high')
        parser.add_argument('-z', '--redshift', dest='redshift', help='Input redshift')
        parser.add_argument('--mass_type', dest='mass_type', default='M500c',help='If you input if it is a M500c or M200c')
        parser.add_argument('-t','--table', dest='table',default=None,help='Text file with .csv extension with 3 columns, ID,mass,redshift ')
        parser.add_argument("-v", "--verbose", type=str2bool, nargs='?', default=True,help="Activate or deactivate verbose mode. Default True")

        args = parser.parse_args()

        #The following 4 lines are for the autocompletion of the command
        #t = tabCompleter.tabCompleter()
        #readline.set_completer_delims('\t')
        #readline.parse_and_bind("tab: complete")
        #readline.set_completer(t.pathCompleter)
        #

        if args.mass is None:
                args.mass=input("Enter Mass (in 1e14 Msol or Msol^h-1): ")
        if args.redshift is None:
                args.redshift=input("Enter the redshifts of the halo: ")
        if args.mass_type is None:
                args.mass_type=input("Enter which mass it is, M500c or M200c ?: ")
        return args


if __name__== "__main__":

    #Either you input M500 or M200 
    #if the tab option is set, there is a table with all the info and the plot is produce, nothing appear in the console. 
    #plot cylindrical mass in funciton of R
    print('Feed like python turn_spherical_to_cylindrical.py -m Mass(in1e14 Msun) -z z -mass_type M500c_or_M200c -t')
    print("The code doesnt really care if it is Msun or Msun*h^-1, it will give you the same unit")
    print(" ")
    print(" ")

    '''FOR X-RAY UPGRADE OF THE CODE
    In Okabe 2020 ^^ "we convert the X-ray
    temperature shown in Table 4 of Postman et al. (2012) to M500 by
    "using an empirical relation (Arnaud et al. 2007).
    '''

    args = define_args_from_parser()
    if args.table is None:
       z=float(args.redshift)
       if args.mass_type=="M500c":
          M500c=float(args.mass)*1e14
          R500c=compute_R500c(M500c,z)
          print("For M500c={:e} R500c= {} kpc".format(M500c,R500c))
          M200c=M200c_from_M500c(M500c,z)
          R200c=compute_R500c(M200c,z)
          print("Converting M500c to M200c, give M200c = {:e} then R200c= {} kpc".format(M200c,R200c))
          c200c=c200c_from_M200c(M200c,z)
          rs200c=rs_from_M200c(M200c,z,c200c)
          print("Therefore, c200c = {} ".format(c200c,rs200c))
          rs_over_R500c=invc500c_from_c200c(c200c) #eq C9
          c500c=1.0/rs_over_R500c
          rs500c=rs_from_M500c(M500c,z,c500c)
          print("Therefore, c500c = {} and rs using M500c = {}".format(c500c,rs500c))
          M500ccylindrical=M500ccylindrical_from_M500cspherical(M500c,z) 
          print("The cylindrical mass is then = {:e} ".format(M500ccylindrical))
          print(" ")
          print(" ")
          errl,errh=args.error.split(',')
          mu, sigma = M500c, float(errl)*1e14 
          s = np.random.normal(mu, sigma, 100)
          smin=s[s<mu]
          sigma=float(errh)*1e14
          s = np.random.normal(mu, sigma, 100)
          smax=s[s>mu]
          stot=np.concatenate((smin, smax), axis=None)   
          fin=[]      
          for s_ind in stot:
            fin.append(M500ccylindrical_from_M500cspherical_tab(s_ind,z))     
          #print(fin)
          print(len(fin))
          
          np.save('mass_mcmc_results_sampling100',fin)
          plt.hist(np.ravel(fin), bins = 1000)  
          print(np.nanmean(np.ravel(fin))/1e14,
            np.nanmedian(np.ravel(fin))/1e14,
            np.nanpercentile(fin, 50)/1e14,
            np.nanpercentile(fin, 50-34.1)/1e14,
            np.nanpercentile(fin, 50+34.1)/1e14,)
          #print('rs500c',rs500c)
          #print('from M500c',rho0_from_M_rs_c(M500c,rs500c,c500c))
          
  




