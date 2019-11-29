import numpy as np
# import cupy as cp #module identique à numpy dans les fonctions qu'il intègre et sa syntaxe mais reposant sur CUDA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
import sys



"""
**************************************************
*          CONSTRUCTION DE LA STRUCTURE          *
**************************************************
"""


def bornes(p,n,r0):
    liste_bornes=[r0]
    r=r0
    for i in range(2*n-1):
        r+=np.pi*r/p
        liste_bornes.append(r)
    return liste_bornes


def intervalle(x,liste):
    n=0
    for k in range(len(liste)):
        if x>liste[n]:
            n+=1
    return n


def capegpu(p,n,r0,Npt):
    N=np.ones((Npt,Npt))
    liste_bornes=bornes(p,n,r0)
    for i in range(Npt):
        for j in range(Npt):
            z=complex(i-(Npt-1)/2,j-(Npt-1)/2)
            anglez=np.angle(z)
            if anglez<0:
                anglez+=2*np.pi
            if int(anglez*p/np.pi)%2==0:
                if intervalle(np.abs(z),liste_bornes)%2==1:
                    N[i,j]=0
    return cp.asarray(N)


def capecpu(p,n,r0,Npt):
    N=np.ones((Npt,Npt))
    liste_bornes=bornes(p,n,r0)
    for i in range(Npt):
        for j in range(Npt):
            z=complex(i-(Npt-1)/2,j-(Npt-1)/2)
            anglez=np.angle(z)
            if anglez<0:
                anglez+=2*np.pi
            if int(anglez*p/np.pi)%2==0:
                if intervalle(np.abs(z),liste_bornes)%2==1:
                    N[i,j]=0
    return N



"""
**************************************************
*                 INITIALISATION                 *
**************************************************
"""


def initgpu (Npt, Nrg, Npl, rpt, L) :
    """ 
    Construit maillage & structure, 
    int Npt: précision du maillage (nombre de points)
    int Nrg: Nombre de rang de la structure
    int Npl: Nombre de points par rangs
 -> (X,Y,H,N) quadruplet de float array de dimension Npt*Npt
    X,Y,H : Composantes en x, y des vecteurs vitesses, et hauteur
    N : Coefficient directeur de la droite normale à la surface de l'obstacle (sans obstacle, vaut 0)"""
    
    X = cp.zeros((Npt,Npt))
    Y = cp.zeros((Npt,Npt))
    H = cp.ones((Npt,Npt))
    N = capegpu(Npt,Nrg,rpt,Npt)
    #N = cp.ones((Npt,Npt))

    N[0]*=0
    N[Npt-1]*=0
    N[:,0]*=0
    N[:,Npt-1]*=0
    return (N*X,N*Y,H,N)


def initcpu (Npt, Nrg, Npl, rpt, L) :
    """ 
    Construit maillage & structure, 
    int Npt: précision du maillage (nombre de points)
    int Nrg: Nombre de rang de la structure
    int Npl: Nombre de points par rangs
 -> (X,Y,H,N) quadruplet de float array de dimension Npt*Npt
    X,Y,H : Composantes en x, y des vecteurs vitesses, et hauteur
    N : Coefficient directeur de la droite normale à la surface de l'obstacle (sans obstacle, vaut 0)"""
    
    X = np.zeros((Npt,Npt))
    Y = np.zeros((Npt,Npt))
    H = np.ones((Npt,Npt))
    N = capecpu(Npt,Nrg,rpt,Npt)
    N[0]*=0
    N[Npt-1]*=0
    N[:,0]*=0
    N[:,Npt-1]*=0
            
    return (N*X,N*Y,H,N)




"""
**************************************************
*        ITERATION ET FONCTION PRINCIPALE        *
**************************************************
"""


def iteration (X0,Y0, H0, N, dt, dx, Npt,t,T,F) :
    """
    Itere une configuration
    X0, Y0, H0 : configuration à l'instant n
    N : Structure
    dt : pas temporel
    F : matrice de forçage
    dx : pas spatial
 -> X1, Y1, H1 : configuration à l'instant n+1"""
        
    Nm = Npt - 1
    
    X0[1:-1, 1:-1],Y0[1:-1, 1:-1],H0[1:-1, 1:-1] = X0[1:-1, 1:-1] - dt/(2*dx) * (X0[1:-1, 1:-1]*(X0[2:, 1:-1] - X0[:-2, 1:-1]) + Y0[1:-1, 1:-1]*(X0[1:-1,2:] - X0[1:-1, :-2]) + 9.81*(H0[2:, 1:-1] - H0[:-2, 1:-1])) + dt*F[1:-1, 1:-1]*np.cos(2*np.pi*t/T),    Y0[1:-1, 1:-1] - dt/(2*dx) * (X0[1:-1, 1:-1]*(Y0[2:, 1:-1] - Y0[:-2, 1:-1]) + Y0[1:-1, 1:-1]*(Y0[1:-1,2:] - Y0[1:-1, :-2]) + 9.81*(H0[1:-1,2:] - H0[1:-1, :-2])),     H0[1:-1, 1:-1] - dt/(2*dx) * (X0[2:, 1:-1] * H0[2:, 1:-1] -  X0[:-2, 1:-1] * H0[:-2, 1:-1] + Y0[1:-1,2:] * H0[1:-1,2:] - Y0[1:-1, :-2] * H0[1:-1, :-2])
    
    return(N*X0,N*Y0,H0)

        
def proggpu(Npt, Nrg, Npl, r0, dt, T, L, tau,periode,A,dimension) :
    start_time=time.time()
    Nm = Npt - 1
    rpt = r0*Npt/L
    dl = L/Npt
    (X,Y,H,N) = initgpu(Npt, Nrg, Npl, rpt, L)
    F=cp.ones((Npt,Npt))
    for i in range(Npt) :
        F[i]*=A*np.exp(-(i-Npt/256)**2/((Npt/10)**2))
    fig=plt.figure()
    enregistrer(cp.asnumpy(H),Npt,L,0,fig,dimension)
    k=0
    st=time.time()
    while k<T-dt :
        i=k
        while i<(k+tau) :
            (X,Y,H)=iteration(X,Y,H,N,dt,dl,Npt,i,periode,F)
            sys.stdout.write("temps restant de  calcul estimé (en secondes) : " +str((time.time()-st)*(T-i+dt)/(i+dt)) + "\r")
            i+=dt
        k+=tau
        enregistrer(cp.asnumpy(H),Npt,L,k,fig,dimension)
    plt.show()
    return time.time() - start_time

def progcpu(Npt, Nrg, Npl, r0, dt, T, L, tau,periode,A,dimension) :
    start_time=time.time()
    rpt = r0*Npt/L
    dl = L/Npt
    (X,Y,H,N) = initcpu(Npt, Nrg, Npl, rpt, L)
    F=np.ones((Npt,Npt))
    for i in range(Npt) :
        F[i]*=A*np.exp(-(i-Npt/256)**2/((Npt/10)**2))
    fig=plt.figure()
    enregistrer(H,Npt,L,0,fig,dimension)
    k=0
    st=time.time()
    while k<T :
        i=k
        while i<(k+tau) :
            (X,Y,H)=iteration(X,Y,H,N,dt,dl,Npt,i,periode,F)
            sys.stdout.write("temps restant de  calcul estimé (en secondes) : " +str((time.time()-st)*(T-i+dt)/(i+dt)) + "\r")
            i+=dt
        k+=tau
        enregistrer(H,Npt,L,k,fig,dimension)
    plt.show()
    return time.time() - start_time



"""
**************************************************
*                 ENREGISTREMENT                 *
**************************************************
"""


def enregistrer(H, Npt,L,i,fig,dimension) :
    plt.clf()
    if dimension == 3 :
        X=np.linspace(0,L,Npt)
        Y=np.linspace(0,L,Npt)
        X,Y=np.meshgrid(X,Y)
        plt.axes(projection='3d').plot_surface(X, Y, H , rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        fig.savefig(str(i)+".png")
    if dimension == 2 :
        plt.imsave(str(i)+"struct.png",H, vmin=0.997,vmax=1.003)
        plt.matshow(H,False,vmin=0.997,vmax=1.003)
