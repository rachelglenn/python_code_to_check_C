################################################################
#
#
#                            Created by: Rachel Glenn
#                                 Date: 12/14/2016
#       This code calculates the converaged CCSD energy, pseudo energy, the t1, t2, lam1, and lam2
#       It also calculates the single particle density matrix using the converged t1, t2, lam1, and lam2 
#
#
#####################################################################
import sys
import os
from copy import deepcopy
import numpy as np
import cmath
import pandas as pd
#from pandas import *
import psi4 as psi4
sys.path.append(os.environ['HOME']+'/miniconda2/lib/python2.7/site-packages')
from opt_einsum import contract
import time
import csv

class CCSD_Helper(object):
    
    def __init__(self,psi): #ndocc=None):
       
        self.counter = 0
        self.mol = psi4.core.get_active_molecule()
        mol = self.mol
        self.wfn = psi4.scf_helper('SCF',  return_wfn = True)
        self.scf_e = psi4.energy('scf')
        self.mints = psi4.core.MintsHelper(self.wfn.basisset())
        self.nmo = self.wfn.nmo()
        self.ccsd_e = psi4.energy('ccsd')
        self.S = np.asarray(self.mints.ao_overlap())
        
        #define ndocc
        # Orthoganlizer
        A = self.mints.ao_overlap()
        A.power(-0.5, 1.e-14)
        self.A = np.asarray(A)
        self.ndocc =int(sum(mol.Z(A) for A in range(mol.natom())) / 2)
        self.wfn.Ca_subset("AO", "ACTIVE").print_out() 
        self.C = self.wfn.Ca()
        V = np.asarray(self.mints.ao_potential())
        T = np.asarray(self.mints.ao_kinetic())
        self.H = T + V
        self.occ = slice(2*self.ndocc)
        self.vir = slice(2*self.ndocc, 2*self.nmo)
        #MO energies
        self.eps = np.asarray(self.wfn.epsilon_a()).repeat(2, axis=0)
        self.TEI = self.TEI_MO()
    

###############Setup the Fock matrix and TEIs #####################
    def TEI_MO(self, C=None):
        if C is None: C = self.C
        return np.asarray(self.mints.mo_spin_eri(C, C))
           
    def GenS12(self): 
        # Update S, transform to MO basis and tile for alpha/beta spin
        S = self.S
        nmo = self.nmo
        S = S.repeat(2, axis=1).repeat(2, axis=0)
        S = S*np.tile(np.identity(2),(nmo,nmo))
        evals, evecs = np.linalg.eigh(S)
        nmo = self.nmo
        
        Ls = np.zeros(shape=(2*nmo,2*nmo))
        Lsplus = np.zeros(shape=(2*nmo,2*nmo))    
          
        for i in range (2*nmo):
            Ls[i][i]= 1/np.sqrt(evals[i])
            Lsplus[i][i]= np.sqrt(evals[i])
            
        S12 = contract('il,lk,jk->ij', evecs, Ls, evecs)
        S12plus = contract('il,lk,jk->ij', evecs, Lsplus, evecs)        
        return S12, S12plus
        
        
    def F_MO(self, H=None, C=None):
        if H is None: H = self.H
        if C is None: C = self.C
        TEI = self.TEI_MO(C)
        occ = self.occ
        nmo =self.nmo
        # Update H, transform to MO basis and tile for alpha/beta spin
        H = contract('vi,uv,uj->ij', C, H, C)
        H = H.repeat(2, axis=1).repeat(2, axis=0)
        H = H*np.tile(np.identity(2),(nmo,nmo))
        F= H + contract('pmqm->pq', TEI[:, occ, :, occ])
        return F
        
    def MO_E(self, H=None, C=None):  
        if H is None: H = self.H
        if C is None: C = self.C 
        F = self.F_MO(H,C)
        evals, evecs = np.linalg.eigh(F)
        return evals
    
    def MP2_E(self, alpha, H=None, C=None):  
        #alpha is a text variable to select the output
        if H is None: H = self.H
        if C is None: C = self.C 
        eps = self.MO_E(H,C)
        o = self.occ
        v = self.vir
        self.TEI = self.TEI_MO(C)
        TEI = self.TEI
        Dem = eps[o].reshape(-1, 1, 1, 1) + eps[o].reshape(-1, 1, 1) - eps[v].reshape(-1, 1) - eps[v]
        Dem = 1/Dem
        T2 = contract('ijab,ijab->ijab', TEI[o, o, v, v],Dem)
        MP2 = contract('ijab,ijab->', T2, TEI[o, o, v, v])
        T2 = TEI[o, o ,v, v]*Dem
        MP2 = np.sum(TEI[o, o, v, v]*T2)
        #print MP2

        MP2_E = self.scf_e + 1/4.0*MP2
        
        if alpha is 'Test':

            pass
        return self.scf_e, MP2_E, T2

############################################################       
#                    
#               T1 and T2-equations (CCSD)
#                   By R. Glenn, I used T. Daniel Crawfords equations
#    
#    
#    
############################################################
    
    #Build Fvv
    def Fae(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = F[v, v].copy()

        term2 = - 0.5*contract('me,ma->ae', F[o, v], t1)
        term3 = contract('mafe,mf->ae', TEI[o, v, v, v], t1)
        tau = t2.copy() + contract('ia,jb->ijab', t1, t1)
        term4 =-0.5*contract('mnef,mnaf->ae', TEI[o, o, v, v], tau)
        total = term1 + term2 + term3 + term4
        return total
    
    #Build Foo
    def Fmi(self, t1, t2, F):
        v = self.vir
        o = self.occ  
        TEI = self.TEI 
        term1 = F[o, o].copy()
        term2 =0.5*contract('me,ie->mi', F[o, v], t1)
        term3 = contract('mnie,ne->mi', TEI[o, o, o, v], t1)
        tau = t2.copy() + contract('ia,jb->ijab', t1, t1)
        term4 = 0.5*contract('mnef,inef->mi', TEI[o, o, v, v], tau)
        total = term1 + term2 + term3 + term4 
        return total
    
    #Build Fov    
    def Fme(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = F[o, v].copy()
        term2 = contract('mnef,nf->me', TEI[o, o, v, v], t1)
        total = term1 + term2
        return total

#####################################################
#
#       T1-rhs-CCSD
#
#####################################################
    def T1eq_rhs(self, t1, t2, F):        
        #All terms in the T1 Equation
        v = self.vir
        o = self.occ
        TEI = self.TEI
        fae = self.Fae(t1, t2, F) 
        fmi = self.Fmi(t1, t2, F)  
        fme = self.Fme(t1, t2, F) 
              
        term1 = F[o, v].copy()
        term2 = contract('ae,ie->ia', fae, t1)
        term3 = -contract('mi,ma->ia', fmi,t1)
        term4 = contract('me,imae->ia', fme, t2)
        #extra terms   
        extra1 = -contract('naif,nf->ia', TEI[o, v, o, v], t1)
        extra2 = -0.5*contract('nmei,mnae->ia', TEI[o, o, v, o], t2)
        extra3 = -0.5*contract('maef,imef->ia', TEI[o, v, v, v], t2)
        
        total = term1 + term2 + term3  + term4  + extra1  + extra2  + extra3
        return total
#######################################################
   #Build Woooo for t2 terms 
    def Wmnij(self, t1 ,t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[o, o, o, o].copy()
        term2a = contract('mnie,je->mnij', TEI[o, o, o, v], t1)
        term2 = term2a - term2a.swapaxes(2,3) #swap ij
        
        tau = 0.25*t2 + 0.5*contract('ia,jb->ijab', t1, t1) - 0.5*contract('ib,ja->ijab', t1, t1) 
        term3 = contract('mnef,ijef->mnij', TEI[o, o, v, v], tau)
        total = term1 + term2 + term3 
        return total    
     
    #Build Woooo for t1 * t1 like terms       
    def Wmnij_2(self, t1 ,t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[o, o, o, o].copy()
        term2a = contract('mnie,je->mnij', TEI[o, o, o, v], t1)
        term2 = term2a - term2a.swapaxes(2,3) #swap ij
        tau = contract('ia,jb->ijab', t1, t1)
        term4a = 0.25*contract('mnef,ijef->mnij', TEI[o, o, v, v], tau)  
        term4 = term4a - term4a.swapaxes(2,3)
        total = term1 + term2 + term4
        return total 
     
    #Build Wvvvv for t2 terms                                                                                                                                                                                                                                                                     
    def Wabef(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[v, v, v, v].copy()
        term2tmp = -contract('amef,mb->abef', TEI[v, o, v, v], t1) 
        term2 = term2tmp - term2tmp.swapaxes(0,1) #swap ab
        tau = contract('ia,jb->ijab', t1, t1) #- contract('ib,ja->ijab', t1, t1) 
        term3 = 0.25*contract('mnef,mnab->abef', TEI[o, o, v, v], t2)
        term4a = 0.5*contract('mnef,mnab->abef', TEI[o, o, v, v], tau)
        term4 = term4a - term4a.swapaxes(0,1)
        total = term1 + term2 + term3 + term4
        return total

    #Build Wvvvv for t1 * t1 like terms
    def Wabef_2(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[v, v, v, v].copy()
        term2tmp = -contract('amef,mb->abef', TEI[v, o, v, v], t1) 
        term2 = term2tmp - term2tmp.swapaxes(0,1) #swap ab
        tau = contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1) 
        term4 = 0.25*contract('mnef,mnab->abef', TEI[o, o, v, v], tau)
        total = term1 + term2 + term4
        return total
    
    #Build Wovvo                                                                                                                                                                                                                                                                    
    def Wmbej(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[o, v, v, o].copy()
        term2 = -contract('mnej,nb->mbej', TEI[o, o, v, o], t1)
        t2t1 = 0.5*t2 + contract('jf,nb->jnfb', t1, t1)
        term34 = -contract('mnef,jnfb->mbej', TEI[o, o, v, v], t2t1)
        term5 = contract('mbef,jf->mbej', TEI[o, v, v, v], t1)
        total = term1 + term2 + term34 + term5 
        return total
 
########### Build T2 Equation################################                                                       
    def T2eq_rhs(self, t1, t2, F):
        v = self.vir
        o = self.occ  
        TEI = self.TEI         
        fae = self.Fae(t1, t2, F) 
        fmi = self.Fmi(t1, t2, F)  
        fme = self.Fme(t1, t2, F) 
        wmnij = self.Wmnij(t1, t2, F)
        wabef = self.Wabef(t1, t2, F)
        wmbej = self.Wmbej(t1, t2, F)
        wabef_2 = self.Wabef_2(t1 ,t2, F)
        wmnij_2 = self.Wmnij_2(t1 ,t2, F)
        #All terms in the T2 Equation
        term1 = TEI[o, o, v, v].copy()
        
        term2tmp = fae - 0.5 *contract('me,mb->be', fme, t1)
        term2a = contract('be,ijae->ijab', term2tmp, t2) 
        term2 = term2a - term2a.swapaxes(2, 3) #swap ab
        
        term3temp = fmi + 0.5 *contract('me,je->mj', fme, t1)
        term3a = -contract('mj,imab->ijab', term3temp, t2) 
        term3 = term3a - term3a.swapaxes(0, 1) #swap ij
             
        tau = contract('ma,nb->mnab', t1, t1) - contract('na,mb->mnab', t1, t1)
        term44 = 0.5*contract('mnij,mnab->ijab', wmnij, t2)
        term55 = 0.5*contract('abef,ijef->ijab', wabef, t2)   
        term44 += 0.5*contract('mnij,mnab->ijab', wmnij_2, tau) 
        term55 +=  0.5*contract('abef,ijef->ijab', wabef_2, tau)  
 
        term6tmp = contract('mbej,imae->ijab', wmbej, t2)
        term6tmp = term6tmp - contract('mbej,ie,ma->ijab', TEI[o, v, v, o], t1, t1)
        term6 =  term6tmp - term6tmp.swapaxes(2, 3)  - term6tmp.swapaxes(0, 1)  + term6tmp.swapaxes(0, 1).swapaxes(2, 3)

                                             
        term7tmp = contract('abej,ie->ijab', TEI[v ,v, v, o], t1) 
        term7 =  term7tmp - term7tmp.swapaxes(0, 1) #swap ij 
                             
        term8tmp = -contract('mbij,ma->ijab', TEI[o, v, o, o], t1) 
        term8 =  term8tmp - term8tmp.swapaxes(2, 3) #swap ab
    
        total = term1 + term2 + term3 + term44 + term55 + term6 + term7 + term8
        return total
    
    #Calculate the CCSD energy 
    def CCSD_Corr_E(self, t1, t2, F):
        o = self.occ
        v = self.vir
        TEI = self.TEI
        term1 = contract('ia,ia->',F[o, v], t1)
        term2 = 0.25*contract('ijab,ijab->', TEI[o, o, v, v], t2)
        term3 = 0.5*contract('ijab,ia,jb->', TEI[o, o, v, v], t1, t1)
        total = term1 + term2
        total = total + term3 
        return total                           
    
    # update the T2 iteratively
    def corrected_T2(self, t2, dt2, F):
        o = self.occ
        v = self.vir
        eps, evecs = np.linalg.eigh(F)
        Dem = eps[o].reshape(-1, 1, 1, 1)
        Dem = Dem + eps[o].reshape(-1, 1, 1)
        Dem = Dem - eps[v].reshape(-1, 1) 
        Dem = Dem - eps[v]
        Dem = 1/Dem
        t2 = t2 + contract('ijab,ijab->ijab', dt2, Dem)
        return t2
     
     # update the T1 iteratively    
    def corrected_T1(self, t1, dt1, F):
        o = self.occ
        v = self.vir
        eps, evecs = np.linalg.eigh(F)
        Dem =  eps[o].reshape(-1, 1) - eps[v]
        Dem = 1/Dem
        t1 = t1 + contract('ia,ia->ia', dt1, Dem)
        return t1
    
    #Routine for DIIS solver, builds all arrays(maxsize) before B is computed    
    def DIIS_solver(self, t1, t2, F, maxsize, maxiter, E_min):
            #Store the maxsize number of t1 and t2
            T1rhs = self.T1eq_rhs(t1, t2, np.longdouble(F))
            T2rhs = self.T2eq_rhs(t1, t2, np.longdouble(F))
            t1 = np.longdouble(self.corrected_T1(t1, T1rhs, F))
            t2 = np.longdouble(self.corrected_T2(t2, T2rhs, F))
            t1stored = [t1.copy()]
            t2stored = [t2.copy()]
            errort1 = []
            errort2 = []
            
            for n in range(1, maxsize+1):  
                T1rhs = self.T1eq_rhs(t1, t2, np.longdouble(F))
                T2rhs = self.T2eq_rhs(t1, t2, np.longdouble(F))
                t1 = np.longdouble(self.corrected_T1(t1, T1rhs, F))
                t2 = np.longdouble(self.corrected_T2(t2, T2rhs, F))
                t1stored.append(t1.copy())
                t2stored.append(t2.copy())
                
                errort1.append(t1stored[n]- t1stored[n-1])
                errort2.append(t2stored[n]- t2stored[n-1])

             # Build B
            B = np.ones((maxsize + 1, maxsize + 1)) * -1
            B[-1, -1] = 0
            for z in range(1, maxiter):
                CCSD_E_old = self.CCSD_Corr_E( t1, t2, F)
                for n in range(maxsize):
                    for m in range(maxsize):
                        a = contract('ia,ia->',errort1[m], errort1[n])
                        b = contract('ijab,ijab->', errort2[m], errort2[n])
                        B[n, m] = a + b
    
                # Build residual vector
                A = np.zeros(maxsize + 1)
                A[-1] = -1

                c = np.linalg.solve(B, A)
                
                # Update t1 and t2 
                t1 = 0.0*t1
                t2 = 0.0*t2
                for n in range(maxsize):
                    t1 += c[n] * t1stored[n+1]
                    t2 += c[n] * t2stored[n+1]

                oldt1 = t1.copy()
                oldt2 = t2.copy()
                #test if converged
                CCSD_E = self.CCSD_Corr_E( t1, t2, F)
                diff_E = CCSD_E - CCSD_E_old
                if (abs(diff_E) < E_min):
                    break
                #update t1 and t2 list
                T1rhs = self.T1eq_rhs(t1, t2, np.longdouble(F))
                T2rhs = self.T2eq_rhs(t1, t2, np.longdouble(F))
                t1 = np.longdouble(self.corrected_T1(t1, T1rhs, F))
                t2 = np.longdouble(self.corrected_T2(t2, T2rhs, F))
                t1stored.append(t1.copy())
                t2stored.append(t2.copy())
                
                errort1.append(t1 - oldt1)
                errort2.append(t2 - oldt2)
                
                print("inter =", z,  "\t", "CCSD_E =", CCSD_E,"diff=", diff_E)
                del t1stored[0]
                del t2stored[0]
                del errort1[0]
                del errort2[0]
            return CCSD_E, t1, t2

    #a regular iterative solver, Slow, don't use        
    def NO_DIIS_solver(self, t1, t2, F, maxsize, maxiter, E_min):    
        i=0
        for x in range (maxiter):
            CCSDE_Em = self.CCSD_Corr_E(t1, t2, F)
            T1rhs = self.T1eq_rhs(t1, t2, np.longdouble(F))
            T2rhs = self.T2eq_rhs(t1, t2, np.longdouble(F))
            t1 = np.longdouble(self.corrected_T1(t1, T1rhs, F))
            t2 = np.longdouble(self.corrected_T2(t2, T2rhs, F))
            CCSD_E = self.CCSD_Corr_E(t1, t2, F)
            diff_E = np.abs( CCSD_E -CCSDE_Em )
            i+=1
            if (abs(diff_E) < E_min):
                break
            print("inter =", i,  "\t", "CCSD_E =", CCSD_E,"diff=", diff_E)
        return CCSD_E, t1, t2
 
##############################################################################
#    
#     
#        
#                       Lambda Equations (CCSD):
#                       Derived by R. Glenn
#     
#   
#      
#######################################################################     

    # Build Fvv for L1 and L2 
    def LRFea(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = F[v, v].copy()
        term2 = - contract('ma,me->ea', F[o, v], t1)
        term3 = contract('emaf,mf->ea', TEI[v, o, v, v], t1)
        tau = 0.5*t2 + contract('ia,jb->ijab', t1, t1)
        term4 =-contract('mnaf,mnef->ea', TEI[o, o, v, v], tau)
        #Fea = term1 + term2 + term3 + term4
        
        total = term1 + term2 + term3 + term4
        return total
        
    #Build Foo for L1 and L2     
    def LRFim(self, t1, t2, F):
        v = self.vir
        o = self.occ  
        TEI = self.TEI 
        term1 = F[o, o].copy()
        term2 = contract('ie,me->im', F[o, v], t1)
        term3 = contract('inmf,nf->im', TEI[o, o, o, v], t1)
        tau = 0.5*t2 + contract('ia,jb->ijab', t1, t1) 
        term4 = contract('inef,mnef->im', TEI[o, o, v, v], tau)
        total = term1 + term2 + term3 + term4 
        return total
        
    #Build Wovvo          
    def LSWieam(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[o, v, v, o].copy()
        term2 = contract('eifa,mf->ieam', TEI[v, o, v, v], t1)
        term3 = -contract('nima,ne->ieam', TEI[o, o, o, v], t1)
        tau = t2 + contract('ia,jb->ijab', t1, t1)
        term4 =  -contract('ijab,mjbe->ieam', TEI[o, o, v, v], tau)
        total = term1 + term2 + term3 + term4
        return total
            
    #Build Wvvvo    
    def LRWefam(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        Fme = self.Fme(t1, t2, F)
        Wabef = self.LSWabef(t1, t2, F)
        

        term1 = 0.5*TEI[v, v, v, o].copy()
        term2 = 0.5*contract('na,mnef->efam', Fme, t2)
        term3 = contract('efab,mb->efam', Wabef, t1)
        term4a = -TEI[o, v, v, o].copy() + contract('jnab,nmfb->jfam', TEI[o, o, v, v], t2)  
        term4 = contract('jfam,je->efam', term4a, t1)
        tau =0.25*t2 + 0.5*contract('ia,jb->ijab', t1, t1) #- contract('ib,ja->ijab', t1, t1)
        term5 = contract('jnam,jnef->efam', TEI[o, o, v, o], tau)
        term6 = -contract('jfab,jmeb->efam', TEI[o, v, v, v], t2) 
        total = term1 + (term2 + term3 + term4 + term5 + term6) #+ extra
        return total
    
       #Build Wovoo                     
    def LRWibjm(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        Fme = self.Fme(t1, t2, F)
        Wmnij = self.LSWmnij(t1, t2, F)
        
        term1 = -0.5*TEI[o, v, o, o].copy()
        term2 = 0.5*contract('ie,jmbe->ibjm', Fme, t2)
        term3 = contract('injm,nb->ibjm', Wmnij, t1)
        term4a = -TEI[o, v, v, o].copy() - contract('inef,nmfb->ibem', TEI[o, o, v, v], t2) 
        term4 = contract('ibem,je->ibjm', term4a, t1)
        tau = 0.25*t2 + 0.5*contract('ia,jb->ijab', t1, t1) #-contract('ib,ja->ijab', t1, t1)
        term5 = -contract('ibef,jmef->ibjm', TEI[o, v, v, v], tau)
        term6 = contract('inem,jneb->ibjm', TEI[o, o, v, o], t2)
        total = term1 + (term2 + term3 + term4 + term5 + term6) 
        return total
                                                                                                                                          
    def Gfe(self, t2, lam2):
        return -0.5*contract('mnfb,mneb->fe', lam2, t2)
        
    def Gmn(self, t2, lam2):
        return 0.5*contract('njed,mjed->nm', lam2, t2)
             
    #Build Wvovv       
    def LWfiea(self, t1):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[v, o, v, v].copy()
        term2 = -contract('jiea,jf->fiea', TEI[o, o, v, v], t1)
        total = term1 + term2
        return total
        
     #Build Wooov   
    def LWmina(self, t1):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = TEI[o, o, o, v].copy()
        term2 = contract('mifa,nf->mina', TEI[o, o, v, v], t1)
        total = term1 + term2
        return total
        
###############Lam1 Equation#####################
 

    def lam_1eq_rhs(self, t1, t2, lam1, lam2, F):   
        v = self.vir
        o = self.occ
        TEI = self.TEI
        Fia = self.Fme(t1, t2, F)
        Fea = self.LRFea(t1, t2, F)
        Fim = self.LRFim(t1, t2, F)
        Wieam = self.LSWieam(t1, t2, F)
        Wefam = self.LRWefam(t1, t2, F)
        Wibjm= self.LRWibjm(t1, t2, F)
                    
        Gef = self.Gfe(t2, lam2)
        Gmn = self.Gmn(t2, lam2)
        Weifa = self.LWfiea(t1)
        Wmina = self.LWmina(t1)
        
        
        term1 = Fia.copy()
        term2 = contract('ea,ie->ia', Fea, lam1)
        term3 = -contract('im,ma->ia', Fim, lam1)
        term4 = contract('ieam,me->ia', Wieam, lam1)
        term5 = contract('efam,imef->ia', Wefam, lam2)
        term6 = contract('ibjm,jmab->ia', Wibjm, lam2) 
        term7 = -contract('fe,fiea->ia', Gef, Weifa)
        term8 = -contract('nm,mina->ia', Gmn, Wmina)
        total = (term1 + (term2 + term3 + term4) + term5 + term6 + term7 + term8) 
        return total
################################################################

    
    # Build Woooo 
    def LSWmnij(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = 0.5*TEI[o, o, o, o].copy()
        term2 = contract('ijme,ne->ijmn', TEI[o, o, o, v], t1)
        tau = 0.25*t2 + 0.5*contract('ia,jb->ijab', t1, t1)
        term3 = contract('ijfe,mnfe->ijmn', TEI[o, o, v, v], tau)
        total = (term1 + term2 + term3)
        return total
             
    #Build Wvvvv          
    def LSWabef(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        term1 = 0.5*TEI[v, v, v, v].copy()
        term2 = -contract('emab,mf->efab', TEI[v, o, v, v], t1)
        tau = 0.25*t2 + 0.5*contract('ia,jb->ijab', t1, t1) 
        term3 = contract('nmab,nmef->efab', TEI[o, o, v, v], tau)
        total = (term1 + term2 + term3  )
        return total
                                
########################Lam 2 Equations################
    def lam2eq_rhs(self, t1, t2, lam1, lam2, F):
        v = self.vir
        o = self.occ  
        TEI = self.TEI     
        Feb = self.LRFea(t1, t2, F)
        Fjm = self.LRFim(t1, t2, F)
        Wijmn = self.LSWmnij(t1, t2, F)
        Wefab = self.LSWabef(t1, t2, F)
        Wjebm = self.LSWieam(t1, t2, F)
        Wejab = self.LWfiea(t1)
        Wijmb = self.LWmina(t1)
        Fjb = self.Fme(t1, t2, F)
        Gbe = self.Gfe(t2, lam2)
        Gmj = self.Gmn(t2, lam2) 
        
        term1 = TEI[o, o, v, v]
        term2a = contract('eb,ijae->ijab', Feb, lam2)
        term2 = term2a - term2a.swapaxes(2,3)
        term3a = -contract('jm,imab->ijab', Fjm, lam2)
        term3 = term3a - term3a.swapaxes(0,1)
        term4 = contract('ijmn,mnab->ijab', Wijmn, lam2)
        term5 = contract('efab,ijef->ijab', Wefab, lam2)
        term6a = contract('ejab,ie->ijab', Wejab, lam1)
        term6 = term6a - term6a.swapaxes(0,1)
        term7a = -contract('ijmb,ma->ijab', Wijmb, lam1)
        term7 = term7a - term7a.swapaxes(2,3)
        #term8 and 9
        term89a = contract('jebm,imae->ijab', Wjebm, lam2) + contract('jb,ia->ijab', Fjb, lam1)
        term89 = term89a 
        term89 = term89 - term89a.swapaxes(2,3) 
        term89 = term89 - term89a.swapaxes(0,1) 
        term89 = term89 + term89a.swapaxes(0,1).swapaxes(2,3) 
        term10a = contract('ijfb,af->ijab', TEI[o, o, v, v], Gbe)
        term10 = term10a - term10a.swapaxes(2,3)
        term11a = -contract('mjab,im->ijab', TEI[o, o, v, v], Gmj)
        term11 = term11a - term11a.swapaxes(0,1)
        total = term1 + term2 + term3 + term4 + (term5 + term6) + term7 + term89
        total = total + term10 + term11
        return total
        
    def CCSD_pseudo_E(self, t1, t2, lam1, lam2, F):
        o = self.occ
        v = self.vir
        TEI = self.TEI
        term1 = contract('ia,ia->', F[o, v], lam1)
        term2 = 0.25*contract('ijab,ijab->', TEI[o, o, v, v], lam2)
        return term1, term2                         
        
    def corrected_lam2(self, lam2, dlam2, F):
        o = self.occ
        v = self.vir
        eps, evecs = np.linalg.eigh(F)
        Dem = eps[o].reshape(-1, 1, 1, 1)
        Dem = Dem + eps[o].reshape(-1, 1, 1)
        Dem = Dem - eps[v].reshape(-1, 1) 
        Dem = Dem - eps[v]
        Dem = 1/Dem
        lam2 = lam2 + contract('ijab,ijab->ijab', dlam2, Dem)
        return lam2
         
    def corrected_lam1(self, lam1, dlam1, F):
        o = self.occ
        v = self.vir
        eps, evecs = np.linalg.eigh(F)
        Dem =  eps[o].reshape(-1, 1) - eps[v]
        Dem = 1/Dem
        lam1 = lam1 + contract('ia,ia->ia', dlam1, Dem)
        return lam1

    
    def NO_DIIS_solve_lamr(self, t1, t2, lam1, lam2, F, maxsize, maxiter, E_min):    
        i=0
        print("this is the convergence", E_min)
        for x in range (maxiter):
            E1, E2 = self.CCSD_pseudo_E(t1, t2, lam1, lam2, F)
            pseudo_Em = E1 +E2
            lam1rhs = self.lam_1eq_rhs(t1, t2, lam1, lam2, np.longdouble(F))
            lam2rhs = self.lam2eq_rhs(t1, t2, lam1, lam2 ,np.longdouble(F))
            lam1 = np.longdouble(self.corrected_lam1(lam1, lam1rhs, F))
            lam2 = np.longdouble(self.corrected_lam2(lam2, lam2rhs, F))
            E1, E2 = self.CCSD_pseudo_E(t1, t2, lam1, lam2, F)
            pseudo_E = E1 +E2
            diff_E = np.abs( pseudo_E -pseudo_Em )
            i+=1
            
            if (abs(diff_E) < E_min):
                break
                #pass
            print("inter =", i,  "\t", "pseudo_E =", pseudo_E,"diff=", diff_E)
        print(E1, E2)
        return pseudo_E, lam1, lam2
        
    def DIIS_solver_Lam(self, t1, t2, lam1, lam2, F, maxsize, maxiter, E_min): 
            #Store the maxsize number of t1 and t2
            lam1rhs = self.lam_1eq_rhs(t1, t2, lam1, lam2, np.longdouble(F))
            lam2rhs = self.lam2eq_rhs(t1, t2, lam1, lam2 ,np.longdouble(F))
            lam1 = np.longdouble(self.corrected_lam1(lam1, lam1rhs, F))
            lam2 = np.longdouble(self.corrected_lam2(lam2, lam2rhs, F))
            lam1stored = [lam1.copy()]
            lam2stored = [lam2.copy()]
            errort1 = []
            errort2 = []
            
            for n in range(1, maxsize+1):  
                lam1rhs = self.lam_1eq_rhs(t1, t2, lam1, lam2, np.longdouble(F))
                lam2rhs = self.lam2eq_rhs(t1, t2, lam1, lam2 ,np.longdouble(F))
                lam1 = np.longdouble(self.corrected_lam1(lam1, lam1rhs, F))
                lam2 = np.longdouble(self.corrected_lam2(lam2, lam2rhs, F))
                lam1stored.append(lam1.copy())
                lam2stored.append(lam2.copy())
                
                errort1.append(lam1stored[n]-lam1stored[n-1])
                errort2.append(lam2stored[n]- lam2stored[n-1])

             # Build B
            B = np.ones((maxsize + 1, maxsize + 1)) * -1
            B[-1, -1] = 0
            for z in range(1, maxiter):
                E1, E2 = self.CCSD_pseudo_E(t1, t2, lam1, lam2, F)
                CCSD_E_old = E1 + E2
                for n in range(maxsize):
                    for m in range(maxsize):
                        a = contract('ia,ia->',errort1[m], errort1[n])
                        b = contract('ijab,ijab->', errort2[m], errort2[n])
                        B[n, m] = a + b
    
                # Build residual vector
                A = np.zeros(maxsize + 1)
                A[-1] = -1

                c = np.linalg.solve(B, A)
                
                # Update t1 and t2 
                lam1 = 0.0*lam1
                lam2 = 0.0*lam2
                for n in range(maxsize):
                    lam1 += c[n] * lam1stored[n+1]
                    lam2 += c[n] * lam2stored[n+1]

                oldlam1 = lam1.copy()
                oldlam2 = lam2.copy()
                #test if converged
                E1, E2 = self.CCSD_pseudo_E(t1, t2, lam1, lam2, F)
                CCSD_E = E1 + E2
                diff_E = CCSD_E - CCSD_E_old
                if (abs(diff_E) < E_min):
                    break
                #update t1 and t2 list
                lam1rhs = self.lam_1eq_rhs(t1, t2, lam1, lam2, np.longdouble(F))
                lam2rhs = self.lam2eq_rhs(t1, t2, lam1, lam2 ,np.longdouble(F))
                lam1 = np.longdouble(self.corrected_lam1(lam1, lam1rhs, F))
                lam2 = np.longdouble(self.corrected_lam2(lam2, lam2rhs, F))
                lam1stored.append(lam1.copy())
                lam2stored.append(lam2.copy())
                
                errort1.append(lam1 - oldlam1)
                errort2.append(lam2 - oldlam2)
                
                print("inter =", z,  "\t", "Pseudo_E =", CCSD_E,"diff=", diff_E)
                #print("inter =", z,  "\t", "CCSD_E =", CCSD_E,"diff=", diff_E, "lam1E=", E1, "lam2E=", E2
                del lam1stored[0]
                del lam2stored[0]
                del errort1[0]
                del errort2[0]
            print("Lambda1 energy =", E1)
            print("Lambda2 energy =", E2)
            return CCSD_E, lam1, lam2

    def print_2(self, t1):
        t1_tmp = t1.ravel()
        sort_t1 = sorted(t1_tmp, reverse=True)
        for x in range(len(sort_t1)-1):
            
            if (round(sort_t1[x], 10) ==0e13 or round(sort_t1[x+1], 13) == round(sort_t1[x], 13)):
                 
                pass
            else:
                print('\t', ('% 5.13f' %  sort_t1[x]))
        print('\t', ('% 5.13f' %  sort_t1[-1]))

    def remove_dup(self, t):
        s = []
        for i in t:
            i = round(i, 10)
            if i not in s:
                s.append(i)
        return s
    
    def print_T_amp(self, t1, t2):
        sort_t1 = sorted(t1.ravel())
        sort_t1 = self.remove_dup(sort_t1)
        sort_t2 = sorted(t2.ravel())
        sort_t2 = self.remove_dup(sort_t2)
        print("\n   The largest T1 values:")
        for x in range(len(sort_t1)):
            #if (round(sort_t1[x], 5) ==0e5 or x % 2 or 30< x < 60 ):
            if (round(sort_t1[x], 5) ==0e5 ):

                pass
            else: 
                print('\t', ('% 5.10f' %  sort_t1[x]))
       
        print("\n   The largest T2 values are:")

        for x in range(len(sort_t2)):
            if (round(sort_t2[x],2) ==0e5 ): #or x % 2 or x > 20):
                pass
            else:
                print('\t', ('% 5.10f' %  sort_t2[x]))  
                
    def print_L_amp(self, lam1, lam2):
        sort_lam1 = sorted(-abs(lam1.ravel()))
        sort_lam1 = self.remove_dup(sort_lam1)
        sort_lam2 = sorted(lam2.ravel())
        sort_lam2 = self.remove_dup(sort_lam2)
        print("\n   The largest lam1 values:")
        for x in range(len(sort_lam1)):
            if (round(sort_lam1[x], 5) ==0e5):# or x % 2 or x >20):
                pass
            else: 
                print('\t', ('% 5.10f' %  sort_lam1[x]))
        
        print("\n   The largest lam2 values are:")
        for x in range(len(sort_lam2)):
            if (round(sort_lam2[x],2) ==0.00 or x % 2 or x > 20):
                pass
            else:
                print('\t', ('% 5.10f' %  sort_lam2[x]))   


############################################################
#
#               T1 and T2-equations (CC2)
#                   By R. Glenn, I used T. Daniel Crawfords equations
#
#
#
############################################################

    
    ########### Build T2 Equation################################
    def T2eq_rhs_CC2(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        
        fae = F[v, v].copy()
        fmi = F[o, o].copy()
        wabef_2 = self.Wabef_2(t1 ,t2, F)
        wmnij_2 = self.Wmnij_2(t1 ,t2, F)
        #All terms in the T2 Equation
        term1 = TEI[o, o, v, v].copy()
        
        term2tmp = fae #- 0.5 *contract('me,mb->be', fme, t1)
        term2a = contract('be,ijae->ijab', term2tmp, t2)
        term2 = term2a - term2a.swapaxes(2, 3) #swap ab
        
        term3temp = fmi #+ 0.5 *contract('me,je->mj', fme, t1)
        term3a = -contract('mj,imab->ijab', term3temp, t2)
        term3 = term3a - term3a.swapaxes(0, 1) #swap ij
        
        tau = contract('ma,nb->mnab', t1, t1) - contract('na,mb->mnab', t1, t1)
        term44 = 0.5*contract('mnij,mnab->ijab', wmnij_2, tau)
        term55 =  0.5*contract('abef,ijef->ijab', wabef_2, tau)
        
        term6tmp = - contract('mbej,ie,ma->ijab', TEI[o, v, v, o], t1, t1)
        term6 =  term6tmp - term6tmp.swapaxes(2, 3)  - term6tmp.swapaxes(0, 1)  + term6tmp.swapaxes(0, 1).swapaxes(2, 3)
        
        
        term7tmp = contract('abej,ie->ijab', TEI[v ,v, v, o], t1)
        term7 =  term7tmp - term7tmp.swapaxes(0, 1) #swap ij
        
        term8tmp = -contract('mbij,ma->ijab', TEI[o, v, o, o], t1)
        term8 =  term8tmp - term8tmp.swapaxes(2, 3) #swap ab
        
        total = term1 + term2 + term3 + term44 + term55 + term6 + term7 + term8
        return total


    #Routine for DIIS solver, builds all arrays(maxsize) before B is computed    
    def DIIS_solver_CC2(self, t1, t2, F, maxsize, maxiter, E_min):
            #Store the maxsize number of t1 and t2
            T1rhs = self.T1eq_rhs(t1, t2, F)
            T2rhs = self.T2eq_rhs_CC2(t1, t2, F)
            t1 = self.corrected_T1(t1, T1rhs, F)
            t2 = self.corrected_T2(t2, T2rhs, F)
            t1stored = [t1.copy()]
            t2stored = [t2.copy()]
            errort1 = []
            errort2 = []
            
            for n in range(1, maxsize+1):  
                T1rhs = self.T1eq_rhs(t1, t2, F)
                T2rhs = self.T2eq_rhs_CC2(t1, t2, F)
                t1 = self.corrected_T1(t1, T1rhs, F)
                t2 = self.corrected_T2(t2, T2rhs, F)
                t1stored.append(t1.copy())
                t2stored.append(t2.copy())
                
                errort1.append(t1stored[n]- t1stored[n-1])
                errort2.append(t2stored[n]- t2stored[n-1])

             # Build B
            B = np.ones((maxsize + 1, maxsize + 1)) * -1
            B[-1, -1] = 0
            for z in range(1, maxiter):
                CC2_E_old = self.CCSD_Corr_E( t1, t2, F)
                for n in range(maxsize):
                    for m in range(maxsize):
                        a = contract('ia,ia->',errort1[m], errort1[n])
                        b = contract('ijab,ijab->', errort2[m], errort2[n])
                        B[n, m] = a + b
    
                # Build residual vector
                A = np.zeros(maxsize + 1)
                A[-1] = -1

                c = np.linalg.solve(B, A)
                
                # Update t1 and t2 
                t1 = 0.0*t1
                t2 = 0.0*t2
                for n in range(maxsize):
                    t1 += c[n] * t1stored[n+1]
                    t2 += c[n] * t2stored[n+1]

                oldt1 = t1.copy()
                oldt2 = t2.copy()
                #test if converged
                CC2_E = self.CCSD_Corr_E( t1, t2, F)
                diff_E = CC2_E - CC2_E_old
                if (abs(diff_E) < E_min):
                    break
                #update t1 and t2 list
                T1rhs = self.T1eq_rhs(t1, t2, F)
                T2rhs = self.T2eq_rhs_CC2(t1, t2, F)
                t1 = self.corrected_T1(t1, T1rhs, F)
                t2 = self.corrected_T2(t2, T2rhs, F)
                t1stored.append(t1.copy())
                t2stored.append(t2.copy())
                
                errort1.append(t1 - oldt1)
                errort2.append(t2 - oldt2)
                
                print("inter =", z,  "\t", "CC2_E =", CC2_E,"diff=", diff_E)
                del t1stored[0]
                del t2stored[0]
                del errort1[0]
                del errort2[0]
            return CC2_E, t1, t2


 ##################################################################
 #
 #
 #                  Single-electron density matrix equations-derived by R. Glenn
 #
 #
 #####################################################################         
    #Dipoles in the MO basis
    def Defd_dipole(self):
        C = np.asarray(self.C)
        nmo = self.nmo
        tmp_dipoles = self.mints.so_dipole()
        dipoles_xyz = []
        for n in range(3):
            temp = contract('li,lk,kj->ij',C,tmp_dipoles[n],C)
            temp = temp.repeat(2, axis=1).repeat(2, axis=0)
            temp = temp*np.tile(np.identity(2),(nmo,nmo))
            dipoles_xyz.append(temp)
        return dipoles_xyz
    

    #Build Dvv 
    def Dij(self, t1, t2, lam1, lam2):
        term1 = contract('je,ie->ij', lam1, t1)
        term2 = 0.5*contract('jmea,imea->ij', lam2, t2)
        total = -(term1 + term2)
        return total
    
      #Build Doo 
    def Dab(self, t1, t2, lam1, lam2):
        term1 = contract('nb,na->ab', lam1, t1)
        term2 = 0.5*contract('mneb,mnea->ab', lam2, t2)
        total = term1 + term2
        return total  
        
      #Build Dvo
    def Dai(self, t1, t2, lam1, lam2):
        term1a = t1 
        term1 = contract('ia->ai', term1a)
        term2 = contract('me,miea->ai', lam1, t2)
        term3 = -contract('me,ma,ie->ai', lam1, t1, t1)
        term4 = -0.5*contract('mnef,mnaf,ie->ai', lam2, t2, t1)
        term5 = -0.5*contract('mnef,inef,ma->ai', lam2, t2, t1)
        total = term1 + term2 + term3 + term4 + term5
        return total
    #Dov is equal to lam1

    def Buildpho(self, F):
        o =self.occ
        S12, S12plus = self.GenS12()
        evals, evecs = np.linalg.eigh(F)
        C = contract('ij,jk->ik', S12, evecs)
        pho = contract('ik,jk->ij', C[:, o], np.conj(C[:, o]))
        return pho

    def pholowdinbasis(self, pho):
        S12, S12plus = self.GenS12()
        pholowdin = contract('il,lk,jk->ij', S12plus, pho, S12plus)
        return pholowdin 
                                     
    #For testing purposes only, to check my density as a function of time 
    def pho_checks(self, HF_p, corr_p, dip_xyz_corr):
        
        ##################################
        #
        #       Check the density, dipole, trace, idempotency
        #
        ####################################
        
        #get the correlated dipoles from psi to compare to
        dip_x = np.asarray(psi4.core.get_variable('CC DIPOLE X'))
        dip_y = np.asarray(psi4.core.get_variable('CC DIPOLE Y'))
        dip_z = np.asarray(psi4.core.get_variable('CC DIPOLE Z'))
        fac = 0.393456#The conversion factor from dybe to a.u.
        x_nuclear_dipole = 0.0 #H2O
        y_nuclear_dipole = 0.0 #H2O
        z_nuclear_dipole = 1.1273 #H2O
        dip_x = dip_x*fac -x_nuclear_dipole
        dip_y = dip_y*fac -y_nuclear_dipole
        dip_z = dip_z*fac -z_nuclear_dipole
        
        #print("\n This is the psi4 electric in a. u. units dipole \n", psi4.core.get_variable('CC DIPOLE Z'))

        #Compare calculated CC dipole to psi4
        #print("This is the calculated electric in a. u. dipole \n", "x=", dip_xyz_corr[0], "y=", dip_xyz_corr[1], "z=", dip_xyz_corr[2])
        #print("\n This is the psi4 electric in a. u. units dipole \n", "x=", dip_x, "y=", dip_y, "z=", dip_z)
        
        #Check that the p_trace_corr = 0, and p_trace_Hf =0
        p_trace_corr = np.sum(contract('ii->i', corr_p))
        p_trace_HF = np.sum(contract('ii->i', HF_p)) 
        print("The trace of pho corr is", p_trace_corr,"\n")
        #print "The trace of pho is", p_trace_tot,"\n"
        #print("The trace of pho HF is", p_trace_HF,"\n")
        
        #Check the idempotency of HF
        p_sqd = contract('ij,kj->ik', HF_p, HF_p)
        #print("The difference between HF density p and p^2 should be zero \n", HF_p-p_sqd, "\n")

        #Check the idempotency of the total density ( It is not idempotent )
        ptot = HF_p + corr_p
        p_sqd = contract('ij,kj->ik', ptot, ptot)
        np.set_printoptions(precision=3)
        #print("The difference between the total p and p^2 should be zero \n", ptot-p_sqd, "\n")
        
    #Build the expectation value of the dipole moment
    def dipole_moment(self, t1, t2, lam1, lam2, F):
        #Build the four blocks of the density matrix
        pai = self.Dai(t1, t2, lam1, lam2)
        pia = lam1 
        pab = self.Dab(t1, t2, lam1, lam2)
        pij = self.Dij(t1, t2, lam1, lam2)
        dipolexyz = self.Defd_dipole()

        #Build the correlated density matrix
        left_p = np.vstack((pij, pai))
        right_p = np.vstack((pia, pab))
        corr_p = np.hstack((left_p, right_p))
        
        #Build the Hartree Fock Density matrix
        HF_p = self.Buildpho(F)
        HF_p = self.pholowdinbasis(HF_p)
        
        #Calculate the corr dipole moment
        dip_xyz_corr = []
        HF_dip = []
        for i in range(3):
            HF_dip.append(contract('ij,ji->', dipolexyz[i], HF_p))
            temp = contract('ij,ji->', dipolexyz[i], corr_p)
            #temp = contract('ij,ji->', dipolexyz[i], corr_p)
            #trace = contract('ii', temp)
            
            dip_xyz_corr.append(temp)
        #print("\n This is the HF dipole in a. u. units dipole \n", "x=", HF_dip[0], "y=", HF_dip[1], "z=", HF_dip[2])
        #Check important characteristics before moving on
        #self.pho_checks(HF_p, corr_p, dip_xyz_corr)
        return dip_xyz_corr             
        
########################################################
#
#
#
#       ###Functions for doing the Time integration
#
#
###########################################################
    def Save_parameters(self, w0, A, t0, t, dt, precs, i, a, CCSD_or_CC2):
        save_dat =  pd.DataFrame( columns = ( 'w0', 'A', 't0','dt','precs', 'i', 'a','CCSD_or_CC2'))
        save_dat.loc[1] = [w0, A, t, dt, precs, i, a, CCSD_or_CC2]
        save_dat.to_csv('Parameters.csv',float_format='%.10f')
        
    def write_2data(self, F, FileName, precs):
        with open(FileName, 'w') as outcsv:
        #configure writer to write standard csv file
            writer = csv.writer(outcsv, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            for i in range(F.shape[0]):
                for a in range(F.shape[1]):
                #Write item to outcsv
                    writer.writerow([i, a, np.around(F[i][a], decimals=precs) ])

    def write_4data(self, F, FileName, precs):
        with open(FileName, 'w') as outcsv:
        #configure writer to write standard csv file
            writer = csv.writer(outcsv, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            for i in range(F.shape[0]):
                for j in range(F.shape[1]):
                    for a in range(F.shape[2]):
                        for b in range(F.shape[3]):
                        #Write item to outcsv
                            writer.writerow([i, j, a, b, np.around(F[i][j][a][b], decimals=precs) ])

    def Save_data(self, F, t1, t2, lam1, lam2, data, timing, precs, restart):
        if restart is None: 
            data.to_csv('H2O.csv')
            timing.to_csv('timing.csv')
        else:
           data.to_csv('H2O.csv', mode='a', header=False)
           timing.to_csv('timing.csv', mode='a', header=False)
           #with open('H2O.csv', 'a') as f:
           #     data.to_csv(f, header=False)
           # with open('timing.csv', 'a') as f:
           #     timing.to_csv(f, header=False)
        self.write_2data(F.real, 'F_real.dat', precs)
        self.write_2data(F.imag, 'F_imag.dat', precs)
        self.write_2data(t1.real, 't1_real.dat', precs)
        self.write_2data(t1.imag, 't1_imag.dat', precs)
        self.write_4data(t2.real, 't2_real.dat', precs)
        self.write_4data(t2.imag, 't2_imag.dat', precs)
        self.write_2data(lam1.real, 'lam1_real.dat', precs)
        self.write_2data(lam1.imag, 'lam1_imag.dat', precs)
        self.write_4data(lam2.real, 'lam2_real.dat', precs)
        self.write_4data(lam2.imag, 'lam2_imag.dat', precs)

###############################################
#        
#            
#     Runge-Kutta time dependent propagator
#
#    
################################################
 
 ########Functions for Runge-Kutta#################
 #########for T1, T2, L1, L2#######################
       #T1 Runge-Kutta function 
    def ft1(self, t, dt, t1, t2, F, Vt):  
        k1 = self.T1eq_rhs(t1, t2, F + Vt(t))
        k2 = self.T1eq_rhs(t1 + dt/2.0*k1, t2, F + Vt(t + dt/2.0)) 
        k3 = self.T1eq_rhs(t1 + dt/2.0*k2, t2, F + Vt(t + dt/2.0))
        k4 = self.T1eq_rhs(t1 + dt*k3, t2, F + Vt(t + dt))
        return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
         
    #T2 Runge-Kutta function 
    def ft2(self, t, dt, t1, t2, F, Vt):
        k1 = self.T2eq_rhs(t1, t2, F + Vt(t))
        k2 = self.T2eq_rhs(t1, t2 + dt/2.0*k1, F + Vt(t + dt/2.0))  
        k3 = self.T2eq_rhs(t1, t2 + dt/2.0*k2, F + Vt(t + dt/2.0)) 
        k4 = self.T2eq_rhs(t1, t2 + dt*k3,  F + Vt(t + dt))
        return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
                 
    #L1 Runge-Kutta function 
    def fL1(self, t, dt, t1, t2, lam1, lam2, F, Vt):
        k1 = self.lam_1eq_rhs(t1, t2, lam1, lam2, F + Vt(t))
        k2 = self.lam_1eq_rhs(t1, t2, lam1 + dt/2.0*k1, lam2, F + Vt(t + dt/2.0))  
        k3 = self.lam_1eq_rhs(t1, t2, lam1 + dt/2.0*k2, lam2, F + Vt(t + dt/2.0)) 
        k4 = self.lam_1eq_rhs(t1, t2, lam1 + dt*k3, lam2, F + Vt(t + dt))
        return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)  
           
   #L2 Runge-Kutta function  
    def fL2(self, t, dt, t1, t2, lam1, lam2, F, Vt):
        k1 = self.lam2eq_rhs(t1, t2, lam1, lam2, F + Vt(t))
        k2 = self.lam2eq_rhs(t1, t2, lam1, lam2 + dt/2.0*k1, F + Vt(t + dt/2.0))  
        k3 = self.lam2eq_rhs(t1, t2, lam1, lam2 + dt/2.0*k2, F + Vt(t + dt/2.0)) 
        k4 = self.lam2eq_rhs(t1, t2, lam1, lam2 + dt*k3, F + Vt(t + dt))
        return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)      
############END functions for Runge-Kutta#############
     
    ####Time propagator#############
    def Runge_Kutta_solver(self, F, t1, t2, lam1, lam2, w0, A, t0, tf, dt, timeout, precs, restart=None):
        t = t0
        data =  pd.DataFrame( columns = ('time', 'mu_real', 'mu_imag')) 
        timing =  pd.DataFrame( columns = ('total','t1', 't2', 'l1','l2')) 
        
        #Electric field, it is in the z-direction for now
        def Vt(t):
            mu = self.Defd_dipole()
            pi = np.cos(-1)
            return -A*mu[2]*cmath.exp(1j*w0*2*np.pi*t)

        i=0
        start = time.time()
        mua = self.dipole_moment(t1, t2, lam1, lam2, F)
        print("time \t\t mu_z_real \t\t mu_z_real")
        print(round(t, 4), '\t', mua[2].real, '\t', mua[2].imag)
        while t < tf:
            L1min = np.around(lam1, decimals=precs) 
            L2min = np.around(lam2, decimals=precs) 
            i += 1
            t += dt
            itertime_t1 = itertime_t2 = 0
            
            t1min = np.around(t1.copy(), decimals=precs)
            t2min = np.around(t2.copy(), decimals=precs)
            itertime = time.time()
            dt1 = -1j*self.ft1(t, dt, t1, t2, F, Vt) #Runge-Kutta
            itertime_t1 = -itertime + time.time()
            itertime = time.time()
            dt2 = -1j*self.ft2(t, dt, t1, t2, F, Vt) #Runge-Kutta
            itertime_t2 = -itertime + time.time()
            itertime = time.time()
            dL1 = 1j*self.fL1(t, dt, t1, t2, lam1, lam2, F, Vt) #Runge-Kutta
            itertime_l1 = -itertime  + time.time()
            itertime = time.time()
            dL2 = 1j*self.fL2(t, dt, t1, t2, lam1, lam2, F, Vt)  #Runge-Kutta
            itertime_l2 = -itertime  + time.time()
            total = itertime_t1 + itertime_t2 + itertime_l1 + itertime_l2
            timing.loc[i] = [total, itertime_t1, itertime_t2, itertime_l1, itertime_l2 ]
            
            t1 = t1min + dt1
            t2 = (t2min + dt2)
            lam1 = (L1min + dL1)
            lam2 = (L2min + dL2)
            
           
            stop = time.time()-start
            
            #self.check_T1_T2_L1_L2(t1, t2, lam1, lam2, F)
            mua = self.dipole_moment(t1, t2, lam1, lam2, F)
            data.loc[i] = [round(t, 4), mua[2].real, mua[2].imag  ]
            print(round(t, 4), '\t', mua[2].real, '\t', mua[2].imag)
            
            if abs(stop)>0.9*timeout*60.0:
                self.Save_data(F, t1min, t2min, L1min, L2min, data, timing, precs, restart)
                self.Save_parameters(w0, A, t0, t-dt, dt, precs, t1.shape[0], t1.shape[1], 'CCSD')
    
                break
            #Calculate the dipole moment using the density matrix
            if abs(mua[2].real) > 100:
                self.Save_data(F, t1min, t2min, L1min, L2min, data, timing, precs, restart)
                self.Save_parameters(w0, A, t0, t-dt, dt, precs, t1.shape[0], t1.shape[1], 'CCSD')
                break
            
        stop = time.time()
        print("total time non-adapative step:", stop-start)
        print("total steps:", i)
        print("step-time:", (stop-start)/i)



