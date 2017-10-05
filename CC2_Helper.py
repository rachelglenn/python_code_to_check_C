
############################################################
#               T1 and T2-equations (CC2)
#                   By R. Glenn
##########################################################
from CCSD_Helper import *
class CC2_Helper(CCSD_Helper):
    def __init__(self, psi):
        super(CC2_Helper, self).__init__(psi)


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
        
        term2tmp = fae
        term2a = contract('be,ijae->ijab', term2tmp, t2)
        term2 = term2a - term2a.swapaxes(2, 3)
        
        term3temp = fmi
        term3a = -contract('mj,imab->ijab', term3temp, t2)
        term3 = term3a - term3a.swapaxes(0, 1)
        
        tau = contract('ma,nb->mnab', t1, t1) - contract('na,mb->mnab', t1, t1)
        term44 = 0.5*contract('mnij,mnab->ijab', wmnij_2, tau)
        term55 =  0.5*contract('abef,ijef->ijab', wabef_2, tau)
        
        term6tmp = - contract('mbej,ie,ma->ijab', TEI[o, v, v, o], t1, t1)
        term6 =  term6tmp - term6tmp.swapaxes(2, 3)  - term6tmp.swapaxes(0, 1)  + term6tmp.swapaxes(0, 1).swapaxes(2, 3)
        
        
        term7tmp = contract('abej,ie->ijab', TEI[v ,v, v, o], t1)
        term7 =  term7tmp - term7tmp.swapaxes(0, 1)
        
        term8tmp = -contract('mbij,ma->ijab', TEI[o, v, o, o], t1)
        term8 =  term8tmp - term8tmp.swapaxes(2, 3)
        
        total = term1 + term2 + term3 + term44 + term55 + term6 + term7 + term8
        return total


    def L1eq_rhs_CC2(self, t1, t2, lam1, lam2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        #RHS = Fme
        FME = self.Fme(t1, t2, F)
        lam1_rhs = FME.copy()
        
        #Check L1 RHS += P(ab) Lie * Fea
        Fea = self.LRFea(t1, t2, F)
        lam1_rhs = lam1_rhs + contract('ea,ie->ia', Fea, lam1)

        #Check L1 RHS -= P(ij) Lma * Fjm
        Fim = self.LRFim(t1, t2, F)
        term = -contract('im,ma->ia', Fim, lam1)
        lam1_rhs = lam1_rhs + term

        #Check L1 RHS += Lme*Wieam
        #match Wmbej
        #** Wmbej = <mb||ej> + t_j^f <mb||ef> - t_n^b <mn||ej>
        #**         - { t_j^f t_n^b } <mn||ef>
        Wmbej = TEI[o, v, v, o].copy()
        Wmbej = Wmbej + contract('mbef,jf->mbej', TEI[o, v, v, v], t1)
        Wmbej = Wmbej - contract('mnej,nb->mbej', TEI[o, o, v, o], t1)
        tau = contract('ia,jb->ijab', t1, t1)
        Wmbej = Wmbej - contract('mnef,jnfb->mbej', TEI[o, o, v, v], tau)
        #Gbj[R] = -Lem[R] tmjbe
        Gbj = -contract('me,mjbe->bj', lam1, t2)
        lam1_rhs = lam1_rhs + contract('ijab,bj->ia', TEI[o, o, v, v], Gbj)

        #Wmbej = self.LSWieam(t1, t2, F)
        lam1_rhs = lam1_rhs + contract('ieam,me->ia', Wmbej, lam1)

        #Check L1 RHS += -1/2 Lmnae*Wiemn
        #Match Wmnij
        #Wmnij = <mn||ij> + P(ij) t_j^e <mn||ie> + t_i^e t_j^f <mn||ef>
        Wmnij = TEI[o, o, o, o]
        Wmnij = Wmnij + contract('mnie,je->mnij', TEI[o, o, o, v], t1)
        Wmnij = Wmnij - contract('mnje,ie->mnij', TEI[o, o, o, v], t1)
        tau = contract('ia,jb->ijab', t1, t1) - contract('ib,ja->ijab', t1, t1)
        Wmnij = Wmnij - 0.5*contract('ijef,mnfe->ijmn', TEI[o, o, v, v], tau)
    
    
        Wmbij = TEI[o, v, o, o].copy()
        Wmbij = Wmbij - contract('nb,mnij->mbij', t1, Wmnij)
        Wmbij = Wmbij + 0.5*contract('mbef,ijef->mbij', TEI[o, v, v, v], tau)
        Zmejb = TEI[o, v, v, o].copy()
        Zmejb_T1 = contract('mbei,je->mbij', TEI[o, v, v, o], t1)
        Zmejb_T1 = Zmejb_T1  - contract('mbej,ie->mbij', Zmejb, t1)
        Wmbij = Wmbij - Zmejb_T1
        term = - 0.5*contract('iemn,mnae->ia', Wmbij, lam2)
        lam1_rhs = lam1_rhs + term
        
        #L1 RHS += 1/2 Limef*Wefam
        # Wabei = 1/2 <Ei|Ab>
        Wabei = TEI[v, v, v, o].copy()
        
        # WEbEi += <Ab|Ef> * t(i,f)
        Wabei = Wabei +  contract('abef,if->abei', TEI[v, v, v, v], t1)
        #Zmbej = TEI[o, v, v, o].copy()--first term
        
        # WEbEi = tma Zmbej
        # Zmbej = = <mb||ej> +  t_j^f[R] <mb||ef> - t_n^b[R] <mn||ej> + t_j^f[R] t_n^b[R] <mn||ef>        
        Zmbej = TEI[o, v, v, o].copy()
        Zmbej = Zmbej + contract('mbef,jf->mbej', TEI[o, v, v, v], t1)
        Zmbej = Zmbej - 0.5*contract('mnej,nb->mbej', TEI[o, o, v, o], t1)
        tau = contract('ia,jb->ijab', t1, t1) # - contract('ib,ja->ijab', t1, t1)
        Zmbej = Zmbej - 0.5*contract('mnef,jnfb->mbej', TEI[o, o, v, v], tau)
        
        Wabei = Wabei + contract('maei,mb->abei', Zmbej, t1)
        Wabei = Wabei - contract('mbei,ma->abei', Zmbej, t1)
    
        term = 0.5*contract('efam,imef->ia', Wabei, lam2)
        lam1_rhs = lam1_rhs + term

        return lam1_rhs

    def L2eq_rhs_CC2(self, t1, t2, lam1, lam2, F):

        v = self.vir
        o = self.occ
        TEI = self.TEI

        term1 = TEI[o, o, v, v].copy()
        Feb = F[v, v].copy()
        term2a = contract('eb,ijae->ijab', Feb, lam2)
        term2 = term2a - term2a.swapaxes(2,3)
        Fjm = F[o, o].copy()
        term3a = -contract('jm,imab->ijab', Fjm, lam2)
        term3 = term3a - term3a.swapaxes(0,1)


        Wijmb = self.LWmina(t1)
        term7a = -contract('ijmb,ma->ijab', Wijmb, lam1)
        term7 = term7a - term7a.swapaxes(2,3)

        Wejab = self.LWfiea(t1)
        term6a = contract('ejab,ie->ijab', Wejab, lam1)
        term6 = term6a - term6a.swapaxes(0,1)

        Fjb = self.Fme(t1, t2, F)
        term89a = contract('jb,ia->ijab', Fjb, lam1)
        term89 = term89a
        term89 = term89 - term89a.swapaxes(2,3)
        term89 = term89 - term89a.swapaxes(0,1)
        term89 = term89 + term89a.swapaxes(0,1).swapaxes(2,3)
        total = term1 + term2 + term3 +term7 + term6 + term89
        return total
    def L2eq_rhs_CC2_T(self, t1, t2, lam1, lam2, F):
    
        v = self.vir
        o = self.occ
        TEI = self.TEI
        
        # RHS = <ij||ab>
        lam2_rhs = TEI[o, o, v, v].copy()
        
        #Check Fae
        #RHS += P(ab) Lijae * Feb
        #Fea = self.LRFea(t1, t2, F)
        #Fea = self.Fae(t1, t2, F)
        Fea = F[v, v].copy()
        lam2_rhs = lam2_rhs + contract('ijab,eb->ijab', lam2, Fea)
        lam2_rhs = lam2_rhs - contract('ijab,eb->ijba', lam2, Fea)
        
        #Check Fmi
        #RHS -= P(ij)*Limab*Fjm
        #Fjm = self.LRFim(t1, t2, F)
        Fjm = F[o, o].copy()
        lam2_rhs = lam2_rhs - contract('imab,jm->ijab', lam2, Fjm)
        lam2_rhs = lam2_rhs + contract('imab,jm->jiab', lam2, Fjm)
        
        
        #Check Wijmb
        #RHS += -P(ab) Lma * Wijmb
        Wmina = self.LWmina(t1)
        lam2_rhs = lam2_rhs - contract('ijmb,ma->ijab', Wmina, lam1)
        lam2_rhs = lam2_rhs + contract('ijmb,ma->ijba', Wmina, lam1)
        
        #Check Wejab
        #RHS += P(ij) Lie[R] * Wejab[R] Weifa = self.LWfiea(t1)
        Weifa = self.LWfiea(t1)
        lam2_rhs = lam2_rhs + contract('ejab,ie->ijab', Weifa, lam1)
        lam2_rhs = lam2_rhs - contract('ejab,ie->jiab', Weifa, lam1)

        
        #Check L_ij^ab <-- P(ij) P(ab) L_i^a Fjb
        #where Fjb = fjb + t_n^f <jn||bf>
        #Fjb = Fa[o, v].copy() + contract('nf,jnbf->jb', t1, TEI[o, o, v, v ])
        Fme = self.Fme(t1, t2, F)
        lam2_rhs = lam2_rhs + contract('ia,jb->ijab', lam1, Fme)
        lam2_rhs = lam2_rhs - contract('ia,jb->ijba', lam1, Fme)
        lam2_rhs = lam2_rhs - contract('ia,jb->jiab', lam1, Fme)
        lam2_rhs = lam2_rhs + contract('ia,jb->jiba', lam1, Fme)
    
        return lam2_rhs


    #Routine for DIIS solver, builds all arrays(maxsize) before B is computed    
    def DIIS_solver_CC2(self, t1, t2, F, maxsize, maxiter, E_min):
            #Store the maxsize number of t1 and t2
            T1rhs = self.T1eq_rhs(t1, t2, np.longdouble(F))
            T2rhs = self.T2eq_rhs_CC2(t1, t2, np.longdouble(F))
            t1 = np.longdouble(self.corrected_T1(t1, T1rhs, F))
            t2 = np.longdouble(self.corrected_T2(t2, T2rhs, F))
            t1stored = [t1.copy()]
            t2stored = [t2.copy()]
            errort1 = []
            errort2 = []
            
            for n in range(1, maxsize+1):  
                T1rhs = self.T1eq_rhs(t1, t2, np.longdouble(F))
                T2rhs = self.T2eq_rhs_CC2(t1, t2, np.longdouble(F))
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
                T1rhs = self.T1eq_rhs(t1, t2, np.longdouble(F))
                T2rhs = self.T2eq_rhs_CC2(t1, t2, np.longdouble(F))
                t1 = np.longdouble(self.corrected_T1(t1, T1rhs, F))
                t2 = np.longdouble(self.corrected_T2(t2, T2rhs, F))
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


    def LRWefam_cc2(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        Fme = self.Fme(t1, t2, F)

        term1 = 0.5*TEI[v, v, v, v].copy()
        term2 = -contract('emab,mf->efab', TEI[v, o, v, v], t1)
        tau = 0.5*contract('ia,jb->ijab', t1, t1)
        term3 = contract('nmab,nmef->efab', TEI[o, o, v, v], tau)
        Wabef = (term1 + term2 + term3  )

        term1 = 0.5*TEI[v, v, v, o].copy()
        term3 = contract('efab,mb->efam', Wabef, t1)
        term4a = -TEI[o, v, v, o].copy()
        term4 = contract('jfam,je->efam', term4a, t1)
        tau = 0.5*contract('ia,jb->ijab', t1, t1)
        term5 = contract('jnam,jnef->efam', TEI[o, o, v, o], tau)

        total = term1 + ( term3 + term4 + term5)
        return total
    

    def LRWibjm_cc2(self, t1, t2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        Fme = self.Fme(t1, t2, F)

        term1 = 0.5*TEI[o, o, o, o].copy()
        term2 = contract('ijme,ne->ijmn', TEI[o, o, o, v], t1)
        tau = 0.5*contract('ia,jb->ijab', t1, t1)
        term3 = contract('ijfe,mnfe->ijmn', TEI[o, o, v, v], tau)
        Wmnij = (term1 + term2 + term3)

        
        term1 = -0.5*TEI[o, v, o, o].copy()
        term3 = contract('injm,nb->ibjm', Wmnij, t1)
        term4a = -TEI[o, v, v, o].copy()
        term4 = contract('ibem,je->ibjm', term4a, t1)
        tau =  0.5*contract('ia,jb->ijab', t1, t1)
        term5 = -contract('ibef,jmef->ibjm', TEI[o, v, v, v], tau)
        total = term1 + (term3 + term4 + term5 )
        return total


    def L1_eq_rhs_cc2(self, t1, t2, lam1, lam2, F):
        v = self.vir
        o = self.occ
        TEI = self.TEI
        Fia = self.Fme(t1, t2, F)
        Fea = self.LRFea(t1, t2, F)
        Fim = self.LRFim(t1, t2, F)
        Wieam = self.LSWieam(t1, t2, F)
        Wefam = self.LRWefam_cc2(t1, t2, F)
        Wibjm= self.LRWibjm_cc2(t1, t2, F)
                    
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
        total = (term1 + (term2 + term3 + term4) + term5 + term6)
        return total

    def DIIS_solver_Lam_CC2(self, t1, t2, lam1, lam2, F, maxsize, maxiter, E_min):
            #lam1rhs = self.L1eq_rhs_CC2(t1, t2, lam1, lam2, F)
            #lam1rhs = self.L1_eq_rhs_cc2(t1, t2, lam1, lam2, F)
            #lam2rhs = self.L2eq_rhs_CC2(t1, t2, lam1, lam2, F)
            #lam2rhs = self.L2eq_rhs_CC2_T(t1, t2, lam1, lam2, F)
            #Store the maxsize number of t1 and t2
            lam1rhs = self.L1eq_rhs_CC2(t1, t2, lam1, lam2, np.longdouble(F))
            lam2rhs = self.L2eq_rhs_CC2(t1, t2, lam1, lam2 ,np.longdouble(F))
            lam1 = np.longdouble(self.corrected_lam1(lam1, lam1rhs, F))
            lam2 = np.longdouble(self.corrected_lam2(lam2, lam2rhs, F))
            lam1stored = [lam1.copy()]
            lam2stored = [lam2.copy()]
            errort1 = []
            errort2 = []

            for n in range(1, maxsize+1):
                lam1rhs = self.L1eq_rhs_CC2(t1, t2, lam1, lam2, np.longdouble(F))
                lam2rhs = self.L2eq_rhs_CC2(t1, t2, lam1, lam2 ,np.longdouble(F))
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
                lam1rhs = self.L1eq_rhs_CC2(t1, t2, lam1, lam2, np.longdouble(F))
                lam2rhs = self.L2eq_rhs_CC2(t1, t2, lam1, lam2 ,np.longdouble(F))
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


########################################################################
#
#
#          CC2-Runge_Kutta
#
#
#
########################################################################3
    #T2 Runge-Kutta function 
    def ft2_CC2(self, t, dt, t1, t2, F, Vt):
        k1 = self.T2eq_rhs_CC2(t1, t2, F + Vt(t))
        k2 = self.T2eq_rhs_CC2(t1, t2 + dt/2.0*k1, F + Vt(t + dt/2.0))  
        k3 = self.T2eq_rhs_CC2(t1, t2 + dt/2.0*k2, F + Vt(t + dt/2.0)) 
        k4 = self.T2eq_rhs_CC2(t1, t2 + dt*k3,  F + Vt(t + dt))
        return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)
    
    #L1 Runge-Kutta function 
    def fL1_CC2(self, t, dt, t1, t2, lam1, lam2, F, Vt):
        k1 = self.L1eq_rhs_CC2(t1, t2, lam1, lam2, F + Vt(t))
        k2 = self.L1eq_rhs_CC2(t1, t2, lam1 + dt/2.0*k1, lam2, F + Vt(t + dt/2.0))
        k3 = self.L1eq_rhs_CC2(t1, t2, lam1 + dt/2.0*k2, lam2, F + Vt(t + dt/2.0))
        k4 = self.L1eq_rhs_CC2(t1, t2, lam1 + dt*k3, lam2, F + Vt(t + dt))
        return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)  
           
   #L2 Runge-Kutta function  
    def fL2_CC2(self, t, dt, t1, t2, lam1, lam2, F, Vt):
        k1 = self.L2eq_rhs_CC2(t1, t2, lam1, lam2, F + Vt(t))
        k2 = self.L2eq_rhs_CC2(t1, t2, lam1, lam2 + dt/2.0*k1, F + Vt(t + dt/2.0))
        k3 = self.L2eq_rhs_CC2(t1, t2, lam1, lam2 + dt/2.0*k2, F + Vt(t + dt/2.0))
        k4 = self.L2eq_rhs_CC2(t1, t2, lam1, lam2 + dt*k3, F + Vt(t + dt))
        return dt/6.0*(k1 + 2.0*k2 + 2.0*k3 + k4)      


    ####Time propagator#############
    def Runge_Kutta_solver_CC2(self, F, t1, t2, lam1, lam2, w0, A, t0, tf, dt, timeout, precs, restart=None):
        t=t0
        data =  pd.DataFrame( columns = ('time', 'mu_real', 'mu_imag')) 
        timing =  pd.DataFrame( columns = ('total','t1', 't2', 'l1','l2')) 
        
        #Electric field, it is in the z-direction for now
        def Vt(t):
            mu = self.Defd_dipole()
            pi = np.cos(-1)
            return -A*mu[2]*cmath.exp(1j*w0*2*np.pi*t)
        i=0
        start = time.time()
        #self.check_T1_T2_L1_L2(t1, t2, lam1, lam2, F)
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
            dt2 = -1j*self.ft2_CC2(t, dt, t1, t2, F, Vt) #Runge-Kutta
            itertime_t2 = -itertime + time.time()
            itertime = time.time()
            dL1 = 1j*self.fL1_CC2(t, dt, t1, t2, lam1, lam2, F, Vt) #Runge-Kutta
            itertime_l1 = -itertime  + time.time()
            itertime = time.time()
            dL2 = 1j*self.fL2_CC2(t, dt, t1, t2, lam1, lam2, F, Vt)  #Runge-Kutta
            itertime_l2 = -itertime  + time.time()
            total = itertime_t1 + itertime_t2 + itertime_l1 + itertime_l2
            timing.loc[i] = [total, itertime_t1, itertime_t2, itertime_l1, itertime_l2 ]

            t1 = t1min + dt1
            t2 = (t2min + dt2)
            lam1 = (L1min + dL1)
            lam2 = (L2min + dL2)
            stop = time.time()-start
            
            mua = self.dipole_moment(t1, t2, lam1, lam2, F)
            data.loc[i] = [round(t, 5), mua[2].real, mua[2].imag  ]
            print(round(t, 5), '\t', mua[2].real, '\t', mua[2].imag)
            
            if abs(stop)>0.9*timeout*60.0:
                self.Save_data(F, t1min, t2min, L1min, L2min, data, timing, precs, restart)
                self.Save_parameters(w0, A, t0, t-dt, dt, precs, t1.shape[0], t1.shape[1], 'CC2')
                break
            #Calculate the dipole moment using the density matrix
            if abs(mua[2].real) > 100:
                self.Save_data(F, t1min, t2min, L1min, L2min, data, timing, precs, restart)
                self.Save_parameters(w0, A, t0, t-dt, dt, precs, t1.shape[0], t1.shape[1], 'CC2')
                break
            
        stop = time.time()
        print("total time non-adapative step:", stop-start)
        print("total steps:", i)
        print("step-time:", (stop-start)/i)



