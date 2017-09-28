# -*- coding: utf-8 -*-
# #####################################################################
#
#
#                            Created by: Rachel Glenn
#                                 Date: 12/14/2016
#This is the driver for CC2 or CCSD_Helper and Runge_Kutta to calculate
#the time dependent dipole moment.
#
# First it calculates the converged t1, t2,
# Second the lam1, lam2
# It uses the converged t1, t2, lam1, lam2 to calculate the real-time
# single-electron density matrix
#
########################################################################
import sys
sys.path.insert(0,'./..')
import psi4 as psi4
import numpy as np
from CCSD_Helper import *
from  CC2_Helper import  CC2_Helper
import pandas as pd
sys.path.append('/home/rglenn/newriver/buildpython/pandas')

########################################################
#                 Setup
########################################################
class CC_Calculator(object):
    
    def __init__(self, psi, **kwargs):
        
        self.mol = CCSD_Helper(psi)
        mol = self.mol
        self.ndocc = mol.ndocc
        try:
            #Start parameters
            self.w0 = kwargs['w0'] #frequency of the oscillation
            self.A = kwargs['A'] #the amplitude of the electric field
            self.t0 = kwargs['t0'] #the start time
            self.tf = 10.1 #the stop time, the actual stop time is governed by the timelength of the job
            #Unless it completes enough steps to get to tf first.
            self.dt = kwargs['dt'] #time step
            self.precs = kwargs['precs'] #precision of the t1, t2, l1, l2 amplitudes
        except:
            pass
            
    def test_MP2(self):
        mol = self.mol
        scf, MP2, T2 = mol.MP2_E('Test')
        return MP2

##############################################
#  Time-dependent dipole matrix(CC2/CCSD):
##############################################
    
    def TDCC(self, timeout, CCSD_or_CC2):
        if CCSD_or_CC2 == 'CC2':
            self.TDCC2(timeout)
        elif CCSD_or_CC2 == 'CCSD':
            self.TDCCSD(timeout)
        else:
            print("Error in specifying whether it is a CCSD or CC2 calculation")
            print("Correct format is:\nmol.TDCC(stop_time, CCSD)\nwhere CCSD='CCSD' or 'CC2'")
    
    
##############################################
#  RESTART Time-dependent dipole matrix(CC2/CCSD):
##############################################

    def TDCC_restart(self, timeout):
        param = pd.read_csv('Parameters.csv')
        #Start parameters
        w0 = param['w0'][0]#frequency of the oscillation and transition frequency
        A = param['A'][0]#the amplitude of the electric field    
        t0 = param['t0'][0]  #the start time
        CC2_or_CCSD = param['CCSD_or_CC2'][0]
        tf = 50.0 + t0 #the stop time, the actual stop time is governed by the timelength of the job
                             #Unless it completes enough steps to get to tf first. 
        dt =  param['dt'][0] #time step
        t0 = t0 + 2*dt
        precs = int(param['precs'][0])
        i = int(param['i'][0])
        a = int(param['a'][0])

        def convert_2data(Filename, i, a):
            F = pd.read_csv(Filename,sep="\t",header=None, names=['i', 'a', 'F'])
            #F = np.genfromtxt(("\t".join(i) for i in csv.reader(open(Filename))), delimiter="\t")
            Freshape = np.zeros(shape=(i, a))
            x = np.around(F['i'].tolist())
            y = np.around(F['a'].tolist())
            Fa = F['F'].tolist()
            for n in range(len(Fa)):
                ni = int(x[n])
                na = int(y[n])
                Freshape[ni][na] = Fa[n]
            return Freshape

        def convert_4data(Filename, i, a):
            F = pd.read_csv(Filename,sep="\t",header=None, names=['i', 'j', 'a', 'b', 'F'])
            #F = np.genfromtxt(("\t".join(i) for i in csv.reader(open(Filename))), delimiter="\t")
            Freshape = np.zeros(shape=(i, i, a, a))
            x = np.around(F['i'].tolist())
            y = np.around(F['j'].tolist())
            t = np.around(F['a'].tolist())
            z = np.around(F['b'].tolist())
            Fa = F['F'].tolist()
            for n in range(len(Fa)):
                ni = int(x[n])
                nj = int(y[n])
                na = int(t[n])
                nb = int(z[n])
                Freshape[ni][nj][na][nb] = Fa[n]
            return Freshape

        #######The data for t1 is "i", "a", "t1-flattend"############################
        t1_real = convert_2data("t1_real.dat", i, a)
        t1_imag = convert_2data("t1_imag.dat", i, a)
        t1 = t1_real + 1.0*1j*t1_imag

        lam1_real = convert_2data("lam1_real.dat", i, a)
        lam1_imag = convert_2data("lam1_imag.dat", i, a)
        lam1 = lam1_real + 1.0*1j*lam1_imag

        t2_real = convert_4data("t2_real.dat", i, a)
        t2_imag = convert_4data("t2_imag.dat", i, a)
        t2 = t2_real + 1.0*1j*t2_imag

        lam2_real = convert_4data("lam2_real.dat", i, a)
        lam2_imag = convert_4data("lam2_imag.dat", i, a)
        lam2 = lam2_real + 1.0*1j*lam2_imag


        F_real = convert_2data("F_real.dat", i+a, i+a )
        F_imag = convert_2data("F_imag.dat", i+a, i+a)
        F = F_real #+ 1.0*1j*F_imag
        mol = CCSD_Helper(psi4)
    
        if CC2_or_CCSD == 'CC2':
            mol_CC2 = CC2_Helper(psi4)
            mol_CC2.Runge_Kutta_solver_CC2(F, t1, t2, lam1, lam2, w0, A, t0, tf, dt, timeout, precs, 'restart')

    
        elif CC2_or_CCSD == 'CCSD':
            mol = CCSD_Helper(psi4)
            mol.Runge_Kutta_solver(F, t1, t2, lam1, lam2, w0, A, t0, tf, dt, timeout, precs, 'restart')
    




##############################################
#              CCSD--Calculations--
##############################################

    def TDCCSD(self, timeout):#T1 equation
        mol = CCSD_Helper(psi4)
        nmo = mol.nmo
        ndocc = mol.ndocc
        F =  mol.F_MO()
        v = 2*(nmo-ndocc)
        o = 2*ndocc
        psienergy = psi4.energy('CCSD')
        
##############################################
#           t1 and t2 Amplitudes (CCSD):
##############################################

        #initialize t1 and t2
        scf, MP2, t2 = mol.MP2_E('Test')
        t1 = np.zeros( shape=(o, v), dtype=np.longdouble) 
        print("Escf=", scf)
        print("Emp2=", MP2-scf)
        print("Etot=", MP2)
        
        maxsize = 7 # number of t1 and t2 to store
        maxiter = 40 #max iterations incase it crashes
        E_min = 1e-15 # minimum energy to match
        
        CC2_E, t1, t2 = mol.DIIS_solver(t1, t2, F, maxsize, maxiter, E_min)
        print("E_ccsd_plugin=", CC2_E + scf)
        print("difference between psi4 and plugin=", psienergy.real - (CC2_E + scf))
        mol.print_T_amp(t1, t2)
        
        psi4.driver.p4util.compare_values(psi4.energy('CCSD'), CC2_E+scf, 10, 'CCSD Energy')
        
##############################################
#           lam1 and lam2 Amplitudes (CCSD):
##############################################

        maxiter = 30
        E_min = 1e-15 # minimum energy to match
        lam1 = t1
        lam2 = t2
        pseudo_E, lam1, lam2 = mol.DIIS_solver_Lam(t1, t2, lam1, lam2, F, maxsize, maxiter, E_min)
        
        ###Print out the L1 and L2 amplitudes and Pseudo energy
        print("E_pseudo_plugin=", pseudo_E)
        mol.print_L_amp(lam1, lam2)

##############################################
#           Time-dependent dipole matrix(CCSD):
##############################################
        mol.Runge_Kutta_solver(F, t1, t2, lam1, lam2, self.w0, \
        self.A, self.t0, self.tf, self.dt, timeout, self.precs)















##############################################
#              CC2--Calculations--
##############################################
    def TDCC2(self, timeout):#T1 equation
        mol_CC2 = CC2_Helper(psi4)
        nmo = mol_CC2.nmo
        ndocc = mol_CC2.ndocc
        F =  mol_CC2.F_MO()
        v = 2*(nmo-ndocc)
        o = 2*ndocc
        psienergy = psi4.energy('CC2')

##############################################
#           t1 and t2 Amplitudes (CC2):
##############################################
        #initialize t1 and t2
        scf, MP2, t2 = mol_CC2.MP2_E('Test')
        t1 = np.zeros( shape=(o, v), dtype=np.longdouble) 
        print("Escf=", scf)
        print("Emp2=", MP2-scf)
        print("Etot=", MP2)
        
        maxsize = 7 # number of t1 and t2 to store
        maxiter = 40 #max iterations incase it crashes
        E_min = 1e-15 # minimum energy to match


        CC2_E, t1, t2 = mol_CC2.DIIS_solver_CC2(t1, t2, F, maxsize, maxiter, E_min)
        print("E_cc2_plugin=", CC2_E + scf)
        print("difference between psi4 and plugin=", psienergy.real - (CC2_E + scf))
        mol_CC2.print_T_amp(t1, t2)
        psi4.driver.p4util.compare_values(psi4.energy('CC2'), CC2_E+scf, 10, 'CCSD Energy')
##############################################
#           lam1 and lam2 Amplitudes (CC2):
##############################################
        maxiter = 30
        E_min = 1e-15 # minimum energy to match
        lam1 = t1
        lam2 = t2
        pseudo_E, lam1, lam2 = mol_CC2.DIIS_solver_Lam_CC2(t1, t2, lam1, lam2, F, maxsize, maxiter, E_min)
        print("E_pseudo_plugin=", pseudo_E)
        mol_CC2.print_L_amp(lam1, lam2)

##############################################
#           Time-dependent dipole matrix(CC2):
##############################################
        mol_CC2.Runge_Kutta_solver_CC2(F, t1, t2, lam1, lam2, \
        self.w0, self.A, self.t0, self.tf, self.dt,timeout, self.precs)
















