import sys
sys.path.insert(1,'./..')
import psi4 as psi4
from CC_Calculator import *

timeout = float(sys.argv[1])/60
print("time in minutes is:", timeout)
numpy_memory = 2
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")


#mol = psi4.geometry("""
#    N           -1.527107413251     0.745960643462     0.766603000356
#    C           -0.075844098953     0.811790225041     0.711418672248
#    C            0.503195220163    -0.247849447550    -0.215671574613
#    O           -0.351261319421    -0.748978309671    -1.089590304723
#    O            1.639498336738    -0.571249748886    -0.174705953194
#    H           -1.207655674855    -0.365913941094    -0.918035522052
#symmetry c1
#""")






opt_dict = {
  "basis": "sto-3g",
  "reference": "RHF",
  "print_MOs" : "True",
  "mp2_type": "conv",
  "scf_type": "pk",
  "roots_per_irrep": [40],
  "e_convergence": 1e-14,
  "r_convergence": 1e-14
}
psi4.set_options(opt_dict)
psi4.properties('ccsd', properties=['dipole','analyze'])
#psi4.properties('cc2', properties=['dipole','analyze'])
#print("The dipole moment is[a.u.]: ", psi4.get_variable('CC DIPOLE Z')*0.393456)
dip_z = psi4.get_variable('CC DIPOLE Z')

fac = 0.393456#The conversion factor from dybe to a.u.
z_nuclear_dipole = 1.1273
z_HF_dipole = -0.52376994809
z_HF_dipole = -0.0795
dip_z = dip_z*fac -z_nuclear_dipole-z_HF_dipole

print("The ccdipole moment is: ", round(dip_z,4) )


#Start parameters
#w0 frequency of the oscillation
#A = 0.005#the amplitude of the electric field
#t0 = 0.0000 #the start time
#dt = 0.0001 #time step
#precs = 15 #precision of the t1, t2, l1, l2 amplitudes

mol = CC_Calculator(psi4, w0=0.968635,A=0.005,t0=0.0,dt=0.0001,precs=15)
#Time-dependent CC2 calculation
#mol.TDCC(timeout, 'CC2')
#Time-dependent CCSD calculation
mol.TDCC(timeout, 'CCSD')
