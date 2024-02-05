#console 1

import phoebe
import numpy as np
import matplotlib.pyplot as plt

kic_path = '/home/mrdecesare/Desktop/learning phoebe/kic08445775.00.lc.data'
data = np.loadtxt(kic_path)
bjd = data[:, 0]
phase = data[:,1]
dtr = data[:, 6]
dtrerr = data[:,7]

logger = phoebe.logger()
b = phoebe.default_binary(semidetached=False, contact_binary=False)

per = 26.5082709 #days
incln = 91.65137637074137 #degrees
#ftPE = 58482.6219577890
tPEbjd = 149.6219577890 #BJD  #21:04:22.85 UT
#tPE = 21.0713833 #21:04:22.85
#tpPE = 0.8780424 #tPE/24 hrs 0.87.... of a day
esinw = -3.68320755880000e-07
ecosw = 9.46471960566000e-07	


per1 = data[0:7000]
bjd1 = bjd[0:7000]
phase1 = phase[0:7000]
dtr1 = dtr[0:7000]
dtrerr1 = dtrerr[0:7000]

b.add_dataset('lc', 
              times = phase1,
              compute_phases = bjd1,
              fluxes = dtr1, 
              sigmas = dtrerr1,
              dataset = 'lc01')

b.set_value('pblum_mode', 'dataset-scaled')
b.set_value('incl', component = 'binary', value = incln)
b.set_value('t0_supconj', orbit = 'binary', value = tPEbjd)


def solver(): 
    b.add_solver('estimator.lc_geometry', solver = 'nm_optim')
    b.run_solver(solution = 'w_solver')
    b.add_constraint('esinw', orbit = 'binary', value = esinw, solver = 'nm_optim')
    b.add_constraint('ecosw', orbit = 'binary',value = ecosw, solver = 'nm_optim')
    return b.adopt_solution('w_solver')


b.run_compute(ntriangles = 1000, ltte = False)
b.plot(x= 'phases', y = 'fluxes', legend = True, show = True)



###
# mcmc and solvers get the model to line up with the data
# solver first them mcmc
# find the best fit model then mcmc around it
## hope feed answer 
