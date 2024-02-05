import phoebe
import numpy as np
import matplotlib.pyplot as plt

# Load your multiple light curve datasets
kic_path = '/home/mrdecesare/Desktop/learning phoebe/kic08445775.00.lc.data'
data = np.loadtxt(kic_path)
bjd = data[:, 0]
phase = data[:,1]
dtr = data[:, 6]
dtrerr = data[:,7]

logger = phoebe.logger('WARNING')
b = phoebe.default_binary(semidetached=False, contact_binary=False)

per = 26.5082709 #days
incln = 91.65137637074137 #degrees
pit = 1.65137637074137

per1 = data[0:7000]
bjd1 = bjd[0:7000]
phase1 = phase[0:7000]
dtr1 = dtr[0:7000]
dtrerr1 = dtrerr[0:7000]

#b.set_value('period', component='binary', value=per)


b.add_dataset('lc', 
              times = phase1,
              compute_phases = bjd1,
              fluxes = dtr1, 
              sigmas = dtrerr1)

b.set_value('pblum_mode', 'dataset-scaled')
b.set_value('incl', component = 'binary', value = incln)

b.run_compute(ntriangles = 1000, ltte = False)
b.plot(x= 'phases', y = 'fluxes',show = True)



###
# mcmc and solvers get the model to line up with the data
# solver first them mcmc
# find the best fit model then mcmc around it
## hope feed answer 
