import numpy as np
import ellc #Maxted (2016)
from scipy import optimize
#import lightkurve as lk
import matplotlib.pyplot as plt
import tqdm
import phoebe
import emcee
import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

###################
#impoting only inphase data
#wimdemuth table
per = 26.5082709 #days
incl = 91.65137637074137 #degrees
# lit_t0 = 149.6219577890 #BJD  #21:04:22.85 UT
#esinw = -3.68320755880000e-07
#ecosw = 9.46471960566000e-07	

teff = 5957 

#inphase only data path
#first part of dan hey's code that technically is done above but we're gonna try without


# dataset uploaded to github.com/mrdecesare/phoebe_codes/kic08445775.00.lc.data
inphase_path = '/home/mrdecesare/Desktop/kic8445775/only_in_phase.txt'
data = np.loadtxt(inphase_path)
kic_path = '/home/mrdecesare/Desktop/learning phoebe/kic08445775.00.lc.data'
fulldata = np.loadtxt(kic_path)
phase = data[:,0]
dtr = data[:,1]
dtrerr = data[:,2]

#####
m = np.argsort(phase) 
phase = phase[m]
dtr = dtr[m]

m = ((phase > -0.015) & (phase < 0.015))
m |= ((phase > 0.49) | (phase < -0.49)) 
phase = phase[m]
flux = dtr[m]
fluxerr = dtrerr[m]
time = bjd = fulldata[:, 0]
###############
#load up phoebe basics
logger = phoebe.logger('error')
logger = phoebe.logger(clevel = 'WARNING')
b = phoebe.default_binary(semidetached=False, contact_binary=False)

######################
#cite: dan hey(2024)
#bunyip used for optimization (was struggling with other phoebe options)
class Bunyip:
    def __init__(self, phase, flux, flux_err=None, shape='sphere'):

        # Assign initial values
        self.phase = phase
        self.flux = (flux - np.median(flux)) + 1.
        if flux_err is None:
            flux_err = np.zeros_like(flux)
        self.flux_err = flux_err

        # Initialise fit parameters
        self.parameters = {
            "q": 1.0,
            "rsum": 0.1,
            "rratio": 1.,
            "fc": 0,
            "fs": 0,
            "sbratio": 1.0,
            "incl": 90.,
            "t0": 0.0,
            "mean": 0,
            "log_f": np.std(self.flux),
        }

        # Parameters specific to ellc
        self.ellc_parameters = {"shape_1": shape, "shape_2": shape}

        # Some wrappers for the optimizers. don't @ me
        self.nll = lambda *args: -self.lnlike(*args)
        self.nll_wrapper = lambda *args: -self.lnlike_wrapper(*args)

    def lc_model(self):
        rsum = self.parameters["rsum"]
        rratio = self.parameters["rratio"]
        r1, r2 = rsum / (1.0 + rratio), rsum * rratio / (1.0 + rratio)

        lc = ellc.lc(
            self.phase,
            t_zero=self.parameters["t0"],
            q=self.parameters["q"],
            radius_1=r1,
            radius_2=r2,
            incl=self.parameters["incl"],
            sbratio=self.parameters["sbratio"],
            f_c=self.parameters["fc"],
            f_s=self.parameters["fs"],
            shape_1=self.ellc_parameters["shape_1"],
            shape_2=self.ellc_parameters["shape_1"],
        )

        return lc + self.parameters["mean"]

    def update_parameters(self, prediction):
        """Update parameters from a prediction given by either the neural
        network, or the KNN classifier

        Parameters
        ----------
        prediction : list
            List of prediction values, should be in the form
            q, r1, r2, tratio, incl, ecc, per0
        """
        old_params = self.parameters.copy()
        q, r1, r2, tratio, incl, ecc, per0 = prediction
        self.parameters.update(
            {
                "q": q,
                "rsum": r1 + r2,
                "rratio": r2 / r1,
                "fc": np.sqrt(ecc) * np.cos(np.radians(per0)),
                "fs": np.sqrt(ecc) * np.sin(np.radians(per0)),
                "sbratio": tratio,
                "incl": incl,
            }
        )

        if not np.isfinite(self.lnprior()):
            print(self.parameters)
            self.parameters = old_params
            print(
                "The network failed to find a solution, defaulting to original values. Or grid search here?"
            )

    def lnprior(self):
        """log prior of the ellc model

        Returns
        -------
        float
            The prior probability
        """
        rsum = self.parameters["rsum"]
        rratio = self.parameters["rratio"]
        r1, r2 = rsum / (1.0 + rratio), rsum * rratio / (1.0 + rratio)

        ecc = self.parameters["fc"] ** 2 + self.parameters["fs"] ** 2
        per0 = np.arctan2(self.parameters["fs"], self.parameters["fc"])
        if (
            (0 < self.parameters["q"])
            & (0 < r1 < 0.5)
            & (0 < r2 < 0.5)
            # & (self.parameters["incl"] <= 90)
            & (0 < self.parameters["sbratio"])
            & ((-0.5) <= self.parameters["t0"] <= (0.5))
            & (0 < r1 < 1)
            & (0 < r2 < 1)
            & (0 <= ecc < 1.0)
        ):
            return 0.0
        else:
            return -np.inf

    def lnlike_wrapper(self, params, *vars):
        return self.lnlike(params, vars)

    def lnlike(self, params, vars):
        """The ln likelihood of the model.

        Parameters
        ----------
        params : list
            List of parameter values corresponding to the var names
        vars : list
            List of parameter names corresponding to the params

        Returns
        -------
        float
            Value of the ln likelihood at the given values
        """
        for param, var in zip(params, vars):
            self.parameters.update({var: param})

        lp = self.lnprior()
        if not np.isfinite(lp):
            return -np.inf
        model_flux = self.lc_model()
        # sigma2 = self.flux_err**2 + model_flux**2 * np.exp(2 * self.parameters["log_f"])
        sigma2 = np.exp(self.parameters['log_f'])
        try:
            ln_lc = -0.5 * np.sum(
                (self.flux - model_flux) ** 2 / sigma2 + np.log(sigma2)
            )
        except:
            return -np.inf
        if np.any(np.isnan(ln_lc)):
            return -np.inf
        return ln_lc

    def optimize(self, vars=None, **kwargs):
        """Optimises the `parameters` of the Bunyip object with Scipy

        Parameters
        ----------
        vars : list, optional
            List of vars to optimize, by default None

        Returns
        -------
        dict
            Results of the optimization
        """
        if vars is None:
            vars = list(self.parameters.keys())

        x0 = [self.parameters[var] for var in vars]
        soln = optimize.minimize(self.nll, x0, args=(vars), **kwargs)
        for var, val in zip(vars, soln.x):
            self.parameters.update({var: val})
        return soln

    def plot_model(self, ax=None, **kwargs):
        """Plot the current model in the parameters dict

        Parameters
        ----------
        ax : matplotlib axis, optional
            axis object on which to plot, by default None

        Returns
        -------
        matplotlib axis
            axis
        """
        model_flux = self.lc_model()

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.phase, model_flux, label="Model", zorder=100)
        ax.plot(self.phase, self.flux, ".k", label="Data")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Flux")
        plt.legend()
        return ax

    def optimize_best(self, **kwargs):
        optimization_path = [
            ["t0"],
            ["mean", "log_f"],
            ["t0", "mean", "log_f", "fc", "fs"],
            ["t0", "mean", "log_f", "fc", "fs", "rsum", "rratio", "incl", "mean"],
            None,  # All parameters
        ]

        for path in tqdm.tqdm(optimization_path):
            soln = self.optimize(vars=path, method="Nelder-Mead", **kwargs)
        return soln
    
    def run_emcee(self, burnin=1000, draws=2000, nwalkers=32, **kwargs):
        """Run emcee initialised around the current parameters

        Parameters
        ----------
        burnin : int, optional
            Number of burn-in values, by default 1000
        draws : int, optional
            Number of draws, by default 2000
        nwalkers : int, optional
            Number of walkers in the chain, by default 32

        Returns
        -------
        [type]
            [description]
        """
        import emcee

        vars = list(self.parameters.keys())
        init = list(self.parameters.values())
        pos = init + 1e-4 * np.random.randn(nwalkers, len(init))
        ndim = len(vars)

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            self.lnlike_wrapper,
            args=(vars),
            moves=[
                (emcee.moves.DEMove(), 0.8),
                (emcee.moves.DESnookerMove(), 0.2),
            ],
        )
        sampler.run_mcmc(pos, burnin + draws, progress=True)
        samples = sampler.get_chain(flat=True, discard=burnin)
        self.sampler = sampler
        return samples

    def corner_plot(self, trace, **kwargs):

        import corner

        ax = corner.corner(
            trace, labels=list(self.parameters.keys()), show_titles=True, **kwargs
        )
        return ax

    def plot_samples_from_trace(self, trace, n=100, ax=None, **kwargs):
        vars = list(self.parameters.keys())
        samples = trace[np.random.choice(len(trace), size=n)]

        if ax is None:
            fig, ax = plt.subplots()
        for i in samples:
            [self.parameters.update({key: val}) for key, val in zip(vars, i)]
            model_flux = self.lc_model()
            plt.plot(self.phase, model_flux, **kwargs)
        return ax

    def get_model_from_trace(self, trace, n=100):
        vars = list(self.parameters.keys())
        samples = trace[np.random.choice(len(trace), size=n)]
        lcs = []
        for i in samples:
            [self.parameters.update({key: val}) for key, val in zip(vars, i)]
            model_flux = self.lc_model()
            lcs.append(model_flux)
        return lcs

###################


bn = Bunyip(phase[m], flux[m], shape='roche') # Initialize the model

bn.parameters.update({
    'rsum': 0.03525, #edited to it fully fits the data
    'incl': 91.65137637074137,
    'period':per})

_ = bn.optimize_best(options={'adaptive': True}) #makes it longer to run

#bn.plot_model()
  
def bn_b(bn):
    b.set_value('q',bn['q']),
    b.set_value('incl@binary@orbit@component', bn['incl'])
    b.set_value('requiv@primary',value = (bn['rsum'])/(1 + bn['rratio'])),
    b.set_value('requiv@secondary', value = (bn['rsum']*bn['rratio'])/(1 + bn['rratio'])),
    #b.set_value('sbratio@binary@component', bn['sbratio']),
    b.set_value('ecc@binary@component', value = np.sqrt(bn['fc'] ** 2 + bn['fs']** 2)),
    b.set_value('t0', bn['t0']), #need to calc t0_superconj
    return b
#commented out params give value error where i dont know/remember how to fix it
#or dont have a 1:1 between b and bn


b.add_dataset('lc', 
              times = phase,
              compute_phases = time,
              fluxes = flux, 
              sigmas = fluxerr,
              passband = 'Kepler:mean')
              #teffratio = teff)
bn_b(bn.parameters)


pre = b.get_parameter('t0_supconj', context = 'component')
print('pre ellc: t0_supcoj = ', pre)

#skipping ld coeffs
#sb['teff@primary'] = teff
b.set_value_all('ld_func', 'linear')
b.set_value_all('ld_mode', 'manual')
b.set_value_all('ld_mode_bol', 'manual')
b.set_value_all('atm', 'ck2004')
#b.set_value_all('ld_coeffs', [0.]) #should just produce warning but should be able to skip ld_coeffs
#b.set_value_all('ld_coeffs_bol', [0.])

b.set_value('pblum_mode', 'dataset-scaled') #covery to component coupled and marginalizing 

b.add_compute('ellc', compute = 'fastcompute') #has emcee built into it
b.run_compute(kind = 'ellc', compute = 'fastcompute') 

post = b.get_parameter('t0_supconj', context = 'component')
print('post ellc: t0_supconj = ', post)
#b.plot(show = True)

