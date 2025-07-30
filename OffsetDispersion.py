import os
import copy as copycp

import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
from matplotlib import pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table
import astropy.units as u
import astropy.constants as cons
import mpfit_single as mpfit

MAGTYPE_GALEXFUV = 0
MAGTYPE_GALEXNUV = 1
MAGTYPE_AB = 2

# Intrinsic dispersion priors
mn_off, mx_off = -0.3, +0.3
mn_sig, mx_sig = 0, +0.1


def log_likelihood(theta, y, yerr):
    off, sig = theta
    sigma2 = yerr ** 2 + sig**2
#     return -0.5 * np.sum(((y - off) ** 2 / sigma2))
    return -0.5 * np.sum(((y - off) ** 2 / sigma2) + np.log(2*np.pi*sigma2))


def log_prior(theta):
    off, sig = theta
    if mn_off < off < mx_off and mn_sig < sig < mx_sig:
        return 0.0
    return -np.inf


def log_probability(theta, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y, yerr)


def myfunct_intdisp(p, fjac=None, x=None, y=None, yerr=None):
    model = p[0]
    sigma2 = yerr**2 + p[1]**2
    # Non-negative status value means MPFIT should
    # continue, negative means stop the calculation.
    chi2 = (y-model)**2/sigma2 + np.log(sigma2)
    status = 0
    return [status, np.sqrt(chi2)]


def get_offset_dispersion_chisq(filt_diffs, filt_diffe, refdirc):
    this_magoffs = np.zeros((nfilts))
    this_magdisp = np.zeros((nfilts))
    use_mpfit = False
    for ii in range(nfilts):
        this_offs, this_disp = filt_diffs[:,ii], filt_diffe[:,ii]
        wgd = np.where(this_disp != 0.0)
        # Set some reasonable starting conditions
        p0=[0.0, 0.005]
        if use_mpfit:
            param_base={'value':0., 'fixed':0, 'limited':[0,0], 'limits':[0.,0.]}
            param_info=[]
            for jj in range(len(p0)):
                param_info.append(copycp.deepcopy(param_base))
                param_info[jj]['value']=p0[jj]
            param_info[1]['limited'] = [1,0]
            param_info[1]['limits']  = [0.0,0.0]

            # Now tell the fitting program what we called our variables
            y, yerr = this_offs[wgd], this_disp[wgd]
            fa = {'x':np.zeros(y.size), 'y':y, 'yerr':yerr}

            # PERFORM THE FIT AND PRINT RESULTS
            m = mpfit.mpfit(myfunct_intdisp, p0, parinfo=param_info,functkw=fa,quiet=True)
            if (m.status <= 0): 
                print('error message = ', m.errmsg)
            print("param: ", m.params)
            print("error: ", m.perror)
            if np.isnan(m.fnorm) or np.isinf(m.fnorm) or (m.perror[1] == 0.0) or (m.params[1]/m.perror[1] <= 2.0):
                this_magoffs[ii] = np.average(this_offs[wgd], weights=1.0/(this_disp[wgd])**2)
                this_magdisp[ii] = 0.0
            else:
                this_magoffs[ii] = m.params[0]
                this_magdisp[ii] = m.params[1]
        else:
            nll = lambda *args: -log_likelihood(*args)
            initial = np.array(p0)
            bnds = ((None, None), (0.0, None))
            # Now tell the fitting program what we called our variables
            y, yerr = this_offs[wgd], this_disp[wgd]
            soln = minimize(nll, initial, args=(y, yerr), bounds=bnds)
            this_magoffs[ii], this_magdisp[ii] = soln.x
    return this_magoffs, this_magdisp


def get_offset_dispersion(filt_diffs, filt_diffe, refdirc):
    nreps, ndim, nwalkers = 10, 2, 100
    this_magoffs = np.zeros((nfilts,nreps))
    this_magdisp = np.zeros((nfilts,nreps))
    for rr in range(nreps):
        print(f"REPITITION {rr+1}/{nreps}")
        for ii in range(nfilts):
            this_offs, this_disp = filt_diffs[:,ii], filt_diffe[:,ii]
            wgd = np.where(this_disp != 0.0)
            # Initialise the MCMC
            pos = [np.array([np.random.uniform(mn_off, mx_off),
                             np.random.uniform(mn_sig, mx_sig)]) for i in range(nwalkers)]
#             print("Initialising sampler")
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(this_offs[wgd], this_disp[wgd]))
#             print("Running MCMC")
            sampler.run_mcmc(pos, 1000, progress=False)
            # Get the chains
            flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)
            np.save(refdirc+"/samples_filter{0:d}.npy".format(ii), flat_samples)
            # Store the median values
            mcmcA = np.percentile(flat_samples[:, 0], [16, 50, 84])
            this_magoffs[ii,rr] = mcmcA[1]
            mcmcB = np.percentile(flat_samples[:, 1], [16, 50, 84])
            this_magdisp[ii,rr] = mcmcB[1]
            tmp_mu = mcmcB[1]
            tmp_std = 0.5*(mcmcB[2]-mcmcB[0])
            if tmp_mu/tmp_std <= 2.0:
                this_magdisp[ii,rr] = 0.0
                this_magoffs[ii,rr] = np.average(this_offs[wgd], weights=1.0/(this_disp[wgd])**2)
            else:
                this_magdisp[ii,rr] = tmp_mu
    # Reject outliers
    final_magoffs = np.median(this_magoffs, axis=1)
    final_magdisp = np.median(this_magdisp, axis=1)
    print("#############################################################################")
    print("#############################################################################")
    print("#############################################################################")
    return final_magoffs, final_magdisp


def get_filter(name):
    # Get the filter name
    tst = name[:4]
    if tst == "phot":
        if name == 'phot_g_mean_ABmag': oput = "Gaia3G.dat"
        elif name == 'phot_bp_mean_ABmag': oput = "Gaia3GBp.dat"
        elif name == 'phot_rp_mean_ABmag': oput = 'Gaia3GRp.dat'
        else:
            print("GAIA FILTER WRONG FORMAT")
            assert False
    elif tst == "DES_":
        if name == 'DES_y': oput = "DES_Y.dat"
        else: oput = name+".dat"
    elif tst == "2MAS":
        if name == '2MASS_j': oput = "TWOMASS_J.dat"
        elif name == '2MASS_h': oput = "TWOMASS_H.dat"
        elif name == '2MASS_k': oput = "TWOMASS_Ks.dat"
        else:
            print("2MASS FILTER WRONG FORMAT")
            assert False
    elif tst == 'GALE':
        if name == 'GALEX_fuv_mag': oput = 'GALEX_FUV.dat'
        elif name == 'GALEX_nuv_mag': oput = 'GALEX_NUV.dat'
        else:
            print("GALEX FILTER WRONG FORMAT")
            assert False
    elif tst == 'WISE':
        if name == 'WISE_W1': oput = 'W1.dat'
        elif name == 'WISE_W2': oput = 'W2.dat'
        else:
            print("WISE FILTER WRONG FORMAT")
            assert False
    else:
        oput = name+".dat"
    # Load it
    wave, tput = np.loadtxt(f"../Fitting/filters/{oput}", unpack=True, usecols=(0,1))
    wcen = np.sum(wave*tput)/np.sum(tput)
    return wave, tput, wcen


def getname(t):
    coo = SkyCoord(ra=t['ra']*u.deg, dec=t['dec']*u.deg, pm_ra_cosdec=t['pmra']*u.mas/u.yr, pm_dec=t['pmdec']*u.mas/u.yr, obstime="J2016.0", equinox="J2016.0")
    cn = coo.apply_space_motion(Time("J2000.0"))
    radecstr = "BB" + cn.to_string('hmsdms').replace('h','').replace('m','').replace('d', '').replace('s', '')
    hms_ra = radecstr.split()[0].split(".")[0]
    dms_dec = radecstr.split()[1].split(".")[0].replace("-","m").replace("+","p")
    this_name = hms_ra + "_" + dms_dec
    return this_name


def load_filters(waves, mask, filts):
    nfilts = np.sum(np.logical_not(mask))
    responses = np.zeros((waves.size, nfilts))
    magtypes = np.zeros(nfilts)
    midwaves = np.zeros(nfilts)
    cntr = 0
    for ff, filt in enumerate(filts):
        if mask[ff]: continue
        if filt == "GALEX_FUV": magtypes[ff] = MAGTYPE_GALEXFUV
        elif filt == "GALEX_NUV": magtypes[ff] = MAGTYPE_GALEXNUV
        else: magtypes[ff] = MAGTYPE_AB
        wave, sens = np.loadtxt('filters/'+filt+'.dat', unpack=True)
        responses[:, cntr] = interpolate.interp1d(wave.copy(), sens.copy(), kind='linear', bounds_error=False, fill_value=0.0)(waves)
        midwaves[ff] = np.sum(wave.copy()*sens.copy())/np.sum(sens.copy())
        cntr += 1
    return midwaves, responses, magtypes


def blackbody_func(a, teff, waves, responses, magtype, GaiaG, fmag=False):
    numfilts = responses.shape[1]
    # Setup the units
    teff *= 1.0E4*u.K
    a *= 1.0E-23
    # Calculate the function
    waveshift = waves*1
    flam = ((a*2*cons.h*cons.c**2)/waveshift**5)/(np.exp((cons.h*cons.c/(waveshift*cons.k_B*teff)).to(u.m/u.m).value)-1.0)
    flam = flam.to(u.erg/u.s/u.cm**2/u.Angstrom).value
    flamref = ((3631.0*u.Jy) * cons.c/(waves**2)).to(u.erg/u.s/u.cm**2/u.Angstrom).value
    # Now calculate the model fluxes
    modelmag = np.zeros(numfilts)
    for ff in range(numfilts):
        if magtype[ff] == MAGTYPE_GALEXFUV:
            engy = (cons.h*cons.c/waves).to(u.erg).value
            cnts_per_sec = np.trapz(flam * responses[:,ff]/engy, x=waves.value)
            modelmag[ff] = -2.5 * np.log10(cnts_per_sec) + 18.82
        elif magtype[ff] == MAGTYPE_GALEXNUV:
            engy = (cons.h*cons.c/waves).to(u.erg).value
            cnts_per_sec = np.trapz(flam * responses[:,ff]/engy, x=waves.value)
            modelmag[ff] = -2.5 * np.log10(cnts_per_sec) + 20.08
        elif magtype[ff] == MAGTYPE_AB:
            modelmag[ff] = -2.5*np.log10(np.trapz(flam * waves * responses[:,ff], x=waves)/np.trapz(flamref * waves * responses[:,ff], x=waves))
        else:
            print("ERROR - NOT EXPECTING THIS KIND OF MAGNITUDE!")
            assert(False)
    # Correct all filters to Gaia G
    diff = 0.0#GaiaG - modelmag[0]
    modelmag += diff
    # Returning flux as well?
    if fmag:
        fmag = -2.5*np.log10( (flam*waves*np.gradient(waves)) / (flamref*waves*np.gradient(waves)) ) + diff
        return modelmag, fmag
    else:
        return modelmag


def myfunct(par_bb, fjac=None, y=None, err=None, id_star=None, id_filt=None, responses=None, waves=None, magtype=None, nfilts=None, idx=None):
    """
    y         :: Measured magnitude
    err       :: Measurement magnitude error
    id_star   :: A list containing the ID number of each star
    id_filt   :: A list containing the ID number of each filt
    responses :: Response functions of each filter
    waves     :: Wavelength of the corresponding responses
    magtype   :: Magnitude type (AB or GALEX)
    """
    # Extract some useful information
    nstars = np.max(id_star)+1
    model = np.zeros(y.size)
    sigma2 = np.zeros(y.size)
    # Loop over all stars
    for ss in range(nstars):
        wresp = np.where(id_star==ss)
        this_filt = id_filt[wresp]
        model[wresp] = blackbody_func(par_bb[2*ss], par_bb[2*ss + 1], waves, responses[:,this_filt], magtype[this_filt], y[0])
        # Calculate the intrinsic dispersion and the total error value
        sigma2[wresp] = err[wresp]**2
    # Non-negative status value means MPFIT should
    # continue, negative means stop the calculation.
    chi2 = np.abs(((y-model)**2)/sigma2)
    status = 0
    return [status, np.sqrt(chi2)]


def LoadData(filttab):
    """
    Load the data from the filter table, and merge all photometry into a single table that is used for the fitting
    """
    # Perform some checks on the input to make sure there's no duplicate filter names

    # Load all information into a single merged table

    return mergetab


if __name__ == "__main__":
    outdirc = "Outputs"

    # Load the full data table
    This needs to be done after reading filter info
    print("Loading data")
    tab = Table.read("../FINAL_TABLE_BEST_CANDIDATES_WITH_PHOTOMETRY.csv")
    nstars = len(tab)
    print("Number of stars = ", nstars)

    filttab = Table.read("filter_input.csv", format='ascii.csv')
    # Load the data
    tab = LoadData(filttab)

    # Setup the wavelength grid
    threshold = 1.0E-4
    numsample = 20000
    sum_chisq_old = np.inf
    filts = list(filttab.keys())
    nfilts = len(filts)
    numiter = 1000  # Iterate this many times until the changes to the magnitudes is negligible
    # Initialize the filter systematic offsets and dispersion
    magoffs = np.zeros(nfilts)
    magdisp = np.zeros(nfilts)
    old_magdisp = magdisp.copy()
    # Initialize the blackbody parameters
    init_a, init_t = np.ones(len(tab)), np.ones(len(tab))
    
    # Load filter responses
    waves = np.linspace(1300.0, 56000.0, numsample)  # Includes the wavelength range for GALEX FUV - WISE W2
    midwave, responses, magtypes = load_filters(waves, np.zeros(len(filts), dtype=bool), filts)
    wise_midwave, wise_responses, wise_magtypes = load_filters(waves, np.zeros(4, dtype=bool), ['GALEX_FUV', 'GALEX_NUV', 'WISE_W1', 'WISE_W2'])
    waves *= u.AA

    # Load data
    print("Loading data...")
    all_mags = -1*np.ones((nstars,nfilts))  # Magnitude data for each star and filter
    all_mage = -1*np.ones((nstars,nfilts))  # Corresponding magnitude errors
    all_magm = np.ones((nstars,nfilts), dtype=bool)  # Mask indicating if a filter has a measurement (True = good data)
    for tt in range(nstars):
        for ff, filt in enumerate(filts):
            # Check if a magnitude is available
            if tab[tt][filttab[filt][0]] <= 0:  # A masked value
                all_magm[tt, ff] = False
                continue
            if filttab[filt][1] is not None:
                if tab[tt][filttab[filt][1]] <= 0:  # A masked value
                    all_magm[tt, ff] = False
                    continue
            # Store the magnitude
            all_mags[tt, ff] = tab[tt][filttab[filt][0]]
            # Store or calculate the error
            if filttab[filt][1] is not None:
                all_mage[tt, ff] = tab[tt][filttab[filt][1]]
            else:
                print("Unsupported filter!")
                embed()
                assert(False)
    
    allcoo = SkyCoord(ra=tab['ra']*u.deg, dec=tab['dec']*u.deg)
    total_chisq = np.zeros((nstars,numiter))
    for nn in range(numiter):
#       tab = Table.read("all_BBtable_v3.csv")
        all_modl = np.zeros((nstars, nfilts))
        filt_diffs = np.zeros((nstars, nfilts))
        filt_diffe = np.zeros((nstars, nfilts))
        all_aval, all_teff = np.zeros(nstars), np.zeros(nstars)
        all_avale, all_teffe = np.zeros(nstars), np.zeros(nstars)
        all_chisq, all_dof, all_redchisq = np.zeros(nstars), np.zeros(nstars), np.zeros(nstars)
        print("Current iteration:", nn)
        for tt in range(nstars):
#             if tt+1==27:
#                 # This has bad SMSS photometry
#                 continue
#             print(f"Plotting star {tt+1}/{len(tab)}")
#             print(magoffs)
            # Get the coordinate of this BB star
            coo = allcoo[tt]
            wisewave, magswise, magewise = np.array([1516,2267,34000,46000]), np.array([tab[tt]['GALEX_fuv_mag'],tab[tt]['GALEX_nuv_mag'],tab[tt]['WISE_W1'],tab[tt]['WISE_W2']]), np.array([tab[tt]['GALEX_fuv_mag_err'],tab[tt]['GALEX_nuv_mag_err'],tab[tt]['WISE_W1_err'],tab[tt]['WISE_W2_err']])

            p0 = [init_a[tt], init_t[tt]]
            # Set some constraints you would like to impose
            param_base={'value':0., 'fixed':0, 'limited':[0,0], 'limits':[0.,0.], 'step':1.0E-6}

            # Make a copy of this 'base' for all of our parameters, and set starting parameters
            param_info=[]
            for i in range(len(p0)):
                param_info.append(copycp.deepcopy(param_base))
                param_info[i]['value']=p0[i]

            # Now put in some constraints on the BB normalisation and temperature
            for pp in range(len(p0)):
                param_info[pp]['limited'] = [1,0]
                param_info[pp]['limits']  = [0.0,0.0]

            # Normalisation
#             param_info[0]['value'] = 1.0
#             param_info[0]['fixed'] = 1

            mags = all_mags[tt,:].reshape((1,nfilts))
            mage = all_mage[tt,:].reshape((1,nfilts))
            magm = all_magm[tt,:].reshape((1,nfilts))

            # Find the good magnitudes, and store the list of magnitudes used in the fit
            goodFilts = np.where(magm)
            id_filt = goodFilts[1]
            
            # Make some corrections
            magcens = mags[goodFilts]+magoffs[id_filt]
            magerrs = np.sqrt(mage[goodFilts]**2 + magdisp[id_filt]**2)
            GaiaG = magcens[0]
            fa = {'y':magcens, 'err':magerrs, 'id_star':goodFilts[0], 'id_filt':goodFilts[1], 'responses':responses, 'waves':waves, 'magtype':magtypes, 'nfilts':nfilts}

            m = mpfit.mpfit(myfunct, p0, parinfo=param_info,functkw=fa,quiet=True)
            print(tt+1, "CHI-SQUARED/dof = ", m.fnorm/m.dof, m.params)
            if m.status <= 0:
                print("ERROR")
                assert False

            # Now perform MCMC
            if False:
                ndim, nwalkers = 2, 10
                priors = np.zeros((ndim,2))
                priors[0,0] = m.params[0]/2
                priors[0,1] = m.params[0]*2
                priors[1,0] = m.params[1]/1.3
                priors[1,1] = m.params[1]*1.3

                pos = [np.array([np.random.uniform(priors[0,0], priors[0,1]),
                                 np.random.uniform(priors[1,0], priors[1,1])
                                 ]) for i in range(nwalkers)]
    #            print("Initialising sampler")
                id_star = goodFilts[0]
                wresp = np.where(id_star==0)

    #             with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, BB_log_probability, args=(magcens, magerrs, waves, responses, magtypes, id_filt[wresp], priors, GaiaG))
    #             sampler = emcee.EnsembleSampler(nwalkers, ndim, BB_log_probability, pool=pool, args=(magcens, magerrs, waves, responses, magtypes, id_filt[wresp], priors, extfunc_arr, extfunc))
                print("Running MCMC")
                sampler.run_mcmc(pos, 1000, progress=True)
    #                 sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    #                 sampler.run_mcmc(initial, nsteps, progress=True)
    #                 multi_time = end - start
    #                 print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    #                 print("{0:.1f} times faster than serial".format(serial_time / multi_time))

                # Get the chains
                flat_samples = sampler.get_chain(discard=500, thin=5, flat=True)
                # Store the median values
                mcmcA = np.percentile(flat_samples[:, 0], [16, 50, 84])
                all_aval[tt] = mcmcA[1]
                all_avale[tt] = 0.5*(mcmcA[2]-mcmcA[0])
                mcmcB = np.percentile(flat_samples[:, 1], [16, 50, 84])
                all_teff[tt] = mcmcB[1]*1.0E4
                all_teffe[tt] = 0.5*(mcmcB[2]-mcmcB[0])*1.0E4

                samples = sampler.get_chain()

            if False:
                fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
                samples = sampler.get_chain()
                labels = ["a", "Teff"]
                for i in range(ndim):
                    ax = axes[i]
                    ax.plot(samples[:, :, i], "k", alpha=0.3)
                    ax.set_xlim(0, len(samples))
                    ax.set_ylabel(labels[i])
                    ax.yaxis.set_label_coords(-0.1, 0.5)

                axes[-1].set_xlabel("step number")
                plt.show()
                tau = sampler.get_autocorr_time()

            if False:
                modmags, fwav, fmags = get_blackbody_samples(flat_samples, waves, all_aval[tt], all_teff[tt]*1.0E-4, responses[:,id_filt], magtypes[id_filt], GaiaG)
                if all_ebv[tt] == 0.0:
                    modmags, fmag = blackbody_func(m.params[0], m.params[1], waves, responses[:,id_filt], magtypes[id_filt], GaiaG, fmag=True)
                    fwavb = waves
                else:
                    fmag = fmags[3,:]
                    fwavb = fwav
            else:
                # Only doing chi-squared minimization
                modmags, fmag = blackbody_func(m.params[0], m.params[1], waves, responses[:,id_filt], magtypes[id_filt], GaiaG, fmag=True)
                fwavb = waves
                all_aval[tt] = m.params[0]
                all_avale[tt] = m.perror[0]
                all_teff[tt] = m.params[1]*1.0E4
                all_teffe[tt] = m.perror[1]*1.0E4
                init_a[tt], init_t[tt] = m.params[0], m.params[1]

            all_mags[tt,id_filt] = mags[goodFilts]
            all_mage[tt,id_filt] = mage[goodFilts]
            all_modl[tt,id_filt] = modmags
            filt_diffs[tt,id_filt] = modmags-magcens
            filt_diffe[tt,id_filt] = mage[goodFilts]
#             all_aval[tt], all_teff[tt] = m.params[0], m.params[1]*1.0E4
#             all_avale[tt], all_teffe[tt] = m.perror[0], m.perror[1]*1.0E4
            all_chisq[tt] = m.fnorm
            all_dof[tt] = m.dof
            all_redchisq[tt] = m.fnorm/m.dof
            total_chisq[tt,nn] = m.fnorm/m.dof
            miny, maxy = np.max(fmag)+0.5, np.min(fmag)-0.5
            if plotit:
                plt.subplot(211)
                astr = "a = {0:.4f} +/- {1:.4f} x 10^-23".format(all_aval[tt], all_avale[tt])
                tstr = "T = {0:.1f} +/- {1:.1f} K ".format(all_teff[tt], all_teffe[tt])
                plt.title(f"{astr}    {tstr}", fontsize=10)
                # Plot the model
                if False:
                    plt.fill_between(fwav, fmags[0,:], y2=fmags[-1,:], color='r', alpha=0.2, zorder=-100)
                    plt.fill_between(fwav, fmags[1,:], y2=fmags[-2,:], color='r', alpha=0.45, zorder=-99)
                plt.plot(fwavb, fmag, 'r-', linewidth=2, label='blackbody', zorder=-97)
                # Plot the model magnitudes, and measured magnitudes
                plt.plot(midwave[id_filt], modmags, 'bx', label='model')
                plt.errorbar(midwave[id_filt], magcens, yerr=magerrs, fmt='rx', label='data')
                tmp_mn, tmp_mx = np.max(magcens + magerrs)+0.5, np.min(magcens - magerrs)-0.5
                if tmp_mn > miny: miny = tmp_mn
                if tmp_mx < maxy: maxy = tmp_mx
                gdw = np.where(magswise!=-1)
                if gdw[0].size != 0:
                    plt.errorbar(wisewave[gdw], magswise[gdw], yerr=magewise[gdw], fmt='mx', label='WISE')
                    tmp_mn, tmp_my = np.max(magswise[gdw] + magewise[gdw])+0.5, np.min(magswise[gdw] - magewise[gdw])-0.5
                    if tmp_mn > miny: miny = tmp_mn
                    if tmp_mx < maxy: maxy = tmp_mx
                plt.ylim(miny, maxy)
                plt.xlim(np.min(waves.value), np.max(waves.value))
                plt.xscale("log")
                plt.legend()
                plt.subplot(212)
                # Plot the model
    #             plt.fill_between(fwav, fmags[3,:]-fmags[0,:], y2=fmags[3,:]-fmags[-1,:], color='r', alpha=0.15, zorder=-100)
                if False:
                    plt.fill_between(fwav, fmags[3,:]-fmags[1,:], y2=fmags[3,:]-fmags[-2,:], color='r', alpha=0.3, zorder=-99)
                    plt.fill_between(fwav, fmags[3,:]-fmags[2,:], y2=fmags[3,:]-fmags[-3,:], color='r', alpha=0.5, zorder=-98)
                plt.axhline(0, color='r', linewidth=2)
                plt.errorbar(midwave[id_filt], modmags-magcens, yerr=magerrs, fmt='bx', label='data')
                tmp_mn, tmp_mx = np.max(magcens + magerrs)+0.5, np.min(magcens - magerrs)-0.5
                if tmp_mn > miny: miny = tmp_mn
                if tmp_mx < maxy: maxy = tmp_mx
                gdw = np.where(magswise!=-1)
                if gdw[0].size != 0:
                    modmags_wise = blackbody_func(all_aval[tt], all_teff[tt]*1.0E-4, waves, wise_responses[:,gdw[0]], wise_magtypes[gdw[0]], GaiaG)
    #                 modmags_wise = blackbody_func(m.params[0], m.params[1], m.params[3], waves, wise_responses[:,gdw[0]], wise_magtypes[gdw[0]], fmag=False)
                    plt.errorbar(wise_midwave[gdw], modmags_wise-magswise[gdw], yerr=magewise[gdw], fmt='mx', label='GALEX+WISE')
                plt.xlim(np.min(waves.value), np.max(waves.value))
                plt.xscale("log")
                # Save the figure
                BBname = getname(tab[tt])
                outname = f'{outdirc}/{BBname}.pdf'
                plt.savefig(outname)
                os.system(f'pdfcrop --margins=2 {outname} {outname}')
                plt.clf()
                if False:
                    np.save(f'{outdirc}/samples_{BBname}.npy', samples)
                    np.save(f'{outdirc}/{BBname}.npy', np.append(fwav.reshape((1, fwav.size)), fmags, axis=0))
        # Calculate the offsets
        this_magoffs, this_magdisp = get_offset_dispersion_chisq(filt_diffs, filt_diffe, outdirc)
#         this_magoffsb, this_magdispb = get_offset_dispersion_chisq(filt_diffs, filt_diffe, refdirc)
#         for rrr in range(nfilts):
#             dev_mu = 100*(this_magoffs[rrr]-this_magoffsb[rrr])/this_magoffs[rrr]
#             dev_std = 100*(this_magdisp[rrr]-this_magdispb[rrr])/this_magdisp[rrr] if this_magdisp[rrr] != 0.0 else 0.0
#             print(filts[rrr], dev_mu, dev_std)
        # Check that these are zero, and therefore everything is relative to Gaia G
        print("THESE SHOULD BE ZERO", this_magoffs[0], this_magdisp[0])
#         this_magoffs -= this_magoffs[0]
#         this_magdisp[0] = 0.0

        # Check if we have converged
        factadj = 1
        magoffs += factadj*this_magoffs
        magdisp = old_magdisp + factadj*(this_magdisp-old_magdisp)
        old_magdisp = magdisp.copy()

        sum_chisq = np.sum(total_chisq, axis=0)
        if (sum_chisq_old < sum_chisq[nn]) and nn > 20:
            plt.subplot(211)
            plt.plot(sum_chisq[:nn])
            plt.subplot(212)
            for ff in range(nfilts):
                plt.plot(total_chisq[ff,:nn])
            plt.show()
            embed()
            break

        sum_chisq_old = sum_chisq[nn]

        if np.all(np.abs(this_magoffs)<threshold):
            break
        else:
            print("MAXIMUM OFFSET ::", np.max(np.abs(this_magoffs)))
            print("MAXIMUM RELATIVE TO SYSTEMATIC ::", np.max(np.abs(this_magoffs/this_magdisp)))
            print("THESE OFFSETS/DISPERSIONS ::")
            for rrr in range(nfilts):
                try:
                    print(filts[rrr], magoffs[rrr], 100*this_magoffs[rrr]/magoffs[rrr], this_magoffs[rrr], this_magdisp[rrr])
                except:
                    embed()
#             print(this_magoffs)
#             print(this_magdisp)
#             print("---")
#             print(magoffs)
#             print(100*this_magoffs/magoffs)
        if True:
            np.savetxt("nogalex_BBparams_FINAL.txt", np.transpose((all_aval, all_avale, all_teff, all_teffe, all_chisq, all_dof, all_redchisq)))
            np.save("nogalex_filt_offset_value_FINAL.npy", magoffs)
            np.save("nogalex_filt_offset_error_FINAL.npy", magdisp)
            np.save("nogalex_filt_diffs_FINAL.npy", filt_diffs)
            np.save("nogalex_filt_diffe_FINAL.npy", filt_diffe)
            np.save("nogalex_all_mags_FINAL.npy", all_mags)
            np.save("nogalex_all_mage_FINAL.npy", all_mage)
            np.save("nogalex_all_modl_FINAL.npy", all_modl)
