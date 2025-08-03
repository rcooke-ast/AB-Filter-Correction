import os, sys
import copy as copycp

import numpy as np
import astropy.units as u
import astropy.constants as cons
from matplotlib import pyplot as plt
from scipy.optimize import minimize

import Utils
import mpfit_single as mpfit

def myfunct_intdisp(p, fjac=None, x=None, y=None, yerr=None):
    model = p[0]
    sigma2 = yerr**2 + p[1]**2
    # Non-negative status value means MPFIT should
    # continue, negative means stop the calculation.
    chi2 = (y-model)**2/sigma2 + np.log(sigma2)
    status = 0
    return [status, np.sqrt(chi2)]


def get_offset_dispersion_chisq(filt_diffs, filt_diffe):
    nfilts = filt_diffs.shape[1]
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
            nll = lambda *args: -Utils.log_likelihood(*args)
            initial = np.array(p0)
            bnds = ((None, None), (0.0, None))
            # Now tell the fitting program what we called our variables
            y, yerr = this_offs[wgd], this_disp[wgd]
            soln = minimize(nll, initial, args=(y, yerr), bounds=bnds)
            this_magoffs[ii], this_magdisp[ii] = soln.x
    return this_magoffs, this_magdisp


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
        if magtype[ff] == Utils.MAGTYPE_GALEXFUV:
            engy = (cons.h*cons.c/waves).to(u.erg).value
            cnts_per_sec = np.trapz(flam * responses[:,ff]/engy, x=waves.value)
            modelmag[ff] = -2.5 * np.log10(cnts_per_sec) + 18.82
        elif magtype[ff] == Utils.MAGTYPE_GALEXNUV:
            engy = (cons.h*cons.c/waves).to(u.erg).value
            cnts_per_sec = np.trapz(flam * responses[:,ff]/engy, x=waves.value)
            modelmag[ff] = -2.5 * np.log10(cnts_per_sec) + 20.08
        elif magtype[ff] == Utils.MAGTYPE_AB:
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
    sigma = np.zeros(y.size)
    # Loop over all stars
    for ss in range(nstars):
        wresp = np.where(id_star==ss)
        this_filt = id_filt[wresp]
        model[wresp] = blackbody_func(par_bb[2*ss], par_bb[2*ss + 1], waves, responses[:,this_filt], magtype[this_filt], y[0])
        # Calculate the intrinsic dispersion and the total error value
        sigma[wresp] = err[wresp]
    # Non-negative status value means MPFIT should
    # continue, negative means stop the calculation.
    status = 0
    return [status, (y-model)/sigma]


def run_iterfit(outdirc, prefix, filttab, plotit=False):
    magtypes = filttab['MagType'].astype(int)
    # Load the data
    phottab = Utils.LoadData(filttab)
    nstars = len(phottab)
    print("Number of stars = ", nstars)

    # Setup the parameters of the fit
    threshold = 1.0E-4
    numsample = 20000
    sum_chisq_old = np.inf
    nfilts = len(filttab)
    numiter = 1000  # Iterate this many times until the changes to the magnitudes is negligible
    # Initialize the filter systematic offsets and dispersion
    magoffs = np.zeros(nfilts)
    magdisp = np.zeros(nfilts)
    old_magdisp = magdisp.copy()
    # Initialize the blackbody parameters
    init_a, init_t = np.ones(nstars), np.ones(nstars)

    # Load filter responses
    waves = np.linspace(1300.0, 56000.0, numsample)  # Includes the wavelength range for GALEX FUV - WISE W2
    midwave, responses = Utils.LoadFilters(waves, np.zeros(nfilts, dtype=bool), filttab)
    waves *= u.AA

    # Load data
    print("Loading photometry...")
    all_mags, all_mage, all_magm = Utils.LoadPhotometry(phottab, filttab)

    total_chisq = np.zeros((nstars, numiter))
    for nn in range(numiter):
        all_modl = np.zeros((nstars, nfilts))
        filt_diffs = np.zeros((nstars, nfilts))
        filt_diffe = np.zeros((nstars, nfilts))
        all_aval, all_teff = np.zeros(nstars), np.zeros(nstars)
        all_avale, all_teffe = np.zeros(nstars), np.zeros(nstars)
        all_chisq, all_dof, all_redchisq = np.zeros(nstars), np.zeros(nstars), np.zeros(nstars)
        print("Current iteration:", nn)
        for tt in range(nstars):
            p0 = [init_a[tt], init_t[tt]]
            # Set some constraints you would like to impose
            param_base = {'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.], 'step': 1.0E-6}

            # Make a copy of this 'base' for all of our parameters, and set starting parameters
            param_info = []
            for i in range(len(p0)):
                param_info.append(copycp.deepcopy(param_base))
                param_info[i]['value'] = p0[i]

            # Now put in some constraints on the BB normalisation and temperature
            for pp in range(len(p0)):
                param_info[pp]['limited'] = [1, 0]
                param_info[pp]['limits'] = [0.0, 0.0]

            mags = all_mags[tt, :].reshape((1, nfilts))
            mage = all_mage[tt, :].reshape((1, nfilts))
            magm = all_magm[tt, :].reshape((1, nfilts))

            # Find the good magnitudes, and store the list of magnitudes used in the fit
            goodFilts = np.where(magm)
            id_filt = goodFilts[1]

            # Make some corrections
            magcens = mags[goodFilts] + magoffs[id_filt]
            magerrs = np.sqrt(mage[goodFilts] ** 2 + magdisp[id_filt] ** 2)
            GaiaG = magcens[0]
            fa = {'y': magcens, 'err': magerrs, 'id_star': goodFilts[0], 'id_filt': goodFilts[1],
                  'responses': responses, 'waves': waves, 'magtype': magtypes, 'nfilts': nfilts}

            m = mpfit.mpfit(myfunct, p0, parinfo=param_info, functkw=fa, quiet=True)
            print(tt + 1, "CHI-SQUARED/dof = ", m.fnorm / m.dof, m.params)
            if m.status <= 0:
                print("FITTING ERROR")
                sys.exit()

            # Only doing chi-squared minimization
            modmags, fmag = blackbody_func(m.params[0], m.params[1], waves, responses[:, id_filt], magtypes[id_filt], GaiaG, fmag=True)
            fwavb = waves
            all_aval[tt] = m.params[0]
            all_avale[tt] = m.perror[0]
            all_teff[tt] = m.params[1] * 1.0E4
            all_teffe[tt] = m.perror[1] * 1.0E4
            init_a[tt], init_t[tt] = m.params[0], m.params[1]

            all_mags[tt, id_filt] = mags[goodFilts]
            all_mage[tt, id_filt] = mage[goodFilts]
            all_modl[tt, id_filt] = modmags
            filt_diffs[tt, id_filt] = modmags - magcens
            filt_diffe[tt, id_filt] = mage[goodFilts]
            all_chisq[tt] = m.fnorm
            all_dof[tt] = m.dof
            all_redchisq[tt] = m.fnorm / m.dof
            total_chisq[tt, nn] = m.fnorm / m.dof
            miny, maxy = np.max(fmag) + 0.5, np.min(fmag) - 0.5
            if plotit:
                plt.subplot(211)
                astr = "a = {0:.4f} +/- {1:.4f} x 10^-23".format(all_aval[tt], all_avale[tt])
                tstr = "T = {0:.1f} +/- {1:.1f} K ".format(all_teff[tt], all_teffe[tt])
                plt.title(f"{astr}    {tstr}", fontsize=10)
                plt.plot(fwavb, fmag, 'r-', linewidth=2, label='blackbody', zorder=-97)
                # Plot the model magnitudes, and measured magnitudes
                plt.plot(midwave[id_filt], modmags, 'bx', label='model')
                plt.errorbar(midwave[id_filt], magcens, yerr=magerrs, fmt='rx', label='data')
                tmp_mn, tmp_mx = np.max(magcens + magerrs) + 0.5, np.min(magcens - magerrs) - 0.5
                if tmp_mn > miny: miny = tmp_mn
                if tmp_mx < maxy: maxy = tmp_mx
                plt.ylim(miny, maxy)
                plt.xlim(np.min(waves.value), np.max(waves.value))
                plt.xscale("log")
                plt.legend()
                plt.subplot(212)
                # Plot the model
                plt.axhline(0, color='r', linewidth=2)
                plt.errorbar(midwave[id_filt], modmags - magcens, yerr=magerrs, fmt='bx', label='data')
                tmp_mn, tmp_mx = np.max(magcens + magerrs) + 0.5, np.min(magcens - magerrs) - 0.5
                if tmp_mn > miny: miny = tmp_mn
                if tmp_mx < maxy: maxy = tmp_mx
                plt.xlim(np.min(waves.value), np.max(waves.value))
                plt.xscale("log")
                # Save the figure
                BBname = Utils.getname(phottab[tt])
                outname = f'{outdirc}/{BBname}.pdf'
                plt.savefig(outname)
                os.system(f'pdfcrop --margins=2 {outname} {outname}')
                plt.clf()
        # Calculate the offsets
        this_magoffs, this_magdisp = get_offset_dispersion_chisq(filt_diffs, filt_diffe)

        # Check if we have converged
        factadj = 1
        magoffs += factadj * this_magoffs
        magdisp = old_magdisp + factadj * (this_magdisp - old_magdisp)
        old_magdisp = magdisp.copy()

        sum_chisq = np.sum(total_chisq, axis=0)
        if (sum_chisq_old < sum_chisq[nn]) and nn > 20:
            print("CHISQ INCREASED, STOPPING")
            if plotit:
                plt.subplot(211)
                plt.plot(sum_chisq[:nn])
                plt.subplot(212)
                for ff in range(nfilts):
                    plt.plot(total_chisq[ff, :nn])
                plt.show()
                break

        sum_chisq_old = sum_chisq[nn]

        if np.all(np.abs(this_magoffs) < threshold):
            print("CONVERGED TO WITHIN THRESHOLD ::", threshold)
            break
        else:
            print("MAXIMUM OFFSET ::", np.max(np.abs(this_magoffs)))
            print("MAXIMUM RELATIVE TO SYSTEMATIC ::", np.max(np.abs(this_magoffs / this_magdisp)))
            print("THESE OFFSETS/DISPERSIONS ::")
            for rrr in range(nfilts):
                print(filttab['Photometry'][rrr], magoffs[rrr], 100 * this_magoffs[rrr] / magoffs[rrr], this_magoffs[rrr], this_magdisp[rrr])
        # Saving the results
        print("Saving results to files")
        np.savetxt(outdirc + prefix + "_BBparams.txt", np.transpose((all_aval, all_avale, all_teff, all_teffe, all_chisq, all_dof, all_redchisq)))
        np.save(outdirc + prefix + "_filt_offset_value.npy", magoffs)
        np.save(outdirc + prefix + "_filt_offset_error.npy", magdisp)
        np.save(outdirc + prefix + "_filt_diffs.npy", filt_diffs)
        np.save(outdirc + prefix + "_filt_diffe.npy", filt_diffe)
        np.save(outdirc + prefix + "_all_mags.npy", all_mags)
        np.save(outdirc + prefix + "_all_mage.npy", all_mage)
        np.save(outdirc + prefix + "_all_modl.npy", all_modl)
