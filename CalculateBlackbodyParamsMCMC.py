import os, sys

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from astropy.table import Table
import astropy.units as u

import emcee

import Utils
from IterFit import blackbody_func

from IPython import embed


def ticks_format(value, index):
    """
    get the value and returns the value as:
       integer: [0,99]
       1 digit float: [0.1, 0.99]
       n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value/10**exp
    if int(base) in [5,7,9]:
        return ''
    if exp == 0 or exp == 1:
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))


def BB_log_likelihood(theta, y, yerr, waves, responses, magtypes, id_filt):
    a, Teff = theta
    model = blackbody_func(a, Teff, waves, responses[:,id_filt], magtypes[id_filt], 0.0)
    sigma2 = yerr ** 2
    return -0.5 * np.sum(((y - model) ** 2 / sigma2))


def BB_log_prior(theta, priors):
    a, Teff = theta
    if priors[0,0] < a < priors[0,1] and priors[1,0] < Teff < priors[1,1]:
        return 0.0
    return -np.inf


def BB_log_probability(theta, y, yerr, waves, responses, magtypes, id_filt, priors):
    lp = BB_log_prior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    return lp + BB_log_likelihood(theta, y, yerr, waves, responses, magtypes, id_filt)


def get_blackbody_samples(flat_samples, waves, resps, magtyps, nwave=1000):
    # First do the median value
    a = np.percentile(flat_samples[:, 0], 50)
    teff = np.percentile(flat_samples[:, 1], 50)
    print("Getting blackbody model")
    modmags = blackbody_func(a, teff, waves, resps, magtyps, 0.0)
    tmpwav = np.linspace(waves[0], waves[-1], nwave)
    tmpresps = np.zeros((nwave,resps.shape[1]))
    for rr in range(resps.shape[1]):
        tmpresps[:,rr] = np.interp(tmpwav, waves, resps[:,rr])
    nsamples = flat_samples.shape[0]
    modmag_arr = np.zeros((nsamples, modmags.size))
    fmag_arr = np.zeros((nsamples, nwave))
    print("Getting blackbody samples")
    for ii in range(nsamples):
        if ii%(nsamples//10) == 0:
            print(ii+1,'/',flat_samples.shape[0])
        modmag_arr[ii,:], fmag_arr[ii,:] = blackbody_func(flat_samples[ii, 0], flat_samples[ii, 1], tmpwav, tmpresps, magtyps, 0.0, fmag=True)
    sig1 = 68.27/2
    sig2 = 95.45/2
    sig3 = 99.73/2
    fmags = np.percentile(fmag_arr, [50-sig3,50-sig2,50-sig1,50,50+sig1,50+sig2,50+sig3], axis=0)
    modmags_err = np.percentile(modmag_arr, [50-sig3,50-sig2,50-sig1,50,50+sig1,50+sig2,50+sig3], axis=0)
    return modmags, modmags_err, tmpwav.value, fmags


def MakeSynthPhotometryTable(phottab, synth_tab):
    """
    Create a synthetic photometry table based on the input photometric table.

    Parameters
    ----------
    phottab : astropy.table.Table
        The input photometric table containing the star data.
    synth_tab : astropy.table.Table
        The output synthetic photometry table with details to be used in th generated table
    """
    # Create a new table for synthetic photometry
    synth_phot_tab = Table()
    synth_phot_tab['source_id'] = phottab['source_id']
    nstars = len(phottab)
    # Add the filter names and errors from the synth_tab
    for filt in range(len(synth_tab)):
        synth_phot_tab[synth_tab['Photometry'][filt]] = -1*np.ones(nstars)
        synth_phot_tab[synth_tab['Photometry Error'][filt]] = -1*np.ones(nstars)
    return synth_phot_tab


def run_blackbody_params_mcmc(outdirc, prefix, filttab, funcform="blackbody", plotit=False, rerun=True):
    """
    Run the MCMC to calculate the blackbody parameters for each star in the input table.

    Parameters
    ----------
    outdirc : str
        The output directory where the results will be saved.
    prefix : str
        The prefix for the output files.
    filttab : str
        The input filter table file name.
    funcform : str, optional
        The functional form to use for the photometric fit. Default is "blackbody".
    plotit : bool, optional
        If True, plots will be generated. Default is False.
    rerun : bool, optional
        If True, the MCMC will be rerun even if output files already exist. Default is True.
        If the files have already been generated and you just wish to update the plots, set this to False.
    """
    print("Calculating final blackbody parameters from MCMC results... This may take a while...")

    infile = outdirc + prefix + "_BBparams.txt"
    all_aval, all_avale, all_teff, all_teffe, all_chisq, all_dof, all_redchisq = np.loadtxt(infile, unpack=True)

    # Load the full data table
    print("Loading data")
    phottab = Utils.LoadData(filttab)
    nstars = len(phottab)
    nfilts = len(filttab)
    magtypes = filttab['MagType'].astype(int)
    wscale = 1.0E4  # Scale the wavelengths from Angstroms to microns
    fs = 10
    print("Number of stars = ", nstars)

    # Setup the wavelength grid
    numsample = 20000
    if funcform == "blackbody":
        # Load the offset and dispersion information (blackbody fits)
        use_chisq = True  # I think this is best, because the magdisp will be the value that minimises the chi-squared (unlike MCMC, which stores a zero value when it's undetected at 2sigma).
        if use_chisq:
            magoffs = np.load(outdirc+prefix+"_filt_offset_value.npy")
            magdisp = np.load(outdirc+prefix+"_filt_offset_error.npy")
        else:
            offs_mcmc = np.mean(np.load(outdirc+prefix+"_filt_offset_value_MCMC.npy"), axis=2)
            disp_mcmc = np.mean(np.load(outdirc+prefix+"_filt_offset_error_MCMC.npy"), axis=2)
            magoffs = offs_mcmc[:, 1]
            magdisp = disp_mcmc[:, 1]
            # Note, magdisp can be zero when the dispersion is not detected with at least 2sigma confidence
    else:
        print("Not supported in this script yet. Please use the blackbody functional form.")
        sys.exit()

    # Load filter responses
    waves = np.linspace(1300.0, 56000.0, numsample)  # Includes GALEX FUV - WISE W2
    midwave, responses = Utils.LoadFilters(waves, np.zeros(nfilts, dtype=bool), filttab)
    # Load the synthetic filter responses
    synth_tab = Table.read("filter_output.csv", format='ascii.csv', comment="#")
    synth_magtypes = synth_tab['MagType'].astype(int)
    nsynth = len(synth_tab)
    synth_midwave, synth_responses = Utils.LoadFilters(waves, np.zeros(nsynth, dtype=bool), synth_tab)
    waves *= u.AA

    synth_phot = MakeSynthPhotometryTable(phottab, synth_tab)

    filtidx, surveys, filters = Utils.LoadIDX(filttab)
    cols = ['black', 'firebrick', 'gold', 'forestgreen', 'dodgerblue', 'mediumorchid']
    sym = ['o', 's', 'p', '*', 'd', 'X']
    symsize = [3, 3, 5, 7, 5, 5]
    assert (filtidx.size == nfilts)
    wavemin, wavemax = 3000.0, 23000.0
    npp = 5  # Number of objects per pdf

    # Load data
    print("Loading data...")
    all_mags, all_mage, all_magm = Utils.LoadPhotometry(phottab, filttab)

    npage = 0
    for tt in range(nstars):

        mags = all_mags[tt, :].reshape((1, nfilts))
        mage = all_mage[tt, :].reshape((1, nfilts))
        magm = all_magm[tt, :].reshape((1, nfilts))

        # Find the good magnitudes, and store the list of magnitudes used in the fit
        goodFilts = np.where(magm)
        id_filt = goodFilts[1]

        # Make some corrections
        magcens = mags[goodFilts] + magoffs[id_filt]
        magerrs = np.sqrt(mage[goodFilts] ** 2 + magdisp[id_filt] ** 2)

        # Now perform MCMC
        if rerun:
            ndim, nwalkers = 2, 10
            priors = np.zeros((ndim, 2))
            priors[0, 0] = all_aval[tt] - 5 * all_avale[tt]
            priors[0, 1] = all_aval[tt] + 5 * all_avale[tt]
            priors[1, 0] = (all_teff[tt] - 5 * all_teffe[tt]) / 1.0E4
            priors[1, 1] = (all_teff[tt] + 5 * all_teffe[tt]) / 1.0E4

            pos = [np.array([np.random.uniform(priors[0, 0], priors[0, 1]),
                             np.random.uniform(priors[1, 0], priors[1, 1])
                             ]) for i in range(nwalkers)]
            id_star = goodFilts[0]
            wresp = np.where(id_star == 0)

            sampler = emcee.EnsembleSampler(nwalkers, ndim, BB_log_probability, args=(
            magcens, magerrs, waves, responses, magtypes, id_filt[wresp], priors))

            print("Running MCMC")
            mcmcsamp = 10000
            sampler.run_mcmc(pos, mcmcsamp, progress=True)
            try:
                tau = sampler.get_autocorr_time()
                print(tau)
            except emcee.autocorr.AutocorrError:
                print("There was an autocorrelation error for star", tt)
                embed()

            # Get the chains
            flat_samples = sampler.get_chain(discard=mcmcsamp//10, thin=10, flat=True)
            # Store the median values
            mcmcA = np.percentile(flat_samples[:, 0], [16, 50, 84])
            all_aval[tt] = mcmcA[1]
            all_avale[tt] = 0.5 * (mcmcA[2] - mcmcA[0])
            mcmcB = np.percentile(flat_samples[:, 1], [16, 50, 84])
            all_teff[tt] = mcmcB[1] * 1.0E4
            all_teffe[tt] = 0.5 * (mcmcB[2] - mcmcB[0]) * 1.0E4

            # Save the full chains
            samples = sampler.get_chain()
            BBname = Utils.getname(phottab[tt])
            np.save(f'{outdirc}{prefix}_samples_{BBname}.npy', samples)
            np.save(f'{outdirc}{prefix}_flatsamples_{BBname}.npy', flat_samples)
        else:
            BBname = Utils.getname(phottab[tt])
            fname_samples = f'{outdirc}{prefix}_samples_{BBname}.npy'
            if os.path.exists(fname_samples):
                samples = np.load(fname_samples)
                flat_samples = np.load(f'{outdirc}{prefix}_flatsamples_{BBname}.npy')
            else:
                print(f"Samples don't exist!  Ignoring {BBname}")
                continue

        modmags, modmags_err, fwav, fmags = get_blackbody_samples(flat_samples, waves, responses[:, id_filt], magtypes[id_filt])
        buff = 0.2
        miny, maxy = np.max(fmags[3, :]) + buff, np.min(fmags[3, :]) - buff
        miny = maxy

        # Plot the results
        msize = 15
        if tt % npp == 0:
            # Start a new figure environment
            fig = plt.figure(figsize=(8, 11))
            gs = gridspec.GridSpec(3 * npp, 1, height_ratios=[4, 1, 0.2] * npp)
            gs.update(wspace=0.0, hspace=0.1, bottom=0.2, top=0.95, left=0.15,
                      right=0.95)  # set the spacing between axes.
            axs = []
            for nn in range(npp):
                #                 axs.append(fig.add_subplot(gs[nn*4:nn*4+3,0]))
                #                 axs.append(fig.add_subplot(gs[nn*4+3,0]))
                if npage == 5 and nn == npp - 1:
                    continue
                else:
                    axs.append(fig.add_subplot(gs[3 * nn, 0]))
                    axs.append(fig.add_subplot(gs[3 * nn + 1, 0]))
        tp = 2 * (tt % npp)
        height_ratios = [1, 1, 3]
        axs[tp].cla()
        #         astr = "a = {0:.4f} +/- {1:.4f} x 10^-23".format(all_aval[tt], all_avale[tt])
        #         tstr = "T = {0:.1f} +/- {1:.1f} K ".format(all_teff[tt], all_teffe[tt])
        #         plt.title(f"{astr}    {tstr}", fontsize=10)

        # Plot the model
        #         axs[0].fill_between(fwav/wscale, fmags[0,:], y2=fmags[-1,:], color='r', alpha=0.2, zorder=-100, linewidth=0.2)
        #         axs[0].fill_between(fwav/wscale, fmags[1,:], y2=fmags[-2,:], color='r', alpha=0.45, zorder=-99, linewidth=0.2)
        #         axs[0].fill_between(fwav/wscale, fmags[2,:], y2=fmags[-3,:], color='r', alpha=0.7, zorder=-98, linewidth=0.2)
        axs[tp].fill_between(fwav / wscale, fmags[2, :], y2=fmags[-3, :], color='r', alpha=0.5, zorder=-98,
                             linewidth=0.2)
        axs[tp].plot(fwav / wscale, fmags[3, :], 'r-', linewidth=0.5, label='Model', zorder=-97)
        # Plot the model magnitudes, and measured magnitudes
        #         plt.plot(midwave[id_filt], modmags, 'bx', label='model')
        this_idx = filtidx[id_filt]
        unq = np.unique(this_idx)
        for ff in range(unq.size):
            mwave = midwave[id_filt]
            wff = np.where(this_idx == unq[ff])[0]
            uu = unq[ff]
            fudge_magcen = np.zeros(wff.size)
            for jjj in range(wff.size):
                amin = np.argmin(np.abs(fwav - mwave[wff[jjj]]))
                fudge_magcen[jjj] = fmags[3, amin] - (modmags[wff[jjj]] - magcens[wff[jjj]])
            axs[tp].errorbar(mwave[wff] / wscale, fudge_magcen, yerr=magerrs[wff], color=cols[uu], linestyle='None',
                             elinewidth=1, marker=sym[uu], markersize=symsize[uu])
        tmp_mn, tmp_mx = np.max(magcens + magerrs) + buff, np.min(magcens - magerrs) - buff
        if tmp_mn > miny: miny = tmp_mn
        if tmp_mx < maxy: maxy = tmp_mx
        # Calculate the synthetic magnitudes
        this_synth_phot, modmags_err_synth, fwav_synth, fmags_synth = get_blackbody_samples(flat_samples, waves, synth_responses, synth_magtypes)
        this_synth_photerr = 0.5 * np.abs(modmags_err_synth[2, :] - modmags_err_synth[-3, :])
        # Store the synthetic magnitudes in the synth_phot table
        for filt in range(len(synth_tab)):
            synth_phot[synth_tab['Photometry'][filt]][tt] = this_synth_phot[filt]
            synth_phot[synth_tab['Photometry Error'][filt]][tt] = this_synth_photerr[filt]
        # Make a legend
        if tp // 2 == 1:
            # Fudge one of each point on the far side of the plot
            unq = np.unique(filtidx)
            for ff in range(unq.size):
                uu = unq[ff]
                axs[tp].errorbar([99999999.9], [99999999.9], yerr=[0.0], color=cols[uu], linestyle='None',
                                 elinewidth=1, marker=sym[uu], markersize=symsize[uu], label=surveys[uu])
            axs[tp].legend(fontsize=fs - 2.5, loc='upper right', ncol=2)
        # Finalise the look of the panel
        axs[tp].set_ylim(miny, maxy)
        axs[tp].set_xlim(wavemin / wscale, wavemax / wscale)
        axs[tp].set_xscale("log")
        axs[tp].yaxis.set_major_formatter(FormatStrFormatter(r'$%.1f$'))
        axs[tp].xaxis.set_minor_formatter(ticker.NullFormatter())
        axs[tp].xaxis.set_ticklabels([])
        axs[tp].set_ylabel("AB mag", fontsize=fs)
        axs[tp].tick_params(labelsize=fs)
        #         print(wavemin+0.005*(wavemax-wavemin), maxy+0.1*(maxy-miny))
        #         print(wavemin, wavemax, miny, maxy)
        bbstr = BBname.replace("p", "$+$").replace("m", "$-$").replace("_", "")
        axs[tp].text(wavemin / wscale + 0.005 * (wavemax - wavemin) / wscale, maxy - 0.05 * (maxy - miny),
                     r'{0:s}'.format(bbstr), va='top', ha='left', fontsize=fs)

        # Plot the model
        #     plt.fill_between(fwav, fmags[3,:]-fmags[0,:], y2=fmags[3,:]-fmags[-1,:], color='r', alpha=0.15, zorder=-100)
        #     plt.fill_between(fwav, fmags[3, :] - fmags[1, :], y2=fmags[3, :] - fmags[-2, :], color='r', alpha=0.3,
        #                      zorder=-99)
        #     plt.fill_between(fwav, fmags[3, :] - fmags[2, :], y2=fmags[3, :] - fmags[-3, :], color='r', alpha=0.5,
        #                      zorder=-98)
        axs[tp + 1].fill_between(fwav / wscale, fmags[3, :] - fmags[2, :], y2=fmags[3, :] - fmags[-3, :], color='r',
                                 alpha=0.5, zorder=-98, linewidth=0.2)
        axs[tp + 1].axhline(0, color='r', linewidth=0.5)
        for ff in range(unq.size):
            mwave = midwave[id_filt]
            wff = np.where(this_idx == unq[ff])
            uu = unq[ff]
            axs[tp + 1].errorbar(mwave[wff] / wscale, modmags[wff] - magcens[wff], yerr=magerrs[wff],
                                 color=cols[uu], linestyle='None', elinewidth=1, marker=sym[uu],
                                 markersize=symsize[uu], label=surveys[uu])
        tmp_mn, tmp_mx = np.max(magcens + magerrs) + buff, np.min(magcens - magerrs) - buff
        if tmp_mn > miny: miny = tmp_mn
        if tmp_mx < maxy: maxy = tmp_mx

        axs[tp + 1].set_xlim(wavemin / wscale, wavemax / wscale)
        axs[tp + 1].set_xscale("log")
        subs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]  # ticks to show per decade
        axs[tp + 1].xaxis.set_minor_locator(ticker.LogLocator(subs=subs))  # set the ticks position
        if tp // 2 == npp - 1 or tt == nstars - 1:
            axs[tp + 1].xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))  # add the custom ticks
            axs[tp + 1].xaxis.set_major_formatter(ticker.FuncFormatter(ticks_format))  # add the custom ticks
        else:
            axs[tp + 1].xaxis.set_minor_formatter(ticker.NullFormatter())
            axs[tp + 1].xaxis.set_ticklabels([])
        axs[tp + 1].set_ylim(-0.05, +0.05)
        axs[tp + 1].tick_params(which='both', labelsize=fs)
        fig.canvas.draw()
        axs[tp + 1].yaxis.set_major_formatter(FormatStrFormatter(r'$%.2f$'))
        axs[tp + 1].set_ylabel("Residual", fontsize=fs)
        Utils.plot_pm(axs[tp + 1], xy="y", zero=False)

        # Set the title of the final x-axis
        if tp // 2 == npp - 1 or tt == nstars - 1:
            axs[tp + 1].set_xlabel(r'Wavelength ($\mu$m)', fontsize=fs)
        # Save the figure
        if tt % npp == npp - 1 or tt == nstars - 1:
            npage += 1
            #             BBname = getname(tab[tt])
            BBname = f"_MultiPanel_{npage}"
            outname = f'{outdirc}{prefix}{BBname}.pdf'
            plt.savefig(outname)
            try:
                os.system(f'pdfcrop --margins=2 {outname} {outname}')
            except:
                # Some users won't have their pdfs nicely cropped...
                pass
            plt.clf()

    # Save the synthetic magnitudes
    synth_phot.write(outdirc+prefix+"_synthetic_mags.csv", overwrite=True)
