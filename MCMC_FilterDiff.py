import numpy as np
from matplotlib import pyplot as plt

import emcee

import Utils


def get_offset_dispersion(filt_diffs, filt_diffe, plotit=False):
    nreps, ndim, nwalkers = 1, 2, 100
    nfilts = filt_diffs.shape[1]
    this_magoffs = np.zeros((nfilts,3,nreps))
    this_magdisp = np.zeros((nfilts,3,nreps))
    all_samples = []
    this_numstars = np.zeros(nfilts)
    if plotit:
        cols = ['r', 'g', 'c', 'b', 'm', 'k']
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        labels = ["offs", "disp"]
    for rr in range(nreps):
        print(f"\n\nREPITITION {rr+1}/{nreps}\n\n")
        for ii in range(nfilts):
            print(f"Calculating the offset and intrinsic dispersion for filter {ii+1}/{nfilts}")
            this_offs, this_disp = filt_diffs[:,ii], filt_diffe[:,ii]
            wgd = np.where(this_disp > 0.0)
            # Initialise the MCMC
            pos = [np.array([np.random.uniform(Utils.MIN_OFF, Utils.MAX_OFF),
                             np.random.uniform(Utils.MIN_SIG, Utils.MAX_SIG)]) for i in range(nwalkers)]
            this_numstars[ii] = this_offs[wgd].size
            sampler = emcee.EnsembleSampler(nwalkers, ndim, Utils.log_probability, args=(this_offs[wgd], this_disp[wgd]))
            sampler.run_mcmc(pos, 10000, progress=True)
            tau = sampler.get_autocorr_time()
            print("Autocorrelation analysis: ", tau)
            # Get the chains
            flat_samples = sampler.get_chain(discard=1000, thin=10, flat=True)
            if plotit:
                samples = sampler.get_chain(discard=1000)
                for i in range(ndim):
                    ax = axes[i]
                    ax.plot(samples[:, :, i], cols[rr], alpha=0.3)
                    ax.set_xlim(0, len(samples))
                    ax.set_ylabel(labels[i])
                axes[-1].set_xlabel("step number")
                all_samples.append(flat_samples.copy())
            # Store the median values
            mcmcA = np.percentile(flat_samples[:, 0], [16, 50, 84])
            this_magoffs[ii,rr] = mcmcA[1]
            mcmcB = np.percentile(flat_samples[:, 1], [16, 50, 84])
            this_magdisp[ii,rr] = mcmcB[1]
            tmp_mu = mcmcB[1]
            tmp_std = 0.5*(mcmcB[2]-mcmcB[0])
            this_magoffs[ii,:,rr] = mcmcA
            if tmp_mu/tmp_std <= 2.0:
                this_magdisp[ii,:,rr] = np.array([np.percentile(flat_samples[:, 1], 95),0.0,0.0])
            else:
                this_magdisp[ii,:,rr] = mcmcB
    if plotit:
        plt.show()
    return this_magoffs, this_magdisp, this_numstars


def run_mcmc_filtdiff(outdirc, prefix, plotit=False):
    """
    Run the MCMC fitting for the filter differences

    Parameters
    ----------
    outdirc : str
        Output directory containing the calculation output, including details of the filter offsets.
    prefix : str
        Prefix for the output files.
    plotit : bool, optional
        If True, plots the results of the MCMC fitting. Default is False.
    """
    print("Running MCMC filter difference calculation for", prefix)
    all_mags = np.load(outdirc + prefix + "_all_mags.npy")
    all_mage = np.load(outdirc + prefix + "_all_mage.npy")
    all_modl = np.load(outdirc + prefix + "_all_modl.npy")

    # Calculate the offset from all values G
    gaia_offs = all_modl[:, 0] - all_mags[:, 0]  # Offset to make Gaia G agree with the model exactly

    filt_diffs = (all_modl - all_mags) - gaia_offs.reshape((gaia_offs.size, 1))
    filt_diffe = all_mage

    this_magoffs, this_magdisp, this_numstars = get_offset_dispersion(filt_diffs, filt_diffe, plotit=plotit)

    np.save(outdirc + prefix + "_filt_offset_value_MCMC.npy", this_magoffs)
    np.save(outdirc + prefix + "_filt_offset_error_MCMC.npy", this_magdisp)
    np.save(outdirc + prefix + "_numstars.npy", this_numstars)
    print("MCMC filter difference calculation completed and saved to files.")
