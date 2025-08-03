import os, sys

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker, colormaps
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

import Utils

plt.rcParams.update({
    "text.usetex": True,
})


def ticks_format(value, index):
    """
    get the value and returns the value as:
       integer: [0,99]
       1 digit float: [0.1, 0.99]
       n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value / 10 ** exp
    if int(base) in [5, 7, 9]:
        return ''
    if exp == 0 or exp == 1:
        return '${0:d}$'.format(int(value))
    if exp == -1:
        return '${0:.1f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))


def gettexline(missionname, filttext, y, errp, errm, se, sep, sem, ns):
    err = 0.5 * (errp + errm)
    nsigfig = 1 + int(abs(np.floor(np.log10(err))))
    #     nsigfig = 1+int(abs(np.floor(np.log10(min(errp,errm)))))
    offtxt = "{0:+." + str(nsigfig) + "f}"
    errtxt = "{0:." + str(nsigfig) + "f}"
    sigtxt = "{0:." + str(1) + "f}\\sigma"
    #     sig = min(abs(y/errp), abs(y/errm))
    #     sig = abs(y/se) if se!=0 else 1
    sig = abs(y / err)
    #     otmpp = errtxt.format(errp)
    #     otmpm = errtxt.format(errm)
    otmpp = errtxt.format(err)
    otmpm = errtxt.format(err)
    if otmpp == otmpm:
        offtxte = "\\pm" + otmpp
    else:
        offtxte = "^{+" + otmpp + "}_{-" + otmpm + "}"
    # Now deal with systematic error (intrinsic dispersion)
    if sep == 0.0:
        nsigfig = 1 + int(abs(np.floor(np.log10(se)))) if se != 0.0 else 1
        setxt = "{0:+." + str(nsigfig) + "f}"
        fullsystxt = "<" + setxt.format(se)
    else:
        nsigfig = 1 + int(abs(np.floor(np.log10(min(sep, sem)))))
        setxt = "{0:+." + str(nsigfig) + "f}"
        serrtxt = "{0:." + str(nsigfig) + "f}"
        stmpp = errtxt.format(sep)
        stmpm = errtxt.format(sem)
        if stmpp == stmpm:
            systxte = "\\pm" + stmpp
        else:
            systxte = "^{+" + stmpp + "}_{-" + stmpm + "}"
        fullsystxt = serrtxt.format(se) + systxte
    return "{0:s} {1:s} & {2:d} & $".format(missionname, filttext, int(ns)) + offtxt.format(
        y) + offtxte + "$ & $" + fullsystxt + "$ & $" + sigtxt.format(sig) + "$ \\\\\n"


def get_textable_preamble():
    """
    Returns the preamble for the LaTeX table.
    """
    textable = ""
    textable += "\\begin{table}\n"
    textable += "    \\centering\n"
    textable += "    \\caption{Filter Corrections (AB = Catalogue + $\\Delta m_{\\rm F,T}$)}\n"
    textable += "    \\begin{tabular}{lcccc}\n"
    textable += "\\hline\n"
    textable += "Survey + Filter & $N$ & $\\Delta m_{\\rm F,T}$ & $\\sigma_{\\rm F,sys}$ & Sig. \\\\\n"
    textable += "\\hline\n"
    textable += "\\hline\n"
    return textable


def get_textable_postamble():
    """
    Returns the postamble for the LaTeX table.
    """
    textable = ""
    textable += "\\end{tabular}\n\n"
    textable += "$^{\\rm a}$ All reported corrections ($\\Delta m_{\\rm F,T}$) and intrinsic dispersions ($\\sigma_{\\rm F,sys}$) are quoted relative to the Gaia $G$ magnitude.\n"
    textable += "    \\label{tab:filtercorr}\n"
    textable += "\\end{table}\n"
    return textable


def prepare_table(outdirc, prefix, filttab, funcform="blackbody"):
    """
    Create the figure with the SEDs and the filters

    Parameters
    ----------
    outdirc : str
        The output directory where the figure will be saved.
    prefix : str
        The prefix for the output file name.
    filttab : astropy.table.Table
        The table containing filter information.
    funcform : str, optional
        The functional form to use for the SED. Options are 'blackbody' or 'atmosphere'. Default is 'blackbody'.
    """
    # Check the inputs
    if funcform not in ['blackbody', 'atmosphere']:
        raise ValueError("funcform must be 'blackbody' or 'atmosphere'")

    # Determine the number of surveys and the filter information
    idx, surveys, filters = Utils.LoadIDX(filttab)
    nfilts = len(filters)

    filtfontsize = 10
    if "GALEX" in surveys:
        xmin, xmax = 1150.0, 24000.0
    else:
        xmin, xmax = 3000.0, 24000.0

    npanel = len(surveys)
    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(npanel, 1)
    gs.update(wspace=0.0, hspace=0.1, bottom=0.1, top=0.95, left=0.25, right=0.75)  # set the spacing between axes.
    axs = [fig.add_subplot(gs[ii]) for ii in range(npanel)]
    msize = 15
    axs[0].cla()

    if funcform == "blackbody":
        # Load the offset and dispersion information (blackbody fits)
        offs_mcmc = np.mean(np.load(outdirc+prefix+"_filt_offset_value_MCMC.npy"), axis=2)
        disp_mcmc = np.mean(np.load(outdirc+prefix+"_filt_offset_error_MCMC.npy"), axis=2)
        nstars = np.load(outdirc+prefix+"_numstars.npy")
        offset = offs_mcmc[:, 1]
        errorsp = offs_mcmc[:, 2] - offs_mcmc[:, 1]
        errorsm = offs_mcmc[:, 1] - offs_mcmc[:, 0]
        syserr = disp_mcmc[:, 1]
        syserrp = disp_mcmc[:, 2] - disp_mcmc[:, 1]
        syserrm = disp_mcmc[:, 1] - disp_mcmc[:, 0]
        # Catch the 2sigma upper limits
        if True:
            wlim = np.where(syserr == 0.0)
            tmp = disp_mcmc[:, 0]
            syserr[wlim] = tmp[wlim]
            syserrp[wlim] = 0
            syserrm[wlim] = 0
    else:
        print("Not supported in this script yet. Please use the blackbody functional form.")
        sys.exit()
        # Load the offset and dispersion information (1D stellar atmosphere fits)
        offset = np.load("../Cukanovaite_WD_models_1D/filt_offset_value.npy")
        errors = np.load("../Cukanovaite_WD_models_1D/filt_offset_error.npy")
        all_mags = np.load("../Cukanovaite_WD_models_1D/all_mags.npy")
        all_mage = np.load("../Cukanovaite_WD_models_1D/all_mage.npy")
        all_modl = np.load("../Cukanovaite_WD_models_1D/all_modl.npy")

    # Load the filter properties
    numsample = 200000
    waves = np.linspace(1300.0, 56000.0, numsample)  # Includes GALEX FUV - WISE W2

    # Load filter responses
    midwave, responses = Utils.LoadFilters(waves, np.zeros(nfilts, dtype=bool), filttab)

    # Convert to microns
    waves /= 1.0E4
    midwave /= 1.0E4
    xmin /= 1.0E4
    xmax /= 1.0E4
    buff = 2

    # Setup the colors
    # cmap = colormaps['Spectral_r']
    cmap = colormaps['jet']
    # cmap = colormaps['nipy_spectral']
    # norm = Normalize(np.min(midwave), np.max(midwave))
    norm = Normalize(np.min(midwave), 1.0)
    colors = cmap(norm(midwave))

    textable = get_textable_preamble()

    # Loop through the surveys and plot the data
    for ax in range(npanel):
        missionname = r"{0:s}".format(surveys[ax])
        wfilt = np.where(idx == ax)
        ylo, yhi = np.max(np.abs(offset[wfilt] - syserr[wfilt])), np.max(np.abs(offset[wfilt] + syserr[wfilt]))
        yex = max(ylo, yhi)
        ymin, ymax = -yex * buff, yex * buff
        filtlev = ymin
        normresp = np.max(responses[:, wfilt[0]]) / yex
        for wf, pos, y, errp, errm, se, sep, sem, ns, color in zip(wfilt[0], midwave[wfilt], offset[wfilt], errorsp[wfilt],
                                                                   errorsm[wfilt], syserr[wfilt], syserrp[wfilt],
                                                                   syserrm[wfilt], nstars[wfilt], colors[wfilt]):
            #     err = np.array([errm,errp])[:,None]
            err = se  # if sem == 0.0 else np.array([sem, sep])[:,None]
            terr = se  # if sem == 0.0 else sep
            filttext = filters[wf]
            lw = 10 if sep == 0 else 2
            axs[ax].errorbar(pos, y, err, lw=lw, capsize=5, capthick=2, color=color)
            axs[ax].text(pos, y + terr + 0.05 * (ymax - ymin), filttext, ha='center', va='bottom', color='k',
                         fontsize=filtfontsize)
            axs[ax].text(pos, y + terr + 0.15 * (ymax - ymin), "{0:d}".format(int(ns)), ha='center', va='bottom', color='k',
                         fontsize=filtfontsize)
            #     print(missionname, filttext, y, err)
            textable += gettexline(missionname, filttext, y, errp, errm, se, sep, sem, ns)
            resp = responses[:, wf] / normresp
            wnz = np.where(resp != 0.0)
            axs[ax].fill_between(waves[wnz], resp[wnz] + filtlev, y2=filtlev, color=color, alpha=0.3)
        #     axs[ax].plot(waves[wnz], resp[wnz], color=color)
        axs[ax].axhline(0.0, color='k', linestyle='--')
        axs[ax].set_xlim(xmin, xmax)
        axs[ax].set_ylim(ymin, ymax)
        axs[ax].text(xmax - 0.05 * (xmax - xmin), ymax - 0.1 * (ymax - ymin), missionname, va='top', ha='right',
                     fontsize=14)
        # axs[ax].text(xmax-0.05*(xmax-xmin), ymax-0.25*(ymax-ymin), "29 stars", va='top', ha='right', fontsize=12)
        axs[ax].set_xscale('log')
        axs[ax].xaxis.set_minor_formatter(ticker.NullFormatter())
        axs[ax].xaxis.set_ticklabels([])
        # axs[ax].xaxis.set_minor_locator(ticker.NullLocator())
        # axs[ax].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x),0)))).format(x)))
        # axs[ax].xaxis.set_ticklabels([1.0,2.0])
        # axs[ax].xaxis.set_ticklabels(["0.3","0.5","0.7","1.0","2.0"])
        # axs[ax].set_xticks([1.0, 2.0])
        # axs[ax].get_xaxis().set_major_formatter(ticker.ScalarFormatter())

        textable += "\\hline\n"

    textable += get_textable_postamble()

    print("\n\n\n\n" + textable + "\n\n\n\n")

    subs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]  # ticks to show per decade
    axs[ax].xaxis.set_minor_locator(ticker.LogLocator(subs=subs))  # set the ticks position
    # axs[ax].xaxis.set_major_formatter(ticker.NullFormatter())   # remove the major ticks
    axs[ax].xaxis.set_minor_formatter(ticker.FuncFormatter(ticks_format))  # add the custom ticks

    # Draw it
    fig.canvas.draw()

    for ax in range(npanel):
        axs[ax].yaxis.set_major_formatter(FormatStrFormatter(r'$%.2f$'))
        #         if ax < npanel-1:
        #             axs[ax].xaxis.set_major_formatter(ticker.NullFormatter())
        #         else:
        #             axs[ax].xaxis.set_major_formatter(FormatStrFormatter(r'$%.1f$'))
        Utils.plot_pm(axs[ax], xy="y", zero=False)

    # Set the title of the final x-axis
    axs[npanel - 1].set_xlabel(r'Wavelength ($\mu$m)', fontsize=16)
    if funcform == 'blackbody':
        outname = outdirc + prefix + '_plot_filter_offsets_blackbody_MCMC.pdf'
    elif funcform == 'atmosphere':
        outname = outdirc + prefix + '_plot_filter_offsets_atmosphere_MCMC.pdf'
    else:
        outname = outdirc + prefix + '_plot_filter_offsets_<MISSING_MODEL>_MCMC.pdf'
    plt.savefig(outname)
    os.system(f'pdfcrop --margins=2 {outname} {outname}')

    print(f"Saved figure to {outname}")
    return
