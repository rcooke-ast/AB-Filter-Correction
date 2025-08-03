import os, sys

import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time
from scipy import interpolate

# Filter types
MAGTYPE_AB = 0
MAGTYPE_GALEXFUV = 1
MAGTYPE_GALEXNUV = 2

# Intrinsic dispersion priors
MIN_OFF = -0.3
MAX_OFF = +0.3
MIN_SIG = 0.0
MAX_SIG = 0.5


def log_likelihood(theta, y, yerr):
    off, sig = theta
    model = off
    sigma2 = yerr ** 2 + sig**2
    return -0.5 * np.sum(((y - model) ** 2 / sigma2) + np.log(2*np.pi*sigma2))


def log_prior(theta):
    off, sig = theta
    if MIN_OFF < off < MAX_OFF and MIN_SIG < sig < MAX_SIG:
        return 0.0
    return -np.inf


def log_probability(theta, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, y, yerr)


def LoadFilters(waves, mask, filttab):
    """
    Load the filter responses for the given wavelengths.

    Parameters
    ----------
    waves : array_like
        Wavelengths at which to evaluate the filter responses.
    mask : array_like
        Boolean mask indicating which filters to include (True for included, False for excluded).
    filttab : AstropyTable
        Table containing filter names and their properties.
    """
    nfilts = np.sum(np.logical_not(mask))
    responses = np.zeros((waves.size, nfilts))
    midwaves = np.zeros(nfilts)
    cntr = 0
    for ff, filt in enumerate(filttab['Filter Response Filename']):
        if mask[ff]:
            continue
        dirc = filttab['Folder'][ff] + '/'
        wave, sens = np.loadtxt('filters/'+dirc+filt, unpack=True)
        responses[:, cntr] = interpolate.interp1d(wave.copy(), sens.copy(), kind='linear', bounds_error=False, fill_value=0.0)(waves)
        midwaves[ff] = np.sum(wave.copy()*sens.copy())/np.sum(sens.copy())
        cntr += 1
    return midwaves, responses


def LoadData(filttab):
    """
    Load the data from the filter table, and merge all photometry into a single table that is used for the fitting

    Parameters
    ----------
    filttab : AstropyTable
        Table containing filter names and their properties.

    Returns
    -------
    AstropyTable containing merged photometry data.
    """
    # Perform some checks on the input to make sure there's no duplicate filter names
    print("Performing checks on the data")
    terminate = False
    for ff, filt in enumerate(filttab['Filter Response Filename']):
        if np.sum(filttab['Filter Response Filename'] == filt) > 1:
            print("ERROR - Duplicate filter name found: ", filt)
            terminate = True
    for ff, phot in enumerate(filttab['Photometry']):
        if np.sum(filttab['Photometry'] == phot) > 1:
            print("ERROR - Duplicate photometry name found: ", phot)
            terminate = True
    for ff, phot in enumerate(filttab['Photometry Error']):
        if np.sum(filttab['Photometry Error'] == phot) > 1:
            print("ERROR - Duplicate photometry error name found: ", phot)
            terminate = True
    # If any of the checks failed, terminate the program
    if terminate:
        print("Exiting due to bad data input.")
        sys.exit()
    # Check that Gaia is the first filter in the table
    if filttab['Filter Response Filename'][0] != 'Gaia3G.dat':
        print("ERROR - Gaia G filter must be the first filter in the table!")
        sys.exit()
    # Load all information into a single merged table
    print("Loading data...")
    mergetab = Table()
    for ff in range(len(filttab)):
        dirc = filttab['Folder'][ff] + '/'
        filt = filttab['Filter Response Filename'][ff]
        if not os.path.exists('filters/' + dirc + filt):
            print("ERROR - Filter file not found: ", "filters/" + dirc + filt)
            sys.exit()
        # Load the photmetry catalogue
        cat = Table.read(filttab['Photometry Catalogue'][ff], format='ascii.csv')
        # Insert the data for all stars with photometry with this filter into the merged table
        if ff == 0:
            # Setup the table with the relevant information
            mergetab['source_id'] = cat['source_id']
            mergetab['ra'] = cat['ra']
            mergetab['dec'] = cat['dec']
            mergetab['pmra'] = cat['pmra']
            mergetab['pmdec'] = cat['pmdec']
            mergetab[filttab['Photometry'][ff]] = cat[filttab['Photometry'][ff]]
            mergetab[filttab['Photometry Error'][ff]] = cat[filttab['Photometry Error'][ff]]
        else:
            # Cross-match the source IDs to ensure we only include stars that have photometry in this filter
            ind = np.array([], dtype=int)
            for ll in range(len(cat)):
                wind = np.where(mergetab['source_id'] == cat['source_id'][ll])[0]
                if len(wind) != 1:
                    print("ERROR - Source ID not found in merged table: ", cat['source_id'][ll])
                    sys.exit()
                ind = np.append(ind, wind[0])
            # Add the photometry for this filter to the merged table
            mergetab[filttab['Photometry'][ff]] = np.zeros(len(mergetab), dtype=float)
            mergetab[filttab['Photometry Error'][ff]] = np.zeros(len(mergetab), dtype=float)
            mergetab[filttab['Photometry'][ff]][ind] = cat[filttab['Photometry'][ff]]
            mergetab[filttab['Photometry Error'][ff]][ind] = cat[filttab['Photometry Error'][ff]]
    return mergetab


def LoadPhotometry(phottab, filttab):
    """
    Load the photometry data from the provided table and filter information.

    Parameters
    ----------
    phottab : astropy.table.Table
        The table containing photometry data.
    filttab : astropy.table.Table
        The table containing filter information.

    Returns
    -------
    all_mags : np.ndarray
        Array of magnitudes for each star and filter.
    all_mage : np.ndarray
        Array of magnitude errors for each star and filter.
    all_magm : np.ndarray
        Boolean mask indicating if a filter has a measurement (True = good data).
    """
    nstars = len(phottab)  # Number of stars in the photometry table
    nfilts = len(filttab)  # Number of filters in the filter table
    all_mags = -1 * np.ones((nstars, nfilts))  # Magnitude data for each star and filter
    all_mage = -1 * np.ones((nstars, nfilts))  # Corresponding magnitude errors
    all_magm = np.ones((nstars, nfilts), dtype=bool)  # Mask indicating if a filter has a measurement (True = good data)
    for tt in range(nstars):
        for ff in range(nfilts):
            # Grab the names for convenience
            photname = filttab['Photometry'][ff]
            photerrs = filttab['Photometry Error'][ff]
            # Check if a magnitude is available
            if phottab[tt][photname] <= 0:  # A masked value
                all_magm[tt, ff] = False
                continue
            if phottab[tt][photerrs] <= 0:  # A masked value
                all_magm[tt, ff] = False
                continue
            # Store the magnitude
            all_mags[tt, ff] = phottab[tt][photname]
            # Store or calculate the error
            all_mage[tt, ff] = phottab[tt][photerrs]
    return all_mags, all_mage, all_magm


def LoadIDX(filttab):
    """
    Load the index of the filters from the provided table.

    Parameters
    ----------
    filttab : astropy.table.Table
        The table containing filter information.

    Returns
    -------
    idx : numpy.ndarray
        An array of indices corresponding to the filters.
    survey : list
        A list of unique survey names extracted from the table.
    """
    # Find the number of unique surveys
    thisidx, thissurvey = 0, str(filttab['Folder'][0])
    # Initialise
    idx = np.array([thisidx], dtype=int)
    survey = [thissurvey]
    filters = np.array(str(filttab['Filter Name'][0]))
    for row in range(1, len(filttab)):
        if str(filttab['Folder'][row]) != thissurvey:
            thisidx += 1
            thissurvey = str(filttab['Folder'][row])
            survey.append(thissurvey)
        idx = np.append(idx, thisidx)
        filters = np.append(filters, str(filttab['Filter Name'][row]))
    return idx, survey, filters


def getname(t):
    coo = SkyCoord(ra=t['ra']*u.deg, dec=t['dec']*u.deg, pm_ra_cosdec=t['pmra']*u.mas/u.yr, pm_dec=t['pmdec']*u.mas/u.yr, obstime="J2016.0", equinox="J2016.0")
    cn = coo.apply_space_motion(Time("J2000.0"))
    radecstr = "BB" + cn.to_string('hmsdms').replace('h','').replace('m','').replace('d', '').replace('s', '')
    hms_ra = radecstr.split()[0].split(".")[0]
    dms_dec = radecstr.split()[1].split(".")[0].replace("-","m").replace("+","p")
    this_name = hms_ra + "_" + dms_dec
    return this_name


def plot_pm(axis, xy="x", zero=False, ftype=False):
    """
    plot "+" and "-" signs on the axis labels (- is always done by matplotlib as a default)
    xy   :: can be one of "x", "y", or "xy" which will put '+' signs on the x, y, or both axes
    zero :: if True, a plus sign is included on a zero if it is present

    #########################
    ##  VERY IMPORTANT!!!  ##
    #########################
    In order for this function to work, you need to format the labels before calling this function.
    For example:
    ax.xaxis.set_major_formatter(FormatStrFormatter(r'$%d$'))
    ax.yaxis.set_major_formatter(FormatStrFormatter(r'$%.1f$'))
    and set ftype=False
    """
    if "x" in xy.lower():
        labels = [item.get_text() for item in axis.get_xticklabels()]
        for i in range(len(labels)):
            if ftype:
                tmp = labels[i].split("{")[1].split("}")[0]
            else:
                tmp = labels[i].strip("$")
            newlbl = "$"+tmp+"$"
            try:
                ftmp = float(tmp)
                if ftmp == 0.0:
                    if zero: newlbl = "$+"+tmp+"$"
                elif ftmp > 0.0:
                    newlbl = "$+"+tmp+"$"
            except:
                pass
            labels[i] = newlbl
        axis.set_xticklabels(labels)
    if "y" in xy.lower():
        labels = [item.get_text() for item in axis.get_yticklabels()]
        for i in range(len(labels)):
            if ftype:
                tmp = labels[i].split("{")[1].split("}")[0]
            else:
                tmp = labels[i].strip("$")
            newlbl = "$"+tmp+"$"
            try:
                ftmp = float(tmp)
                if ftmp == 0.0:
                    if zero: newlbl = "$+"+tmp+"$"
                elif ftmp > 0.0:
                    newlbl = "$+"+tmp+"$"
            except:
                pass
            labels[i] = newlbl
        axis.set_yticklabels(labels)
    return labels
