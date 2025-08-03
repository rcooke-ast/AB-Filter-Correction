import os

from astropy.table import Table

import IterFit
import MCMC_FilterDiff
import PlotFilterOffsets
import CalculateBlackbodyParamsMCMC


if __name__ == "__main__":
    outdirc = "Outputs/"
    prefix = "csp"
    plotit = False
    cleanold = False

    # Remove old output files if requested
    if cleanold:
        os.system(f"rm {outdirc}{prefix}_*")

    # Load the filter input table
    filttab = Table.read("filter_input.csv", format='ascii.csv', comment="#")

    # Run the filter difference calculation
    # IterFit.run_iterfit(outdirc, prefix, filttab, plotit=plotit)

    # Run the MCMC filter difference calculation
    # MCMC_FilterDiff.run_mcmc_filtdiff(outdirc, prefix)

    # Plot the filter offsets and prepare a latex table
    # PlotFilterOffsets.prepare_table(outdirc, prefix, filttab)

    # Calculate the final blackbody parameters
    CalculateBlackbodyParamsMCMC.run_blackbody_params_mcmc(outdirc, prefix, filttab, plotit=plotit, rerun=True)

    print("All calculations completed successfully.")