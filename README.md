# AB-Filter-Correction
A lightweight package to iteratively calculate the systematic offsets and uncertainties for photometric survey data.

### Step 1:
Add your own survey photometry. The minimum that you need to list is the Gaia
source_id (this must be one of the blackbody stars in the Cooke, Suzuki, Prochaska 2026
sample), as well as the corresponding photometry and uncertainty in the AB system.
There should only be one object per line, and the columns should be separated by commas.

_NOTE:_ Do not alter the `CookeSuzukiProchaska2026.csv` file unless you know what you're doing,
as this contains the photometry used by Cooke, Suzuki, & Prochaska (2026). If you wish
to add new photometry, create a new file with the same format as the example file:
`example_photometry.csv`.

### Step 2:
Make a copy of the example `filter_input_example.csv` file 
and rename it to `filter_input.csv`. This file contains the survey
names, their corresponding filter names, the file containing the
photometry, and how to read the photometry.

### Step 3:
Edit the `filter_input.csv` file to include your survey names,
filter names, etc. following the format of this file. If you
leave this file as is, you will use the same filter information
that was used by Cooke, Suzuki, & Prochaska (2026).

The order that you list the surveys in the `filter_input.csv` file
is the same order they will be printed in the output table and
shown on the output plot.

### Step 4:
Edit the file `filter_output.csv` to include the filters you want to
calculate the synthetic photometry for. The format of this file is
similar to the `filter_input.csv` file.

### Step 5:
If you want to change the output directory, or the prefix used, change
these values in the `main.py` file. The default output directory is
`Outputs/` and the default prefix is `csp`.

### Step 6:
Run the code in the `main.py` file (i.e. `python main.py`). This will
read the `filter_input.csv` file and the photometry files, and then
calculate the systematic offsets and systematic uncertainties for each
filter in each survey.
