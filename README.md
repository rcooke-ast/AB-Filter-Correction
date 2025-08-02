# AB-Filter-Correction
A lightweight package to iteratively calculate the systematic offsets and uncertainties for photometric survey data.

### Step 1:
Add your own survey photometry. The minimum that you need to list is the Gaia
source_id, as well as the corresponding photometry and uncertainty in the AB system.
There should only be one object per line, and the columns should be separated by commas.
Do not alter the CookeSuzukiProchaska2026.csv file unless you know what you're doing,
as this contains the photometry used by Cooke, Suzuki, & Prochaska (2026). If you wish
to add new photometry, create a new file with the same format as the example file:
example_photometry.csv.

### Step 1:
Make a copy of the example filter_input_example.csv file 
and rename it to filter_input.csv. This file contains the survey
names, their corresponding filter names, and the zero points for
each filter.

### Step 2:
Edit the filter_input.csv file to include your survey names,
filter names, and zero points, following the format of this
file. If you leave this file as is, you will use the same
filter information that was used by Cooke, Suzuki, & Prochaska (2026).

The order that you list the surveys in the filter_input.csv file is the same order they will be plotted.