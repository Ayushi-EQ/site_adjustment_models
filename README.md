# Parametric Site Adjustment Models

This repository contains the accompanying script for calculating the site adjustment factors, Electronic Supplements B and G in csv formats, observed intensity measures (IMs), simulated IMs, and adjusted IMs datasets, raw residuals from Linear Mixed-Effects regression (MERA), residuals after applying the required site adjustment factors, data structures, scripts for figures in the paper and other supporting materials accompanying the paper:
“Parameterized models of systematic site effects from hybrid ground-motion simulations in New Zealand” (Manuscript submitted for publication).

# Repository Overview
The repository is organized into modular components reflecting the workflow used in the paper - IMs computation to residual calculations to model development and visualization.

# Folder descriptions
# 1. Adjustment_models_scripts/

Running the site classification models

##
The code for calculating the site adjustment factors is found in `site_adjustment_models/models.py`. It requires `python>=3.11`, `pandas` and `numpy`. The implemented functions are extensively documented, and contain examples and test cases. The test cases test the functions against values from an independent implementation for the results providing the figures in the paper. You can run the tests from the root of this repository like so,

``` shell
$ python site_adjustment_models/test_models.py
......
----------------------------------------------------------------------
Ran 6 tests in 2.645s

OK
```

The tests also test the provided docstring examples. See the module level docstring in `models.py` for further details on each function.

# 2. Electronic_Supplements 
## 
This folder provides the Electronic Supplements B and G of the paper in a csv format.

# 3. IMs
## 
Observed_IMs `site_adjustment_models/IMs/Observed_IMs.csv` -  pseudo-spectral acceleration(pSA) IMs and non-pSA IMs for observed IMs.
Raw_simulated_IMs `site_adjustment_models/IMs/Raw_simulated_IMs.csv` -  pseudo-spectral acceleration(pSA) IMs and non-pSA IMs for simulated IMs. These are the raw IMs on which site adjustments are applied.
Adjusted_simulated_IMs `site_adjustment_models/IMs/Adjusted_simulated_IMs.csv` -  pseudo-spectral acceleration(pSA) IMs and non-pSA IMs for simulated IMs. These are the adjustment IMs after applying the site adjustment factors as implemented in `site_adjustment_models/Adjustment_models_scripts/models.py` .
Other csv's like `events.csv`, `stations.csv`, and `gm.csv` are an accompaniment to these IMs. More details on the observed and simulated database can be found in Lee et al. (2022)

# 4. Residuals
##
This folder contains two subfolders, each subfolder containing residuals for `raw` and `adjusted` IMs. Each of them with a suffix where this could be raw/adjusted:
a. `bias_std_{suffix}.csv` :  Bias, standard deviation, and its components for 112 periods across all sites.
b. `event_residuals_{suffix}.csv` : Event residuals for all 479 events used.
c. `site_to_site_residuals_{suffix}.csv`: Site-to-site residuals for all 212 stations used.
d. `error_site_to_site_residuals_{suffix}.csv`: Uncertainty in the point estimate of the site-to-site residuals for all 212 stations used.
e. `remaining_residuals_{suffix}.csv`: "remaining" within-event residuals for a given earthquake and station

# 5. Scripts_for_figures
Contains the plotting and figure-generation scripts used to produce all visualizations presented in the paper.
Note, that these scripts are not clean/comprehensive and are for demonstration and appropriate data files need to be located in the repo (names might be different) and some might still be missing. Several other details not required for the figure generation are also there.