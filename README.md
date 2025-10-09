# Parametric Site Adjustment Models

This repository contains the scripts, data, and supporting materials accompanying the paper:
“Parameterized models of systematic site effects from hybrid ground-motion simulations in New Zealand” (manuscript submitted for publication).
- Scripts for calculating site adjustment factors

- Electronic Supplements B and G in CSV format

- Observed intensity measures (IMs)

- Simulated IMs and adjusted IMs datasets

- Raw residuals from Linear Mixed-Effects regression (MERA)

- Residuals after applying site adjustment factors

- Scripts for figure generation

# Repository Overview
The repository is organized into modular components reflecting the workflow used in the paper from IM computation to residual calculation, parametric model development, and visualization.

# Folder descriptions
# 1. Adjustment_models_scripts/

This folder contains scripts for calculating the site adjustment factors.

##
- `models.py`:  The code for calculating the site adjustment factors. It requires `python>=3.11`, `pandas` and `numpy`. The implemented functions are extensively documented, and contain examples and test cases.
-  `test_models.py`: Test cases are provided to test the functions against values from an independent implementation for the results providing the figures in the paper. You can run the tests from the root of this repository like so,

``` shell
$ python site_adjustment_models/test_models.py
......
----------------------------------------------------------------------
Ran 6 tests in 2.645s

OK
```

The tests also test the provided docstring examples. See the module level docstring in `models.py` for further details on each function.

# 2. Electronic_Supplements/
## 
Contains Electronic Supplements B and G of the paper in CSV format.

# 3. IMs/
## 
Contains observed and simulated intensity measures (IMs) used in the analyses:

- Observed_IMs `site_adjustment_models/IMs/Observed_IMs.csv` -  pseudo-spectral acceleration (pSA) IMs and non-pSA IMs for observed ground motions.
- Raw_simulated_IMs `site_adjustment_models/IMs/Raw_simulated_IMs.csv` -  pSA and non-pSA IMs for simulated ground motions. These are the raw IMs on which site adjustments are applied.
- Adjusted_simulated_IMs `site_adjustment_models/IMs/Adjusted_simulated_IMs.csv` - Adjusted pSA and non-pSA IMs for simulated ground motions. These are the adjustment IMs after applying the site adjustment factors as implemented in `site_adjustment_models/Adjustment_models_scripts/models.py` .
- Other csv's  (`events.csv`, `stations.csv`, and `gm.csv`) : Metadata accompanying the IM datasets. 

More details on the observed and simulated database can be found in Lee et al. (2022)

# 4. Residuals/
##
This folder contains two subfolders, each subfolder containing residuals for `raw` and `adjusted` IMs. Each file uses a suffix indicating whether it is raw or adjusted:
- `bias_std_{suffix}.csv` :  Bias, standard deviation, and its components for 112 periods across all sites.
- `event_residuals_{suffix}.csv` : Event residuals for all 479 events used.
- `site_to_site_residuals_{suffix}.csv`: Site-to-site residuals for all 212 stations used.
- `error_site_to_site_residuals_{suffix}.csv`: Uncertainty in the point estimate of the site-to-site residuals for all 212 stations used.
- `remaining_residuals_{suffix}.csv`: "remaining" within-event residuals for a given earthquake and station

# 5. Scripts_for_figures/
Contains scripts used to generate the figures presented in the paper.

Note: These scripts are for demonstration and may require adjustments:

- Filenames of input data may differ.

- Some scripts may be incomplete or contain extra code unrelated to figure generation.

- Ensure required data files are present in the repository before running the scripts.