# Site_Adjustment_Models
Parameterized models of systematic site effects from hybrid ground-motion simulations in New Zealand

## Running the site classification models

The code for calculating the site adjustment factors is found in `site_adjustment_models/models.py`. It requires `python>=3.11`, `pandas` and `numpy`. The implemented functions are extensively documented, and contain examples and test cases. The test cases test the functions against values from an independent implementation for the results providing the figures in the paper. You can run the tests from the root of this repository like so,

``` shell
$ python site_adjustment_models/test_models.py
......
----------------------------------------------------------------------
Ran 6 tests in 2.645s

OK
```

The tests also test the provided docstring examples. See the module level docstring in `models.py` for further details on each function.
