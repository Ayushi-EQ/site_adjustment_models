import doctest
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import models

ADJUSTMENT_FACTORS = pd.read_csv( Path(__file__).parent / "data" / "Adjfactor.csv").set_index("stat_id")
SITES = pd.read_csv(Path(__file__).parent / "data" / "sites.csv").set_index( "stat_id")


class TestSiteAdjustmentModels(unittest.TestCase):
    def test_site_adjustment_models(self):
        periods = ADJUSTMENT_FACTORS.columns.values

        for stat_id, station in SITES.iterrows():
            site_class = None
            geomorphology = station["Geomorphology"]
            h_1250 = station["H1250"]
            t0 = station["T0"]
            basin_type = station["Basin Type"]
            if geomorphology == "Hill":
                site_class = models.SiteClass.HILL
            elif basin_type == "Unmodeled Basin":
                site_class = models.SiteClass.UNMODELED_BASIN
            elif geomorphology == "Basin Edge":
                site_class = models.SiteClass.BASIN_EDGE
            elif geomorphology == "Basin":
                site_class = models.SiteClass.BASIN
            else: # ONLY VALLEY REMAINING
                site_class = models.SiteClass.VALLEY

            for period in periods:
                correct_adjustment = ADJUSTMENT_FACTORS.loc[stat_id, period]
                with self.subTest(
                    station=station["stat_name"],
                    period=float(period),
                    site_class=site_class,
                    t0=t0,
                    h_1250=h_1250,
                ):
                    site_adjustment_factor = models.site_adjustment(
                        site_class,
                        float(period),
                        t0=t0,
                        h_1250=h_1250,
                    )
                    err_message = f"Expected {correct_adjustment:.5e} but received {site_adjustment_factor:.5e}, difference of {correct_adjustment - site_adjustment_factor:.5e}."
                    self.assertTrue(
                        np.isclose(
                            site_adjustment_factor, correct_adjustment, atol=1e-7
                        ),
                        msg=err_message,
                    )


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSiteAdjustmentModels)
    suite.addTests(doctest.DocTestSuite(models))
    runner = unittest.TextTestRunner()
    runner.run(suite)
