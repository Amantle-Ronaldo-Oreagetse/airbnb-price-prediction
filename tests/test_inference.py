import sys
import unittest

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_DIR / "src"))

from importlib import import_module

inference = import_module("10_predict_listing_price")

load_city_model = inference.load_city_model
predict_listing_price = inference.predict_listing_price


class TestAirbnbPredictionSystem(unittest.TestCase):

    def setUp(self):

        self.valid_listing = pd.DataFrame([

            {
                "accommodates": 2,
                "bedrooms": 1,
                "bedrooms_missing": 0,
                "minimum_nights": 2,
                "maximum_nights": 30,
                "amenities_count": 20,
                "latitude": 48.8566,
                "longitude": 2.3522,
                "property_type": "Entire rental unit",
                "room_type": "Entire home/apt",
                "neighbourhood": "Louvre",
                "instant_bookable": "f"
            }

        ])

    def test_all_city_models_load(self):

        for city in inference.CITY_CURRENCIES:

            with self.subTest(city=city):

                model = load_city_model(city)

                self.assertIsNotNone(model)

    def test_valid_prediction(self):

        result = predict_listing_price(
            "Paris",
            self.valid_listing
        )

        self.assertEqual(len(result), 1)

        self.assertTrue(
            pd.notna(result["predicted_price"].iloc[0])
        )

        self.assertGreaterEqual(
            result["predicted_price"].iloc[0],
            0
        )

        self.assertEqual(
            result["currency"].iloc[0],
            "EUR"
        )

    def test_invalid_city(self):

        with self.assertRaises(ValueError):

            predict_listing_price(
                "London",
                self.valid_listing
            )

    def test_missing_features(self):

        incomplete_listing = self.valid_listing.drop(
            columns=["bedrooms"]
        )

        with self.assertRaises(ValueError):

            predict_listing_price(
                "Paris",
                incomplete_listing
            )

    def test_negative_accommodates(self):

        invalid_listing = self.valid_listing.copy()

        invalid_listing.loc[0, "accommodates"] = -2

        with self.assertRaises(ValueError):

            predict_listing_price(
                "Paris",
                invalid_listing
            )


if __name__ == "__main__":

    unittest.main(verbosity=2)