"""Tests for lat/lon pair selection in detect_spatial_columns.

Regression coverage: columns named like a coordinate axis must win over
science/index columns that merely fall inside the coordinate value range
(e.g. Pangaea ``Temp``/``Sample ID``), while the iNaturalist datasets keep
resolving ``decimalLatitude``/``decimalLongitude``.
"""
import glob
import os
import sys
import unittest
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.context.context_factory import create_context
from src.tools import context_tools

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def _detect_pair(csv_path: str, resource_name: str = "file_1"):
    ctx = create_context({resource_name: csv_path}, name="pairing_test")
    key = context_tools.register_context(
        f"ctx_pairing_{uuid.uuid4().hex[:8]}", ctx
    )
    det = context_tools.detect_spatial_columns.invoke(
        {"context_key": key, "resource": resource_name}
    )
    return det


class TestSpatialColumnPairing(unittest.TestCase):
    def test_pangaea_prefers_named_lat_lon(self):
        csv = os.path.join(DATA_DIR, "experiment", "pangaea_datasets", "897882.csv")
        if not os.path.isfile(csv):
            self.skipTest(f"Pangaea fixture missing: {csv}")

        det = _detect_pair(csv)
        self.assertNotIn("error", det)
        self.assertEqual(
            det["detected_coordinate_pairs"],
            [{"latitude": "Latitude", "longitude": "Longitude"}],
        )

    def test_inaturalist_files_resolve_decimal_lat_lon(self):
        inat_dir = os.path.join(
            DATA_DIR,
            "experiment",
            "inaturalist_100_species_1000_obs",
            "inaturalist_100_species_1000_obs",
        )
        files = sorted(glob.glob(os.path.join(inat_dir, "*.csv")))
        if not files:
            self.skipTest(f"iNaturalist fixtures missing: {inat_dir}")

        expected = [{"latitude": "decimalLatitude", "longitude": "decimalLongitude"}]
        wrong = []
        for csv in files:
            det = _detect_pair(csv)
            if det.get("detected_coordinate_pairs") != expected:
                wrong.append((os.path.basename(csv), det.get("detected_coordinate_pairs")))

        self.assertEqual(wrong, [], f"Unexpected pairs: {wrong[:5]}")


if __name__ == "__main__":
    unittest.main()
