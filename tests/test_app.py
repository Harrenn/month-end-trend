import copy
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from app import (
    LocalFileStorageAdapter,
    TREND_CONFIG,
    get_data_status_for_trend,
    get_required_min_period,
    trend_store,
)


class LocalStorageAdapterTests(unittest.TestCase):
    def test_write_then_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = copy.deepcopy(TREND_CONFIG)
            target_path = Path(temp_dir) / 'collection' / 'collection_data.csv'
            cfg['collection']['file_path'] = str(target_path)

            adapter = LocalFileStorageAdapter(cfg)
            payload = b"date,collection\n2026-01-01,10\n"
            adapter.write_trend_csv('collection', payload)

            self.assertEqual(adapter.read_trend_csv('collection'), payload)

    def test_read_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = copy.deepcopy(TREND_CONFIG)
            target_path = Path(temp_dir) / 'releases' / 'releases_data.csv'
            cfg['releases']['file_path'] = str(target_path)

            adapter = LocalFileStorageAdapter(cfg)
            self.assertIsNone(adapter.read_trend_csv('releases'))


class DataFreshnessTests(unittest.TestCase):
    def test_required_min_period_previous_month(self):
        now = datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(str(get_required_min_period(now)), '2026-01')

    def test_stale_when_latest_older_than_previous_month(self):
        original_segments = trend_store['collection']['segments']
        try:
            trend_store['collection']['segments'] = {'__all__': {'latest_period': '2025-12'}}
            now = datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)
            status = get_data_status_for_trend('collection', now)
            self.assertTrue(status['is_stale'])
            self.assertEqual(status['required_min_period'], '2026-01')
        finally:
            trend_store['collection']['segments'] = original_segments

    def test_fresh_when_latest_is_previous_month(self):
        original_segments = trend_store['collection']['segments']
        try:
            trend_store['collection']['segments'] = {'__all__': {'latest_period': '2026-01'}}
            now = datetime(2026, 2, 16, 12, 0, tzinfo=timezone.utc)
            status = get_data_status_for_trend('collection', now)
            self.assertFalse(status['is_stale'])
            self.assertEqual(status['required_min_period'], '2026-01')
        finally:
            trend_store['collection']['segments'] = original_segments


if __name__ == '__main__':
    unittest.main()
