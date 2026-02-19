import copy
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app import (
    LocalFileStorageAdapter,
    TREND_CONFIG,
    VercelBlobStorageAdapter,
    app as flask_app,
    create_storage_adapter,
    get_data_status_for_trend,
    get_required_min_period,
    storage_backend,
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


class FakeHTTPResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class VercelBlobAdapterTests(unittest.TestCase):
    def test_write_path_mismatch_raises(self):
        cfg = copy.deepcopy(TREND_CONFIG)
        with patch.dict(os.environ, {'BLOB_READ_WRITE_TOKEN': 'token-123'}, clear=False):
            adapter = VercelBlobStorageAdapter(cfg)

        payload = json.dumps({
            'url': 'https://example.com/blob/random.csv',
            'pathname': 'random.csv',
        }).encode('utf-8')
        with patch('app.urllib_request.urlopen', return_value=FakeHTTPResponse(payload)):
            with self.assertRaises(RuntimeError):
                adapter.write_trend_csv('collection', b'date,collection\n2026-01-01,10\n')


class StorageConfigTests(unittest.TestCase):
    def test_create_storage_adapter_blob_requires_token(self):
        cfg = copy.deepcopy(TREND_CONFIG)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                create_storage_adapter('vercel_blob', cfg)


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


class ApiBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.client = flask_app.test_client()
        self._collection_context = copy.deepcopy(trend_store['collection'])

    def tearDown(self):
        trend_store['collection'] = self._collection_context

    def test_live_trend_error_includes_trend_and_backend_when_data_missing(self):
        with self.client.session_transaction() as sess:
            sess['selected_trend'] = 'collection'
            sess['selected_segment'] = '__all__'

        trend_store['collection']['error'] = 'Data file not found'
        trend_store['collection']['segments'] = {
            '__all__': {
                'error': 'Data file not found',
                'legacy_df': None,
                'monthly_totals': None,
            }
        }
        trend_store['collection']['segment_labels'] = {'__all__': 'Collection'}

        payload = {
            'trend_month': '2026-01',
            'day_to_trend': 1,
            'current_mtd': 1,
            'n_months_to_use': 6,
        }
        with patch('app.sync_trend_data', return_value=None):
            response = self.client.post('/api/live_trend', json=payload)

        self.assertEqual(response.status_code, 400)
        error_msg = response.get_json().get('error', '')
        self.assertIn('No data found for this trend', error_msg)
        self.assertIn('trend=collection', error_msg)
        self.assertIn(f'backend={storage_backend}', error_msg)

    def test_storage_health_returns_diagnostics(self):
        with self.client.session_transaction() as sess:
            sess['selected_trend'] = 'collection'
            sess['selected_segment'] = '__all__'

        trend_store['collection']['error'] = None
        trend_store['collection']['segments'] = {
            '__all__': {
                'latest_period': '2026-01',
                'error': None,
            }
        }
        trend_store['collection']['segment_labels'] = {'__all__': 'Collection'}
        trend_store['collection']['storage_observability'] = {
            'last_read_at': '2026-02-19T10:00:00+00:00',
            'last_read_status': 'found',
            'last_read_bytes': 128,
            'last_upload_at': '2026-02-19T10:01:00+00:00',
            'last_upload_bytes': 128,
            'last_error': None,
        }

        with patch('app.sync_trend_data', return_value=None):
            response = self.client.get('/api/storage_health?trend=collection')

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['backend'], storage_backend)
        self.assertEqual(payload['trend'], 'collection')
        self.assertEqual(payload['selected_trend'], 'collection')
        self.assertEqual(payload['read_status'], 'found')
        self.assertEqual(payload['latest_period'], '2026-01')
        self.assertIn('storage_target', payload)

if __name__ == '__main__':
    unittest.main()
