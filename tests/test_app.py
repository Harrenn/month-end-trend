import copy
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib import error as urllib_error

import pandas as pd

from app import (
    LocalFileStorageAdapter,
    TREND_CONFIG,
    VercelBlobStorageAdapter,
    app as flask_app,
    create_storage_adapter,
    get_data_status_for_trend,
    get_required_min_period,
    load_and_prepare_data,
    normalize_uploaded_table_bytes,
    read_tabular_dataframe,
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


def build_http_error(url, status_code, body=''):
    return urllib_error.HTTPError(
        url=url,
        code=status_code,
        msg='error',
        hdrs=None,
        fp=BytesIO(body.encode('utf-8')),
    )


class VercelBlobAdapterTests(unittest.TestCase):
    def test_write_path_mismatch_raises(self):
        cfg = copy.deepcopy(TREND_CONFIG)
        with patch.dict(os.environ, {'BLOB_READ_WRITE_TOKEN': 'token-123'}, clear=False):
            adapter = VercelBlobStorageAdapter(cfg)

        payload = json.dumps({
            'url': 'https://example.com/blob/collection/collection_data-abc.csv',
            'pathname': 'collection/collection_data-abc.csv',
        }).encode('utf-8')
        with patch('app.urllib_request.urlopen', return_value=FakeHTTPResponse(payload)):
            with self.assertRaises(RuntimeError):
                adapter.write_trend_csv('collection', b'date,collection\n2026-01-01,10\n')

    def test_write_path_exact_match_succeeds(self):
        cfg = copy.deepcopy(TREND_CONFIG)
        with patch.dict(os.environ, {'BLOB_READ_WRITE_TOKEN': 'token-123'}, clear=False):
            adapter = VercelBlobStorageAdapter(cfg)

        payload = json.dumps({
            'url': 'https://example.com/blob/collection/collection_data.csv',
            'pathname': 'collection/collection_data.csv',
        }).encode('utf-8')
        with patch('app.urllib_request.urlopen', return_value=FakeHTTPResponse(payload)) as mock_urlopen:
            adapter.write_trend_csv('collection', b'date,collection\n2026-01-01,10\n')

        req = mock_urlopen.call_args[0][0]
        self.assertTrue(req.full_url.startswith('https://blob.vercel-storage.com/collection/collection_data.csv'))
        target = adapter.storage_target('collection')
        self.assertEqual(target['expected_path'], 'collection/collection_data.csv')
        self.assertEqual(target['last_uploaded_pathname'], 'collection/collection_data.csv')

    def test_read_404_returns_none_without_fallback(self):
        cfg = copy.deepcopy(TREND_CONFIG)
        with patch.dict(os.environ, {'BLOB_READ_WRITE_TOKEN': 'token-123'}, clear=False):
            adapter = VercelBlobStorageAdapter(cfg)

        adapter._last_uploaded_urls['collection'] = 'https://example.com/blob/collection/collection_data-fallback.csv'
        list_payload = json.dumps({'blobs': []}).encode('utf-8')
        with patch(
            'app.urllib_request.urlopen',
            return_value=FakeHTTPResponse(list_payload),
        ) as mock_urlopen:
            self.assertIsNone(adapter.read_trend_csv('collection'))

        # Should rely on canonical path resolution only (no in-memory URL fallback reads).
        self.assertEqual(mock_urlopen.call_count, 1)

    def test_read_resolves_blob_url_from_pathname(self):
        cfg = copy.deepcopy(TREND_CONFIG)
        with patch.dict(os.environ, {'BLOB_READ_WRITE_TOKEN': 'token-123'}, clear=False):
            adapter = VercelBlobStorageAdapter(cfg)

        list_payload = json.dumps({
            'blobs': [
                {
                    'pathname': 'collection/collection_data-older.csv',
                    'url': 'https://store.public.blob.vercel-storage.com/collection/collection_data-older.csv',
                    'uploadedAt': '2026-02-19T00:00:00.000Z',
                },
                {
                    'pathname': 'collection/collection_data-newer.csv',
                    'url': 'https://store.public.blob.vercel-storage.com/collection/collection_data-newer.csv',
                    'uploadedAt': '2026-02-19T12:00:00.000Z',
                }
            ]
        }).encode('utf-8')
        file_payload = b'date,collection\n2026-01-01,10\n'

        with patch(
            'app.urllib_request.urlopen',
            side_effect=[FakeHTTPResponse(list_payload), FakeHTTPResponse(file_payload)],
        ) as mock_urlopen:
            payload = adapter.read_trend_csv('collection')

        self.assertEqual(payload, file_payload)
        list_req = mock_urlopen.call_args_list[0][0][0]
        read_req = mock_urlopen.call_args_list[1][0][0]
        self.assertIn('prefix=collection%2Fcollection_data', list_req.full_url)
        self.assertEqual(
            read_req.full_url,
            'https://store.public.blob.vercel-storage.com/collection/collection_data-newer.csv',
        )


class StorageConfigTests(unittest.TestCase):
    def test_create_storage_adapter_blob_requires_token(self):
        cfg = copy.deepcopy(TREND_CONFIG)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                create_storage_adapter('vercel_blob', cfg)

    def test_create_storage_adapter_blob_requires_base_url_in_production(self):
        cfg = copy.deepcopy(TREND_CONFIG)
        with patch.dict(
            os.environ,
            {'BLOB_READ_WRITE_TOKEN': 'token-123'},
            clear=True,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                create_storage_adapter('vercel_blob', cfg, deployment_env='production')
        self.assertIn('VERCEL_BLOB_BASE_URL is missing', str(ctx.exception))

    def test_create_storage_adapter_blob_allows_missing_base_url_outside_production(self):
        cfg = copy.deepcopy(TREND_CONFIG)
        with patch.dict(
            os.environ,
            {'BLOB_READ_WRITE_TOKEN': 'token-123'},
            clear=True,
        ):
            backend, adapter = create_storage_adapter('vercel_blob', cfg, deployment_env='preview')
        self.assertEqual(backend, 'vercel_blob')
        self.assertIsInstance(adapter, VercelBlobStorageAdapter)


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


class CsvParsingTests(unittest.TestCase):
    def test_load_and_prepare_data_accepts_bom_date_header(self):
        csv_bytes = '\ufeffdate,collection\n2025-12-31,10\n'.encode('utf-8')
        overall_df, segment_frames = load_and_prepare_data(csv_bytes, 'collection')

        self.assertIsNotNone(overall_df)
        self.assertEqual(len(overall_df), 1)
        self.assertIn('collection', overall_df.columns)
        self.assertEqual(segment_frames, {})

    def test_read_tabular_dataframe_supports_semicolon_csv(self):
        csv_bytes = 'date;collection\n2025-12-31;10\n'.encode('utf-8')
        df = read_tabular_dataframe(csv_bytes, filename='sample.csv')
        self.assertEqual(list(df.columns), ['date', 'collection'])
        self.assertEqual(float(df.iloc[0]['collection']), 10.0)

    def test_normalize_uploaded_table_bytes_supports_excel(self):
        source_df = pd.DataFrame({
            'date': ['2025-12-31'],
            'collection': [10],
        })
        excel_buffer = BytesIO()
        source_df.to_excel(excel_buffer, index=False)
        excel_bytes = excel_buffer.getvalue()

        normalized_csv = normalize_uploaded_table_bytes(excel_bytes, filename='sample.xlsx')
        parsed_df = read_tabular_dataframe(normalized_csv, filename='normalized.csv')

        self.assertEqual(list(parsed_df.columns), ['date', 'collection'])
        self.assertEqual(float(parsed_df.iloc[0]['collection']), 10.0)

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
        with patch('app.sync_trend_data', return_value=None), patch('app.storage_backend', 'vercel_blob'):
            response = self.client.post('/api/live_trend', json=payload)

        self.assertEqual(response.status_code, 400)
        error_msg = response.get_json().get('error', '')
        self.assertIn('No data found for this trend', error_msg)
        self.assertIn('trend=collection', error_msg)
        self.assertIn('backend=vercel_blob', error_msg)

    def test_data_status_error_includes_trend_and_backend_when_data_missing(self):
        with self.client.session_transaction() as sess:
            sess['selected_trend'] = 'collection'
            sess['selected_segment'] = '__all__'

        trend_store['collection']['error'] = 'Data file not found'
        trend_store['collection']['segments'] = {
            '__all__': {
                'latest_period': None,
                'error': 'Data file not found',
            }
        }
        trend_store['collection']['segment_labels'] = {'__all__': 'Collection'}

        with patch('app.sync_trend_data', return_value=None), patch('app.storage_backend', 'vercel_blob'):
            response = self.client.get('/api/data_status')

        self.assertEqual(response.status_code, 200)
        error_msg = response.get_json().get('error', '')
        self.assertIn('No data found for this trend', error_msg)
        self.assertIn('trend=collection', error_msg)
        self.assertIn('backend=vercel_blob', error_msg)

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

        storage_target = {
            'type': 'vercel_blob',
            'expected_path': 'collection/collection_data.csv',
            'expected_url': 'https://store-id.public.blob.vercel-storage.com/collection/collection_data.csv',
            'last_uploaded_pathname': 'collection/collection_data.csv',
            'last_uploaded_url': 'https://store-id.public.blob.vercel-storage.com/collection/collection_data.csv',
        }

        with (
            patch('app.sync_trend_data', return_value=None),
            patch('app.storage_backend', 'vercel_blob'),
            patch('app.storage_adapter') as mock_storage_adapter,
        ):
            mock_storage_adapter.storage_target.return_value = storage_target
            response = self.client.get('/api/storage_health?trend=collection')

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['backend'], 'vercel_blob')
        self.assertEqual(payload['trend'], 'collection')
        self.assertEqual(payload['selected_trend'], 'collection')
        self.assertEqual(payload['read_status'], 'found')
        self.assertEqual(payload['latest_period'], '2026-01')
        self.assertIn('storage_target', payload)
        self.assertEqual(payload['storage_target']['expected_path'], 'collection/collection_data.csv')
        self.assertEqual(
            payload['storage_target']['expected_url'],
            'https://store-id.public.blob.vercel-storage.com/collection/collection_data.csv',
        )

    def test_upload_accepts_excel_and_writes_normalized_csv_bytes(self):
        with self.client.session_transaction() as sess:
            sess['selected_trend'] = 'collection'
            sess['selected_segment'] = '__all__'

        normalized_csv_bytes = b'date,collection\n2025-12-31,10\n'
        upload_form = {
            'file': (BytesIO(b'fake-excel-bytes'), 'collection.xlsx'),
        }
        with (
            patch('app.normalize_uploaded_table_bytes', return_value=normalized_csv_bytes) as mock_normalize,
            patch('app.refresh_trend_data', return_value=None),
            patch('app.get_trend_context', return_value={'error': None}),
            patch('app.get_selected_segment_key', return_value='__all__'),
            patch('app.storage_adapter.write_trend_csv') as mock_write,
        ):
            response = self.client.post(
                '/api/upload',
                data=upload_form,
                content_type='multipart/form-data',
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn('data updated successfully', response.get_json().get('message', '').lower())
        self.assertEqual(mock_normalize.call_count, 1)
        self.assertEqual(mock_normalize.call_args.kwargs.get('filename'), 'collection.xlsx')
        self.assertEqual(mock_write.call_count, 1)
        self.assertEqual(mock_write.call_args[0][0], 'collection')
        self.assertEqual(mock_write.call_args[0][1], normalized_csv_bytes)

    def test_upload_returns_503_when_blob_unavailable(self):
        with self.client.session_transaction() as sess:
            sess['selected_trend'] = 'collection'
            sess['selected_segment'] = '__all__'

        upload_form = {
            'file': (BytesIO(b'date,collection\n2025-12-31,10\n'), 'collection.csv'),
        }
        with (
            patch('app.normalize_uploaded_table_bytes', return_value=b'date,collection\n2025-12-31,10\n'),
            patch(
                'app.storage_adapter.write_trend_csv',
                side_effect=build_http_error(
                    'https://blob.vercel-storage.com/collection/collection_data.csv',
                    503,
                ),
            ),
        ):
            response = self.client.post(
                '/api/upload',
                data=upload_form,
                content_type='multipart/form-data',
            )

        self.assertEqual(response.status_code, 503)
        error_msg = response.get_json().get('error', '')
        self.assertIn('Storage is temporarily unavailable', error_msg)
        self.assertIn('trend=collection', error_msg)
        self.assertIn('backend=', error_msg)

if __name__ == '__main__':
    unittest.main()
