"""Stream and audit every raw BACI row; cache HS84/85 without building labels."""
from pathlib import Path
import hashlib
import io
import json
import sys
import time
import zipfile
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research' / 'data'
AUDIT = ROOT / 'research' / 'audit' / 'baci'
OUT.mkdir(parents=True, exist_ok=True)
AUDIT.mkdir(parents=True, exist_ok=True)
RAW = ROOT / 'paper' / 'BACI_HS92_V202601.zip'
SCRIPT_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def stop(year, kind, details):
    payload = {'year': year, 'kind': kind, 'details': details, 'status': 'STOP_REVIEW_REQUIRED'}
    write_json(AUDIT / 'STOP.json', payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    raise SystemExit(2)

with zipfile.ZipFile(RAW) as z:
    countries = pd.read_csv(io.BytesIO(z.read('country_codes_V202601.csv')), keep_default_na=False)
    products = pd.read_csv(io.BytesIO(z.read('product_codes_HS92_V202601.csv')), dtype={'code': str})
    countries.to_csv(OUT / 'countries.csv', index=False, encoding='utf-8-sig')
    products.to_csv(OUT / 'products_hs92.csv', index=False, encoding='utf-8-sig')
    valid_countries = set(countries.country_code.astype(int))
    valid_products = set(products.code)
    product_index = {code: idx for idx, code in enumerate(products.code)}
    summaries = []
    for year in range(1995, 2025):
        member = f'BACI_HS92_Y{year}_V202601.csv'
        info = z.getinfo(member)
        checkpoint = AUDIT / f'{year}.json'
        target = OUT / f'hs84_85_hs4_{year}.csv.gz'
        if checkpoint.exists() and target.exists():
            old = json.loads(checkpoint.read_text(encoding='utf-8'))
            if old.get('source_crc') == info.CRC and old.get('script_sha256') == SCRIPT_HASH:
                summaries.append(old)
                print('REUSE', year, flush=True)
                continue
        started = time.perf_counter()
        stats = {'year': year, 'source_crc': info.CRC, 'source_bytes': info.file_size,
                 'script_sha256': SCRIPT_HASH, 'rows': 0, 'value_kusd': 0.0,
                 'q_missing': 0, 'q_zero': 0, 'q_negative': 0, 'self_trade': 0}
        keys, aggregates, country_x, country_m = [], [], [], []
        seen_i, seen_j = set(), set()
        with z.open(member) as stream:
            for chunk in pd.read_csv(stream, chunksize=400000,
                    dtype={'t': 'int32', 'i': 'int32', 'j': 'int32', 'k': str,
                           'v': 'float64', 'q': 'float64'}, na_values=['NA']):
                invalid = (chunk.t != year) | (~np.isfinite(chunk.v)) | (chunk.v <= 0)
                if invalid.any():
                    stop(year, 'invalid_year_or_value', chunk.loc[invalid].head(10).to_dict('records'))
                unknown_i = set(chunk.i.unique()) - valid_countries
                unknown_j = set(chunk.j.unique()) - valid_countries
                unknown_k = set(chunk.k.unique()) - valid_products
                if unknown_i or unknown_j or unknown_k:
                    stop(year, 'unmapped_codes', {'exporter': sorted(map(int, unknown_i)),
                         'importer': sorted(map(int, unknown_j)), 'product': sorted(unknown_k)})
                q_bad = (~np.isfinite(chunk.q) & chunk.q.notna()) | (chunk.q < 0)
                if q_bad.any():
                    stop(year, 'invalid_weight', chunk.loc[q_bad].head(10).to_dict('records'))
                stats['rows'] += len(chunk)
                stats['value_kusd'] += float(chunk.v.sum())
                stats['q_missing'] += int(chunk.q.isna().sum())
                stats['q_zero'] += int((chunk.q == 0).sum())
                stats['self_trade'] += int((chunk.i == chunk.j).sum())
                seen_i.update(map(int, chunk.i.unique()))
                seen_j.update(map(int, chunk.j.unique()))
                key = (chunk.i.to_numpy(dtype='uint64') * 1000 + chunk.j.to_numpy(dtype='uint64')) * 1000000 + chunk.k.map(product_index).to_numpy(dtype='uint64')
                keys.append(key)
                country_x.append(chunk.groupby('i').agg(v=('v','sum'), rows=('v','size')))
                country_m.append(chunk.groupby('j').agg(v=('v','sum'), rows=('v','size')))
                subset = chunk.loc[chunk.k.str.startswith(('84','85'))].copy()
                subset['k4'] = subset.k.str[:4]
                subset['q_missing'] = subset.q.isna().astype('int32')
                aggregates.append(subset.groupby(['i','j','k4'], as_index=False).agg(
                    v=('v','sum'), q_known=('q','sum'), q_missing=('q_missing','sum'), hs6_rows=('v','size')))
        all_keys = np.concatenate(keys)
        del keys
        if len(all_keys) > 1 and np.any(all_keys[1:] <= all_keys[:-1]):
            unique, counts = np.unique(all_keys, return_counts=True)
            if (counts > 1).any():
                stop(year, 'duplicate_keys', {'extra_rows': int((counts-1).sum()),
                     'encoded_key_examples': unique[counts > 1][:10].tolist()})
        del all_keys
        agg = pd.concat(aggregates, ignore_index=True).groupby(['i','j','k4'], as_index=False).sum()
        del aggregates
        agg.insert(0, 't', year)
        agg['k4'] = agg.k4.astype(str).str.zfill(4)
        agg.to_csv(target, index=False, compression={'method':'gzip','compresslevel':1}, float_format='%.12g')
        for direction, frames, index in [('exports',country_x,'i'), ('imports',country_m,'j')]:
            totals = pd.concat(frames).groupby(level=0).sum().reset_index()
            totals.insert(0, 't', year)
            totals.to_csv(OUT / f'country_{direction}_{year}.csv', index=False)
        stats.update({'exporters': len(seen_i), 'importers': len(seen_j),
                      'hs84_85_hs4_rows': len(agg), 'hs84_85_value_kusd': float(agg.v.sum()),
                      'duplicate_keys': 0, 'crc_checked': True,
                      'seconds': round(time.perf_counter()-started,2)})
        write_json(checkpoint, stats)
        summaries.append(stats)
        pd.DataFrame(summaries).to_csv(AUDIT / 'annual_audit.csv', index=False)
        print(json.dumps(stats, ensure_ascii=False), flush=True)
    digest = hashlib.sha256()
    with RAW.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    write_json(AUDIT / 'manifest.json', {'source': str(RAW.relative_to(ROOT)),
        'sha256': digest.hexdigest(), 'size_bytes': RAW.stat().st_size,
        'script_sha256': SCRIPT_HASH, 'years_completed': len(summaries),
        'total_rows': sum(s['rows'] for s in summaries), 'scope': 'all products audited; HS84/85 cached; no labels'})
    print('COMPLETE', sum(s['rows'] for s in summaries), 'rows', flush=True)
