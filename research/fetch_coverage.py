"""Retrieve public official documentation and a Comtrade availability probe."""
from pathlib import Path
from urllib.request import Request, urlopen
from datetime import datetime, timezone
import hashlib
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parent / 'sources'
OUT.mkdir(exist_ok=True)
sources = {
 'baci_webpage.html': 'https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/baci_webpage.html',
 'baci_release_notes_202601.pdf': 'https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/release_notes_202601.pdf',
 'comtrade_availability_2024_probe.json': 'https://comtradeapi.un.org/public/v1/getDA/C/A/HS?period=2024&reporterCode=156',
}
for year in range(1995,2025):
    sources[f'comtrade_availability_{year}.json']=f'https://comtradeapi.un.org/public/v1/getDA/C/A/HS?period={year}'
manifest = []
for name, url in sources.items():
    try:
        if (OUT/name).exists():
            body=(OUT/name).read_bytes(); status='cached'
        else:
            with urlopen(Request(url, headers={'User-Agent':'TradeResearch/1.0'}), timeout=25) as response:
                body = response.read()
                status = response.status
            (OUT / name).write_bytes(body)
        item = {'file':name,'url':url,'utc':datetime.now(timezone.utc).isoformat(),
                'http_status':status,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}
        print(name, len(body), body[:180] if name.endswith('.json') else '', flush=True)
    except Exception as e:
        item = {'file':name,'url':url,'error':str(e)}
        print('FETCH_ERROR', name, str(e), flush=True)
    manifest.append(item)
(OUT / 'coverage_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
