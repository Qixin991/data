"""Read-only source inspection; no sample selection, labels, or model fitting."""
from pathlib import Path
import importlib.util
import json
import zipfile
import xml.etree.ElementTree as ET
import sys

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'research' / '.deps'))
OUT = ROOT / 'research' / 'audit'
OUT.mkdir(parents=True, exist_ok=True)
print('DEPENDENCIES', {x: bool(importlib.util.find_spec(x)) for x in
      ['pypdf', 'pymupdf', 'pandas', 'xlrd', 'python_calamine', 'torch', 'sklearn']})
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math'}
for path in [ROOT / 'courses.docx', ROOT / 'paper/研究方案.docx',
             ROOT / 'paper/论文初稿_至模型部分.docx']:
    with zipfile.ZipFile(path) as z:
        tree = ET.fromstring(z.read('word/document.xml'))
    paragraphs = []
    for p in tree.findall('.//w:body//w:p', ns):
        value = ''.join(el.text or '' for el in p.iter()
                        if el.tag in {f"{{{ns['w']}}}t", f"{{{ns['m']}}}t"})
        if value:
            paragraphs.append(value)
    (OUT / (path.stem + '.txt')).write_text('\n'.join(paragraphs), encoding='utf-8')
    print('DOCX', path.name, 'paragraphs', len(paragraphs))

if importlib.util.find_spec('pypdf'):
    from pypdf import PdfReader
    for path in ROOT.glob('*.pdf'):
        pdf = PdfReader(path)
        pages = [p.extract_text() or '' for p in pdf.pages]
        (OUT / (path.stem + '.txt')).write_text(
            '\n'.join(f'PAGE {i+1}\n{p}' for i, p in enumerate(pages)), encoding='utf-8')
        print('PDF', path.name, 'pages', len(pages), 'opening', pages[0][:1300])

if importlib.util.find_spec('xlrd'):
    import xlrd
    for path in (ROOT / 'paper').glob('*.xls'):
        wb = xlrd.open_workbook(path)
        print('XLS', path.name, 'sheets', wb.sheet_names())
        for sh in wb.sheets():
            print('SHEET', sh.name, sh.nrows, sh.ncols)
            for row in range(min(6, sh.nrows)):
                print('ROW', row+1, sh.row_values(row)[:8])
            if sh.name == 'Data':
                print('YEAR_HEADERS', sh.row_values(3)[4:])
                print('INDICATORS', sorted({str(sh.cell_value(r, 3)) for r in range(4, sh.nrows)}))
                macro_codes = {str(sh.cell_value(r, 1)) for r in range(4, sh.nrows)}
                for code in ['CHN', 'USA', 'DEU', 'TWN', 'HKG']:
                    print('MACRO_CODE_EXISTS', code, code in macro_codes)

with zipfile.ZipFile(ROOT / 'paper/BACI_HS92_V202601.zip') as z:
    inventory = [{'name': e.filename, 'bytes': e.file_size,
                  'compressed_bytes': e.compress_size} for e in z.infolist()]
    (OUT / 'baci_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    print('BACI_README', z.read('Readme.txt').decode('utf-8-sig'))
    print('BACI_TOTAL_BYTES', sum(e.file_size for e in z.infolist()))
