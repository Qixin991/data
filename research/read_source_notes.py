from pathlib import Path
from html.parser import HTMLParser
import sys
from pypdf import PdfReader
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.skip=0; self.parts=[]; self.links=[]
    def handle_starttag(self, tag, attrs):
        if tag in ('script','style'): self.skip += 1
        if tag == 'a':
            self.links.extend(v for k,v in attrs if k=='href' and v)
    def handle_endtag(self, tag):
        if tag in ('script','style'): self.skip=max(0,self.skip-1)
    def handle_data(self, data):
        if not self.skip and data.strip(): self.parts.append(data.strip())
p=TextParser()
p.feed((ROOT/'sources/baci_webpage.html').read_text(encoding='utf-8'))
(ROOT/'sources/baci_webpage.txt').write_text('\n'.join(p.parts),encoding='utf-8')
for i, s in enumerate(p.parts):
    if any(w in s.lower() for w in ('missing','zero','reporting','2024','preliminary')):
        print('\n'.join(p.parts[max(0,i-1):i+3]))
print('DATA_LINKS',[s for s in p.links if any(w in s.lower() for w in ('report','zero','csv','note','coverage'))])
pdf=PdfReader(ROOT/'sources/baci_release_notes_202601.pdf')
text='\n'.join(x.extract_text() or '' for x in pdf.pages)
(ROOT/'sources/baci_release_notes_202601.txt').write_text(text,encoding='utf-8')
print('RELEASE_NOTES',text[:5000])
