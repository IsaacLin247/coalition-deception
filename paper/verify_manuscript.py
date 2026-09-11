#!/usr/bin/env python3
"""Check synchronized manuscript sources, result-table identity, citations and PDF bounds."""
import argparse, hashlib, json, re, subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

P=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path);args=ap.parse_args()
    md=(P/"when_credibility_collapses.md").read_text()
    tex=(P/"when_credibility_collapses.tex").read_text()
    values=json.loads((P/"data/final_table_sources.json").read_text())
    for name,table in values["generated_tables"].items():
        assert table["markdown"] in md, f"Table {name} changed from its validated source"
    figure_names=re.findall(r'^Figure (\d+)\.',md,re.M)
    table_names=re.findall(r'^Table (\d+|B1)\.',md,re.M)
    assert figure_names==list(map(str,range(1,20)))
    assert table_names==list(map(str,range(1,18)))+["B1"]
    assert tex.count(r"\begin{figure}")==19
    assert tex.count(r"\caption{")==37
    assert re.search(r"\\documentclass\[\s*12pt,",tex)
    abstract=md.split("abstract: |",1)[1].split("\n---",1)[0]
    assert len(abstract.split())<=300
    assert not re.search(r"archived measurements|remain under replication|inference.*pending|still running",md,re.I)
    images=re.findall(r"!\[[^\]]*\]\(([^)]+)\)",md)
    assert len(images)==19 and all((P/p).is_file() for p in images)
    citations=set(re.findall(r"@([\w-]+)",md))
    bib=set(re.findall(r"@\w+\s*\{\s*([^,\s]+)",(P/"references.bib").read_text()))
    assert citations <= bib, citations-bib
    assert len(re.findall(r"^\\CSLLeftMargin",tex,re.M))==len(citations)
    pdf=P/"when_credibility_collapses.pdf"
    bbox=subprocess.check_output(["pdftotext","-bbox-layout",str(pdf),"-"])
    ns={"x":"http://www.w3.org/1999/xhtml"}
    pages=ET.fromstring(bbox).findall(".//x:page",ns)
    info=subprocess.check_output(["pdfinfo","-f","1","-l",str(len(pages)),str(pdf)],text=True)
    rotations={int(n):int(r) for n,r in re.findall(r"Page\s+(\d+) rot:\s+(\d+)",info)}
    outside=[]
    for number,page in enumerate(pages,1):
        width,height=(float(page.attrib[k]) for k in ("width","height"))
        if rotations.get(number,0)%180:width,height=height,width
        for word in page.findall(".//x:word",ns):
            b=[float(word.attrib[k]) for k in ("xMin","yMin","xMax","yMax")]
            if word.text!=str(number) and (b[0]<69 or b[1]<69 or b[2]>width-69 or b[3]>height-69):
                outside.append({"page":number,"text":word.text,"bounds":b})
    assert not outside,outside[:20]
    text=subprocess.check_output(["pdftotext","-layout",str(pdf),"-"],text=True)
    seen={}
    for i,page in enumerate(text.split("\f"),1):
        for label in re.findall(r"^\s*((?:Figure|Table) (?:\d+|B1))\.\s+",page,re.M):
            seen.setdefault(label,i)
    assert len(seen)==37,seen
    paths=[P/f"when_credibility_collapses.{x}" for x in ["md","tex","pdf"]]
    paths += [P/x for x in ["build.sh","numbered_captions.lua","caption_layout.tex","references.bib","ieee.csl","data/final_table_sources.json","data/custom_figure_sources.json","plot_final_controls.py","verify_manuscript.py"]]
    paths += [P/x for x in images]
    report={"status":"passed","pages":len(pages),"figures":19,"tables":18,
            "generated_tables_verified":len(values["generated_tables"]),
            "selected_source_rows":len(values["selected_rows"]),"abstract_words":len(abstract.split()),
            "citation_keys":sorted(citations),"caption_pages":seen,"words_outside_margins":outside,
            "files":{str(p.relative_to(P)):sha(p) for p in paths}}
    if args.output:args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ["files","caption_pages"]},indent=2))
if __name__=="__main__":main()
