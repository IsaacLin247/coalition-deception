#!/usr/bin/env python3
"""Check the shorter manuscript against preserved evidence and rendered output."""
import argparse
import hashlib
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

P = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_lines(text):
    return [line.strip() for line in text.splitlines() if line.lstrip().startswith('|')]


def table_before_caption(md, number):
    before = md.split(f'\nTable {number}.', 1)[0].rstrip()
    lines = before.splitlines()
    rows = []
    while lines and lines[-1].lstrip().startswith('|'):
        rows.insert(0, lines.pop().strip())
    assert rows, f'Missing table before caption {number}'
    return rows


def cells(line):
    return [value.strip() for value in line.strip('|').split('|')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    sources = json.loads((P / 'data/paper_v2_sources.json').read_text())
    for name, expected in sources['original_files'].items():
        assert sha(P / name) == expected, f'Original manuscript/build changed: {name}'
    assert sha(P / 'data/final_table_sources.json') == sources['table_manifest_sha256']
    old = (P / 'when_credibility_collapses.md').read_text()
    md = (P / 'paper_v2.md').read_text()
    tex = (P / 'paper_v2.tex').read_text()
    evidence = json.loads((P / 'data/final_table_sources.json').read_text())['generated_tables']

    # The six dependence contrasts are retained in full, including the null result.
    assert table_before_caption(md, '3') == table_lines(evidence['15']['markdown'])
    selection = sources['tables']['2']
    expected_rows = []
    for line in table_lines(evidence[selection['original_table']]['markdown']):
        row = cells(line)
        if row[selection['filter_column']] == selection['filter_value']:
            expected_rows.append([row[i] for i in selection['columns']])
    observed_rows = [cells(row) for row in table_before_caption(md, '2')[2:]]
    assert observed_rows == expected_rows and len(observed_rows) == 3
    assert len(table_before_caption(md, '3')) - 2 == 6
    assert len(table_before_caption(md, '1')) - 2 == 7

    # Retain the original mathematical statement and proof, not just its conclusion.
    proof = old.split('**Setting.**', 1)[1].split('\nThe tests check the Gaussian', 1)[0].strip()
    assert proof in md, 'Analytical statement/proof differs from preserved original'
    images = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', md)
    assert images == [item['path'] for item in sources['figures']]
    for item in sources['figures']:
        assert sha(P / item['path']) == item['sha256']
        if 'original_path' in item:
            assert sha(P / item['original_path']) == item['original_sha256']
    # Reconstruct every simplified figure cell independently from retained seeds.
    plotted = json.loads((P / 'data/paper_v2_figure_sources.json').read_text())
    for name, expected in plotted['input_sha256'].items():
        assert sha(P.parent / 'data' / name) == expected
    assert sha(P / 'plot_paper_v2.py') == plotted['plotter_sha256']
    assert sha(P / 'data' / plotted['csv_file']) == plotted['csv_sha256']
    def identity(row):
        return row['group'], row['table'], row['cell'], row['metric']
    summaries = json.loads((P.parent / 'data/final_analysis/cell_summary.json').read_text())['rows']
    published = {identity(row): row for row in summaries}
    seed_rows = json.loads((P.parent / 'data/final_analysis/per_seed.json').read_text())['rows']
    original_seeds = {}
    for row in seed_rows:
        original_seeds.setdefault(identity(row), []).append(row)
    plotted_cells = 0
    for filename, figure in plotted['figures'].items():
        assert figure['seeds'] == list(range(10))
        assert len(figure['cells']) == len(figure['rows']) * len(figure['columns'])
        for cell in figure['cells']:
            key = (figure['group'], 'crossplay', f"{cell['row']}|{cell['column']}", figure['metric'])
            assert cell['summary'] == published[key]
            seeds = sorted(original_seeds[key], key=lambda row: row['seed'])
            assert cell['per_seed'] == seeds and [r['seed'] for r in seeds] == list(range(10))
            values = [row['value'] for row in seeds]
            mean = math.fsum(values) / 10
            sd = math.sqrt(math.fsum((v - mean)**2 for v in values) / 9)
            row, column = figure['rows'].index(cell['row']), figure['columns'].index(cell['column'])
            assert math.isclose(mean, figure['mean_matrix'][row][column], rel_tol=0, abs_tol=1e-12)
            assert math.isclose(sd, figure['sample_sd_matrix'][row][column], rel_tol=0, abs_tol=1e-12)
            plotted_cells += 1
    assert plotted_cells == 130
    figures = re.findall(r'^Figure (\d+)\.', md, re.M)
    tables = re.findall(r'^Table (\d+)\.', md, re.M)
    assert figures == ['1', '2', '3', '4', '5']
    assert tables == ['1', '2', '3']
    assert set(re.findall(r'\bFigure (\d+)\b', md)) <= set(figures)
    assert set(re.findall(r'\bTable (\d+)\b', md)) <= set(tables)
    assert not re.search(r'\{\{|\bTODO\b|see (?:the )?supplement|in the supplement', md, re.I)

    citations = set(re.findall(r'@([\w-]+)', md))
    old_citations = set(re.findall(r'@([\w-]+)', old))
    bib = set(re.findall(r'@\w+\s*\{\s*([^,\s]+)', (P / 'references.bib').read_text()))
    assert citations == old_citations and citations <= bib
    assert len(re.findall(r'^\\CSLLeftMargin', tex, re.M)) == len(citations)
    assert tex.count(r'\begin{figure}') == 5
    assert tex.count(r'\begin{table}') == 3
    assert tex.count(r'\caption{') == 8
    assert re.search(r'\\documentclass\[\s*12pt,', tex)
    assert r'font=normalsize' in tex
    abstract = md.split('abstract: |', 1)[1].split('\n---', 1)[0]
    assert len(abstract.split()) <= 300
    assert len(md.split()) < len(old.split())

    pdf = P / 'paper_v2.pdf'
    bbox = subprocess.check_output(['pdftotext', '-bbox-layout', str(pdf), '-'])
    ns = {'x': 'http://www.w3.org/1999/xhtml'}
    pages = ET.fromstring(bbox).findall('.//x:page', ns)
    outside = []
    for number, page in enumerate(pages, 1):
        width, height = (float(page.attrib[key]) for key in ('width', 'height'))
        for word in page.findall('.//x:word', ns):
            bounds = [float(word.attrib[k]) for k in ('xMin', 'yMin', 'xMax', 'yMax')]
            if word.text != str(number) and (bounds[0] < 69 or bounds[1] < 69
                    or bounds[2] > width - 69 or bounds[3] > height - 69):
                outside.append({'page': number, 'text': word.text, 'bounds': bounds})
    assert not outside, outside[:30]
    rendered = subprocess.check_output(['pdftotext', '-layout', str(pdf), '-'], text=True)
    caption_pages = {}
    for number, page in enumerate(rendered.split('\f'), 1):
        for label in re.findall(r'^\s*((?:Figure|Table) \d+)\.\s+', page, re.M):
            assert label not in caption_pages, f'Duplicate rendered caption: {label}'
            caption_pages[label] = number
    assert set(caption_pages) == {f'Figure {n}' for n in range(1, 6)} | {f'Table {n}' for n in range(1, 4)}
    for title in ['Appendix A.', 'Appendix B.']:
        assert title in rendered
    artifacts = ['paper_v2.md', 'paper_v2.tex', 'paper_v2.pdf', 'build_v2.sh',
                 'paper_v2_captions.lua', 'paper_v2_layout.tex',
                 'data/paper_v2_sources.json', 'verify_paper_v2.py', 'plot_paper_v2.py',
                 'data/paper_v2_figure_sources.json', 'data/paper_v2_figure_sources.csv',
                 'figures/v2/hypothesis_crossplay.pdf', 'figures/v2/finite_crossplay.pdf']
    report = {'status': 'passed', 'original_preserved': True,
              'original_words': sources['original_words'], 'v2_words': len(md.split()),
              'word_reduction_percent': round(100 * (1 - len(md.split()) / sources['original_words']), 1),
              'original_pages': sources['original_pages'], 'v2_pages': len(pages),
              'figures': 5, 'tables': 3, 'retained_numeric_table_rows': 9,
              'reconstructed_figure_cells': plotted_cells,
              'original_proposition_and_proof_preserved': True,
              'abstract_words': len(abstract.split()), 'citation_keys': sorted(citations),
              'caption_pages': caption_pages, 'words_outside_margins': outside,
              'new_validation_incorporated': False,
              'files': {name: sha(P / name) for name in artifacts}}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('files', 'caption_pages')}, indent=2))


if __name__ == '__main__':
    main()
