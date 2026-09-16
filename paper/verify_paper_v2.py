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


def verify_reward_incident_evidence(original_seeds, published):
    """Independently reconstruct the reward/incident receipt from retained seeds."""
    receipt = json.loads((P / 'data/reward_incident_decomposition.json').read_text())
    expected_sources = {
        'data/final_analysis/cell_summary.json',
        'data/final_analysis/per_seed.json',
        'data/final_analysis/submission_protocol.json',
        'code/scripts/train_f3_matched_cycle.py',
        'code/src/social_collusion/env/rewards.py',
        'code/src/social_collusion/metrics/outcome_metrics.py',
    }
    assert set(receipt['source_sha256']) == expected_sources
    for name, expected in receipt['source_sha256'].items():
        assert sha(P.parent / name) == expected, f'Reward/incident evidence source changed: {name}'
    metric_map = {
        'false_ejection': 'false_ejection_rate',
        'incident_and_false_ejection': 'any_incident_false_ejection_rate',
        'no_incident_and_false_ejection': 'any_incident_free_false_ejection_rate',
        'incident': 'incident_rate',
    }
    conditional_map = {
        'false_ejection_given_incident': 'false_ejections_per_incident_meeting',
        'false_ejection_given_no_incident': 'false_ejections_per_incident_free_meeting',
    }
    pairs = {
        'C0D0': 'coalition0_vs_soft_credibility',
        'C0D1': 'coalition0_vs_learned_crew1',
        'C1D1': 'coalition1_vs_learned_crew1',
        'C1D0': 'coalition1_vs_soft_credibility',
    }
    expected_cells = {(f'{crew}+2', pair) for crew in (3, 5, 7) for pair in pairs}
    receipt_cells = {(cell['crew_plus_coalition'], cell['cell']): cell for cell in receipt['cells']}
    assert len(receipt['cells']) == len(receipt_cells) == 12
    assert set(receipt_cells) == expected_cells
    seed_cells = 0
    mean_checks = 0
    recomputed_rates = {}
    for (population, pair), cell in receipt_cells.items():
        group = f"f3_tenseed_crew{population.split('+')[0]}"
        source_cell = pairs[pair]
        assert cell['group'] == group and cell['source_cell'] == source_cell
        assert cell['n_seeds'] == 10 and cell['episodes_per_seed'] == 1000
        assert cell['total_episodes'] == 10000
        by_metric = {}
        for name, metric in metric_map.items():
            key = (group, 'crossplay', source_cell, metric)
            rows = sorted(original_seeds[key], key=lambda row: row['seed'])
            assert len(rows) == 10 and [r['seed'] for r in rows] == list(range(10))
            by_metric[name] = [r['value'] for r in rows]
            mean = math.fsum(by_metric[name]) / 10
            assert math.isclose(mean, published[key]['mean'], rel_tol=0, abs_tol=1e-12)
            assert math.isclose(mean, cell['unconditional_rates'][name], rel_tol=0, abs_tol=1e-12)
            mean_checks += 1
        seed_counts = []
        for seed in range(10):
            counts = {name: round(values[seed] * 1000) for name, values in by_metric.items()}
            for name, values in by_metric.items():
                assert math.isclose(counts[name] / 1000, values[seed], rel_tol=0, abs_tol=1e-12)
            counts['no_incident'] = 1000 - counts['incident']
            assert counts['false_ejection'] == counts['incident_and_false_ejection'] + counts['no_incident_and_false_ejection']
            assert 0 <= counts['incident_and_false_ejection'] <= counts['incident'] <= 1000
            assert 0 <= counts['no_incident_and_false_ejection'] <= counts['no_incident'] <= 1000
            seed_counts.append({'seed': seed, 'episodes': 1000, **counts})
            seed_cells += 1
        assert cell['per_seed_counts_reconstructed_from_published_rates'] == seed_counts
        totals = {name: sum(row[name] for row in seed_counts) for name in seed_counts[0] if name not in ('seed', 'episodes')}
        assert cell['counts'] == totals
        rates = {name: count / 10000 for name, count in totals.items()}
        assert cell['unconditional_rates'] == rates
        recomputed_rates[(population, pair)] = rates
        share = totals['no_incident_and_false_ejection'] / totals['false_ejection']
        assert math.isclose(cell['share_of_false_ejections_in_no_incident_games'], share, rel_tol=0, abs_tol=1e-12)
        for name, metric in conditional_map.items():
            condition = name.removeprefix('false_ejection_given_')
            defined = [r['seed'] for r in seed_counts if r[condition]]
            values = [r[condition + '_and_false_ejection'] / r[condition] for r in seed_counts if r[condition]]
            key = (group, 'crossplay', source_cell, metric)
            original = sorted(original_seeds[key], key=lambda row: row['seed'])
            assert [r['seed'] for r in original] == defined
            for computed, row in zip(values, original):
                assert math.isclose(computed, row['value'], rel_tol=0, abs_tol=1e-12)
            assert cell['defined_seeds_for_conditional_means'][name] == defined
            mean = math.fsum(values) / len(values)
            assert math.isclose(mean, published[key]['mean'], rel_tol=0, abs_tol=1e-12)
            assert math.isclose(mean, cell['mean_seed_conditional_rates'][name], rel_tol=0, abs_tol=1e-12)
            pooled = totals[condition + '_and_false_ejection'] / totals[condition]
            assert math.isclose(pooled, cell['pooled_conditional_rates'][name], rel_tol=0, abs_tol=1e-12)
            mean_checks += 1
    deltas = {r['crew_plus_coalition']: r for r in receipt['defender_change_decomposition']}
    assert len(receipt['defender_change_decomposition']) == len(deltas) == 3
    assert set(deltas) == {'3+2', '5+2', '7+2'}
    for population, delta in deltas.items():
        assert delta['contrast'] == 'C0D1 minus C0D0'
        for name in metric_map:
            expected = recomputed_rates[(population, 'C0D1')][name] - recomputed_rates[(population, 'C0D0')][name]
            assert math.isclose(delta['change_in_unconditional_rate'][name], expected, rel_tol=0, abs_tol=1e-12)
    assert seed_cells == 120 and mean_checks == 72
    return {'reward_incident_cells': len(receipt_cells),
            'reward_incident_seed_cells': seed_cells,
            'reward_incident_mean_checks': mean_checks,
            'reward_incident_source_hashes': len(expected_sources)}


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
    reward_incident_checks = verify_reward_incident_evidence(original_seeds, published)
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
                 'verify_reward_incident_decomposition.py', 'data/reward_incident_decomposition.json',
                 'data/paper_v2_figure_sources.json', 'data/paper_v2_figure_sources.csv',
                 'figures/v2/hypothesis_crossplay.pdf', 'figures/v2/finite_crossplay.pdf']
    report = {'status': 'passed', 'original_preserved': True,
              'original_words': sources['original_words'], 'v2_words': len(md.split()),
              'word_reduction_percent': round(100 * (1 - len(md.split()) / sources['original_words']), 1),
              'original_pages': sources['original_pages'], 'v2_pages': len(pages),
              'figures': 5, 'tables': 3, 'retained_numeric_table_rows': 9,
              'reconstructed_figure_cells': plotted_cells,
              **reward_incident_checks,
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
