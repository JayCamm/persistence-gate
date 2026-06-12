from __future__ import annotations
import argparse, csv, random, sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / 'src', ROOT / 'benchmarks'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from persistence_memory.api import PersistenceGate
from dirty_metadata_trap_benchmark import CASE_SPECS, evaluate_selection, keyword_caution_filter, make_dirty_workspace, metadata_status_filter, recency_filter, retrieve_candidates, top_k, write_csv

VARIANTS = ['full_signals_role_hidden', 'warning_risk_stress', 'neutral_signals', 'shuffled_signals']


def transform_docs(docs, variant, seed):
    rows = [deepcopy(d) for d in docs]
    if variant == 'full_signals_role_hidden':
        return rows
    if variant == 'warning_risk_stress':
        for d in rows:
            if d['doc_id'].endswith('_current_warning'):
                d['risk'], d['harm_score'], d['usefulness_score'] = 0.55, 0.0, 0.90
            if d['doc_id'].endswith('_historical_comparison'):
                d['risk'], d['harm_score'], d['usefulness_score'] = 0.28, 0.0, 0.84
        return rows
    if variant == 'neutral_signals':
        for d in rows:
            d['risk'], d['harm_score'], d['usefulness_score'] = 0.10, 0.0, 0.50
        return rows
    if variant == 'shuffled_signals':
        rng = random.Random(seed)
        sig = [(d['risk'], d['harm_score'], d['usefulness_score']) for d in rows]
        rng.shuffle(sig)
        for d, s in zip(rows, sig):
            d['risk'], d['harm_score'], d['usefulness_score'] = s
        return rows
    raise ValueError(variant)


def strip_role(row):
    r = dict(row)
    md = dict(r.get('metadata') or {})
    md.pop('role', None)
    r['metadata'] = md
    r.pop('role', None)
    return r


def filt_risk(rows, k):
    return top_k([r for r in rows if float(r.get('risk', 0)) <= 0.50], k)


def filt_useful(rows, k):
    return top_k([r for r in rows if float(r.get('usefulness_score', 0)) >= 0.50], k)


def gate_rows(result):
    return [{'id': i.id, 'doc_id': i.metadata.get('doc_id', i.id), 'text': i.text, 'source': i.source, 'metadata': i.metadata, 'relevance': i.relevance, 'risk': i.risk, 'harm_score': i.harm_score, 'usefulness_score': i.usefulness_score} for i in result.allowed_items]


def run_case(case, docs, variant, seed, profile, k, pool):
    cands = retrieve_candidates(case.query, [d for d in docs if d['family'] == case.family], pool)
    selections = {
        'ordinary_top_k': top_k(cands, k),
        'metadata_status_filter': metadata_status_filter(cands, k),
        'recency_filter': recency_filter(cands, k),
        'keyword_caution_filter': keyword_caution_filter(cands, k),
        'risk_threshold_filter': filt_risk(cands, k),
        'usefulness_threshold_filter': filt_useful(cands, k),
    }
    gate = PersistenceGate(profile=profile, top_k=k, context_scope='global')
    result = gate.filter(query=case.query, retrieved_items=[strip_role(r) for r in cands], context_scope='global')
    selections[f'persistence_gate:{profile}'] = gate_rows(result)
    detail = []
    for method, selected in selections.items():
        detail.append({'variant': variant, 'seed': seed, 'case_id': case.id, 'family': case.family, 'query': case.query, 'method': method, 'required_doc_ids': ';'.join(case.required_docs), 'excluded_doc_ids': ';'.join(case.excluded_docs), **evaluate_selection(case, selected), 'notes': case.notes})
    audit = [{'variant': variant, 'seed': seed, 'case_id': case.id, 'family': case.family, 'query': case.query, 'profile': profile, **r, 'doc_id': r['id'].split('::')[0]} for r in result.audit_log]
    return detail, audit


def summarize(rows):
    out = []
    for variant, method in sorted({(r['variant'], r['method']) for r in rows}):
        sub = [r for r in rows if r['variant'] == variant and r['method'] == method]
        n = len(sub)
        out.append({'variant': variant, 'method': method, 'cases': n, 'excluded_info_exposed_cases': sum(int(r['excluded_info_exposed']) for r in sub), 'excluded_info_exposure_rate': round(sum(int(r['excluded_info_exposed']) for r in sub) / n, 4), 'required_info_retained_cases': sum(int(r['required_info_retained']) for r in sub), 'required_info_retention_rate': round(sum(int(r['required_info_retained']) for r in sub) / n, 4), 'clean_context_cases': sum(int(r['clean_context']) for r in sub), 'clean_context_rate': round(sum(int(r['clean_context']) for r in sub) / n, 4), 'total_context_tokens': sum(int(r['context_tokens']) for r in sub)})
    return out


def write_csv2(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8'); return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='conservative', choices=['permissive','balanced','conservative'])
    ap.add_argument('--top-k', type=int, default=3)
    ap.add_argument('--candidate-pool', type=int, default=8)
    ap.add_argument('--shuffle-replicates', type=int, default=10)
    ap.add_argument('--seed', type=int, default=20260612)
    ap.add_argument('--out-dir', default='benchmark_results/dirty_metadata_bias_audit')
    ap.add_argument('--cache-dir', default='benchmark_data/dirty_metadata_public_cache')
    ap.add_argument('--refresh', action='store_true')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    base_docs, source_manifest = make_dirty_workspace(Path(args.cache_dir), args.refresh)
    jobs = [('full_signals_role_hidden', args.seed), ('warning_risk_stress', args.seed), ('neutral_signals', args.seed)] + [('shuffled_signals', args.seed+i) for i in range(args.shuffle_replicates)]
    details, audits = [], []
    for variant, seed in jobs:
        docs = transform_docs(base_docs, variant, seed)
        for case in CASE_SPECS:
            d, a = run_case(case, docs, variant, seed, args.profile, args.top_k, args.candidate_pool)
            details += d; audits += a
    summary = summarize(details)
    write_csv(out / 'source_manifest.csv', source_manifest)
    write_csv2(out / 'case_details.csv', details)
    write_csv2(out / 'gate_audit_log.csv', audits)
    write_csv2(out / 'summary.csv', summary)
    (out / 'bias_audit_interpretation.md').write_text('Bias audit variants: full signals with role hidden; warning risk stress; neutral signals; shuffled signals. Strong full/stress plus weak neutral/shuffled means the gate depends on governance signals rather than hidden role labels.\n', encoding='utf-8')
    print('\nDirty Metadata Bias Audit')
    print(f'Output: {out}')
    print('variant,method,exposure_rate,retention_rate,clean_context_rate,total_context_tokens')
    for r in summary:
        print(f"{r['variant']},{r['method']},{r['excluded_info_exposure_rate']},{r['required_info_retention_rate']},{r['clean_context_rate']},{r['total_context_tokens']}")

if __name__ == '__main__':
    main()
