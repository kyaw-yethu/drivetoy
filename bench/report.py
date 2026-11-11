"""Tables of a bench.score file as Markdown (Task_specific_JDM docs/TASK_ENCODER_EVAL.md, Report).

Per source: the score of its plan and the probability-weighted score of its paths, the mean of each
term for the plan, and the share of futures in which the plan breaks each TRC rule; overall and by
situation, cause (aspect B) and template. gt_self is reported without the human
filter, so its terms show the errors of the conversion itself.

    python -m bench.report --scores FILE --data DIR --out FILE [--names NAMES.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .score import TERMS, episode_score
from .trc import RULES

REFS = ('gt_rec', 'cv', 'stop', 'gt_self')


def load(path):
    z = np.load(path)
    sources = sorted({k.split('__')[0] for k in z.files if '__' in k}, key=lambda s: (s in REFS, REFS.index(s) if s in REFS else 0, s))
    per = {}
    for s in sources:
        filtered = f'{s}__terms' in z.files
        terms = z[f'{s}__terms' if filtered else f'{s}__raw']            # (N, 16, M, 6)
        w = z[f'{s}__weights']                                          # (N, M)
        sc = np.nan_to_num(episode_score(terms)).mean(1)                # (N, M)
        per[s] = dict(plan=sc[:, 0], weighted=(sc * w).sum(1), terms=terms[:, :, 0].mean(1),
                      rules=z[f'{s}__rules'][:, :, 0].mean(1), filtered=filtered)
    return per, z['diversity'], z['template'], z['family'], [str(x) for x in z['families']]


def pct(x):
    return f'{100 * x:.1f}'


def table(head, rows):
    out = ['| ' + ' | '.join(head) + ' |', '|---|' + '---:|' * (len(head) - 1)]
    return out + ['| ' + ' | '.join(r) + ' |' for r in rows]


def grouped(per, names, codes, labels, key, title):
    lines = [f'### {title}', '']
    rows = []
    for g, lab in enumerate(labels):
        keep = codes == g
        if keep.sum() == 0:
            continue
        rows.append([lab, f'{keep.sum():,}'] + [f'{per[s][key][keep].mean():.3f}' if key != 'any' else
                                              pct(per[s]['rules'][keep].max(-1).mean()) for s in per])
    rows.append(['All', f'{len(codes):,}'] + [f'{per[s][key].mean():.3f}' if key != 'any' else
                                            pct(per[s]['rules'].max(-1).mean()) for s in per])
    return lines + table([title.split(' by ')[-1].capitalize(), 'Samples'] + [names.get(s, s) for s in per], rows) + ['']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--names', default=None)
    args = ap.parse_args()
    meta = json.loads((Path(args.data) / 'meta.json').read_text())
    names = json.loads(Path(args.names).read_text()) if args.names else {}
    per, div, tmpl, fam, families = load(args.scores)
    order = list(names) + [s for s in per if s not in names]
    per = {s: per[s] for s in order if s in per}
    lines = ['# NAVSIM-style Evaluation', '',
             'Scores follow `Task_specific_JDM docs/TASK_ENCODER_EVAL.md`. The plan is the first path of a source; the weighted score '
             'averages the scores of its paths by their probabilities. Each sample scores the mean over its 16 futures. '
             'gt_self is shown without the human filter.', '', '## Overall', '']
    rows = [[names.get(s, s), f"{p['plan'].mean():.3f}", f"{p['weighted'].mean():.3f}"] +
            [f'{v:.3f}' for v in p['terms'].mean(0)] for s, p in per.items()]
    lines += table(['Source', 'Score', 'Weighted score'] + list(TERMS), rows) + ['']
    lines += ['## Traffic Rules', '', 'Share of futures (%) in which the plan breaks each rule.', '']
    rows = [[names.get(s, s)] + [pct(v) for v in p['rules'].mean(0)] + [pct(p['rules'].max(-1).mean())] for s, p in per.items()]
    lines += table(['Source'] + [r.capitalize() for r in RULES] + ['Any'], rows) + ['']
    groups = [('S', fam, families, 'by situation'),
              ('B', div[:, 1], meta['diversity']['B'], 'by cause'),
              ('T', tmpl, meta['templates'], 'by template')]
    lines += ['## Score', '']
    for _, codes, labels, title in groups:
        lines += grouped(per, names, codes, labels, 'plan', f'Score {title}')
    lines += ['## Any Rule Broken (%)', '']
    for _, codes, labels, title in groups:
        lines += grouped(per, names, codes, labels, 'any', f'Rules {title}')
    Path(args.out).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:40]))


if __name__ == '__main__':
    main()
