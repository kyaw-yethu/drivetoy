#!/usr/bin/env python
"""Write the data folder's README.md from the code README, with the built splits' counts and
shares (from <split>/annotations.npz, meta.json and checks.json) and links to the GIFs in viz/;
the code README gets the same tables, linking the GIFs next to it.

    python scripts/v2s_readme.py --data /root/projects/drive_v2s --code /root/projects/drivetoy_v2
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

NL = chr(10)
SPLITS = ("train", "val", "test")
GIFS = dict(A="diversity_A_light_v2s.gif", B="diversity_B_cause_v2s.gif", C="diversity_C_speed_v2s.gif",
            D="diversity_D_path_v2s.gif", E="diversity_E_choice_v2s.gif")
E_ROWS = [("pending", "a major choice is cued on screen by frame 31, with odds between 0.1 and 0.9, and falls in frames 32 to 40"),
          ("decided", "the major choice is made on screen, and only minor variation is left"),
          ("none", "the clip has no major choice")]


def pct(x):
    return f"{100 * x:.1f}%"


def section(text, start, end):
    i0 = text.index(start)
    i1 = text.index(end, i0 + len(start)) if end else len(text)
    return text[i0:i1]


def aspect_rows(div_text, key):
    """(label, description, target) of an aspect's table in the code README."""
    m = re.search(r"\*\*" + key + r"\. [^\n]*\n\n((?:\|[^\n]*\n)+)", div_text)
    rows = []
    for line in m.group(1).strip().split(NL)[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append((cells[0], cells[1], cells[2]))
    return m.group(0).split(NL)[0], rows


def build(code_readme, data, gif_prefix, title):
    text = code_readme
    meta = {s: json.loads((data / s / "meta.json").read_text()) for s in SPLITS}
    checks = {s: json.loads((data / s / "checks.json").read_text()) for s in SPLITS}
    ann = {s: np.load(data / s / "annotations.npz", mmap_mode="r") for s in SPLITS}
    templates = meta["train"]["templates"]
    intro = text[text.index(NL) + 1:text.index("**Contents**")].strip()
    head = [f"# {title}", "", intro, "",
            "**Contents**", "", "- [Data](#data)", "- [Diversity](#diversity)", "- [Scripts](#scripts)",
            "- [Stochasticity](#stochasticity)"]
    for h in re.findall(r"^## (.+)$", section(text, "# Stochasticity", None), re.M):
        head.append(f"  - [{h}](#{re.sub(r'[^a-z0-9 -]', '', h.lower()).replace(' ', '-')})")
    # data
    rows = ["| Split | Clips | Samples | " + " | ".join(f"`{t}`" for t in templates) + " |",
            "|---|---:|---:|" + "---:|" * len(templates)]
    for s in SPLITS:
        tpl = np.asarray(ann[s]["template"])
        N, K = len(tpl), meta[s]["branches"]
        rows.append(f"| {s} | {N // K:,} | {N:,} | " + " | ".join(f"{int((tpl == i).sum()):,}" for i in range(len(templates))) + " |")
    data_sec = ["## Data", "", "The splits are in `/root/projects/drive_v2s/` on yethu-workspace.", ""] + rows + ["",
        "Each sample has 32 visible frames at $128 \\times 128$ and 5 Hz (6.4 s), followed by 32 label-only frames. "
        "A test clip has two futures that share frames 0 to 4, so it gives two samples. Each split directory holds "
        "`annotations.npz` (per-sample labels, including `diversity` for aspects A to E, the decision log `decisions` "
        "and the per-driver values `drivers`), `future_set.npz` (16 futures of frames 32 to 63 per sample, forked at "
        "frame 31; future 0 is the sample's own label), `meta.json`, `checks.json`, and the per-sample `labels/`, "
        "`scenes/` and `render/` files. Val and test also hold `videos.npy` (samples, 3, 32, 128, 128); train keeps its "
        "frames in `render/<n>.npz` only. The GIFs are in `viz/`.", ""]
    # diversity
    div_text = section(text, "## Diversity", "## Scripts")
    div = ["## Diversity", "",
           "Every sample has one label per aspect, computed from frames 0 to 31 (cause from frames 5 to 31; E from the "
           "choices in frames 0 to 40), so the shares of an aspect sum to 100%. The tables give the sampler's target "
           "share and the share actually built in each split; the test shares count both futures of a clip. Under each "
           "table, a GIF shows one val clip per label, the ego view beside the top-down map, with the clip's labels above it.", ""]
    for key in "ABCDE":
        if key == "E":
            header = "**E. Choice at frame 31** (the state of the ego's major choices, see [Stochasticity](#stochasticity))."
            rows_k = [(l, d, pct(meta["train"]["cfg"]["DT_SHARES"]["E"][l])) for l, d in E_ROWS]
        else:
            header, rows_k = aspect_rows(div_text, key)
            header = header.replace(" The shares are met exactly.", "")
        div += [header, "", "| Label | Description | Target | Train | Val | Test |", "|---|---|---:|---:|---:|---:|"]
        for l, d, t in rows_k:
            div.append(f"| {l} | {d} | {t} | " + " | ".join(pct(checks[s]["shares"][key][l]["built"]) for s in SPLITS) + " |")
        div += ["", f"![One val clip per {key} label: ego view and top-down map]({gif_prefix}{GIFS[key]})", ""]
    note = div_text[div_text.rindex("The sampler meets A first"):].strip()
    div += [note, ""]
    # checks summary, appended to the Build section of Stochasticity
    v = checks["val"]
    odds = "; ".join(f"{b}: {x['mean_p']:.2f} drawn, {x['realised']:.2f} realised" for b, x in v["odds"].items() if x["n"])
    summary = [
        "", "### Built data", "",
        f"In val, {pct(v['pending']['both_outcomes'])} of the pending samples show both outcomes of their pending choice among "
        f"their 16 futures, and the realised outcome shares follow the drawn odds ({odds}). The ego ends more than 2 m apart "
        f"across the futures at frame 63 in {pct(v['spread63']['pending']['over_2m'])} of pending samples, "
        f"{pct(v['spread63']['decided']['over_2m'])} of decided and {pct(v['spread63']['none']['over_2m'])} of none. "
        f"{v.get('resim_equal', 0):,} of {v['n']:,} val samples re-simulate exactly from their seed and target, and "
        f"`rules.check` reports {v.get('violations', 0)} violations. In test, {pct(checks['test'].get('siblings_ego_over_2m', 0))} "
        "of sibling pairs keep the ego more than 2 m apart somewhere in frames 5 to 31.", "",
        f"![One pending val clip per kind of choice; the map runs on to frame 40]({gif_prefix}pending_v2s.gif)", "",
        f"![Four futures of pending val clips, forked at frame 31]({gif_prefix}forks_v2s.gif)", ""]
    rest = section(text, "## Scripts", None)
    if "### Built data" in rest:
        rest = rest[:rest.index("### Built data")].rstrip() + NL
    return NL.join(head) + NL + NL + NL.join(data_sec) + NL + NL.join(div) + NL + rest.rstrip() + NL + NL.join(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/projects/drive_v2s")
    ap.add_argument("--code", default="/root/projects/drivetoy_v2")
    args = ap.parse_args()
    data, code = Path(args.data), Path(args.code)
    src = (code / "README.md").read_text()
    (data / "README.md").write_text(build(src, data, "viz/", "DriveToy v2s"))
    (code / "README.md").write_text(build(src, data, "", src.split(NL)[0].lstrip("# ").strip()))
    print("wrote", data / "README.md", "and", code / "README.md")


if __name__ == "__main__":
    main()
