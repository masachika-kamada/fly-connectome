"""Render the experiment results as SVG figures.

SVG rather than a plotting library: crisp at any size, diffable in git, renders
inline on GitHub, and no binary dependency. Each figure carries its own light and
dark palette via prefers-color-scheme, so it reads on either background.

Writes figures/fig1..fig6.svg.
"""
import collections
import json
import os

import numpy as np

from flymb import data as mbdata, odors
from flymb.paths import FIGURES as FIG, RESULTS as RES, ensure

# Palette slots, light / dark. Categorical order is fixed, never cycled.
S1 = ("#2a78d6", "#3987e5")   # blue
S2 = ("#eb6834", "#d95926")   # orange
S3 = ("#1baf7a", "#199e70")   # aqua
S4 = ("#eda100", "#c98500")   # yellow
INK = ("#0b0b0b", "#ffffff")
SEC = ("#52514e", "#c3c2b7")
MUT = ("#898781", "#898781")
GRID = ("#e1e0d9", "#2c2c2a")
BASE = ("#c3c2b7", "#383835")
SURF = ("#fcfcfb", "#1a1a19")

FONT = ("ui-sans-serif,system-ui,'Segoe UI',Helvetica,Arial,sans-serif")
MONO = ("ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace")


def head(w, h, roles):
    """SVG open tag plus a style block defining every colour role in both modes."""
    light = ";".join(f"--{k}:{v[0]}" for k, v in roles.items())
    dark = ";".join(f"--{k}:{v[1]}" for k, v in roles.items())
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" font-family="{FONT}" role="img">'
        f"<style>"
        f":root{{{light}}}"
        f"@media (prefers-color-scheme:dark){{:root{{{dark}}}}}"
        f".t{{fill:var(--ink)}}.s{{fill:var(--sec)}}.m{{fill:var(--mut)}}"
        f".mono{{font-family:{MONO}}}"
        f"</style>"
        f'<rect width="{w}" height="{h}" fill="var(--surf)"/>'
    )


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def txt(x, y, s, size=12, cls="t", anchor="start", weight=400, mono=False):
    m = " mono" if mono else ""
    return (f'<text x="{x}" y="{y}" font-size="{size}" class="{cls}{m}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'dominant-baseline="middle">{esc(s)}</text>')


def hbar(x, y, w, h, fill, r=4):
    """Horizontal bar with rounded data-end, square against the baseline."""
    w = max(w, 0.01)
    if w <= r:
        return f'<rect x="{x}" y="{y}" width="{w:.2f}" height="{h}" fill="{fill}"/>'
    return (f'<path d="M{x} {y} H{x+w-r:.2f} a{r} {r} 0 0 1 {r} {r} '
            f'V{y+h-r} a{r} {r} 0 0 1 {-r} {r} H{x} Z" fill="{fill}"/>')


def title(x, y, main, sub=None):
    out = txt(x, y, main, 15, "t", weight=600)
    if sub:
        out += txt(x, y + 20, sub, 12, "s")
    return out


# ----------------------------------------------------------------- figure 1
def fig1():
    d = json.load(open(os.path.join(RES, "exp1_hash.json"), encoding="utf-8"))
    order = ["gaussian_lsh", "uniform", "kc_degree_kept", "binary_real",
             "pn_degree_kept", "degree_swap", "real"]
    label = {"real": "real  実測の配線", "degree_swap": "degree_swap  次数保存で張替",
             "pn_degree_kept": "pn_degree_kept  PN側のみ保存",
             "binary_real": "binary_real  重みを1に", "kc_degree_kept": "kc_degree_kept  KC側のみ保存",
             "uniform": "uniform  一様ランダム", "gaussian_lsh": "gaussian_lsh  密なランダム射影"}
    g, p = d["map"]["olfactory"], d["map"]["glomerular"]

    W, H = 760, 452
    L, R, T = 232, 72, 100
    rowh, gap = 42, 4
    xmax = 0.72
    sx = lambda v: L + v / xmax * (W - L - R)

    out = [head(W, H, {"ink": INK, "sec": SEC, "mut": MUT, "grid": GRID,
                       "base": BASE, "surf": SURF, "c1": S1, "c2": S2})]
    out.append(title(20, 24, "実配線は、次数を保ったランダム配線と区別できない",
                     "近傍検索の平均適合率 mAP@50。高いほど良い。誤差棒は5試行の標準偏差"))
    for v in (0.0, 0.2, 0.4, 0.6):
        x = sx(v)
        out.append(f'<line x1="{x}" y1="{T-10}" x2="{x}" y2="{T+len(order)*rowh-8}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')
        out.append(txt(x, T - 20, f"{v:.1f}", 10.5, "m", "middle", mono=True))

    for i, k in enumerate(order):
        y = T + i * rowh
        out.append(txt(L - 12, y + 12, label[k], 11.5, "s", "end"))
        for j, (src, col) in enumerate(((g, S1), (p, S2))):
            m, s = src[k]["mean"], src[k]["std"]
            by = y + j * (11 + gap)
            out.append(hbar(L, by, sx(m) - L, 11, f"var(--c{j+1})"))
            out.append(f'<line x1="{sx(max(0,m-s))}" y1="{by+5.5}" x2="{sx(m+s)}" '
                       f'y2="{by+5.5}" stroke="var(--base)" stroke-width="1.5"/>')
            out.append(txt(sx(m + s) + 6, by + 6, f"{m:.3f}", 11 if j == 0 else 10,
                           "t" if j == 0 else "m", weight=500 if j == 0 else 400, mono=True))

    ly = T + len(order) * rowh + 12
    for j, name in enumerate(("匂いを運ぶ糸球体だけに入力", "全チャンネルに入力（多糸球体PN・温湿度を含む）")):
        out.append(f'<rect x="{L + j*200}" y="{ly}" width="9" height="9" rx="2" fill="var(--c{j+1})"/>')
        out.append(txt(L + j * 200 + 15, ly + 5, name, 11, "s"))
    out.append(txt(L, ly + 28, "橙は旧来の採点。キノコ体がほとんど読まない次元まで正解に含めるので、実配線に不利に働く",
                   10.5, "m"))
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------- figure 2
ROLE_SLOT = {"food": 0, "pheromone": 1, "thermo": 2, "other": 3}


def glom_role(g):
    return ROLE_SLOT[odors.glomerulus_role(g)]


def fig2():
    skew = json.load(open(os.path.join(RES, "exp1_hash.json"), encoding="utf-8"))["glomerulus_outdegree"]
    W_, _, meta = mbdata.load_mb()
    outdeg = (W_ > 0).sum(axis=1)
    agg = collections.defaultdict(int)
    for p, od in zip(meta["pn"], outdeg):
        t = p["type"] or "?"
        agg["multiglomerular" if t.startswith("M_") else t.split("_")[0]] += int(od)
    rows = sorted(agg.items(), key=lambda kv: -kv[1])
    total = sum(v for _, v in rows)
    top, bottom = rows[:16], rows[-8:]

    W, H = 760, 680
    L, R, T = 138, 96, 78
    rowh = 20
    xmax = max(v for _, v in rows) * 1.02
    sx = lambda v: L + v / xmax * (W - L - R)
    cols = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--mut)"]

    out = [head(W, H, {"ink": INK, "sec": SEC, "mut": MUT, "grid": GRID,
                       "base": BASE, "surf": SURF, "c1": S1, "c2": S2, "c3": S3})]
    out.append(title(20, 24, "温度と湿度の入口は、ほとんど読まれていない",
                     f"糸球体ごとのケニヨン細胞への結合数。全{len(rows)}糸球体のうち上位16と下位8"))

    y = T
    for name, v in top:
        c = cols[glom_role(name)]
        out.append(txt(L - 10, y + 7, name, 11, "s", "end", mono=True))
        out.append(hbar(L, y, sx(v) - L, 13, c))
        out.append(txt(sx(v) + 8, y + 7, f"{v}", 11, "t", weight=500, mono=True))
        out.append(txt(W - 22, y + 7, f"{100*v/total:.1f}%", 10.5, "m", "end", mono=True))
        y += rowh

    y += 6
    out.append(f'<line x1="{L}" y1="{y}" x2="{W-R+40}" y2="{y}" stroke="var(--grid)" '
               f'stroke-width="1" stroke-dasharray="3 4"/>')
    out.append(txt(L - 10, y, "…", 12, "m", "end"))
    y += 16

    for name, v in bottom:
        c = cols[glom_role(name)]
        out.append(txt(L - 10, y + 7, name, 11, "s", "end", mono=True))
        out.append(hbar(L, y, max(sx(v) - L, 1.5), 13, c))
        out.append(txt(sx(v) + 8, y + 7, f"{v}", 11, "t", weight=500, mono=True))
        y += rowh

    y += 18
    legend = [("食物の匂い（酸・エステル）", 0), ("フェロモン", 1),
              ("温度・湿度", 2), ("その他の匂い", 3)]
    lx = L
    for name, role in legend:
        out.append(f'<rect x="{lx}" y="{y}" width="9" height="9" rx="2" fill="{cols[role]}"/>')
        out.append(txt(lx + 15, y + 5, name, 11, "s"))
        lx += 22 + len(name) * 11.2
    y += 26
    out.append(txt(L, y + 4, f"上位5糸球体で全結合の {100*sum(v for _,v in rows[:5])/total:.1f}%"
                             f"（一様なら {100*5/len(rows):.1f}%）", 11.5, "t", weight=500))
    med = skew["median"]
    out.append(txt(L, y + 22, f"中央値: 食物 {med['food']:.0f} / フェロモン {med['pheromone']:.0f} / "
                              f"その他の匂い {med['other']:.0f} / 温度・湿度 {med['thermo']:.0f}。"
                              f"食物＋フェロモンが多いとは言えない（片側置換検定 p={skew['p_food_pheromone']:.2f}）",
                   10.5, "m"))
    out.append(txt(L, y + 40, "分類は文献知識によるもので、このデータセットから導いたものではない",
                   10.5, "m"))
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------- figure 3
def fig3():
    d = json.load(open(os.path.join(RES, "exp2_capacity.json"), encoding="utf-8"))
    ks = sorted(int(k) for k in d["auc"]["real"])
    series = [("real  実測", "c1"), ("degree_swap  張替", "c2"), ("uniform  一様", "c3")]
    keys = ["real", "degree_swap", "uniform"]

    W, H = 760, 430
    L, R, T, B = 66, 132, 78, 62
    sx = lambda k: L + (np.log(k) - np.log(ks[0])) / (np.log(ks[-1]) - np.log(ks[0])) * (W - L - R)
    sy = lambda v: T + (1.0 - v) / (1.0 - 0.6) * (H - T - B)

    out = [head(W, H, {"ink": INK, "sec": SEC, "mut": MUT, "grid": GRID,
                       "base": BASE, "surf": SURF, "c1": S1, "c2": S2, "c3": S3})]
    out.append(title(20, 24, "容量の壁は、解析近似より遠い",
                     "学習済みの匂いと未学習の匂いをMBON入力で分離できるか（AUC）。1.0が完全分離"))

    for v in (1.0, 0.9, 0.8, 0.7, 0.6):
        yy = sy(v)
        out.append(f'<line x1="{L}" y1="{yy}" x2="{W-R}" y2="{yy}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(txt(L - 10, yy, f"{v:.1f}", 10.5, "m", "end", mono=True))
    for k in (1, 3, 8, 20, 50, 100):
        out.append(txt(sx(k), H - B + 18, str(k), 10.5, "m", "middle", mono=True))
    out.append(txt((L + W - R) / 2, H - B + 40, "覚えさせた匂いの数 k", 11.5, "s", "middle"))

    # analytic reference, drawn recessive
    pts = " ".join(f"{sx(k):.1f},{sy(max(0.6,(1-d['sparsity'])**k)):.1f}" for k in ks)
    out.append(f'<polyline points="{pts}" fill="none" stroke="var(--mut)" stroke-width="1.5" '
               f'stroke-dasharray="4 4"/>')
    out.append(txt(W - R + 10, sy(max(0.6, (1 - d["sparsity"]) ** ks[-1])) - 2,
                   "解析近似 (1-f)^k", 10.5, "m"))

    for (name, c), key in zip(series, keys):
        pts, last = [], None
        for k in ks:
            m = d["auc"][key][str(k)]["mean"]
            pts.append(f"{sx(k):.1f},{sy(m):.1f}")
            last = m
        out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="var(--{c})" '
                   f'stroke-width="2" stroke-linejoin="round"/>')
        for k in ks:
            m = d["auc"][key][str(k)]["mean"]
            out.append(f'<circle cx="{sx(k):.1f}" cy="{sy(m):.1f}" r="4" fill="var(--{c})" '
                       f'stroke="var(--surf)" stroke-width="2"/>')
        out.append(txt(W - R + 10, sy(last), name, 11, "s"))

    thr = sy(0.95)
    out.append(f'<line x1="{L}" y1="{thr}" x2="{W-R}" y2="{thr}" stroke="var(--base)" '
               f'stroke-width="1" stroke-dasharray="2 3"/>')
    out.append(txt(L + 6, thr - 11, "AUC 0.95 — 実測容量 k=20", 10.5, "m"))
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------- figure 4
def fig4():
    d = json.load(open(os.path.join(RES, "exp3_compartments.json"), encoding="utf-8"))
    rows = d["compartments"]
    W, H = 760, 452
    L, R, T, B = 78, 44, 104, 62
    xs = [r["kc_inputs"] for r in rows]
    lo, hi = np.log(min(xs) * 0.9), np.log(max(xs) * 1.1)
    sx = lambda v: L + (np.log(v) - lo) / (hi - lo) * (W - L - R)
    sy = lambda v: T + (1.0 - v) / (1.0 - 0.7) * (H - T - B)

    out = [head(W, H, {"ink": INK, "sec": SEC, "mut": MUT, "grid": GRID,
                       "base": BASE, "surf": SURF, "c1": S1})]
    out.append(title(20, 24, "容量は区画ごとに10倍ちがう",
                     f"51区画それぞれの、読み出すケニヨン細胞数と記憶20件での分離性能"
                     f"（相関 {d['correlation_log_size_vs_auc20']:+.2f}）"))
    for v in (1.0, 0.9, 0.8, 0.7):
        yy = sy(v)
        out.append(f'<line x1="{L}" y1="{yy}" x2="{W-R}" y2="{yy}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(txt(L - 10, yy, f"{v:.1f}", 10.5, "m", "end", mono=True))
    for k in (125, 250, 500, 1000, 1700):
        out.append(txt(sx(k), H - B + 18, str(k), 10.5, "m", "middle", mono=True))
    out.append(txt((L + W - R) / 2, H - B + 40, "その区画が読むケニヨン細胞の数", 11.5, "s", "middle"))
    out.append(txt(20, T - 24, "AUC @ 記憶20件", 11, "m"))

    for r in rows:
        x, y = sx(r["kc_inputs"]), sy(min(1.0, max(0.7, r["auc_at_20"])))
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="var(--c1)" '
                   f'fill-opacity="0.85" stroke="var(--surf)" stroke-width="2"/>')
    big = max(rows, key=lambda r: r["kc_inputs"])
    bx, by = sx(big["kc_inputs"]), sy(min(1.0, big["auc_at_20"]))
    ay = by - 46
    out.append(f'<path d="M{bx} {by-8} V{ay+6}" stroke="var(--base)" stroke-width="1" fill="none"/>')
    out.append(txt(W - R, ay - 8, "MBON11  嫌悪記憶の主役", 11, "t", "end", weight=500))
    out.append(txt(W - R, ay + 8, f'{big["kc_inputs"]} KC入力 / 容量50件以上', 10.5, "m", "end"))
    small = min(rows, key=lambda r: r["kc_inputs"])
    sx_, sy_ = sx(small["kc_inputs"]), sy(min(1.0, max(0.7, small["auc_at_20"])))
    out.append(f'<path d="M{sx_} {sy_+8} V{sy_+30}" stroke="var(--base)" stroke-width="1" fill="none"/>')
    out.append(txt(sx_, sy_ + 42, f'最小 {small["kc_inputs"]} KC入力', 10.5, "m", "middle"))
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------- figure 5
def fig5():
    d = json.load(open(os.path.join(RES, "exp4_real_odors.json"), encoding="utf-8"))
    A = d["similarity_preservation"]
    order = ["uniform", "kc_degree_kept", "binary_real", "pn_degree_kept", "degree_swap", "real"]
    label = {"real": "実測の配線", "degree_swap": "次数保存で張替", "pn_degree_kept": "PN側のみ保存",
             "binary_real": "重みを1に", "kc_degree_kept": "KC側のみ保存", "uniform": "一様ランダム"}

    W, H = 760, 388
    L, R, T = 150, 40, 100
    rowh = 38
    lo, hi = 0.80, 0.95
    sx = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)

    out = [head(W, H, {"ink": INK, "sec": SEC, "mut": MUT, "grid": GRID,
                       "base": BASE, "surf": SURF, "c1": S1, "c2": S2})]
    out.append(title(20, 24, f"実在の匂い{d['n_odors']}種でも、実配線は張り替えた配線の真ん中にいる",
                     "受容体レベルで似ている匂いほど、KCコードも重なるか（全ペアの順位相関）。"
                     f"点は乱数配線1本ずつ、各{d['draws']}本"))
    for v in (0.80, 0.85, 0.90, 0.95):
        x = sx(v)
        out.append(f'<line x1="{x}" y1="{T-12}" x2="{x}" y2="{T+len(order)*rowh-14}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')
        out.append(txt(x, T - 24, f"{v:.2f}", 10.5, "m", "middle", mono=True))
    real = A["real"]["mean"]
    out.append(f'<line x1="{sx(real)}" y1="{T-12}" x2="{sx(real)}" y2="{T+len(order)*rowh-14}" '
               f'stroke="var(--c2)" stroke-width="1" stroke-dasharray="3 3"/>')
    for i, k in enumerate(order):
        y = T + i * rowh
        out.append(txt(L - 14, y + 6, label[k], 11.5, "s" if k != "real" else "t", "end",
                       weight=400 if k != "real" else 600))
        pts = A[k].get("draws", [A[k]["mean"]])
        for j, v in enumerate(pts):
            jitter = ((j * 7) % 5 - 2) * 2.2 if len(pts) > 1 else 0
            col = "var(--c2)" if k == "real" else "var(--c1)"
            r = 5.5 if len(pts) == 1 else 3.5
            out.append(f'<circle cx="{sx(v):.1f}" cy="{y + 6 + jitter:.1f}" r="{r}" fill="{col}" '
                       f'fill-opacity="{1 if len(pts) == 1 else 0.6}"/>')
        out.append(txt(W - R + 4, y + 6, f"{A[k]['mean']:.3f}", 10.5, "t", mono=True))
    pct = 100 * A["real_percentile_in_degree_swap"]
    out.append(txt(L, T + len(order) * rowh + 8,
                   f"実配線は、次数保存で張り替えた{d['draws']}本の {pct:.0f} パーセンタイル。"
                   "個々の配線は、実在の匂いに対しても何も足していない", 11, "t", weight=500))
    out.append(txt(L, T + len(order) * rowh + 28,
                   "一様ランダムとの差は、合成データのときと同じく、PN側の出次数の偏りで説明できる", 10.5, "m"))
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------- figure 6
def fig6():
    d = json.load(open(os.path.join(RES, "exp5_errand.json"), encoding="utf-8"))
    names = [p["name"] for p in d["products"]]
    n, per = len(names), d["layouts"] * d["flies"]
    conf = d["confusion"]

    W, H = 760, 820
    L, T, cs = 128, 150, 30
    out = [head(W, H, {"ink": INK, "sec": SEC, "mut": MUT, "grid": GRID,
                       "base": BASE, "surf": SURF, "c1": S1, "c2": S2, "c3": S3})]
    out.append(title(20, 24, f"おつかいの正答率 {100*d['overall']:.0f}%。取り違えるのは、香りの成分がかぶる組だけ",
                     f"行が頼んだ品、列が持ってきた品。各行 {per} 匹（棚の並び {d['layouts']} 通り × {d['flies']} 匹）"))
    for j, name in enumerate(names + ["なし"]):
        x = L + j * cs + cs / 2
        out.append(f'<text x="{x}" y="{T - 8}" font-size="11" class="s" text-anchor="start" '
                   f'transform="rotate(-55 {x} {T - 8})">{esc(name)}</text>')
    for i, name in enumerate(names):
        y = T + i * cs
        out.append(txt(L - 8, y + cs / 2, name, 11.5, "s", "end"))
        for j in range(n + 1):
            v = conf[i][j] / per
            x = L + j * cs
            fill = "var(--c3)" if j == i else ("var(--mut)" if j == n else "var(--c2)")
            out.append(f'<rect x="{x+1}" y="{y+1}" width="{cs-2}" height="{cs-2}" rx="3" '
                       f'fill="{fill}" fill-opacity="{0 if v == 0 else 0.12 + 0.88 * v:.3f}"/>')
            if v >= 0.1:
                out.append(txt(x + cs / 2, y + cs / 2 + 1, f"{100*v:.0f}", 10.5,
                               "t" if v < 0.55 else "s", "middle", weight=600, mono=True))
    ly = T + n * cs + 16
    for k, (label, col) in enumerate((("頼んだ品を持ってきた", "c3"), ("別の品を持ってきた", "c2"))):
        out.append(f'<rect x="{L + k*170}" y="{ly}" width="10" height="10" rx="2" fill="var(--{col})"/>')
        out.append(txt(L + k * 170 + 16, ly + 5, label, 11, "s"))
    out.append(txt(L + 340, ly + 5, f"数字は%。取り違えと学習後の「欲しさ」の相関 r = {d['want_vs_mistakes_r']:.2f}",
                   10.5, "m"))

    # Bottom: the same errands on rewired brains.
    B = d["wiring"]
    y0 = ly + 58
    n_draws = len(B["swap"]["draws"]) + len(B["uniform"]["draws"])
    below = B["swap"]["below_real"] + B["uniform"]["below_real"]
    beats = "すべて" if below == n_draws else f"のうち{below}個"
    out.append(txt(20, y0, f"実測の配線は、張り替えた脳{n_draws}個{beats}より成績がよい", 13.5, "t", weight=600))
    out.append(txt(20, y0 + 20, f"同じ棚・同じハエで比べた正答率。点は脳1個ずつ。順位検定 p = "
                               f"{B['swap']['p_rank']:.2f}（次数保存）、{B['uniform']['p_rank']:.2f}（一様）", 11.5, "s"))
    lo, hi = 0.40, 0.80
    sx = lambda v: 150 + (v - lo) / (hi - lo) * (W - 190)
    rows = [("実測の配線", [B["real"]], "c3"), ("次数保存で張替", B["swap"]["draws"], "c1"),
            ("一様ランダム", B["uniform"]["draws"], "c1")]
    for v in (0.4, 0.5, 0.6, 0.7, 0.8):
        out.append(f'<line x1="{sx(v)}" y1="{y0+40}" x2="{sx(v)}" y2="{y0+40+3*30}" stroke="var(--grid)"/>')
        out.append(txt(sx(v), y0 + 40 + 3 * 30 + 12, f"{100*v:.0f}%", 10.5, "m", "middle", mono=True))
    for i, (label, vals, col) in enumerate(rows):
        y = y0 + 55 + i * 30
        out.append(txt(136, y, label, 11.5, "t" if i == 0 else "s", "end", weight=600 if i == 0 else 400))
        for j, v in enumerate(vals):
            jit = ((j * 7) % 5 - 2) * 2.2 if len(vals) > 1 else 0
            out.append(f'<circle cx="{sx(v):.1f}" cy="{y + jit:.1f}" r="{5.5 if len(vals) == 1 else 3.5}" '
                       f'fill="var(--{col})" fill-opacity="{1 if len(vals) == 1 else 0.6}"/>')
    out.append(txt(20, H - 16, "歩き方と匂いの広がりは仮定（docs/RESULTS.md 実験5）。脳の配線と学習則は実測・標準", 10.5, "m"))
    out.append("</svg>")
    return "".join(out)


def main():
    ensure(FIG)
    for i, fn in enumerate((fig1, fig2, fig3, fig4, fig5, fig6), start=1):
        path = os.path.join(FIG, f"fig{i}.svg")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(fn())
        print(f"  fig{i}.svg  {os.path.getsize(path)/1024:.1f} KB")


if __name__ == "__main__":
    main()
