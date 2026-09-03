"""s/step per variant per card, from the trainer's own timestamped `Step N:` lines.

The atmosphere loader is bimodal, not noisy: aug26 measured "twenty steps at
17-18 s, then one interval at 163-216 s" as the time_buffer window refills
against CFS.  Over a 70-batch probe exactly one such stall lands in one run and
not another, so an end-to-end mean is not a step-time measurement -- it is a
coin flip on whether the refill was caught.  Report the MEDIAN of the per-window
rates, and report the max alongside it so the stall stays visible.
"""
import glob, os, re, sys, statistics as st, datetime as dt

pat = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+) .*?Step (\d+):")

def windows(path):
    pts = []
    for line in open(path, errors="ignore"):
        m = pat.match(line)
        if m:
            t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            pts.append((t.timestamp() + int(m.group(2)) / 1000, int(m.group(3))))
    # skip the first point: the interval into it carries dataloader warm-up
    return [(b1 - b0 and (t1 - t0) / (b1 - b0))
            for (t0, b0), (t1, b1) in zip(pts[1:], pts[2:])]

rows = {}
for tag in sorted(os.listdir(f"{sys.argv[1]}/memprobe")):
    for f in sorted(glob.glob(f"{sys.argv[1]}/memprobe/{tag}/*.log")):
        w = [x for x in windows(f) if x]
        if len(w) >= 3:
            rows.setdefault(os.path.basename(f)[:-4], {})[tag] = w

tags = sorted({t for v in rows.values() for t in v})
if not rows:
    sys.exit("no variant has enough step lines yet")
kw = max(len(k) for k in rows)
print(f"{'variant':<{kw}} " + " ".join(f"{t+' median':>15}{'max':>8}" for t in tags)
      + "   ratio(median)")
for k, v in rows.items():
    cells = []
    for t in tags:
        cells.append(f"{st.median(v[t]):>15.3f}{max(v[t]):>8.2f}" if t in v
                     else f"{'-':>15}{'-':>8}")
    r = ""
    if len(tags) == 2 and all(t in v for t in tags):
        r = f"{st.median(v[tags[0]]) / st.median(v[tags[1]]):.3f}"
    print(f"{k:<{kw}} " + " ".join(cells) + f"   {r:>10}")
