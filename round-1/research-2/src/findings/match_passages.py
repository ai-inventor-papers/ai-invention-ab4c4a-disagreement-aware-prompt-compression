"""Keep only passages that occur verbatim (whitespace-collapsed) in the fetched source text."""
import json, re, pathlib, unicodedata
D = pathlib.Path(__file__).parent
srcs = json.load(open(D / "merged_sources.json"))

def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("–", "-").replace("—", "-").replace("­", "")
    t = re.sub(r"-\n\s*", "", t)          # de-hyphenate line breaks
    return re.sub(r"\s+", " ", t).strip()

report = []
for s in srcs:
    i = s["_idx"]
    f = D / "fetched" / f"{i}.txt"
    text = norm(f.read_text(errors="ignore")) if f.exists() else ""
    kept = []
    for p in s["supporting_passages"]:
        q = norm(p["quote"])
        found = None
        # try full quote, then the longest fragment between ellipses
        cands = [q] + sorted([c.strip(" .;:,") for c in re.split(r"\.\.\.|…|\[\.\.\.\]", q)], key=len, reverse=True)
        for c in cands:
            if len(c) < 30: continue
            # exact
            k = text.find(c)
            if k >= 0:
                found = c; break
            # case-insensitive
            k = text.lower().find(c.lower())
            if k >= 0:
                found = text[k:k+len(c)]; break
        if not found:
            # longest verbatim prefix (word boundary), or longest verbatim suffix
            for c in cands:
                if len(c) < 30: continue
                lo, hi = 0, len(c)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if text.find(c[:mid]) >= 0: lo = mid
                    else: hi = mid - 1
                pref = c[:lo]
                if " " in pref: pref = pref[:pref.rfind(" ")]
                lo2, hi2 = 0, len(c)
                while lo2 < hi2:
                    mid = (lo2 + hi2 + 1) // 2
                    if text.find(c[-mid:]) >= 0: lo2 = mid
                    else: hi2 = mid - 1
                suf = c[-lo2:] if lo2 else ""
                if " " in suf: suf = suf[suf.find(" ")+1:]
                best = max([pref, suf], key=len)
                if len(best) >= 50:
                    found = best.strip(" ,;:"); break
        if found:
            kept.append({"quote": found, "locator": p.get("locator")})
        else:
            report.append((i, "DROP", q[:90]))
    s["supporting_passages"] = kept
    if not kept: report.append((i, "NO-PASSAGE", s["title"][:60], len(text)))
json.dump(srcs, open(D / "merged_sources_checked.json", "w"), indent=1, ensure_ascii=False)
for r in report: print(*r)
print("sources with >=1 passage:", sum(1 for s in srcs if s["supporting_passages"]), "/", len(srcs))
