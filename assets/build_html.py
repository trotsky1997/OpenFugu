#!/usr/bin/env python3
"""Build an HTML version of twitter-ready.md with all LaTeX math rendered to
embedded PNG images (latex -> dvi -> dvipng -> base64 data URI). No JS, no CDN."""
import re, os, sys, hashlib, base64, subprocess, tempfile, html

SRC = "twitter-ready.md"
OUT = "twitter-ready.html"
DPI = 150
CACHE = {}

PREAMBLE = r"""\documentclass[12pt]{article}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage[utf8]{inputenc}
\pagestyle{empty}
\begin{document}
"""

CSS = """
:root{--fg:#1a1a1a;--mut:#666;--bg:#fff;--accent:#0b62d6;--line:#e2e2e2;--code:#f4f4f5}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:17px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
article{max-width:720px;margin:0 auto;padding:48px 24px 96px}
h1{font-size:2em;line-height:1.25;margin:.2em 0 .6em}
h2{font-size:1.45em;margin:1.8em 0 .5em;padding-top:.4em;border-top:1px solid var(--line)}
h3{font-size:1.15em;margin:1.4em 0 .4em}
p{margin:0 0 1em}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
strong{font-weight:650}
code{background:var(--code);padding:.1em .35em;border-radius:4px;
  font:.88em "SF Mono",ui-monospace,Menlo,Consolas,monospace}
blockquote{margin:1.2em 0;padding:.6em 1em;border-left:3px solid var(--accent);
  background:#f8fafd;color:#333;border-radius:0 6px 6px 0}
blockquote p{margin:0}
ul{margin:0 0 1em;padding-left:1.4em}li{margin:.3em 0}
hr{border:0;border-top:1px solid var(--line);margin:2em 0}
table{border-collapse:collapse;width:100%;margin:1.2em 0;font-size:.92em}
th,td{border:1px solid var(--line);padding:.5em .7em;text-align:left}
th{background:var(--code);font-weight:650}
img.math-inline{vertical-align:middle;height:1.05em;margin:0 .1em}
.disp{text-align:center;margin:1.4em 0;overflow-x:auto}
img.math-disp{max-width:100%}
@media (prefers-color-scheme:dark){
  :root{--fg:#e6e6e6;--mut:#999;--bg:#16171a;--accent:#5b9dff;--line:#2c2e33;--code:#23252a}
  blockquote{background:#1c2733}
  /* PNGs are black-on-transparent; invert so they read on dark bg */
  img.math-inline,img.math-disp{filter:invert(1) hue-rotate(180deg)}
}
"""

def render_math(tex, display):
    key = (tex, display)
    if key in CACHE:
        return CACHE[key]
    body = (r"\[" + tex + r"\]") if display else (r"$" + tex + r"$")
    doc = PREAMBLE + body + "\n\\end{document}\n"
    d = tempfile.mkdtemp()
    base = os.path.join(d, "f")
    open(base + ".tex", "w").write(doc)
    try:
        subprocess.run(["latex", "-interaction=nonstopmode", "-halt-on-error",
                        "-output-directory", d, base + ".tex"],
                       capture_output=True, cwd=d, timeout=30, check=True)
        # --depth gives baseline offset for inline vertical-align
        r = subprocess.run(["dvipng", "-D", str(DPI), "-T", "tight", "-bg", "Transparent",
                            "-z", "9", "--depth", "-o", base + ".png", base + ".dvi"],
                           capture_output=True, cwd=d, timeout=30, check=True)
        depth = 0
        m = re.search(rb"depth=(-?\d+)", r.stdout)
        if m: depth = int(m.group(1))
        png = open(base + ".png", "rb").read()
        b64 = base64.b64encode(png).decode()
        CACHE[key] = (b64, depth)
        return CACHE[key]
    except subprocess.CalledProcessError:
        CACHE[key] = (None, 0)
        return CACHE[key]

# PLACEHOLDER_HELPERS

def img_tag(tex, display):
    b64, depth = render_math(tex, display)
    if b64 is None:
        return f'<code>{html.escape(tex)}</code>'   # fallback if latex failed
    cls = "math-disp" if display else "math-inline"
    style = "" if display else f' style="vertical-align:-{depth}px"'
    return (f'<img class="{cls}" alt="{html.escape(tex)}" '
            f'src="data:image/png;base64,{b64}"{style}>')

def inline_fmt(s):
    # protect inline math first
    maths = []
    def stash(m):
        maths.append(m.group(1)); return f"\x00{len(maths)-1}\x00"
    s = re.sub(r'\$([^$]+?)\$', stash, s)
    s = html.escape(s)
    # bold, italic, inline code
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'`([^`]+?)`', r'<code>\1</code>', s)
    s = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', s)
    # links [text](url)
    s = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', s)
    # restore math as images
    s = re.sub(r'\x00(\d+)\x00', lambda m: img_tag(maths[int(m.group(1))], False), s)
    return s

def main():
    lines = open(SRC, encoding="utf-8").read().split("\n")
    out = ['<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>How Fugu Is Implemented</title>', "<style>", CSS, "</style></head><body><article>"]
    i, n = 0, len(lines)
    while i < n:
        l = lines[i]
        s = l.strip()
        if s == "":
            i += 1; continue
        # display math block
        if s == "$$":
            buf = []; i += 1
            while i < n and lines[i].strip() != "$$":
                buf.append(lines[i]); i += 1
            i += 1
            out.append('<div class="disp">' + img_tag("\n".join(buf), True) + "</div>")
            continue
        # heading
        m = re.match(r'^(#{1,6})\s+(.*)', l)
        if m:
            lv = len(m.group(1)); out.append(f"<h{lv}>{inline_fmt(m.group(2))}</h{lv}>"); i += 1; continue
        # blockquote (may span lines)
        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(re.sub(r'^\s*>\s?', '', lines[i])); i += 1
            out.append("<blockquote>" + inline_fmt(" ".join(buf)) + "</blockquote>")
            continue
        # table
        if s.startswith("|"):
            buf = []
            while i < n and lines[i].strip().startswith("|"):
                buf.append(lines[i].strip()); i += 1
            out.append(render_table(buf)); continue
        # list
        if re.match(r'^\s*([-*+]|\d+\.)\s', l):
            buf = []
            while i < n and re.match(r'^\s*([-*+]|\d+\.)\s', lines[i]):
                buf.append(re.sub(r'^\s*([-*+]|\d+\.)\s+', '', lines[i])); i += 1
            out.append("<ul>" + "".join(f"<li>{inline_fmt(x)}</li>" for x in buf) + "</ul>")
            continue
        # horizontal rule
        if re.match(r'^-{3,}$', s):
            out.append("<hr>"); i += 1; continue
        # paragraph
        out.append("<p>" + inline_fmt(l) + "</p>"); i += 1
    out.append("</article></body></html>")
    open(OUT, "w", encoding="utf-8").write("\n".join(out))
    print(f"wrote {OUT}; {len(CACHE)} unique formulas rendered")

def render_table(rows):
    cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.match(r'^:?-+:?$', c or '-') for c in r)]
    head, body = cells[0], cells[1:]
    h = "<tr>" + "".join(f"<th>{inline_fmt(c)}</th>" for c in head) + "</tr>"
    b = "".join("<tr>" + "".join(f"<td>{inline_fmt(c)}</td>" for c in r) + "</tr>" for r in body)
    return f"<table>{h}{b}</table>"

if __name__ == "__main__":
    main()

