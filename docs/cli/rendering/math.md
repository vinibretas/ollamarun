# 2D math

Display math (`$$…$$`, `\[…\]`) is laid out by a self-contained parser/layout
engine — stacked fractions, sqrt overlines, sum/prod/lim limits above-below,
integral side-limits, matrices, cases, aligned, `\boxed`, `\binom`, tall
delimiters, accents, `\mathbb`/`\mathcal`/`\mathfrak`/`\mathbf` — centered in
cyan, with flattened Unicode for inline math and `pylatexenc` as the
never-crashing fallback (unknown env/over-wide/over-budget). `--no-math` /
`/math` disable the conversion.
