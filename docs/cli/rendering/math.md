# 2D math

Display math is not printed as LaTeX. A self-contained parser and layout
engine (`ollamarun_lib/mathrender.py`) turns it into 2D Unicode art —
stacked fractions, radical overlines, limits above and below big operators —
centered in cyan. Inline math is flattened to a single line instead.

The engine handles display math written as `$$…$$` or `\[…\]` on its own
line(s); math inside a sentence, list item or table cell goes through the
inline path so it stays inside its block.

## What it renders

Real output from the engine (the text here; it is centered and cyan on a
terminal):

| LaTeX | Rendered as |
| --- | --- |
| `\frac{a+b}{2}` | A stacked fraction with a rule. |
| `\sqrt{x^2+y^2}` | A radical with an overline. |
| `\sum_{i=1}^{n} …` | `∑` with limits above and below. |
| `\int_0^\infty …` | `∫₀^∞`, tight side limits. |
| `\lim_{x \to 0} …` | `lim` with `x → 0` beneath. |
| `\begin{pmatrix} … \end{pmatrix}` | A parenthesised matrix. |
| `\begin{cases} … \end{cases}` | A brace with stacked cases. |
| `\boxed{E = mc^2}` | A rounded box. |
| `\binom{n}{k}` | Binomial in big parentheses. |
| `\vec{v}\cdot\hat{n} + \bar{x}` | Accents above the letters. |

Verbatim, as the layout engine returns them:

```text
\frac{a+b}{2}
   a + b
   ─────
     2

\sqrt{x^2+y^2}
   ───────
   √x² + y²

\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
     n       n(n + 1)
     ∑   i = ────────
   i = 1        2

\lim_{x \to 0} \frac{\sin x}{x} = 1
         sin x
    lim  ───── = 1
   x → 0   x

\begin{pmatrix} a & b \\ c & d \end{pmatrix}
   ⎛ a  b ⎞
   ⎝ c  d ⎠

\begin{cases} x & x>0 \\ -x & x\le 0 \end{cases}
   ⎧ x  x > 0
   ⎩ -x  x ≤ 0

\begin{aligned} a &= b+c \\ d &= e \end{aligned}
   a = b + c
   d = e

\boxed{E = mc^2}
   ╭─────────╮
   │ E = mc² │
   ╰─────────╯

\binom{n}{k} = \frac{n!}{k!(n-k)!}
   ⎛n⎞       n!
   ⎜ ⎟ = ──────────
   ⎝k⎠   k!(n - k)!

\mathbb{R} \subset \mathbb{C}, \; \mathcal{L}
   ℝ ⊂ ℂ,  ℒ

\alpha + \beta = \gamma \ne \delta \approx \epsilon
   α + β = γ ≠ δ ≈ ε
```

Accents (`\vec`, `\hat`, `\bar`), Greek letters, `\mathbb`/`\mathcal`,
`\left…\right` delimiters, sub/superscripts (stacked for big operators,
raised otherwise) and `\,`-style spacing come along too.

## Inline math

`$x^2 + y_i$` → `x² + yᵢ`; `$\alpha\beta$` → `αβ`; `$\frac{1}{2}$` → `1/2`.
Inline conversion is line-based, so it stays safe inside tables and
sentences.

## Fallbacks, not crashes

Anything the layout engine cannot lay out — an unknown environment, an
unknown macro, a formula too wide for the terminal, an internal over-budget
— falls back to the `pylatexenc` flattening path and is shown literally
instead of half-rendered:

```text
$$ \begin{foo} x + 1 \end{foo} $$     →     x + 1      (literal block)
$$ \notacommand{x} $$                 →     x          (literal block)
```

Nothing unsupported ever takes the rendering down; the expression is
reproduced as best it can be.

## Live example

Asked to repeat a formula exactly, the model's answer is laid out by the
engine on the way in:

```console
$ ollamarun --no-tools llama3.2:3b "Repeat this exact line, with no extra text: \$\$e^{i\pi} + 1 = 0\$\$"
llama3.2:3b ›
                            e^(iπ) + 1 = 0
```

The same run with `--no-math` shows the source instead:

```text
$$e^{i\pi} + 1 = 0$$
```

`/math [on|off]` toggles conversion mid-session; `--no-math` is the
per-run flag. Conversion runs in the renderer's markdown path, so it applies
to live streaming, [transcript replays](transcript.md) and HTML exports
alike.
