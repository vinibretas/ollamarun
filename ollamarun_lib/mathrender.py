"""LaTeX math rendering: regex-level Unicode conversion plus the 2D display
math layout engine.

These two pieces are combined in one module because they call into each
other: `convert_math` (regex-level) falls through to `render_inline_math`
(layout engine) for inline math, and the layout engine falls back to
`latex_to_unicode`/`convert_math` (regex-level) wherever a construct is not
supported -- `_tokenize_math` -> `_MathParser` -> an AST of `M*` node classes
(`MFrac`, `MSqrt`, `MScript`, `MBigOp`, `MMatrix`, `MCases`, `MAligned`,
`MBoxed`, etc.) -> `_layout()` builds a `_Box` (list-of-strings + width)
recursively -> `_flatten()` is the plain-text fallback path.
`render_display_math()`/`render_inline_math()` are the public entry points;
both must never raise -- anything unsupported (unknown environment,
over-wide, over-budget) falls through to `pylatexenc` via `convert_math()`.
Changes here need care: the layout is hand-rolled string-grid arithmetic,
not a real typesetting library.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from rich.align import Align
from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text

from .constants import HAS_PYLATEXENC, LatexNodes2Text

SUPERSCRIPTS = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "=": "⁼",
    "(": "⁽", ")": "⁾", "n": "ⁿ", "i": "ⁱ", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ",
    "d": "ᵈ", "e": "ᵉ", "f": "ᶠ", "g": "ᵍ", "h": "ʰ", "j": "ʲ", "k": "ᵏ",
    "l": "ˡ", "m": "ᵐ", "o": "ᵒ", "p": "ᵖ", "r": "ʳ", "s": "ˢ", "t": "ᵗ",
    "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
}
SUBSCRIPTS = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
    "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "=": "₌",
    "(": "₍", ")": "₎", "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
    "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ",
    "s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ", "x": "ₓ",
}

_CODE_RE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]*`", re.S)
_SCRIPT_RE = re.compile(
    r"(?P<base>[^\s])(?P<op>[_^])(?:\{(?P<braced>[^{}]+)\}|(?P<plain>-?[A-Za-z0-9]))")


def _map_scripts(text: str) -> str:
    def repl(match: re.Match) -> str:
        table = SUPERSCRIPTS if match.group("op") == "^" else SUBSCRIPTS
        target = match.group("braced") or match.group("plain") or ""
        mapped = [table.get(ch) for ch in target]
        if all(mapped):
            return match.group("base") + "".join(mapped)
        return match.group(0)

    for _ in range(3):  # chained scripts like x_i^2 need repeated passes
        updated = _SCRIPT_RE.sub(repl, text)
        if updated == text:
            break
        text = updated
    return text


def latex_to_unicode(tex: str) -> str:
    """Best-effort LaTeX -> readable Unicode; never raises."""
    converted = None
    if HAS_PYLATEXENC:
        try:
            converted = LatexNodes2Text(math_mode="text").latex_to_text(tex)
        except Exception:
            converted = None
    if converted is None:
        converted = tex
        for macro, char in (("\\alpha", "α"), ("\\beta", "β"), ("\\gamma", "γ"),
                            ("\\delta", "δ"), ("\\epsilon", "ε"), ("\\theta", "θ"),
                            ("\\lambda", "λ"), ("\\mu", "μ"), ("\\pi", "π"),
                            ("\\sigma", "σ"), ("\\phi", "φ"), ("\\omega", "ω"),
                            ("\\times", "×"), ("\\cdot", "·"), ("\\leq", "≤"),
                            ("\\geq", "≥"), ("\\neq", "≠"), ("\\approx", "≈"),
                            ("\\infty", "∞"), ("\\sum", "∑"), ("\\prod", "∏"),
                            ("\\int", "∫"), ("\\sqrt", "√"), ("\\partial", "∂"),
                            ("\\nabla", "∇"), ("\\rightarrow", "→"), ("\\to", "→")):
            converted = converted.replace(macro, char)
    return _map_scripts(converted)


_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.S)
_BARE_BOXED_RE = re.compile(r"\\boxed\s*\{")


def _replace_bare_boxed(text: str) -> str:
    """Renders `\\boxed{...}` that is not inside $...$; handles nested braces."""
    search_from = 0
    while True:
        match = _BARE_BOXED_RE.search(text, search_from)
        if match is None:
            return text
        if match.start() > 0 and text[match.start() - 1] == "$":
            search_from = match.end()
            continue
        start = match.end() - 1
        depth = 0
        end = -1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end == -1:
            return text
        replacement = "[ " + render_inline_math(text[start + 1:end]) + " ]"
        text = text[:match.start()] + replacement + text[end + 1:]
        search_from = match.start() + len(replacement)
_INLINE_MATH_RE = re.compile(
    r"\\\((.+?)\\\)"
    r"|\$(?!\$)([^\s$][^$\n]*?)(?<!\s)\$(?!\$)"
    r"|\$(?!\$)\s+([^$\n]+?)\s+\$(?!\$)")


def _display_repl(match: re.Match) -> str:
    tex = match.group(1) or match.group(2) or ""
    return "\n\n```\n" + latex_to_unicode(tex.strip()) + "\n```\n\n"


def _inline_repl(match: re.Match) -> str:
    tex = (match.group(1) or match.group(2) or match.group(3) or "").strip()
    if re.fullmatch(r"[\d,.;:]+", tex):
        return match.group(0)
    return render_inline_math(tex)


def _convert_math_segment(segment: str, math: bool) -> str:
    if not math:
        return segment
    segment = _DISPLAY_MATH_RE.sub(_display_repl, segment)
    segment = _replace_bare_boxed(segment)
    return _INLINE_MATH_RE.sub(_inline_repl, segment)


def convert_math(text: str, math: bool = True) -> str:
    """Convert math outside code fences/inline spans; leaves code untouched."""
    if not math:
        return text
    out: list[str] = []
    pos = 0
    for match in _CODE_RE.finditer(text):
        out.append(_convert_math_segment(text[pos:match.start()], math))
        out.append(match.group(0))
        pos = match.end()
    out.append(_convert_math_segment(text[pos:], math))
    return "".join(out)


MATH_SYMBOLS: dict[str, str] = {
    # greek lower
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "omicron": "ο", "pi": "π", "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ",
    "sigma": "σ", "varsigma": "ς", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "varphi": "ϕ", "chi": "χ", "psi": "ψ", "omega": "ω",
    # greek upper
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω",
    # operators and relations
    "pm": "±", "mp": "∓", "times": "×", "div": "÷", "cdot": "·", "ast": "∗",
    "star": "⋆", "circ": "∘", "bullet": "•", "oplus": "⊕", "ominus": "⊖",
    "otimes": "⊗", "oslash": "⊘", "odot": "⊙", "le": "≤", "leq": "≤",
    "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠", "equiv": "≡", "approx": "≈",
    "sim": "∼", "simeq": "≃", "cong": "≅", "propto": "∝", "ll": "≪", "gg": "≫",
    "subset": "⊂", "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇",
    "in": "∈", "notin": "∉", "ni": "∋", "cup": "∪", "cap": "∩",
    "setminus": "∖", "emptyset": "∅", "varnothing": "∅",
    "forall": "∀", "exists": "∃", "nexists": "∄", "neg": "¬", "lnot": "¬",
    "land": "∧", "wedge": "∧", "lor": "∨", "vee": "∨", "models": "⊨",
    "vdash": "⊢", "dashv": "⊣", "perp": "⊥", "parallel": "∥",
    # arrows
    "to": "→", "rightarrow": "→", "leftarrow": "←", "gets": "←",
    "leftrightarrow": "↔", "Rightarrow": "⇒", "Leftarrow": "⇐",
    "Leftrightarrow": "⇔", "mapsto": "↦", "uparrow": "↑", "downarrow": "↓",
    "longrightarrow": "⟶", "longleftarrow": "⟵", "longmapsto": "⟼",
    "nearrow": "↗", "searrow": "↘", "hookrightarrow": "↪",
    # misc
    "infty": "∞", "partial": "∂", "nabla": "∇", "angle": "∠", "degree": "°",
    "deg": "°", "prime": "′", "ldots": "…", "dots": "…", "cdots": "⋯",
    "vdots": "⋮", "ddots": "⋱", "aleph": "ℵ", "hbar": "ℏ", "ell": "ℓ",
    "Re": "ℜ", "Im": "ℑ", "wp": "℘", "surd": "√", "checkmark": "✓",
    "dagger": "†", "ddagger": "‡", "S": "§", "P": "¶", "copyright": "©",
    "flat": "♭", "sharp": "♯", "natural": "♮", "clubsuit": "♣",
    "diamondsuit": "♦", "heartsuit": "♥", "spadesuit": "♠",
    "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉",
    "langle": "⟨", "rangle": "⟩", "lVert": "‖", "rVert": "‖",
}

MATH_FUNCTIONS = {
    "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "coth", "log", "ln", "lg", "exp", "det", "dim",
    "ker", "hom", "arg", "deg", "gcd", "lcm", "mod", "bmod", "pmod",
}

BIG_OPS: dict[str, str] = {
    "sum": "∑", "prod": "∏", "coprod": "∐", "int": "∫", "iint": "∬",
    "iiint": "∭", "oint": "∮", "oiint": "∯", "bigcup": "⋃", "bigcap": "⋂",
    "bigvee": "⋁", "bigwedge": "⋀", "bigoplus": "⊕", "bigotimes": "⊗",
    "bigodot": "⊙", "biguplus": "⨄", "bigsqcup": "⨆",
}

LIMIT_SIDE_OPS = {"int", "iint", "iiint", "oint", "oiint"}

LIMIT_NAME_OPS = {"lim", "max", "min", "sup", "inf", "argmax", "argmin", "limsup", "liminf"}

ACCENT_COMBINING = {
    "vec": "\u20d7", "hat": "\u0302", "widehat": "\u0302", "bar": "\u0304",
    "tilde": "\u0303", "widetilde": "\u0303", "dot": "\u0307",
    "ddot": "\u0308", "acute": "\u0301", "grave": "\u0300", "check": "\u030c",
    "breve": "\u0306", "overline": "\u0305", "underline": "\u0332",
}

_SPACED_ATOMS = {
    "=", "+", "-", "−", "±", "∓", "×", "·", "÷", "≤", "≥", "≠", "≈", "≡",
    "<", ">", "→", "←", "↔", "⇒", "⇔", "∈", "∉", "⊂", "⊆", "⊃", "⊇",
    "∪", "∩", "∧", "∨", "∼", "≃", "≅", "∝", "⊢", "⊨", "≪", "≫",
}

_DOUBLE_STRUCK = {
    "A": "𝔸", "B": "𝔹", "C": "ℂ", "D": "𝔻", "E": "𝔼", "F": "𝔽", "G": "𝔾",
    "H": "ℍ", "I": "𝕀", "J": "𝕁", "K": "𝕂", "L": "𝕃", "M": "𝕄", "N": "ℕ",
    "O": "𝕆", "P": "ℙ", "Q": "ℚ", "R": "ℝ", "S": "𝕊", "T": "𝕋", "U": "𝕌",
    "V": "𝕍", "W": "𝕎", "X": "𝕏", "Y": "𝕐", "Z": "ℤ",
}
_SCRIPT_CAPS = {
    "A": "𝒜", "B": "ℬ", "C": "𝒞", "D": "𝒟", "E": "ℰ", "F": "ℱ", "G": "𝒢",
    "H": "ℋ", "I": "ℐ", "J": "𝒥", "K": "𝒦", "L": "ℒ", "M": "ℳ", "N": "𝒩",
    "O": "𝒪", "P": "𝒫", "Q": "𝒬", "R": "ℛ", "S": "𝒮", "T": "𝒯", "U": "𝒰",
    "V": "𝒱", "W": "𝒲", "X": "𝒳", "Y": "𝒴", "Z": "𝒵",
}
_FRAKTUR_CAPS = {
    "A": "𝔄", "B": "𝔅", "C": "ℭ", "D": "𝔇", "E": "𝔈", "F": "𝔉", "G": "𝔊",
    "H": "ℌ", "I": "ℑ", "J": "𝔍", "K": "𝔎", "L": "𝔏", "M": "𝔐", "N": "𝔑",
    "O": "𝔒", "P": "𝔓", "Q": "𝔔", "R": "ℜ", "S": "𝔖", "T": "𝔗", "U": "𝔘",
    "V": "𝔙", "W": "𝔚", "X": "𝔛", "Y": "𝔜", "Z": "ℨ",
}


def _bold_letters(text: str) -> str:
    out = []
    for ch in text:
        if "A" <= ch <= "Z":
            out.append(chr(0x1D400 + ord(ch) - ord("A")))
        elif "a" <= ch <= "z":
            out.append(chr(0x1D41A + ord(ch) - ord("a")))
        elif "0" <= ch <= "9":
            out.append(chr(0x1D7CE + ord(ch) - ord("0")))
        else:
            out.append(ch)
    return "".join(out)


_MATH_STYLES: dict[str, Any] = {
    "mathbb": _DOUBLE_STRUCK, "mathcal": _SCRIPT_CAPS, "mathscr": _SCRIPT_CAPS,
    "mathfrak": _FRAKTUR_CAPS, "mathbf": _bold_letters, "boldsymbol": _bold_letters,
}

_RAW_TEXT_CMDS = {"text", "mathrm", "operatorname", "mbox", "textrm", "mathit",
                  "mathbf", "mathsf", "mathtt", "textnormal", "textup", "textbf"}

_TALL_DELIMS: dict[str, tuple[str, str, str]] = {
    "(": ("⎛", "⎜", "⎝"), ")": ("⎞", "⎟", "⎠"),
    "[": ("⎡", "⎢", "⎣"), "]": ("⎤", "⎥", "⎦"),
    "{": ("⎧", "⎪", "⎩"), "}": ("⎫", "⎪", "⎭"),
    "|": ("│", "│", "│"), "‖": ("║", "║", "║"),
    "⟨": ("⎛", "⎜", "⎝"), "⟩": ("⎞", "⎟", "⎠"),
}


@dataclass
class _Tok:
    kind: str
    value: str = ""


@dataclass
class MAtom:
    text: str


@dataclass
class MSpace:
    width: int = 1


@dataclass
class MRow:
    items: list[Any] = field(default_factory=list)


@dataclass
class MFrac:
    num: Any
    den: Any
    bar: bool = True


@dataclass
class MSqrt:
    body: Any
    index: Optional[Any] = None


@dataclass
class MScript:
    base: Any
    sup: Optional[Any] = None
    sub: Optional[Any] = None


@dataclass
class MBigOp:
    symbol: str
    sub: Optional[Any] = None
    sup: Optional[Any] = None
    above: bool = True


@dataclass
class MDelim:
    left: str
    body: Any
    right: str


@dataclass
class MMatrix:
    rows: list[list[Any]] = field(default_factory=list)
    kind: str = "pmatrix"


@dataclass
class MCases:
    rows: list[list[Any]] = field(default_factory=list)


@dataclass
class MAligned:
    rows: list[list[Any]] = field(default_factory=list)


@dataclass
class MAccent:
    kind: str
    body: Any


@dataclass
class MBoxed:
    body: Any


@dataclass
class MText:
    text: str


_CMD_NAME_RE = re.compile(r"[A-Za-z]+")


def _tokenize_math(tex: str) -> list[_Tok]:
    tokens: list[_Tok] = []
    i = 0
    n = len(tex)
    while i < n:
        ch = tex[i]
        if ch == "\\":
            if i + 1 < n and tex[i + 1] == "\\":
                tokens.append(_Tok("rowbreak"))
                i += 2
                continue
            j = i + 1
            if j < n and tex[j].isalpha():
                match = _CMD_NAME_RE.match(tex, j)
                name = match.group(0)
                tokens.append(_Tok("cmd", name))
                i = match.end()
                if name in _RAW_TEXT_CMDS and i < n and tex[i] == "{":
                    depth = 0
                    start = i
                    while i < n:
                        if tex[i] == "{":
                            depth += 1
                        elif tex[i] == "}":
                            depth -= 1
                            if depth == 0:
                                break
                        i += 1
                    tokens.append(_Tok("rawgroup", tex[start + 1:i]))
                    i = min(i + 1, n)
                continue
            if j < n:
                tokens.append(_Tok("cmd", tex[j]))
                i = j + 1
                continue
            i += 1
            continue
        if ch == "{":
            tokens.append(_Tok("lbrace"))
        elif ch == "}":
            tokens.append(_Tok("rbrace"))
        elif ch == "^":
            tokens.append(_Tok("sup"))
        elif ch == "_":
            tokens.append(_Tok("sub"))
        elif ch == "&":
            tokens.append(_Tok("amp"))
        elif ch == "~":
            tokens.append(_Tok("space", "1"))
        elif ch.isspace():
            pass
        else:
            tokens.append(_Tok("char", ch))
        i += 1
    return tokens


class _MathParser:
    def __init__(self, tokens: list[_Tok]):
        self.toks = tokens
        self.pos = 0
        self.budget = 400
        self.stop_at_right = False

    # ── token helpers ───────────────────────────────────────────────────────

    def peek(self, kind: Optional[str] = None) -> Optional[_Tok]:
        if self.pos >= len(self.toks):
            return None
        tok = self.toks[self.pos]
        if kind is not None and tok.kind != kind:
            return None
        return tok

    def next(self) -> Optional[_Tok]:
        tok = self.peek()
        if tok is not None:
            self.pos += 1
        return tok

    def expect_group(self) -> Any:
        tok = self.peek()
        if tok is None:
            return MRow([])
        if tok.kind == "lbrace":
            self.next()
            body = self.parse_row()
            if self.peek("rbrace"):
                self.next()
            return body
        atom = self.parse_atom()
        return atom if atom is not None else MRow([])

    # ── grammar ─────────────────────────────────────────────────────────────

    def parse(self) -> Any:
        return self.parse_row()

    def parse_row(self) -> MRow:
        items: list[Any] = []
        while self.pos < len(self.toks):
            tok = self.toks[self.pos]
            if tok.kind == "rbrace":
                break
            if tok.kind == "cmd" and tok.value == "right" and self.stop_at_right:
                break
            item = self.parse_atom()
            if item is None:
                break
            items.append(item)
            self.budget -= 1
            if self.budget <= 0:
                raise ValueError("math token budget exceeded")
        return MRow(items)

    def parse_atom(self) -> Any:
        tok = self.next()
        if tok is None:
            return None
        if tok.kind == "lbrace":
            body = self.parse_row()
            if self.peek("rbrace"):
                self.next()
            return self.parse_scripts(body)
        if tok.kind == "char":
            return self.parse_scripts(MAtom(tok.value))
        if tok.kind == "space":
            return MSpace(int(tok.value or "1"))
        if tok.kind == "sup" or tok.kind == "sub":
            self.pos -= 1
            return self.parse_scripts(MAtom(""))
        if tok.kind == "cmd":
            node = self.parse_command(tok.value)
            if node is None:
                return None
            return self.parse_scripts(node)
        return None

    def parse_scripts(self, base: Any) -> Any:
        sup = sub = None
        while True:
            tok = self.peek()
            if tok is None or tok.kind not in ("sup", "sub"):
                break
            self.next()
            argument = self.parse_script_argument()
            if tok.kind == "sup":
                sup = argument
            else:
                sub = argument
        if sup is None and sub is None:
            return base
        if isinstance(base, MBigOp):
            if sup is not None:
                base.sup = sup
            if sub is not None:
                base.sub = sub
            return base
        return MScript(base, sup=sup, sub=sub)

    def parse_script_argument(self) -> Any:
        tok = self.peek()
        if tok is not None and tok.kind == "lbrace":
            self.next()
            body = self.parse_row()
            if self.peek("rbrace"):
                self.next()
            return body
        # a single token argument, without swallowing following ^/_ scripts
        tok = self.next()
        if tok is None:
            return MAtom("")
        if tok.kind == "char":
            return MAtom(tok.value)
        if tok.kind == "space":
            return MSpace(int(tok.value or "1"))
        if tok.kind == "cmd":
            return self.parse_command(tok.value) or MAtom("")
        if tok.kind in {"sup", "sub"}:
            self.pos -= 1
            return MAtom("")
        if tok.kind == "lbrace":
            body = self.parse_row()
            if self.peek("rbrace"):
                self.next()
            return body
        return MAtom("")

    def parse_command(self, name: str) -> Any:
        if name in {"frac", "dfrac", "tfrac", "cfrac"}:
            return MFrac(self.expect_group(), self.expect_group())
        if name in {"binom", "dbinom", "tbinom", "choose"}:
            return MDelim("(", MFrac(self.expect_group(), self.expect_group(), bar=False), ")")
        if name == "sqrt":
            index = None
            tok = self.peek()
            if tok is not None and tok.kind == "char" and tok.value == "[":
                self.next()
                collected: list[_Tok] = []
                while self.pos < len(self.toks):
                    item = self.next()
                    if item is None:
                        break
                    if item.kind == "char" and item.value == "]":
                        break
                    collected.append(item)
                index = _MathParser(collected).parse()
            return MSqrt(self.expect_group(), index=index)
        if name in BIG_OPS:
            return MBigOp(BIG_OPS[name], above=name not in LIMIT_SIDE_OPS)
        if name in LIMIT_NAME_OPS:
            return MBigOp(name, above=True)
        if name in MATH_FUNCTIONS:
            return MAtom(name + " ")
        if name in MATH_SYMBOLS:
            return MAtom(MATH_SYMBOLS[name])
        if name in _MATH_STYLES:
            tok = self.peek("rawgroup")
            if tok is not None:
                self.next()
                content = tok.value
            else:
                content = _flatten(self.expect_group())
            style = _MATH_STYLES[name]
            if callable(style):
                return MAtom(style(content))
            return MAtom("".join(style.get(ch, ch) for ch in content))
        if name in _RAW_TEXT_CMDS:
            tok = self.peek("rawgroup")
            if tok is not None:
                self.next()
                return MText(tok.value)
            return MText(_flatten(self.expect_group()))
        if name in ACCENT_COMBINING:
            return MAccent(name, self.expect_group())
        if name in {"boxed", "fbox"}:
            return MBoxed(self.expect_group())
        if name in {"left", "right", "big", "Big", "bigl", "bigr", "Bigl", "Bigr",
                    "bigg", "Bigg", "middle"}:
            if name == "left":
                left = self.read_delimiter()
                previous = self.stop_at_right
                self.stop_at_right = True
                body = self.parse_row()
                self.stop_at_right = previous
                right = "."
                tok = self.peek()
                if tok is not None and tok.kind == "cmd" and tok.value == "right":
                    self.next()
                    right = self.read_delimiter()
                return MDelim(left, body, right)
            return MAtom("")
        if name == "begin":
            return self.parse_environment()
        if name == "end":
            return MAtom("")
        if name in {",", ":", "thinspace", "enspace"}:
            return MSpace(1)
        if name in {";", " ", "medspace", "thickspace"}:
            return MSpace(2)
        if name == "!":
            return MSpace(0)
        if name == "quad":
            return MSpace(3)
        if name == "qquad":
            return MSpace(6)
        if name in {"displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle",
                    "limits", "nolimits", "notag", "nonumber", "label", "tag"}:
            return MAtom("")
        if name in {"%", "&", "$", "#", "_", "{", "}"}:
            return MAtom(name)
        if name in {"vert", "mid"}:
            return MAtom("|")
        if name in {"Vert"}:
            return MAtom("‖")
        if name == " ":
            return MSpace(1)
        # unknown macro: let pylatexenc resolve it (never raises)
        converted = latex_to_unicode("\\" + name)
        return MAtom(converted if converted and converted != "\\" + name else "\\" + name)

    def read_delimiter(self) -> str:
        tok = self.peek()
        if tok is None:
            return "."
        self.next()
        if tok.kind == "cmd":
            if tok.value in {"{", "}"}:
                return tok.value
            return MATH_SYMBOLS.get(tok.value, tok.value)
        return tok.value or "."

    def _read_env_name(self) -> str:
        if not self.peek("lbrace"):
            return ""
        self.next()
        parts: list[str] = []
        while self.pos < len(self.toks):
            tok = self.next()
            if tok is None or tok.kind == "rbrace":
                break
            parts.append(tok.value)
        return "".join(parts)

    def _read_raw_braced(self) -> str:
        if not self.peek("lbrace"):
            return ""
        self.next()
        parts: list[str] = []
        depth = 1
        while self.pos < len(self.toks):
            tok = self.next()
            if tok is None:
                break
            if tok.kind == "lbrace":
                depth += 1
            elif tok.kind == "rbrace":
                depth -= 1
                if depth == 0:
                    break
            parts.append(tok.value)
        return "".join(parts)

    def parse_environment(self) -> Any:
        env = self._read_env_name()
        if env == "array":
            self._read_raw_braced()
        body: list[_Tok] = []
        depth = 0
        while self.pos < len(self.toks):
            tok = self.toks[self.pos]
            if tok.kind == "cmd" and tok.value == "begin":
                depth += 1
            elif tok.kind == "cmd" and tok.value == "end":
                if depth == 0:
                    self.next()
                    self._read_env_name()
                    break
                depth -= 1
            body.append(tok)
            self.pos += 1
        rows = self._split_token_rows(body)
        parsed = [[self._parse_tokens(cell) for cell in row] for row in rows]
        if env in {"pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "matrix", "array"}:
            kind = "matrix" if env == "array" else env
            return MMatrix(parsed, kind)
        if env in {"cases", "dcases", "rcases"}:
            return MCases(parsed)
        if env in {"aligned", "align", "align*", "split", "gathered", "gather", "eqnarray"}:
            return MAligned(parsed)
        raise ValueError(f"unsupported environment {env!r}")

    @staticmethod
    def _parse_tokens(tokens: list[_Tok]) -> Any:
        return _MathParser(tokens).parse()

    @staticmethod
    def _split_token_rows(tokens: list[_Tok]) -> list[list[list[_Tok]]]:
        raw_rows: list[list[_Tok]] = [[]]
        depth = 0
        for tok in tokens:
            if tok.kind == "lbrace":
                depth += 1
            elif tok.kind == "rbrace":
                depth = max(0, depth - 1)
            if tok.kind == "rowbreak" and depth == 0:
                raw_rows.append([])
                continue
            raw_rows[-1].append(tok)
        rows: list[list[list[_Tok]]] = []
        for row in raw_rows:
            cells: list[list[_Tok]] = [[]]
            for tok in row:
                if tok.kind == "amp":
                    cells.append([])
                else:
                    cells[-1].append(tok)
            rows.append(cells)
        return rows


# ── box layout ──────────────────────────────────────────────────────────────

@dataclass
class _Box:
    lines: list[str]
    baseline: int = 0


def _box_width(box: _Box) -> int:
    return max((len(line) for line in box.lines), default=0)


def _box_pad(box: _Box, width: int, align: str = "left") -> _Box:
    lines: list[str] = []
    for line in box.lines:
        extra = width - len(line)
        if extra <= 0:
            lines.append(line)
        elif align == "right":
            lines.append(" " * extra + line)
        elif align == "center":
            left = extra // 2
            lines.append(" " * left + line + " " * (extra - left))
        else:
            lines.append(line + " " * extra)
    return _Box(lines, box.baseline)


def _box_hstack(parts: list[_Box], gaps: Optional[list[int]] = None) -> _Box:
    if not parts:
        return _Box([""])
    gaps = gaps or [0] * len(parts)
    maybe_baselines = [box.baseline for box in parts if box.lines]
    baseline = max(maybe_baselines) if maybe_baselines else 0
    height = max((len(box.lines) for box in parts), default=1)
    lines = [""] * height
    for index, box in enumerate(parts):
        gap = gaps[index] if index < len(gaps) else 0
        if gap:
            lines = [line + " " * gap for line in lines]
        top = baseline - box.baseline
        width = _box_width(box)
        for row in range(height):
            if top <= row < top + len(box.lines):
                line = box.lines[row - top]
                lines[row] += line + " " * max(0, width - len(line))
            else:
                lines[row] += " " * width
    return _Box(lines, baseline)


def _box_vstack(boxes: list[_Box], align: str = "center") -> _Box:
    boxes = [box for box in boxes if box.lines]
    if not boxes:
        return _Box([""])
    width = max(_box_width(box) for box in boxes)
    lines: list[str] = []
    baseline = 0
    for index, box in enumerate(boxes):
        if index == 0:
            baseline = len(lines) + box.baseline
        lines.extend(_box_pad(box, width, align).lines)
    return _Box(lines, baseline)


def _map_chars(text: str, table: dict[str, str]) -> Optional[str]:
    mapped = [table.get(ch) for ch in text]
    if not text or not all(mapped):
        return None
    return "".join(mapped)  # type: ignore[arg-type]


def _script_text(value: Any, kind: str) -> str:
    flat = _flatten(value).strip()
    if not flat:
        return ""
    table = SUPERSCRIPTS if kind == "sup" else SUBSCRIPTS
    mapped = _map_chars(flat, table)
    if mapped is None and " " in flat:
        mapped = _map_chars(flat.replace(" ", ""), table)
    if mapped is not None:
        return mapped
    if len(flat) == 1:
        return ("^" if kind == "sup" else "_") + flat
    return ("^(" if kind == "sup" else "_(") + flat + ")"


def _tall_delim(char: str, height: int) -> list[str]:
    parts = _TALL_DELIMS.get(char)
    if not parts or height <= 1:
        return [char] * max(1, height)
    top, middle, bottom = parts
    if height == 2:
        return [top, bottom]
    return [top] + [middle] * (height - 2) + [bottom]


def _case_brace(height: int) -> list[str]:
    if height <= 1:
        return ["{"]
    if height == 2:
        return ["⎧", "⎩"]
    return ["⎧"] + ["⎪"] * (height - 2) + ["⎩"]


_MATRIX_BRACKETS = {
    "pmatrix": ("(", ")"), "bmatrix": ("[", "]"), "Bmatrix": ("{", "}"),
    "vmatrix": ("|", "|"), "Vmatrix": ("‖", "‖"), "matrix": (" ", " "),
}


def _layout(node: Any) -> _Box:
    if node is None:
        return _Box([""])
    if isinstance(node, MAtom):
        return _Box([node.text])
    if isinstance(node, MText):
        return _Box([node.text])
    if isinstance(node, MSpace):
        return _Box([" " * max(0, node.width)])
    if isinstance(node, MRow):
        parts: list[_Box] = []
        gaps: list[int] = []
        pending = 0
        for item in node.items:
            if isinstance(item, MSpace):
                pending += max(0, item.width)
                continue
            gap = pending
            atom_text = item.text if isinstance(item, MAtom) else None
            leading_sign = atom_text in {"-", "+"} and not parts
            if atom_text in _SPACED_ATOMS and not leading_sign:
                gap = max(gap, 1)
                pending = 1
            elif isinstance(item, MBigOp):
                pending = 1
            else:
                pending = 0
            parts.append(_layout(item))
            gaps.append(gap)
        if not parts:
            return _Box([""])
        gaps[0] = 0
        return _box_hstack(parts, gaps)
    if isinstance(node, MFrac):
        num = _layout(node.num)
        den = _layout(node.den)
        width = max(_box_width(num), _box_width(den), 1)
        top = _box_pad(num, width, "center")
        bottom = _box_pad(den, width, "center")
        bar = "─" * width if node.bar else " " * width
        return _Box(top.lines + [bar] + bottom.lines, len(top.lines))
    if isinstance(node, MSqrt):
        body = _layout(node.body)
        width = max(_box_width(body), 1)
        lines = ["─" * width]
        for index, line in enumerate(body.lines):
            prefix = "√" if index == body.baseline else "│"
            lines.append(prefix + line.ljust(width))
        box = _Box(lines, 1 + body.baseline)
        if node.index is not None:
            index_text = _flatten(node.index).strip()
            if index_text:
                merged = [index_text + lines[0]]
                merged += [" " * len(index_text) + line for line in lines[1:]]
                box = _Box(merged, box.baseline)
        return box
    if isinstance(node, MScript):
        box = _layout(node.base)
        suffix = ""
        if node.sup is not None:
            suffix += _script_text(node.sup, "sup")
        if node.sub is not None:
            suffix += _script_text(node.sub, "sub")
        if not suffix:
            return box
        return _box_hstack([box, _Box([suffix])])
    if isinstance(node, MBigOp):
        symbol = _Box([node.symbol])
        sub_text = _script_text(node.sub, "sub") if node.sub is not None else ""
        sup_text = _script_text(node.sup, "sup") if node.sup is not None else ""
        if node.above and (sub_text or sup_text):
            boxes: list[_Box] = []
            if sup_text:
                boxes.append(_Box([_flatten(node.sup).strip() or sup_text]))
            boxes.append(symbol)
            if sub_text:
                boxes.append(_Box([_flatten(node.sub).strip() or sub_text]))
            stacked = _box_vstack(boxes, "center")
            offset = 1 if sup_text else 0
            return _Box(stacked.lines, offset)
        suffix = sub_text + sup_text
        if not suffix:
            return symbol
        return _box_hstack([symbol, _Box([suffix])])
    if isinstance(node, MDelim):
        body = _layout(node.body)
        height = max(len(body.lines), 1)
        left = "." if node.left in {"", "."} else node.left
        right = "." if node.right in {"", "."} else node.right
        left_chars = [" "] * height if left == "." else _tall_delim(left, height)
        right_chars = [" "] * height if right == "." else _tall_delim(right, height)
        width = _box_width(body)
        lines = [left_chars[i] + body.lines[i].ljust(width) + right_chars[i]
                 for i in range(height)]
        return _Box(lines, body.baseline)
    if isinstance(node, MMatrix):
        return _layout_matrix(node)
    if isinstance(node, MCases):
        row_boxes: list[_Box] = []
        for row in node.rows:
            cells = [_layout(cell) for cell in row] or [_Box([""])]
            gaps = [0] + [2] * (len(cells) - 1)
            row_boxes.append(_box_hstack(cells, gaps))
        body = _box_vstack(row_boxes, "left") if row_boxes else _Box([""])
        height = max(len(body.lines), 1)
        brace = _case_brace(height)
        width = _box_width(body)
        lines = [brace[i] + " " + body.lines[i].ljust(width) for i in range(height)]
        return _Box(lines, body.baseline)
    if isinstance(node, MAligned):
        return _layout_columns(node.rows, gaps=1, align_alternate=True)
    if isinstance(node, MBoxed):
        body = _layout(node.body)
        width = max(_box_width(body), 1)
        top = "╭" + "─" * (width + 2) + "╮"
        bottom = "╰" + "─" * (width + 2) + "╯"
        lines = [top] + ["│ " + line.ljust(width) + " │" for line in body.lines] + [bottom]
        return _Box(lines, body.baseline + 1)
    if isinstance(node, MAccent):
        body = _layout(node.body)
        combining = ACCENT_COMBINING.get(node.kind)
        if combining and len(body.lines) == 1:
            text = body.lines[0]
            if not text:
                return body
            if node.kind == "underline":
                return _Box([text + combining * len(text)])
            return _Box(["".join(ch + combining for ch in text)])
        if node.kind in {"underline"}:
            width = _box_width(body)
            return _Box(body.lines + ["─" * width], body.baseline)
        width = _box_width(body)
        return _Box(["─" * width] + body.lines, body.baseline + 1)
    return _Box([""])


def _layout_matrix(node: MMatrix) -> _Box:
    rows: list[list[_Box]] = []
    for row in node.rows:
        rows.append([_layout(cell) for cell in row] or [_Box([""])])
    if not rows:
        rows = [[_Box([""])]]
    columns = max(len(row) for row in rows)
    column_widths = [
        max((_box_width(row[i]) if i < len(row) else 0) for row in rows)
        for i in range(columns)
    ]
    row_boxes: list[_Box] = []
    for row in rows:
        cells = []
        for index in range(columns):
            cell = row[index] if index < len(row) else _Box([""])
            cells.append(_box_pad(cell, column_widths[index], "center"))
        row_boxes.append(_box_hstack(cells, [0] + [2] * (columns - 1)))
    body = _box_vstack(row_boxes, "left")
    height = max(len(body.lines), 1)
    left_char, right_char = _MATRIX_BRACKETS.get(node.kind, ("(", ")"))
    left_chars = _tall_delim(left_char, height)
    right_chars = _tall_delim(right_char, height)
    width = _box_width(body)
    lines = [left_chars[i] + " " + body.lines[i].ljust(width) + " " + right_chars[i]
             for i in range(height)]
    return _Box(lines, body.baseline)


def _layout_columns(rows: list[list[Any]], gaps: int = 1,
                    align_alternate: bool = False) -> _Box:
    if not rows:
        return _Box([""])
    cells_rows: list[list[_Box]] = []
    for row in rows:
        cells = [_layout(cell) for cell in row] or [_Box([""])]
        cells_rows.append(cells)
    columns = max(len(row) for row in cells_rows)
    column_widths = [
        max((_box_width(row[i]) if i < len(row) else 0) for row in cells_rows)
        for i in range(columns)
    ]
    row_boxes: list[_Box] = []
    for row in cells_rows:
        padded = []
        for index in range(columns):
            cell = row[index] if index < len(row) else _Box([""])
            align = "right" if (align_alternate and index % 2 == 0) else "left"
            padded.append(_box_pad(cell, column_widths[index], align))
        row_boxes.append(_box_hstack(padded, [0] + [gaps] * (columns - 1)))
    return _box_vstack(row_boxes, "left")


def _flatten(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, (MAtom, MText)):
        return node.text
    if isinstance(node, MSpace):
        return " " * max(0, node.width)
    if isinstance(node, MRow):
        parts: list[str] = []
        pending = ""
        last_was_value = False
        for item in node.items:
            if isinstance(item, MSpace):
                pending += " " * max(0, item.width)
                continue
            atom_text = item.text if isinstance(item, MAtom) else None
            is_op = atom_text in _SPACED_ATOMS
            is_sign = atom_text in {"-", "+"}
            spaced = is_op and (last_was_value or (not is_sign and bool(parts)))
            if spaced and not pending:
                pending = " "
            parts.append(pending + _flatten(item))
            pending = " " if (is_op and (not is_sign or spaced)) else ""
            last_was_value = not is_op
        return "".join(parts)
    if isinstance(node, MFrac):
        num = _flatten(node.num)
        den = _flatten(node.den)
        if len(num) > 1 or any(ch in num for ch in "+- "):
            num = f"({num})"
        if len(den) > 1 and not den.isalnum():
            den = f"({den})"
        return f"{num}/{den}"
    if isinstance(node, MSqrt):
        body = _flatten(node.body)
        if node.index is not None:
            index = _flatten(node.index).strip()
            return f"{index}√({body})"
        return f"√({body})"
    if isinstance(node, MScript):
        out = _flatten(node.base)
        if node.sub is not None:
            out += _script_text(node.sub, "sub")
        if node.sup is not None:
            out += _script_text(node.sup, "sup")
        return out
    if isinstance(node, MBigOp):
        out = node.symbol
        if node.sub is not None:
            out += _script_text(node.sub, "sub")
        if node.sup is not None:
            out += _script_text(node.sup, "sup")
        return out
    if isinstance(node, MDelim):
        left = "" if node.left in {"", "."} else node.left
        right = "" if node.right in {"", "."} else node.right
        return left + _flatten(node.body) + right
    if isinstance(node, MMatrix):
        rows = [" ".join(_flatten(cell) for cell in row) for row in node.rows]
        left, right = _MATRIX_BRACKETS.get(node.kind, ("(", ")"))
        return f"{left} " + "; ".join(rows) + f" {right}"
    if isinstance(node, MCases):
        rows = [" ".join(_flatten(cell) for cell in row) for row in node.rows]
        return "{ " + "; ".join(rows) + " }"
    if isinstance(node, MAligned):
        rows = [" ".join(_flatten(cell) for cell in row) for row in node.rows]
        return "; ".join(rows)
    if isinstance(node, MBoxed):
        return f"[ {_flatten(node.body)} ]"
    if isinstance(node, MAccent):
        body = _flatten(node.body)
        combining = ACCENT_COMBINING.get(node.kind)
        if combining and body:
            return "".join(ch + combining for ch in body)
        return body
    return ""


def render_display_math(tex: str, width: Optional[int] = None) -> Optional[list[str]]:
    """LaTeX -> 2D Unicode lines, or None when unsupported (caller falls back)."""
    try:
        tokens = _tokenize_math(tex)
        if not tokens or len(tokens) > 400:
            return None
        node = _MathParser(tokens).parse()
        box = _layout(node)
        lines = [line.rstrip() for line in box.lines]
        if not any(lines):
            return None
        if width and max(len(line) for line in lines) > width:
            return None
        return lines
    except Exception:
        return None


def render_inline_math(tex: str) -> str:
    """Single-line Unicode for inline math; falls back to pylatexenc."""
    try:
        tokens = _tokenize_math(tex)
        if tokens and len(tokens) <= 400:
            node = _MathParser(tokens).parse()
            flat = _flatten(node)
            if flat.strip():
                return flat
    except Exception:
        pass
    return latex_to_unicode(tex)


# ── content rendering (markdown + centered display math) ────────────────────

def _split_display_plain(text: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    pos = 0
    for match in _DISPLAY_MATH_RE.finditer(text):
        if match.start() > pos:
            segments.append(("text", text[pos:match.start()]))
        segments.append(("math", match.group(1) or match.group(2) or ""))
        pos = match.end()
    if pos < len(text):
        segments.append(("text", text[pos:]))
    return segments


def _split_display_math(text: str) -> list[tuple[str, str]]:
    """Splits into ('text', ...) / ('math', ...) with code spans protected."""
    segments: list[tuple[str, str]] = []
    pos = 0
    for code in _CODE_RE.finditer(text):
        segments.extend(_split_display_plain(text[pos:code.start()]))
        segments.append(("text", code.group(0)))
        pos = code.end()
    segments.extend(_split_display_plain(text[pos:]))
    return segments


def render_content(text: str, *, math: bool = True, style: Optional[str] = None,
                   width: Optional[int] = None) -> Group:
    """Markdown with display math as centered 2D Unicode blocks."""
    markdown_style = style or "none"
    parts: list[Any] = []
    if not math:
        if text.strip():
            parts.append(Markdown(text, hyperlinks=True, style=markdown_style))
        return Group(*parts) if parts else Group(Text("", style=markdown_style))
    for kind, chunk in _split_display_math(text):
        if kind == "math":
            lines = render_display_math(chunk, width=width)
            if lines:
                parts.append(Align.center(Text("\n".join(lines), style="math")))
                continue
            chunk = "```\n" + latex_to_unicode(chunk) + "\n```"
        rendered = convert_math(chunk, math)
        if rendered.strip():
            parts.append(Markdown(rendered, hyperlinks=True, style=markdown_style))
    if not parts:
        parts.append(Text("", style=markdown_style))
    return Group(*parts)


def render_thinking(text: str, *, preview_lines: Optional[int] = None) -> Group:
    if preview_lines:
        lines = text.splitlines()
        if len(lines) > preview_lines:
            text = "…\n" + "\n".join(lines[-preview_lines:])
    return render_content(text, style="dim")


