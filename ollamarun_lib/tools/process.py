"""Shared output plumbing for the subprocess-backed tools."""
from __future__ import annotations


def combine_exit(returncode: int, stdout: str, stderr: str) -> str:
    parts = []
    if stdout.strip():
        parts.append(stdout.rstrip("\n"))
    if stderr.strip():
        parts.append("[stderr]\n" + stderr.rstrip("\n"))
    body = "\n".join(parts) or "(no output)"
    return f"exit {returncode}\n{body}"
