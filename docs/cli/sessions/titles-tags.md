# Titles and tags

Every session gets a title and can carry tags; both show up in the
[/sessions picker](save-load.md) and `--list`, and both survive in the index.

## Titles

A new session is titled from its **first prompt** (`naive_title()` — the
first line, trimmed to a sane length). `--title TEXT` sets one up front, and
`--auto-title` asks the model for a better one via the
[title prompt](../../config/prompts.md):

```console
$ ollamarun --no-tools --auto-title llama3.2:3b "Explain in one sentence what a haiku is."
llama3.2:3b ›
A haiku is a traditional Japanese poetry form consisting of three lines with
a syllable count of 5-7-5, often capturing a moment or image from nature.
$ ollamarun --list
| 20261018-122724-96a7 | Understanding Haiku Poetry Basics | llama3.2:3b | …
```

In the REPL:

| Command | Effect |
| --- | --- |
| `/title` | Show the current title. |
| `/title TEXT` | Set it. |
| `/title --auto` | Ask the model to (re)title from the first message. |

The title is also the header of a [transcript replay](search-export.md)
and the first column everywhere sessions are listed. Renaming is cheap — the
title lives on the session, not in the filename.

## Tags

Tags are a free-form list on the session. Set them at launch (`--tag` is
repeatable) or in the REPL:

```text
llama3.2:3b › /tag add demo docs
tags demo, docs
llama3.2:3b › /tag list
tags: demo, docs
llama3.2:3b › /tag rm docs
tags demo
```

`/tag` with no argument lists them; `add`/`rm` take several at once. Tags
appear in `--list`/`--search` output and in the exported JSON, which makes
them the light-weight way to group work (one tag per project, per machine,
per experiment).

## Names, titles and ids

Three related things, easy to mix up:

- **id** — `20261018-122556-a496`, system-generated; any unique prefix works
  as a reference.
- **name** — what `--session NAME` keys on; it is also stored as the title
  when the session is created that way.
- **title** — the display label, derived from the prompt or set by
  `--title`/`/title`/`--auto-title`.

`--list` shows the title; `--show`, `--export` and `-c` accept ids, id
prefixes, names and title references.
