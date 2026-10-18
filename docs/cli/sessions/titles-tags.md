# Titles and tags

Sessions carry a title and free-form tags:

- `--title TEXT` and `--tag TAG` (repeatable) set them at launch;
  `--auto-title` asks the model for a title after the first turn (config
  `auto_title`).
- In the REPL, `/title [TEXT|--auto]` shows or sets the title and
  `/tag [list|add TAG…|rm TAG]` manages tags.

`--search TEXT` matches titles, tags and message content; see
[Search and export](search-export.md).
