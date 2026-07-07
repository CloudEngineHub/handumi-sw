# Agent instructions

This file is read automatically by coding agents that support the `AGENTS.md`
convention (e.g. Codex CLI). Claude Code users get the same behavior via the
`/handumi-commit` skill in `.claude/skills/handumi-commit/SKILL.md` — this
file exists so the same conventions apply for agents that don't read that
skill format.

## Commits

When asked to commit changes in this repo, follow these conventions.

### Behavior

1. Inspect `git status --short`.
2. Analyze staged changes with `git diff --staged`.
3. If nothing is staged, inspect unstaged changes and stage only the files that belong
   to the requested commit.
4. Group changes into meaningful commits that are easy to track.
5. Generate a conventional commit message.
6. Include an English body that summarizes what changed and what was verified.
7. Create the commit with proper formatting.

### Commit cadence

Commit at meaningful checkpoints, not after every tiny edit.

Do create a commit for a coherent unit of work, such as:

- A bug fix that changes behavior.
- A small feature or user-visible improvement.
- A refactor that changes structure without changing behavior.
- A test or build fix that belongs with the related code change.
- A documentation update that explains a completed implementation change.

Do not create a separate commit for incidental edits, such as only changing a button
color, adjusting whitespace, renaming a local variable, or fixing a typo, unless the
user explicitly asks for that granularity or the change is independently useful to review.

### Commit format

```text
<type>(<scope>): <description>

What changed:
- <specific file-oriented summary>
- <specific file-oriented summary>

Verification passed:
- <command>
- <command>

[optional footer]
```

If verification was not run, use:

```text
Verification not run:
- <concrete reason>
```

### Types

- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes
- refactor: Code refactoring
- test: Adding or modifying tests
- chore: Maintenance tasks

### Body guidelines

Write the body like a handoff summary for Codex, Claude Code, or another LLM coding tool that may inspect
the commit later.

- Keep bullets specific and file-oriented.
- Mention behavior changes, not just edited filenames.
- Include important file paths and line numbers when useful.
- Do not include unrelated user changes.
- If generated files or temporary state were changed only by verification, restore or
  exclude them unless the user asked to keep them.

### Example output

```text
fix(parser): handle empty configuration files

What changed:
- Added an explicit empty-file guard in src/config/parser.ts:42 so the parser returns the default configuration instead of throwing.
- Updated src/config/loader.ts:88 to pass the source path into parser errors, which keeps diagnostics useful for invalid non-empty files.
- Added regression coverage in tests/config-parser.test.ts:17 for empty, whitespace-only, and invalid configuration files.

Verification passed:
- <project build command>
- <project lint command>
- <project test command>
```
