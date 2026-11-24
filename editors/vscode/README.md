# Tenon VS Code Extension

First-class IDE support for [Tenon](../../README.md) powered by the built-in `tenon lsp` language server:

- **Syntax Highlighting** (`syntaxes/tenon.tmLanguage.json`): keywords, `comptime`, `macro`, `unsafe`, `defer`, `where`/`requires`/`ensures`, `@repr(C)`/`@packed`/`@get`/`@post`, SIMD vector types, triple-quoted strings.
- **Real-Time Diagnostics**: syntax errors + L0/L1 formal verification diagnostics with Z3 counterexamples as you type (unsaved buffers supported via package overlays).
- **Rich Hover**: function signatures, `requires`/`ensures` contracts, refinement predicates, `@repr(C)`/`@packed`/`union`/`bitfield` exact byte/bit layout tables, and L0/L1 proof summaries.
- **Go to Definition**: jump to declarations within the current file or imported packages.
- **Completion**: keywords, primitive & SIMD types, in-scope items, `@attribute` completion, and `.` member / package completion.
- **Document Symbols & Canonical Formatting**: outline view + `tenon fmt`.

## Development & Verification

```bash
cd editors/vscode
npm test
```

Configure `tenon.executable` in VS Code settings if `tenon` is not on your `PATH` (by default it auto-detects `target/release/tenon` or `target/debug/tenon` inside the workspace).
