# hasse

Japanese voice input IME for Windows and macOS. The user guide is in the README.

## Directory layout

- `app/` — the Rust binary. The recognition core and the state machine are
  platform-neutral; the Windows and macOS backends are selected with `cfg`.
- `xtask/` — the build helper. It fetches ONNX Runtime for the current OS and
  builds the app.
- `engine/` — the exported model artifacts and the export-only Python
  environment. The large files are gitignored.
- `tools/` — the model export script.
- `runtime/` — the fetched ONNX Runtime library, gitignored.

Build, architecture and platform notes live in the source comments.
