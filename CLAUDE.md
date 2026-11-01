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

## Build and install

`cargo xtask build` builds with the ONNX Runtime fetched by the helper; on an
already-prepared tree, `cargo build --release -p hasse` is enough. Either way
the binary is `target/release/hasse.exe` on Windows and `target/release/hasse`
on macOS.

Install with `target/release/hasse.exe install` on Windows or
`./target/release/hasse install` on macOS — run it from the build output, not
from the installed copy. The same `install` subcommand exists on both and does
the same thing there: it stops the running instance, copies the binary and the
model files to the install folder, registers autostart (a Task Scheduler task on
Windows, a LaunchAgent on macOS) and starts the app, so it is also how a build is
deployed and restarted; there is no separate restart step. macOS additionally
needs Input Monitoring and Accessibility permission for the app. The install folder is
`%LOCALAPPDATA%\hasse` on Windows and `~/Library/Application Support/hasse` on
macOS, and the same folder holds `bias.yaml` and `corrections.jsonl`.

`default.yaml` is embedded into the binary at build time, so a change to it
needs a rebuild and an install. A `hasse.yaml` in the user's config folder is
read at startup and needs only a restart.
