# hasse

Japanese voice input for Windows and macOS. Hold a key and speak, then release.
The text appears in a small overlay; confirm it and it is inserted into the app
you are using.

## Install

On Windows, open a terminal in the folder with the app, then run:

```bat
hasse.exe install
```

On macOS, open a terminal in the folder with the app, then run:

```sh
./hasse install
```

macOS also needs **Input Monitoring** and **Accessibility** for hasse. Allow both in
System Settings → Privacy & Security.

The app starts at login and lives in the system tray (Windows) or menu bar (macOS).
`hasse uninstall` turns off the automatic start.

## Usage

1. Hold **Right Shift** and speak.
2. When you release, the text appears in an overlay. Edit it if it is wrong,
   then press **Ctrl+Enter**.
3. The text is inserted into the app you were using.
4. Quit from the hasse icon in the tray or menu bar.

The overlay at the bottom of the screen shows the recording state and the text
waiting to be confirmed.

## Changing the key or microphone

The default key is Right Shift, and the system default microphone is used. To change
either, create `.hasse.yaml` in your home folder, or `.config/hasse/hasse.yaml`. On
Windows, `%APPDATA%\hasse\hasse.yaml` also works:

```yaml
hotkey:
  key: RightShift        # RightShift, RightCtrl, LeftAlt, F8, ...
audio:
  device: ""             # empty uses the system default
```

Run `hasse --devices` to list the microphone names.

## Correcting mistakes

After you release the key, the text waits in the overlay. Press **Ctrl+Enter**
to insert it, or **Esc** to cancel. Enter is left to the input method, so it
confirms a conversion as usual.

If the text came out wrong, edit it in the overlay before confirming. hasse
compares what it heard with what you typed and remembers the difference, so the
corrected wording is more likely the next time you say it. Only the phrases that
differ are kept.

A correction is kept as one rule: what was recognized, what you typed in its
place, and the words on both sides. A rule is at full strength when what
precedes it matches, still pulls with half of that context, and pulls a little
with none of it, so a word learned in one sentence can help in another. The
words after it rank the finished candidates: a transcript that continues the
way you did wins over one that does not. Everything is cut on the model's token
boundaries, so a rule lines up with what the decoder actually emits.

A few corrections cannot be reached through the decoder at all. An uppercase
Latin letter is a byte piece that the language prior never picks over the
lowercase one, and a Latin word is a reading the prior prefers to spell in
Japanese, whatever weight the decoder is given. Those rules are applied to the
finished text as well. A rule that only changes case applies wherever the word
appears on its own, and so does a word corrected in three different sentences:
by then the correction is about the word and not about where it stood. Every
other rule needs the words it was learned with.

`hasse bias rebuild` replays `corrections.jsonl` and rewrites `bias.yaml` with
the current rules. Use it after updating hasse, or to recover a file that was
edited by hand. `hasse bias rewrite "TEXT"` shows what the rules do to a line
without recording anything.

From the folder that holds the app, `hasse.exe bias` lists the learned phrases,
`hasse.exe bias add "..."` adds one by hand, and `hasse.exe bias remove "..."`
drops one. On macOS, use `./hasse bias`. The phrases and the correction log stay
on this computer, in `%LOCALAPPDATA%\hasse` on Windows and
`~/Library/Application Support/hasse` on macOS.

To insert the text without reviewing it first, turn the review off in the
configuration file:

```yaml
review:
  enabled: false
```

If recognition feels slow, set `decode.beam` to `1` in the same file. That
switches to greedy decoding, which is faster and slightly less accurate.

## If it does not work

- **Windows:** apps running as administrator, and apps that ignore `Ctrl+V`, cannot receive
  the text.
- **macOS:** without the two permissions above, the key or the insertion does nothing.
  If macOS blocks the app, right-click it and choose Open.

## License

hasse uses Haas-ja-v1.0 (Apache-2.0) and ONNX Runtime (MIT).
