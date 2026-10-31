# hasse

Japanese voice input for Windows and macOS. Hold a key and speak, then release. The
recognized text is inserted into the app you are using.

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
2. When you release, the text is inserted into the focused app.
3. Quit from the hasse icon in the tray or menu bar.

A small indicator at the bottom of the screen shows when it is recording.

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

## If it does not work

- **Windows:** apps running as administrator, and apps that ignore `Ctrl+V`, cannot receive
  the text.
- **macOS:** without the two permissions above, the key or the insertion does nothing.
  If macOS blocks the app, right-click it and choose Open.

## License

hasse uses Haas-ja-v1.0 (Apache-2.0) and ONNX Runtime (MIT).
