# Release artifacts

Windows `.exe` files are produced by GitHub Actions, not committed manually.

Copy `docs/workflows/build-windows-exe.yml` to `.github/workflows/build-windows-exe.yml`, run the **Build Windows EXE** workflow, and download the artifact:

```text
PersianSignLanguageTranslator-windows-exe
```

Local PyInstaller builds write here too. On Linux/macOS the output is a native binary for that OS, not a real Windows exe.
