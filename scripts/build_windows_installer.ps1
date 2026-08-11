$ErrorActionPreference = "Stop"

Write-Host "Building GUI executable..."
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python scripts/build_exe.py --mode gui

$compiler = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-Not (Test-Path $compiler)) {
    $compiler = "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
}
if (-Not (Test-Path $compiler)) {
    throw "Inno Setup 6 not found. Install it from https://jrsoftware.org/isinfo.php then run this script again."
}

Write-Host "Building installer..."
& $compiler installer\PersianSignLanguageTranslator.iss
Write-Host "Done: release\PersianSignLanguageTranslatorSetup.exe"