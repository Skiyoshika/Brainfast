# Brainfast Code Signing

This repository now includes code-signing integration points for Windows
desktop releases.

## What is already wired

- Release workflow: [.github/workflows/release.yml](.github/workflows/release.yml)
- Signing helper: [project/scripts/sign_release_assets.py](project/scripts/sign_release_assets.py)
- Desktop build output:
  - `project/frontend/dist/BrainfastUI/BrainfastUI.exe`
  - `project/frontend/dist/Brainfast-<version>-setup.exe`

If signing secrets are present in GitHub Actions, the workflow will:

1. decode the PFX certificate into the runner temp directory
2. sign the desktop EXE
3. build the NSIS installer from the already signed desktop bundle
4. sign the NSIS installer EXE
5. archive and publish the signed artifacts

If signing secrets are not present, the build still succeeds and simply skips
the signing step.

## Required GitHub secrets

- `BRAINFAST_CODESIGN_PFX_BASE64`
  - base64-encoded `.pfx` certificate file
- `BRAINFAST_CODESIGN_PASSWORD`
  - password for the `.pfx`
- `BRAINFAST_CODESIGN_TIMESTAMP_URL`
  - optional RFC3161 timestamp endpoint
  - if omitted, the workflow falls back to `http://timestamp.digicert.com`

## Local signing

You can sign local release artifacts with:

```powershell
python project\scripts\sign_release_assets.py ^
  --file project\frontend\dist\BrainfastUI\BrainfastUI.exe ^
  --file project\frontend\dist\Brainfast-0.3.0-desktop-setup.exe ^
  --pfx C:\path\to\brainfast.pfx ^
  --password "<pfx-password>"
```

Or let the local installer build script do it automatically:

```powershell
$env:BRAINFAST_CODESIGN_PFX = "C:\path\to\brainfast.pfx"
$env:BRAINFAST_CODESIGN_PASSWORD = "<pfx-password>"
project\frontend\build_installer.bat
```

Optional environment variables:

- `BRAINFAST_SIGNTOOL`
- `BRAINFAST_CODESIGN_PFX`
- `BRAINFAST_CODESIGN_PASSWORD`
- `BRAINFAST_CODESIGN_TIMESTAMP_URL`
- `BRAINFAST_CODESIGN_DESCRIPTION`
- `BRAINFAST_CODESIGN_PUBLISHER_URL`

## Notes

- The repository does not bundle a certificate. You must provide your own.
- A standard OV certificate signs binaries successfully, but SmartScreen
  reputation may still take time to build.
- An EV certificate improves SmartScreen trust but costs more.
- Signing is the release blocker for broad Windows distribution, not for local
  lab use.
