# Brainfast Manual Acceptance

## Browser acceptance

1. Start the server:
   ```powershell
   python project/frontend/server.py
   ```
2. Open `http://127.0.0.1:8787`.
3. Load a known-good sample config.
4. Run detector preview once and confirm:
   - overlay renders
   - CSV download works
   - detector name is shown
5. Run one whole-brain pipeline sample and confirm:
   - QC overlay renders
   - registration CSVs exist
   - cell count CSVs download
   - methods/export output is non-empty
