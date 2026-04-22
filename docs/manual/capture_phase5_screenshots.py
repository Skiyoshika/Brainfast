"""Capture fresh screenshots for Phase 5 (dual-channel wizard) + dual-channel
overlay UI state for the manual update.

Produces:
    12_wizard_dual_channel.png  — wizard with "Add second channel" expanded,
                                  real TIFF paths filled, both Inspects run
    13_liquify3d_overlay.png    — 3D Liquify with overlay toggle visible
"""

from __future__ import annotations

from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "http://127.0.0.1:8787"
REAL_JOB = "real-35-full"
C0_TIFF = (
    "D:/Brainfast/Sample/ChATe27/"
    "35_High_1000ms_560nm_640nm_150W_z5um_Bothlaser - Pos 3 4 [1] "
    "3DMontage_XY1763150824_Z000_T0_C0.tif"
)
C1_DIR = "D:/Brainfast/project/data/35_C1_full"


def shot(page, name, full=False):
    p = OUT / name
    page.screenshot(path=str(p), full_page=full)
    print(f"  [ok] {name} ({p.stat().st_size / 1024:.0f} KB)")


def main():
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        ctx.add_init_script(
            """try {
                localStorage.setItem('idlebrain.tourDone', '1');
                localStorage.setItem('brainfast.tourDone', '1');
            } catch (_) {}"""
        )
        page = ctx.new_page()

        # ---- 12 Wizard dual-channel ----
        print("-> 12 Wizard: fill C0 + expand 2nd channel + point at C1 dir")
        page.goto(f"{BASE}/")
        page.wait_for_load_state("networkidle", timeout=20000)
        page.wait_for_timeout(500)
        page.click('.nav-btn[data-tab="newsample"]')
        page.wait_for_timeout(600)
        # Fill 1st channel + Inspect
        page.fill("#wizSourcePath", C0_TIFF)
        page.wait_for_timeout(300)
        page.click("#wizInspectBtn")
        page.wait_for_function(
            "document.querySelector('#wizInspectStatus')?.textContent?.length > 10",
            timeout=30000,
        )
        page.wait_for_timeout(500)
        # Expand 2nd channel block + point at C1 dir
        page.check("#wizAddSecondChannel")
        page.wait_for_timeout(300)
        page.fill("#wiz2SourcePath", C1_DIR)
        page.wait_for_timeout(300)
        page.click("#wiz2InspectBtn")
        page.wait_for_function(
            "document.querySelector('#wiz2InspectStatus')?.textContent?.length > 10",
            timeout=20000,
        )
        page.wait_for_timeout(500)
        # Pick 'farred' for 2nd channel (C1 is far-red on Sample 35)
        page.select_option("#wiz2Channel", "farred")
        page.wait_for_timeout(300)
        page.fill("#wizSampleId", "35_dual")
        page.wait_for_timeout(300)
        shot(page, "12_wizard_dual_channel.png", full=True)

        # ---- 13 3D Liquify overlay toggle visible ----
        print("-> 13 3D Liquify: load real-35-full + show 'Overlay 2nd channel' row")
        page.goto(f"{BASE}/?job={REAL_JOB}")
        page.wait_for_load_state("networkidle", timeout=20000)
        page.wait_for_timeout(1500)
        page.click('.nav-btn[data-tab="liquify3d"]')
        page.wait_for_timeout(800)
        page.fill("#liq3dJobId", REAL_JOB)
        page.wait_for_timeout(300)
        page.click("#liq3dReloadBtn")
        page.wait_for_function(
            """() => {
                const el = document.querySelector('#liq3dSliceStatus');
                return el && !/No slices/i.test(el.textContent || '');
            }""",
            timeout=20000,
        )
        page.wait_for_timeout(2000)
        # Toggle the overlay on if the row is now visible
        row = page.evaluate("document.querySelector('#liq3dOverlayRow')?.style.display")
        if row != "none":
            page.check("#liq3dOverlayToggle")
            page.wait_for_timeout(2500)
        shot(page, "13_liquify3d_overlay.png", full=True)

        ctx.close()
        browser.close()
    print("\nDone. Screenshots in", OUT)


if __name__ == "__main__":
    main()
