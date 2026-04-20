"""Capture all screenshots needed for the Brainfast Operations Manual PPT.

Connects to the already-running Brainfast server (127.0.0.1:8787) via Playwright
Chromium and walks through the UI using the real real-35-full job (646 slices,
5.93M cells, 60 landmarks already added) for populated views, and the empty
state for the wizard intro.

Output: D:/Brainfast/docs/manual/screenshots/01..XX_*.png
"""

from __future__ import annotations

from pathlib import Path
from playwright.sync_api import sync_playwright, Page

OUT = Path(__file__).parent / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

VP_W, VP_H = 1440, 900
BASE = "http://127.0.0.1:8787"

REAL_JOB = "real-35-full"
REAL_TIFF = (
    "D:/Brainfast/Sample/ChATe27/"
    "35_High_1000ms_560nm_640nm_150W_z5um_Bothlaser - Pos 3 4 [1] "
    "3DMontage_XY1763150824_Z000_T0_C0.tif"
)


def shot(page: Page, name: str, full: bool = False) -> None:
    path = OUT / name
    page.screenshot(path=str(path), full_page=full)
    size = path.stat().st_size / 1024
    print(f"  [ok] {name} ({size:.0f} KB)")


def wait(page: Page, ms: int = 500) -> None:
    page.wait_for_timeout(ms)


def click_tab(page: Page, tab: str) -> None:
    """Switch to a top-level nav tab by data-tab attribute."""
    page.click(f'.nav-btn[data-tab="{tab}"]')
    wait(page, 600)


def main() -> None:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(viewport={"width": VP_W, "height": VP_H})
        ctx.add_init_script(
            """try {
                localStorage.setItem('idlebrain.tourDone', '1');
                localStorage.setItem('brainfast.tourDone', '1');
            } catch (_) {}"""
        )
        page = ctx.new_page()

        # ============================================================
        # 01 Home / Registration Workflow
        # ============================================================
        print("-> 01 Registration Workflow home (default job)")
        page.goto(f"{BASE}/")
        page.wait_for_load_state("networkidle", timeout=20000)
        wait(page, 800)
        shot(page, "01_home.png")

        # ============================================================
        # 02 New Sample wizard empty
        # ============================================================
        print("-> 02 New Sample wizard (empty)")
        click_tab(page, "newsample")
        wait(page, 600)
        shot(page, "02_wizard_empty.png")

        # ============================================================
        # 03 New Sample wizard after Inspect on real TIFF
        # ============================================================
        print("-> 03 Wizard: Inspect real 2.87GB TIFF")
        page.fill("#wizSourcePath", REAL_TIFF)
        wait(page, 300)
        page.click("#wizInspectBtn")
        # Inspect takes ~3-8s to parse a 2.87GB multi-page TIFF header
        page.wait_for_function(
            "document.querySelector('#wizInspectStatus')?.textContent?.length > 10",
            timeout=30000,
        )
        wait(page, 600)
        shot(page, "03_wizard_inspected.png")

        # ============================================================
        # 04 Wizard filled (Step 2 — auto-filled after inspect)
        # ============================================================
        print("-> 04 Wizard: Step 2 auto-filled + Launch button enabled")
        # Set a friendly sample id so the screenshot is clear
        page.fill("#wizSampleId", "real-35-full")
        wait(page, 400)
        shot(page, "04_wizard_ready.png")

        # ============================================================
        # 05 Progress panel (reuse completed real-35-full for populated view)
        # ============================================================
        print("-> 05 Progress panel + ETA on real-35-full")
        page.goto(f"{BASE}/?job={REAL_JOB}")
        page.wait_for_load_state("networkidle", timeout=20000)
        wait(page, 2000)
        # Registration Workflow tab with job context
        click_tab(page, "workflow")
        wait(page, 800)
        shot(page, "05_progress_eta.png")

        # ============================================================
        # 06 Results tab — cell counts bar chart + region hierarchy
        # ============================================================
        print("-> 06 Results tab (real-35-full 5.93M cells)")
        click_tab(page, "results")
        wait(page, 3000)
        shot(page, "06_results_chart.png", full=True)

        # ============================================================
        # 07 3D Liquify — populated with real job
        # ============================================================
        print("-> 07 3D Liquify: load real job + slice list")
        click_tab(page, "liquify3d")
        wait(page, 800)
        # Fill Job ID and click Reload
        page.fill("#liq3dJobId", REAL_JOB)
        wait(page, 300)
        page.click("#liq3dReloadBtn")
        # Wait for slice list to load (status changes from "No slices loaded")
        page.wait_for_function(
            """() => {
                const el = document.querySelector('#liq3dSliceStatus');
                return el && !/No slices/i.test(el.textContent || '');
            }""",
            timeout=20000,
        )
        wait(page, 2000)
        shot(page, "07_liquify3d_loaded.png")

        # ============================================================
        # 08 3D Liquify — landmark pairs table visible
        # ============================================================
        print("-> 08 3D Liquify: scroll to show landmark pairs table")
        page.evaluate("window.scrollBy(0, 400)")
        wait(page, 400)
        shot(page, "08_liquify3d_pairs_table.png")

        # ============================================================
        # 09 3D Liquify — full-page view (everything together)
        # ============================================================
        print("-> 09 3D Liquify: full-page overview")
        page.evaluate("window.scrollTo(0, 0)")
        wait(page, 400)
        shot(page, "09_liquify3d_full.png", full=True)

        # ============================================================
        # 10 3D Liquify — class prior section
        # ============================================================
        print("-> 10 3D Liquify: class prior + warm start buttons")
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        wait(page, 500)
        shot(page, "10_classprior.png")

        # ============================================================
        # 11 Sidebar overview (for slide 1 / context)
        # ============================================================
        print("-> 11 Sidebar close-up")
        page.evaluate("window.scrollTo(0, 0)")
        wait(page, 300)
        # Clip the sidebar region by using a custom clip
        path = OUT / "11_sidebar.png"
        page.screenshot(path=str(path), clip={"x": 0, "y": 0, "width": 220, "height": 900})
        size = path.stat().st_size / 1024
        print(f"  [ok] 11_sidebar.png ({size:.0f} KB)")

        ctx.close()
        browser.close()
    print("\nAll screenshots saved to", OUT)


if __name__ == "__main__":
    main()
