"""Check offline report rendering with installed Edge; no external page dependencies."""

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    errors, network = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: network.append(request.url)
                if request.url.startswith("http") else None)
        page.goto(args.report.resolve().as_uri(), wait_until="networkidle")
        expect(page.locator("h1")).to_contain_text("CT-MAE")
        assert page.locator("table").count() >= 2
        assert "gradient accumulation" in page.locator("body").inner_text()
        for image in page.locator("img").all():
            assert image.evaluate("image => image.complete && image.naturalWidth > 0")
        for width in (320, 768, 1440):
            page.set_viewport_size({"width": width, "height": 1000})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(args.screenshot), full_page=False)
        browser.close()
    assert not errors, errors
    assert not network, network
    print("Offline report passed: loaded images/tables, responsive widths, no JS/network errors")


if __name__ == "__main__":
    main()
