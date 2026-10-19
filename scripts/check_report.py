"""Exercise the generated offline explorer in installed Microsoft Edge."""

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.report.resolve().as_uri(), wait_until="networkidle")
        sizes = page.locator("#dataset").evaluate("el => JSON.parse(el.textContent).sizes")
        expect(page.locator("#class-select option")).to_have_count(11)
        expect(page.locator("#image-grid img")).to_have_count(len(sizes))
        expect(page.locator("#stats-table tr")).to_have_count(3 * len(sizes))
        expect(page.locator("#class-table tr")).to_have_count(11)
        for size in sizes:
            image = page.locator(f'img[data-resolution="{size}"]')
            expect(image).to_be_visible()
            assert image.evaluate("el => el.naturalWidth") == size
        page.select_option("#split-select", "val")
        page.select_option("#class-select", "9")
        expect(page.locator("#selection")).to_contain_text("Tụy / pancreas · val")
        before = page.locator("#selection").inner_text()
        page.locator("#next").click()
        assert page.locator("#selection").inner_text() != before
        page.locator("#previous").click()
        assert page.locator("#selection").inner_text() == before
        page.select_option("#view-select", "native")
        for size in sizes:
            box = page.locator(f'img[data-resolution="{size}"]').bounding_box()
            assert box is not None and abs(box["width"] - size) < 1
        page.locator("#split-select").focus()
        page.keyboard.press("Tab")
        expect(page.locator("#class-select")).to_be_focused()
        for width in (320, 768, 1024, 1440):
            page.set_viewport_size({"width": width, "height": 1000})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
        page.select_option("#split-select", "train")
        page.select_option("#class-select", "6")
        page.select_option("#view-select", "equal")
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(args.screenshot), full_page=True)
        browser.close()
    assert not errors, errors
    print(
        f"Browser checks passed: {sizes}, class/split/sample controls, native pixels, responsive."
    )


if __name__ == "__main__":
    main()
