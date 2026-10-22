"""Exercise the offline analysis controls and canvas in installed Edge."""

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def check(report: Path, output: Path):
    errors, network = [], []
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: network.append(request.url)
                if request.url.startswith('http') else None)
        page.goto(report.resolve().as_uri(), wait_until='networkidle')
        expect(page.locator('h1')).to_contain_text('Partial fine-tuning')
        assert page.locator('table').count() >= 5
        for image in page.locator('figure img').all():
            assert image.evaluate('image => image.complete && image.naturalWidth > 0')
        page.screenshot(path=str(output / 'report_top.png'))
        page.locator('#case-group').select_option('regression')
        page.locator('#case-model').select_option('partial')
        assert page.locator('.case').count() > 0
        page.locator('#gallery').scroll_into_view_if_needed()
        page.screenshot(path=str(output / 'report_cases.png'))
        page.locator('#case-label').select_option('3')
        page.locator('#case-label').select_option('all')
        page.locator('#embedding-stage').select_option('partial')
        page.locator('#embedding-method').select_option('svd')
        page.locator('#embedding-filter').select_option('wrong')
        canvas = page.locator('#embedding-canvas')
        canvas.scroll_into_view_if_needed()
        point = page.evaluate('displayed[0]')
        assert point is not None
        bounds = canvas.bounding_box()
        page.mouse.move(bounds['x'] + point['x'], bounds['y'] + point['y'])
        expect(page.locator('#point-detail')).to_contain_text('Test #')
        expect(page.locator('#point-tooltip')).to_be_visible()
        first_button = page.locator('#embedding-legend button').first
        first_button.click()
        expect(first_button).to_have_attribute('aria-pressed', 'false')
        first_button.click()
        page.locator('#embedding-filter').select_option('all')
        canvas.scroll_into_view_if_needed()
        page.screenshot(path=str(output / 'report_embedding.png'))
        for width in (320, 768, 1440):
            page.set_viewport_size({'width': width, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), width
        browser.close()
    assert not errors, errors
    assert not network, 'Report made external network requests'
    print('Offline report verified: images, gallery, PCA/SVD controls, hover, mobile widths')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.report, args.output)


if __name__ == '__main__':
    main()
