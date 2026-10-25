"""Check the completed adaptation report offline in Edge, including native-pixel galleries."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    errors, network = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1100})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: network.append(request.url)
                if request.url.startswith('http') else None)
        page.goto(args.report.resolve().as_uri(), wait_until='networkidle')
        expect(page.locator('h1')).to_contain_text('Học tiếp checkpoint')
        tables = page.locator('section').first.locator('table')
        assert tables.count() == 2
        for result in tables.all():
            assert result.locator('tbody tr').count() == 6
        matrices = page.locator('summary').filter(has_text='confusion matrix')
        assert matrices.count() == 12
        page.screenshot(path=str(args.output / 'overview.png'))
        page.get_by_role('heading', name='Organ và kiểu nhầm nào thay đổi?').evaluate(
            'element => element.scrollIntoView({block: "start"})')
        page.screenshot(path=str(args.output / 'per_class.png'))
        gallery = page.locator('summary').filter(has_text='Full · A · Được sửa')
        gallery.click()
        gallery.evaluate('element => element.scrollIntoView({block: "start"})')
        page.screenshot(path=str(args.output / 'recovered.png'))
        regressed = page.locator('summary').filter(has_text='Full · A · Mới sai')
        regressed.click()
        regressed.evaluate('element => element.scrollIntoView({block: "start"})')
        page.screenshot(path=str(args.output / 'regressed.png'))
        page.locator('details').evaluate_all('items => items.forEach(item => item.open = true)')
        for image in page.locator('img').all():
            assert image.evaluate('img => img.complete && img.naturalWidth > 0')
        assert page.locator('.case img').count() > 0
        for image in page.locator('.case img').all():
            assert image.evaluate('img => img.naturalWidth === 128 && img.naturalHeight === 128')
            assert image.evaluate('img => getComputedStyle(img).width === "128px"')
        for width in (320, 768, 1024, 1440):
            page.set_viewport_size({'width': width, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), width
        evidence = {'browser': browser.version, 'result_rows': 12, 'confusion_matrices': 12,
                    'images': page.locator('img').count(), 'native_gallery_pixels': 128,
                    'responsive_widths': [320, 768, 1024, 1440],
                    'javascript_errors': errors, 'external_requests': network}
        browser.close()
    assert not errors, errors
    assert not network, 'Offline report requested external resources'
    (args.output / 'browser_verification.json').write_text(json.dumps(evidence, indent=2), 'utf-8')
    print('Adaptation report passed: 12 results/matrices, native pixels, responsive, offline')


if __name__ == '__main__':
    main()
