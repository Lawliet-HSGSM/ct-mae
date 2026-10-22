"""Verify the offline label-budget report and all confusion-matrix selectors in Edge."""

import argparse
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
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: network.append(request.url)
                if request.url.startswith('http') else None)
        page.goto(args.report.resolve().as_uri(), wait_until='networkidle')
        expect(page.locator('h1')).to_contain_text('Thêm nhãn')
        assert page.locator('table').count() >= 4
        page.screenshot(path=str(args.output / 'report_top.png'))
        for stage in ('scratch', 'linear', 'finetune', 'partial'):
            page.locator('#matrix-stage').select_option(stage)
            for budget in ('5', '10'):
                page.locator('#matrix-budget').select_option(budget)
                page.wait_for_function('''() => {
                    const image=document.getElementById('matrix-image');
                    return image.complete && image.naturalWidth > 0;
                }''')
                expect(page.locator('#matrix-caption')).to_contain_text(budget + '%')
        page.locator('#matrix-image').scroll_into_view_if_needed()
        page.screenshot(path=str(args.output / 'report_confusion.png'))
        for image in page.locator('img').all():
            assert image.evaluate('image => image.complete && image.naturalWidth > 0')
        for width in (320, 768, 1440):
            page.set_viewport_size({'width': width, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), width
        browser.close()
    assert not errors, errors
    assert not network, 'Offline report requested external resources'
    print('Label report passed: 8 matrices, images, responsive layout, no JS/network errors')


if __name__ == '__main__':
    main()
