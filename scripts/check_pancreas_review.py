"""Verify pancreas gallery selection, source dimensions, layout and offline operation."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report = args.report.resolve()
    data = json.loads(report.with_suffix('.json').read_text('utf-8'))
    output = report.parent / 'pancreas-browser-qa'
    output.mkdir(exist_ok=True)
    errors, requests = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: requests.append(request.url)
                if request.url.startswith('http') else None)
        page.goto(report.as_uri(), wait_until='networkidle')
        expect(page.locator('h1')).to_have_text('Ảnh tụy trong OrganAMNIST')
        reference = page.locator('#reference article')
        expect(reference).to_have_attribute('data-index', str(data['reference']))
        expect(reference).to_have_attribute('data-label', str(data['reference_label']))
        for name, key in (('random', 'random_indices'), ('nearest', 'nearest_indices')):
            actual = page.locator(f'#{name} article').evaluate_all(
                'nodes => nodes.map(n => Number(n.dataset.index))',
            )
            assert actual == data[key][:15]
            page.locator(f'#{name}').screenshot(path=str(output / f'{name}.png'))
        images = page.locator('#nearest img')
        assert images.evaluate_all('nodes => nodes.every(n => n.naturalWidth === 128 '
                                   '&& n.naturalHeight === 128)'), 'Source size changed'
        page.locator('#all summary').click()
        cards = page.locator('#all article')
        assert cards.count() == data['test_organ_count']
        assert cards.evaluate_all('nodes => nodes.every(n => n.dataset.label === "9")')
        indices = cards.evaluate_all('nodes => nodes.map(n => Number(n.dataset.index))')
        assert len(set(indices)) == len(indices) and indices == sorted(indices)
        page.locator('#all summary').click()
        for width in (320, 768, 1024, 1440):
            page.set_viewport_size({'width': width, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), width
        page.locator('a[href="followup_report.html"]').click()
        expect(page.locator('h1')).to_contain_text('Partial fine-tuning')
        browser.close()
    assert not errors, errors
    assert not requests, requests
    print(f'Passed: {len(indices)} true-pancreas crops, two 15-image selections, '
          'source dimensions, responsive layout, report link, no JS/network errors')


if __name__ == '__main__':
    main()
