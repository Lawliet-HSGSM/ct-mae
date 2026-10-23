"""Check source dimensions, matched controls and offline crop-review interactions in Edge."""

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
    data = json.loads(args.report.with_suffix('.json').read_text('utf-8'))
    errors, network = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: network.append(request.url)
                if request.url.startswith('http') else None)
        page.goto(args.report.resolve().as_uri(), wait_until='networkidle')
        expect(page.locator('h1')).to_have_text('Ảnh mờ do đâu?')
        page.locator('a[href="followup_report.html"]').click()
        expect(page.locator('h1')).to_contain_text('Partial fine-tuning')
        page.locator('a[href="crop_review.html"]').click()
        expect(page.locator('h1')).to_have_text('Ảnh mờ do đâu?')
        for reference in data['references']:
            page.locator('#reference').select_option(str(reference['index']))
            for rank in range(3):
                page.locator('#rank').select_option(str(rank))
                expected = [reference['index']] + [
                    values[rank]['index'] if len(values) > rank else None
                    for values in reference['matches'].values()
                ]
                actual = page.locator('#comparisons article').evaluate_all(
                    'nodes => nodes.map(n => n.dataset.index ? Number(n.dataset.index) : null)',
                )
                assert actual == expected
                page.wait_for_function('''() => [...document.images].every(
                    image => image.complete && image.naturalWidth > 0)''')
                assert page.locator('#comparisons img').evaluate_all(
                    'nodes => nodes.every(n => n.naturalWidth === 128 && n.naturalHeight === 128)',
                )
        page.locator('#reference').select_option(str(data['references'][0]['index']))
        page.locator('#rank').select_option('0')
        page.locator('#comparisons').screenshot(path=str(args.output / 'matched_native.png'))
        other = data['references'][min(1, len(data['references']) - 1)]['index']
        page.locator('#reference').select_option(str(other))
        page.locator('#comparisons').screenshot(path=str(args.output / 'matched_second.png'))
        page.locator('#resolutions').screenshot(path=str(args.output / 'resolutions.png'))
        for scale in (1, 2, 3):
            page.locator('#scale').select_option(str(scale))
            for smoothing in ('auto', 'pixelated'):
                page.locator('#smoothing').select_option(smoothing)
                first = page.locator('#comparisons img').first
                assert first.evaluate('n => n.getBoundingClientRect().width') == 128 * scale
                assert first.evaluate('n => getComputedStyle(n).imageRendering') == smoothing
                assert first.evaluate('n => n.naturalWidth') == 128
        for width in (320, 768, 1024, 1440):
            page.set_viewport_size({'width': width, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), width
        page.locator('#reference').focus()
        expect(page.locator('#reference')).to_be_focused()
        browser.close()
    assert not errors, errors
    assert not network, 'Offline review requested external resources'
    print(f'Crop review passed: {len(data["references"]) * 3} matched selections, '
          'report navigation, native dimensions, zoom/smoothing, responsive widths, '
          'no JS/network errors')


if __name__ == '__main__':
    main()
