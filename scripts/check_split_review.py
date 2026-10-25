"""Exercise every organ/model/error mode and paired augmentation in the offline split review."""

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
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('request', lambda r: network.append(r.url) if r.url.startswith('http') else None)
        page.goto(args.report.resolve().as_uri(), wait_until='networkidle')
        expect(page.locator('h1')).to_contain_text('validation cao hơn test')
        assert page.locator('#organ option').count() == 11
        page.screenshot(path=str(args.output / 'overview.png'))
        page.locator('#comparison').evaluate('e=>e.scrollIntoView({block:"start"})')
        page.screenshot(path=str(args.output / 'kidney_comparison.png'))
        checked = 0
        for model in ('full', 'partial'):
            page.locator('#model').select_option(model)
            for organ in range(11):
                page.locator('#organ').select_option(str(organ))
                for mode in ('random', 'correct', 'wrong'):
                    page.locator('#mode').select_option(mode)
                    assert page.locator('#comparison article').count() == 32
                    cards = page.locator('#comparison article').all()
                    for row in range(8):
                        assert cards[4*row].get_attribute('data-key') == (
                            cards[4*row+1].get_attribute('data-key'))
                        for column in (2, 3):
                            card = cards[4*row+column]
                            if mode != 'random' and card.locator('img').count():
                                state = '.good' if mode == 'correct' else '.bad'
                                assert card.locator(state).count()
                    page.wait_for_function('''() => [...document.images].every(
                        x=>x.complete && x.naturalWidth>0)''')
                    checked += 1
        page.locator('#organ').select_option('4')
        page.locator('#mode').select_option('random')
        for pool in ('train5', 'train_all', 'added5'):
            page.locator('#train-pool').select_option(pool)
            for variant in ('0', '1'):
                page.locator('#view').select_option(variant)
                assert page.locator('#comparison article[data-augmented=true] img').count() == 8
        page.locator('#train-pool').select_option('train5')
        page.locator('#page').select_option('1')
        page.locator('#scale').select_option('256')
        assert page.locator('#comparison img').first.evaluate('e=>e.width') == 256
        assert page.locator('#comparison img').first.evaluate('e=>e.naturalWidth') == 128
        for metric in ('mean', 'std', 'dark', 'edge'):
            page.locator('#metric').select_option(metric)
            assert page.locator('#conditional-table tbody tr').count() == 4
        for width in (320, 768, 1024, 1440):
            page.set_viewport_size({'width': width, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'), width
        result = {'browser': browser.version, 'organ_model_mode_combinations': checked,
                  'train_pools': 3, 'augmentation_views': 2, 'source_pixels': 128,
                  'responsive_widths': [320, 768, 1024, 1440],
                  'page_errors': errors, 'external_requests': network}
        browser.close()
    assert not errors, errors
    assert not network, network
    (args.output / 'browser_verification.json').write_text(json.dumps(result, indent=2), 'utf-8')
    print('Split review passed: 66 combinations, paired views, pools, pixels and offline layout')


if __name__ == '__main__':
    main()
