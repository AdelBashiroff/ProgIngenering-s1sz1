import pandas as pd
import pytest
from image.capture_screenshots import image_key, load_urls
from image.labels import normalize_label


def test_normalization():
    assert normalize_label('  Интернет-магазин ') == 'shop'
    assert normalize_label('unknown') is None
    assert normalize_label(None) is None


def test_stable_keys():
    assert image_key('https://site.test/a', 'shop') != image_key('https://site.test/b', 'shop')
    assert image_key('https://site.test/a', 'shop') == image_key('https://site.test/a', 'shop')


def test_duplicate_urls(tmp_path):
    p = tmp_path / 'urls.csv'
    p.write_text('url,label\nhttps://example.com,лендинг\nhttps://example.com,landing\n')
    df = load_urls(p)
    assert len(df) == 1
    assert df.iloc[0]['label'] == 'landing'


@pytest.mark.parametrize('row', ['ftp://example.com,blog', 'https://example.com,other'])
def test_invalid_source(tmp_path, row):
    p = tmp_path / 'urls.csv'
    p.write_text('url,label\n' + row)
    with pytest.raises(ValueError):
        load_urls(p)
