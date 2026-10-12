import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import MAX_UPLOAD_BYTES, app, get_predictor


@pytest.fixture
def client():
    def fake(image):
        assert image.mode == 'RGB'
        return {'label': 'shop', 'score': .8, 'probabilities': {'landing': .1, 'shop': .8, 'blog': .1}}
    app.dependency_overrides[get_predictor] = lambda: fake
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def image_bytes(fmt='PNG', mode='RGB'):
    output = io.BytesIO()
    Image.new(mode, (64, 48), 'white').save(output, format=fmt)
    return output.getvalue()


def test_health_and_schema(client):
    assert client.get('/health').json()['status'] == 'ok'
    assert client.get('/model').json()['labels'] == ['landing', 'shop', 'blog']
    assert '/predict' in client.get('/openapi.json').json()['paths']


@pytest.mark.parametrize('fmt,mode', [('PNG', 'RGB'), ('JPEG', 'RGB'), ('WEBP', 'RGB'), ('PNG', 'RGBA')])
def test_predict_contract(client, fmt, mode):
    response = client.post('/predict', files={'file': ('screen', image_bytes(fmt, mode))})
    assert response.status_code == 200
    result = response.json()
    assert result['label'] == 'shop'
    assert result['score'] == .8
    assert sum(result['probabilities'].values()) == pytest.approx(1)
    assert result['model'] == 'openai/clip-vit-base-patch32'


@pytest.mark.parametrize('payload,status', [(b'', 400), (b'not an image', 400), (b'x' * (MAX_UPLOAD_BYTES + 1), 413)])
def test_bad_upload(client, payload, status):
    assert client.post('/predict', files={'file': ('screen.png', payload, 'image/png')}).status_code == status


def test_missing_file(client):
    assert client.post('/predict').status_code == 422


def test_unsupported_format(client):
    assert client.post('/predict', files={'file': ('screen.gif', image_bytes('GIF'))}).status_code == 415


def test_truncated_image(client):
    assert client.post('/predict', files={'file': ('screen.png', image_bytes()[:40])}).status_code == 400


def test_pixel_limit(client, monkeypatch):
    monkeypatch.setattr('api.main.MAX_IMAGE_PIXELS', 100)
    assert client.post('/predict', files={'file': ('screen.png', image_bytes())}).status_code == 413


def test_unavailable_model(client):
    def broken(_):
        raise OSError('weights missing')
    app.dependency_overrides[get_predictor] = lambda: broken
    r = client.post('/predict', files={'file': ('screen.png', image_bytes())})
    assert r.status_code == 503
    assert 'weights missing' not in r.text
