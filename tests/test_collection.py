import base64
from io import BytesIO

import numpy as np
from PIL import Image

from ct_mae.collect import png_data_url


def test_embedded_png_preserves_every_original_pixel():
    original = np.array([[100, 101, 102], [103, 104, 105]], dtype=np.uint8)
    encoded = png_data_url(original).split(",", 1)[1]
    decoded = np.array(Image.open(BytesIO(base64.b64decode(encoded))))
    np.testing.assert_array_equal(decoded, original)
