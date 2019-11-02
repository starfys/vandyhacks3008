#!/usr/bin/env python3

from base64 import b64decode
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps
from sanic import Sanic, response
from sanic.log import logger

# ======================
# Define the app
# ======================
app = Sanic()

# Serve index.html statically
app.static("/", "../static")


@app.route('/transform', methods=["POST"])
async def test(request):
    logger.info("received trans request")
    # Get files from input
    logger.info(request.files)
    decoded_file = b64decode(request.body.split(b',')[1])
    image = Image.open(BytesIO(decoded_file))
    inverted = ImageOps.invert(image)
    out_bytes = BytesIO()
    inverted.save(out_bytes, format='jpeg')
    # Get the file
    return response.raw(out_bytes.getvalue())


# ============================
# Run the app
# ============================
if __name__ == '__main__':
    ssl = {'cert': "certs/MyCertificate.crt", 'key': "certs/MyKey.key"}
    app.run(host='0.0.0.0', port=3000, ssl=ssl)
