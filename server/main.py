#!/usr/bin/env python3

from io import BytesIO
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
    image = Image.open(BytesIO(request.files["i"][0].body))
    inverted = ImageOps.invert(image)
    out_bytes = BytesIO()
    inverted.save(out_bytes, format='PNG')
    # Get the file
    return response.raw(out_bytes.getvalue())


# ============================
# Run the app
# ============================
if __name__ == '__main__':
    ssl = {'cert': "certs/MyCertificate.crt", 'key': "certs/MyKey.key"}
    app.run(host='0.0.0.0', port=3000, ssl=ssl)
