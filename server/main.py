#!/usr/bin/env python3

from PIL import Image
from sanic import Sanic, response

# ======================
# Define the app
# ======================
app = Sanic()

# Serve index.html statically
app.static("/", "../static")


@app.route('/transform')
async def test(request):
    print("received trans request")
    return await response.file('../static/uwuwu.jpg')


# ============================
# Run the app
# ============================
if __name__ == '__main__':
    ssl = {'cert': "certs/MyCertificate.crt", 'key': "certs/MyKey.key"}
    app.run(host='0.0.0.0', port=3000, ssl=ssl)
