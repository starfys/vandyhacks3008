#!/usr/bin/env python3

from PIL import Image
from sanic import Sanic
from sanic.response import json

app = Sanic()

# Serve index.html statically
app.static("/", "../static")


@app.route('/upload')
async def test(request):
    return json({'hello': 'world'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
