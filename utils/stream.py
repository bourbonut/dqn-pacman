import asyncio
from PIL import Image
import io
from quart import Quart, render_template, websocket

def initialize_app(stream_function):
    app = Quart(__name__)

    @app.route("/")
    async def hello():
        return await render_template("index.html")

    @app.websocket("/ws")
    async def ws():
        while True:
            data = stream_function()
            await websocket.send(data)

    return app
