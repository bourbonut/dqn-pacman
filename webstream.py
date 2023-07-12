from main import (
    DataHandler, Buffer, args, start, ALEInterface, Pacman, gym, params, device, ReplayMemory, DQN
)

from quart import Quart, render_template, websocket

paths = start(args)
# Set environment
ale = ALEInterface()
ale.loadROM(Pacman)
env = gym.make("MsPacman-v0")

policy = DQN(params.N_ACTIONS).to(device)
target = DQN(params.N_ACTIONS).to(device)
memory = ReplayMemory(params.REPLAY_MEMORY_SIZE, params.BATCH_SIZE)

buffer = Buffer()
datahandler = DataHandler(env, policy, target, memory, buffer, paths)

app = Quart(__name__)

@app.route("/")
async def hello():
    return await render_template("index.html")

@app.websocket("/ws")
async def ws():
    i = 0
    for _ in datahandler.run():
        await websocket.send(buffer.json())
        i += 1
        if i == 10000:
            raise

if __name__ == "__main__":
    app.run(port=5000)
