import os
import shutil
import uvicorn
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .solver import Solver
from ..poker.component.street import Street

class PolicyQuery(BaseModel):
    board_cards: list[str]
    action_history: list[int]
    solver: str = 'showdown'
    street: str = 'showdown'

origins = [
    "http://localhost",
    "http://localhost:18890",
    "http://103.170.5.183:18890",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
# app.solver_manager = SolverManager()

def pretty_floats(obj, ndigits=2):
    if isinstance(obj, float):
        return round(obj, ndigits=ndigits)
    elif isinstance(obj, dict):
        return dict((k, pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, list):
        return list(map(pretty_floats, obj))
    return obj

@app.post('/export')
def export_policy(run: str, save_name: str):
    max_id = -1
    checkpoint = ''
    for sub_dir in os.listdir(run):
        if sub_dir.startswith('PPO'):
            for checkpoint_dir in os.listdir(os.path.join(run, sub_dir)):
                if checkpoint_dir.startswith('checkpoint'):
                    print(checkpoint_dir, int(checkpoint_dir[11:]))
                    if int(checkpoint_dir[11:]) > max_id:
                        max_id = int(checkpoint_dir[11:])
                        checkpoint = os.path.join(run, sub_dir, checkpoint_dir)
    src_pt = os.path.join(checkpoint, 'policies', 'learned', 'model.pt')
    if not os.path.exists(src_pt):
        return {'result': 'Checkpoint not found'}
    else:
        os.makedirs(os.path.join('./checkpoint', save_name), exist_ok=True)
        dst_pt = os.path.join('./checkpoint', save_name, 'model.pt')
        shutil.copyfile(src_pt, dst_pt)
        return {'result': 'Done'}

@app.post("/policy")
async def get_policy(query: PolicyQuery):
    solver = Solver(
        model_path=os.path.join('./checkpoint', query.solver, 'model.pt'),
        showdown_street=Street.from_str(query.street)
    )
    policy, observation = solver.query(
        board_cards=query.board_cards,
        action_history=query.action_history,
    ) 
    return pretty_floats({"policy": policy.tolist(), "observation": observation})

def main():
    uvicorn.run(app='alphaholdem.solver.main:app', host='0.0.0.0', port=18889, reload=True)
