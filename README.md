# roadagent-boardgame-simulation
A partially observable multi-agent game environment and reinforcement learning project built around my board game. This project trains a recurrent MAPPO policy for 3-player team play, using legal action masks, private observations, and centralised training with decentralised execution. I also implemented different missions, items and characters

## Established mission rules
`missions.py` defines these before training:
1. `standard`: all secret goals must be met.
2. `minimum_total`: every player must have at least 15 total locked points, plus all goals.
3. `team_sixes`: the team must lock at least three 6s, plus all goals.
They are test missions because the provided rulebook does not include individual Mission Card text. Replace or add `Mission` entries in `missions.py` before training an official campaign.

## Install requirements
python3 -m pip install torch numpy

## Play a human mission
python3 play_game.py

## Verify the code
python3 -m unittest -v test_smoke.py

## Where to start
Read README.md first to understand the project goal and how it is intended to be used
Read road_agent.py next since this tells you the rules of the game
Read missions.py
Read MAPO
Run the smoke test
Train a model
Compare models
