"""Build the two browser models and verify exported policy actions against PyTorch."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import torch
from stable_baselines3 import PPO
from ultralytics import YOLO
from ultralytics.utils import ASSETS

root = Path(__file__).resolve().parent
assets = root / "assets"
assets.mkdir(exist_ok=True)
torch.set_num_threads(1)
started = time.perf_counter()
policy = PPO("MlpPolicy", "CartPole-v1", seed=7, device="cpu", verbose=0)
policy.learn(total_timesteps=50000)
policy.save(assets / "cartpole-ppo")

class Actor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        features = self.policy.extract_features(observation)
        return self.policy.action_net(self.policy.mlp_extractor.forward_actor(features))

actor = Actor(policy.policy)
actor.train(False)
torch.onnx.export(actor, torch.zeros(1, 4), assets / "cartpole-ppo.onnx",
                  input_names=["observation"], output_names=["logits"],
                  opset_version=17, dynamo=False)
session = ort.InferenceSession(str(assets / "cartpole-ppo.onnx"))
observations = np.random.default_rng(1000).uniform(-0.2, 0.2, (100, 4)).astype(np.float32)
fixtures = []
for observation in observations:
    with torch.no_grad():
        expected = actor(torch.from_numpy(observation[None])).numpy()[0]
    actual = session.run(None, {"observation": observation[None]})[0][0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert np.argmax(actual) == int(policy.predict(observation, deterministic=True)[0])
    fixtures.append({"observation": observation.tolist(), "logits": expected.tolist(), "action": int(np.argmax(expected))})

env = gym.make("CartPole-v1")
returns = {"trained": [], "random": []}
physics = []
for seed in range(1000, 1020):
    for mode in returns:
        observation, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        total = 0
        for step in range(500):
            action = int(policy.predict(observation, deterministic=True)[0]) if mode == "trained" else env.action_space.sample()
            before = env.unwrapped.state.tolist()
            observation, reward, terminated, truncated, _ = env.step(action)
            if seed == 1000 and step < 10:
                physics.append({"before": before, "action": int(action), "after": env.unwrapped.state.tolist()})
            total += reward
            if terminated or truncated:
                break
        returns[mode].append(total)
env.close()
assert np.mean(returns["trained"]) > 400, returns
assert np.mean(returns["trained"]) > np.mean(returns["random"]), returns

detector = YOLO("yolo11n.pt")
exported = detector.export(format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False)
shutil.copyfile(exported, assets / "yolo11n.onnx")
shutil.copyfile(ASSETS / "bus.jpg", assets / "bus.jpg")
report = {
    "seed": 7, "requested_timesteps": 50000, "actual_timesteps": policy.num_timesteps,
    "evaluation_seeds": list(range(1000, 1020)), "returns": returns,
    "mean_returns": {name: float(np.mean(values)) for name, values in returns.items()},
    "policy_export_parity_observations": len(fixtures),
    "versions": {name: importlib.metadata.version(name) for name in ["torch", "stable-baselines3", "gymnasium", "onnx", "onnxruntime", "ultralytics"]},
    "build_seconds": round(time.perf_counter() - started, 1),
    "files": {file.name: {"sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "bytes": file.stat().st_size}
              for file in assets.iterdir() if file.suffix in [".onnx", ".jpg", ".zip"]},
}
(assets / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
(assets / "parity.json").write_text(json.dumps({"policy": fixtures, "physics": physics}, indent=2) + "\n")
print(json.dumps(report, indent=2))
