# Browser detector and policy inference

This Wendy fork adds an ONNX Runtime Web backend alongside yzma's llama.cpp backend. It runs YOLO11n detection and a trained CartPole PPO policy from model weights. It does not send images or observations to a model server. This is a JavaScript browser extension; the native Go llama.cpp API is unchanged.

The package exports `createDetector`, `createCartPolePolicy`, and the preprocessing and simulation functions from `runtime.mjs`. Pass an ONNX Runtime Web module and model bytes to the factory. Run these calls in a module worker. Use `ort.env.wasm.numThreads = 1` for pages without cross-origin isolation. The initial backend is WASM CPU; it does not claim GPU acceleration.

```js
import * as ort from 'onnxruntime-web/wasm'
import { createDetector, createCartPolePolicy } from '@wendy/yzma-browser'

ort.env.wasm.numThreads = 1
const detector = await createDetector(ort, detectorModelBytes)
const result = await detector.detect(await createImageBitmap(imageFile), 0.25)
const policy = await createCartPolePolicy(ort, policyModelBytes)
const { action, probabilities } = await policy.predict([0, 0, 0.03, 0])
await detector.dispose()
await policy.dispose()
```

## Model contracts

### Python in the browser

`@wendy/yzma-browser/python` exports `installPythonInference(pyodide, {detector, bitmap, policy})`. Initialize the desired model in the same worker, then install this bridge before calling `pyodide.runPythonAsync`. Python can import `detect`, `predict`, `reset`, and `step` from `wendy_inference`. `await detect(threshold=0.25)` runs YOLO on the supplied bitmap and returns detections in original image pixels. `await predict(observation)` runs the trained PPO network and returns its action and probabilities. `reset(seed)` and `step(observation, action)` use the same CartPole simulator as JavaScript. The host supplies Pyodide and must terminate the worker to stop unbounded Python code. A run is limited to 10 detector calls and 10,000 policy calls. This bridge performs inference; it does not install PyTorch or train PPO in the browser.

YOLO11n takes RGB floats in `[1,3,640,640]`, scaled to 0–1. The image is resized with its aspect ratio preserved and padded with 114. Output `[1,84,8400]` contains center coordinates, dimensions and 80 COCO scores. The decoder applies class-aware nonmaximum suppression and returns `xyxy` coordinates in the original image. It supports this detector export, not arbitrary YOLO output formats.

The PPO actor takes `[x, x_velocity, angle_radians, angular_velocity]` as `[1,4]` float32 and returns two logits. Argmax selects action 0 for left or 1 for right. The simulator uses CartPole-v1's Euler integration at 0.02 seconds per step, termination at 2.4 metres or 12 degrees, and reward 1 per step. The caller must truncate at 500 steps. The seeded browser reset generator differs from Gymnasium's, so numerical seed values do not reproduce Gymnasium starts. Paired policies use identical browser starts.

## Reproduce and verify

Use Python 3.12 and install `requirements.txt` in a virtual environment, then run `python wasm/onnx/export-models.py` from the repository root. This downloads the upstream YOLO11n weights, trains PPO for 50,000 requested steps with seed 7, exports both models, checks 100 policy outputs against PyTorch, and evaluates trained and random policies on 20 held-out Gymnasium seeds. The PPO library may complete its rollout batch beyond the requested step count. `assets/manifest.json` records actual steps, versions, evaluation results and SHA-256 hashes. `assets/cartpole-ppo.zip` is the reloadable training checkpoint.

Run `npm install` and `npm test` to check coordinate mapping, suppression, input validation, Gymnasium physics parity, and real WASM policy inference against the PyTorch fixtures. The WASM test also compares the policy with random actions over 20 paired starts.

## Sources and licenses

The runtime extension and the trained CartPole checkpoint are Apache-2.0 under the repository license. CartPole equations follow [Gymnasium](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py); its MIT notice is in `GYMNASIUM-LICENSE`.

`assets/yolo11n.onnx` is an export of [Ultralytics YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt). That model and the bundled Ultralytics `bus.jpg` example retain their upstream AGPL-3.0 terms; see `ULTRALYTICS-LICENSE` and the [upstream source](https://github.com/ultralytics/ultralytics). They are separate assets and are not relicensed Apache-2.0. `export-models.py` supplies the conversion procedure. ONNX Runtime is MIT-licensed and is supplied by the consuming application.

Browser clients should show download progress, verify model hashes, and terminate the worker when the user stops or leaves the exercise. Detection and policy inference are real. The pretrained policy does not learn while the browser episode runs.
