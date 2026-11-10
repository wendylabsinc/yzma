import { resetCartPole, stepCartPole } from './runtime.mjs'

const pythonModule = `import json
from _wendy_inference import detect_json, predict_json, reset_json, step_json

async def detect(threshold=0.25):
    """Run YOLO11n on the image selected in the lesson. Boxes use original image pixels."""
    return json.loads(await detect_json(threshold))

async def predict(observation):
    """Run the pretrained PPO actor. Return action and left/right probabilities."""
    return json.loads(await predict_json(json.dumps(observation, allow_nan=False)))

def reset(seed=1000):
    """Return the same seeded CartPole starting observation as the guided exercise."""
    return json.loads(reset_json(seed))

def step(observation, action):
    """Advance CartPole by 0.02 seconds. The caller enforces the 500-step limit."""
    return json.loads(step_json(json.dumps(observation, allow_nan=False), action))
`

export function installPythonInference(python, { detector, bitmap, policy } = {}) {
  let detections = 0, predictions = 0
  python.registerJsModule('_wendy_inference', {
    async detect_json(threshold) {
      if (!detector || !bitmap) throw new Error('YOLO inference is available in the detection lesson.')
      if (++detections > 10) throw new Error('Limit each run to 10 detector calls.')
      return JSON.stringify(await detector.detect(bitmap, threshold))
    },
    async predict_json(observation) {
      if (!policy) throw new Error('PPO inference is available in the policy lessons.')
      if (++predictions > 10000) throw new Error('Limit each run to 10000 policy calls.')
      return JSON.stringify(await policy.predict(JSON.parse(observation)))
    },
    reset_json(seed) {
      if (!Number.isInteger(seed) || seed < 0 || seed > 1000000) throw new Error('Seed must be an integer from 0 to 1000000.')
      return JSON.stringify(resetCartPole(seed))
    },
    step_json(observation, action) {
      return JSON.stringify(stepCartPole(JSON.parse(observation), action))
    },
  })
  python.FS.writeFile('/home/pyodide/wendy_inference.py', pythonModule)
  python.runPython('import sys\nsys.modules.pop("wendy_inference", None)\nsys.modules.pop("_wendy_inference", None)')
  return { counts: () => ({ detections, predictions }) }
}
