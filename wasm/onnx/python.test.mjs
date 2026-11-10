import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import { loadPyodide } from 'pyodide'
import * as ort from 'onnxruntime-web/wasm'
import { createCartPolePolicy, resetCartPole } from './runtime.mjs'
import { installPythonInference } from './python.mjs'

test('Python awaits real PPO inference and uses the same simulator', async () => {
  ort.env.wasm.numThreads = 1
  const python = await loadPyodide()
  const policy = await createCartPolePolicy(ort, await readFile(new URL('./assets/cartpole-ppo.onnx', import.meta.url)))
  try {
    const bridge = installPythonInference(python, { policy })
    const json = await python.runPythonAsync(`
from wendy_inference import predict, reset, step
import json
observation = reset(1000)
prediction = await predict(observation)
transition = step(observation, prediction["action"])
json.dumps({"prediction": prediction, "transition": transition})
`)
    const actual = JSON.parse(json)
    assert.deepEqual(actual.prediction, await policy.predict(resetCartPole(1000)))
    assert.equal(actual.transition.reward, 1)
    assert.equal(actual.transition.observation.length, 4)
    assert.equal(bridge.counts().predictions, 1)
    await assert.rejects(python.runPythonAsync('await predict([0, 0, 0])'), /four finite/)
    await assert.rejects(python.runPythonAsync('await detect()'), /not defined/)
    await assert.rejects(python.runPythonAsync('from wendy_inference import detect\nawait detect()'), /detection lesson/)
    assert.throws(() => python.runPython('reset(-1)'), /Seed must/)
  } finally { await policy.dispose() }
})

test('Python detector wrapper passes thresholds and preserves pixel coordinates', async () => {
  const python = await loadPyodide()
  let thresholdSeen
  installPythonInference(python, { bitmap: {}, detector: { async detect(_bitmap, threshold) {
    thresholdSeen = threshold
    return { width: 640, height: 480, detections: [{ label: 'test', xyxy: [20, 30, 80, 90], confidence: 0.91 }] }
  } } })
  const result = JSON.parse(await python.runPythonAsync('from wendy_inference import detect\nimport json\njson.dumps(await detect(0.7))'))
  assert.equal(thresholdSeen, 0.7)
  assert.deepEqual(result.detections[0].xyxy, [20, 30, 80, 90])
})
