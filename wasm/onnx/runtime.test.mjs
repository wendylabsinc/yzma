import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import * as ort from 'onnxruntime-web/wasm'
import { createCartPolePolicy, decodeYolo, intersectionOverUnion, letterbox, resetCartPole, seededRandom, stepCartPole } from './runtime.mjs'

test('letterboxing maps boxes back to a portrait image and suppresses only same-class overlaps', () => {
  const geometry = letterbox(320, 640)
  assert.equal(geometry.left, 160)
  const data = new Float32Array(84 * 3)
  for (let i = 0; i < 3; i++) {
    data[i] = 240; data[3 + i] = 100; data[6 + i] = 80; data[9 + i] = 100
  }
  data[4 * 3] = 0.9
  data[4 * 3 + 1] = 0.8
  data[5 * 3 + 2] = 0.7
  const boxes = decodeYolo({ data, dims: [1, 84, 3] }, geometry)
  assert.equal(boxes.length, 2)
  assert.deepEqual(boxes[0].xyxy, [40, 50, 120, 150])
  assert.equal(boxes[0].label, 'person')
  assert.equal(boxes[1].label, 'bicycle')
  assert.equal(decodeYolo({ data, dims: [1, 84, 3] }, geometry, 0.85).length, 1)
  assert.equal(intersectionOverUnion([0, 0, 10, 10], [10, 10, 20, 20]), 0)
})

test('invalid image, tensor, threshold and action contracts fail explicitly', () => {
  assert.throws(() => letterbox(0, 640))
  assert.throws(() => decodeYolo({ data: [], dims: [1, 85, 1] }, letterbox(640, 640)))
  assert.throws(() => decodeYolo({ data: [], dims: [] }, letterbox(640, 640), NaN))
  assert.throws(() => stepCartPole([0, 0, 0, NaN], 0))
  assert.throws(() => stepCartPole([0, 0, 0, 0], 2))
})

test('CartPole transitions match saved Gymnasium trajectories', async () => {
  const fixtures = JSON.parse(await readFile(new URL('./assets/parity.json', import.meta.url)))
  for (const row of fixtures.physics) {
    const actual = stepCartPole(row.before, row.action).observation
    actual.forEach((value, i) => assert.ok(Math.abs(value - row.after[i]) < 1e-12))
  }
  assert.equal(stepCartPole([2.41, 0, 0, 0], 0).terminated, true)
  assert.deepEqual(resetCartPole(7), resetCartPole(7))
})

test('real WASM policy inference matches PyTorch exports and beats random actions on held-out starts', async () => {
  ort.env.wasm.numThreads = 1
  const fixtures = JSON.parse(await readFile(new URL('./assets/parity.json', import.meta.url)))
  const model = await readFile(new URL('./assets/cartpole-ppo.onnx', import.meta.url))
  const policy = await createCartPolePolicy(ort, model)
  try {
    for (const row of fixtures.policy) {
      const actual = await policy.predict(row.observation)
      assert.equal(actual.action, row.action)
      const exp = row.logits.map(n => Math.exp(n - Math.max(...row.logits)))
      const sum = exp[0] + exp[1]
      actual.probabilities.forEach((p, i) => assert.ok(Math.abs(p - exp[i] / sum) < 1e-5))
    }
    const totals = { trained: 0, random: 0 }
    for (let seed = 1000; seed < 1020; seed++) {
      for (const mode of ['trained', 'random']) {
        let observation = resetCartPole(seed)
        const random = seededRandom(seed + 10000)
        for (let i = 0; i < 500; i++) {
          const action = mode === 'trained' ? (await policy.predict(observation)).action : Math.floor(random() * 2)
          const next = stepCartPole(observation, action)
          totals[mode] += next.reward
          observation = next.observation
          if (next.terminated) break
        }
      }
    }
    assert.ok(totals.trained / 20 > 400, JSON.stringify(totals))
    assert.ok(totals.trained > totals.random, JSON.stringify(totals))
    process.stdout.write(`WASM mean returns: trained ${totals.trained / 20}, random ${totals.random / 20}\n`)
  } finally {
    await policy.dispose()
  }
})
