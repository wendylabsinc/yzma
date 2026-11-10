export const COCO_LABELS = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
  'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
  'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]

export function letterbox(width, height, size = 640) {
  if (![width, height, size].every(n => Number.isInteger(n) && n > 0 && n <= 8192)) {
    throw new Error('Image dimensions must be positive integers at most 8192 pixels.')
  }
  const scale = Math.min(size / width, size / height)
  const resizedWidth = Math.max(1, Math.round(width * scale))
  const resizedHeight = Math.max(1, Math.round(height * scale))
  return { width, height, size, resizedWidth, resizedHeight,
    left: Math.floor((size - resizedWidth) / 2), top: Math.floor((size - resizedHeight) / 2) }
}

export function yoloInput(ort, bitmap) {
  const geometry = letterbox(bitmap.width, bitmap.height)
  const { size, resizedWidth, resizedHeight, left, top } = geometry
  const canvas = new OffscreenCanvas(size, size)
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('This browser cannot prepare images in a worker.')
  context.fillStyle = 'rgb(114,114,114)'
  context.fillRect(0, 0, size, size)
  context.drawImage(bitmap, left, top, resizedWidth, resizedHeight)
  const rgba = context.getImageData(0, 0, size, size).data
  const pixels = size * size
  const rgb = new Float32Array(pixels * 3)
  for (let i = 0; i < pixels; i++) {
    rgb[i] = rgba[i * 4] / 255
    rgb[i + pixels] = rgba[i * 4 + 1] / 255
    rgb[i + 2 * pixels] = rgba[i * 4 + 2] / 255
  }
  return { tensor: new ort.Tensor('float32', rgb, [1, 3, size, size]), geometry }
}

export function intersectionOverUnion(a, b) {
  const intersection = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) *
    Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]))
  const area = box => Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1])
  return intersection / Math.max(Number.EPSILON, area(a) + area(b) - intersection)
}

export function decodeYolo(output, geometry, threshold = 0.25, iouThreshold = 0.45) {
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new Error('Invalid confidence threshold.')
  if (!Number.isFinite(iouThreshold) || iouThreshold < 0 || iouThreshold > 1) throw new Error('Invalid IoU threshold.')
  const { data, dims } = output
  if (dims.length !== 3 || dims[0] !== 1 || dims[1] !== 84 || dims[2] < 1 || data.length !== 84 * dims[2]) {
    throw new Error('Expected YOLO11 detection output [1, 84, N].')
  }
  const { width, height, resizedWidth, resizedHeight, left, top } = geometry
  const count = dims[2]
  const candidates = []
  for (let i = 0; i < count; i++) {
    let classId = 0
    let confidence = data[4 * count + i]
    for (let c = 1; c < 80; c++) {
      const score = data[(c + 4) * count + i]
      if (score > confidence) { confidence = score; classId = c }
    }
    if (!Number.isFinite(confidence) || confidence < threshold) continue
    const [x, y, w, h] = [data[i], data[count + i], data[count * 2 + i], data[count * 3 + i]]
    if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) continue
    const clip = (value, max) => Math.min(max, Math.max(0, value))
    const xyxy = [clip((x - w / 2 - left) * width / resizedWidth, width),
      clip((y - h / 2 - top) * height / resizedHeight, height),
      clip((x + w / 2 - left) * width / resizedWidth, width),
      clip((y + h / 2 - top) * height / resizedHeight, height)]
    if (xyxy[2] <= xyxy[0] || xyxy[3] <= xyxy[1]) continue
    candidates.push({ label: COCO_LABELS[classId], classId, confidence, xyxy })
  }
  candidates.sort((a, b) => b.confidence - a.confidence)
  const selected = []
  for (const candidate of candidates.slice(0, 3000)) {
    if (selected.every(box => box.classId !== candidate.classId || intersectionOverUnion(box.xyxy, candidate.xyxy) <= iouThreshold)) {
      selected.push(candidate)
      if (selected.length === 300) break
    }
  }
  return selected
}

export async function createDetector(ort, model) {
  const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] })
  return {
    async detect(bitmap, threshold = 0.25) {
      const { tensor, geometry } = yoloInput(ort, bitmap)
      let outputs
      try {
        const started = performance.now()
        outputs = await session.run({ [session.inputNames[0]]: tensor })
        const inferenceMs = performance.now() - started
        return { detections: decodeYolo(outputs[session.outputNames[0]], geometry, threshold),
          width: bitmap.width, height: bitmap.height, inferenceMs }
      } finally {
        tensor.dispose()
        if (outputs) Object.values(outputs).forEach(tensor => tensor.dispose())
      }
    },
    dispose: () => session.release(),
  }
}

function validateObservation(observation) {
  if (observation.length !== 4 || !Array.from(observation).every(Number.isFinite)) {
    throw new Error('Expected four finite CartPole observations [x, velocity, angle, angular velocity].')
  }
}

export async function createCartPolePolicy(ort, model) {
  const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] })
  return {
    async predict(observation) {
      validateObservation(observation)
      const input = new ort.Tensor('float32', Float32Array.from(observation), [1, 4])
      let outputs
      try {
        outputs = await session.run({ observation: input })
        const logits = Array.from(outputs.logits.data)
        if (logits.length !== 2 || !logits.every(Number.isFinite)) throw new Error('Expected two finite policy logits.')
        const max = Math.max(...logits)
        const exps = logits.map(value => Math.exp(value - max))
        const total = exps[0] + exps[1]
        return { action: logits[0] >= logits[1] ? 0 : 1, probabilities: exps.map(value => value / total) }
      } finally {
        input.dispose()
        if (outputs) Object.values(outputs).forEach(tensor => tensor.dispose())
      }
    },
    dispose: () => session.release(),
  }
}

// CartPole-v1 equations and constants follow Gymnasium's MIT-licensed environment.
export function stepCartPole(observation, action) {
  validateObservation(observation)
  if (action !== 0 && action !== 1) throw new Error('CartPole action must be 0 or 1.')
  const [x, velocity, angle, angularVelocity] = observation
  const sin = Math.sin(angle), cos = Math.cos(angle)
  const temp = ((action === 1 ? 10 : -10) + 0.05 * angularVelocity ** 2 * sin) / 1.1
  const angleAcceleration = (9.8 * sin - cos * temp) / (0.5 * (4 / 3 - 0.1 * cos ** 2 / 1.1))
  const acceleration = temp - 0.05 * angleAcceleration * cos / 1.1
  const next = [x + 0.02 * velocity, velocity + 0.02 * acceleration,
    angle + 0.02 * angularVelocity, angularVelocity + 0.02 * angleAcceleration]
  return { observation: next, reward: 1, terminated: Math.abs(next[0]) > 2.4 || Math.abs(next[2]) > 12 * Math.PI / 180 }
}

export function seededRandom(seed) {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6D2B79F5) >>> 0
    let n = Math.imul(state ^ (state >>> 15), 1 | state)
    n ^= n + Math.imul(n ^ (n >>> 7), 61 | n)
    return ((n ^ (n >>> 14)) >>> 0) / 4294967296
  }
}

export function resetCartPole(seed) {
  const random = seededRandom(seed)
  return Array.from({ length: 4 }, () => random() * 0.1 - 0.05)
}
