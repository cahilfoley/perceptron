import Perceptron from './perceptron.mjs'

// Training config
const batchSize = 100
const pointsToShow = 500
const learningRate = 0.0001
const trainingSize = 100000
const testSize = 1000

// Coordinate space
const xmin = -1
const xmax = 1
const ymin = -1
const ymax = 1

// Globals
let f
let ptron
let trainingSet = []
let testSet = []
let count
let total
let renderer

const accuracyOut = document.getElementById('accuracy')
function updateAccuracy(value) {
  const accuracyVal = Number(value * 100).toFixed(2)
  accuracyOut.innerHTML = `${accuracyVal}%`
  if (accuracyVal === '100.00') {
    noLoop()
  }
}

const pointsOut = document.getElementById('points')
function updatePoints(value) {
  pointsOut.innerHTML = `${value} (batches of ${batchSize})`
}

function generateSample() {
  const x = random(xmin, xmax)
  const y = random(ymin, ymax)
  const answer = y < f(x) ? -1 : 1
  return {
    input: [x, y, 1],
    output: answer
  }
}

function reset() {
  count = 0
  total = 0

  // Linear function
  const vals = [
    (Math.random() * (xmax - xmin) - xmax) * 0.75,
    (Math.random() * (xmax - xmin) - xmax) * 0.75
  ]
  f = x => vals[0] * x + vals[1]

  // The perceptron has 3 inputs -- x, y, and bias
  // Second value is "Learning Constant"
  ptron = new Perceptron(3, learningRate)

  // Create training data
  const iterations = Math.max(trainingSize, testSize)
  trainingSet = []
  testSet = []
  for (let i = 0; i < iterations; i++) {
    if (i < trainingSize) trainingSet.push(generateSample())
    if (i < testSize) testSet.push(generateSample())
  }

  updateAccuracy(ptron.calculateAccuracy(testSet))
  updatePoints(0)
}

window.setup = () => {
  renderer = createCanvas(
    document.getElementById('outputWrapper').getBoundingClientRect().width,
    600
  )
  renderer.canvas.id = 'mainCanvas'

  reset()
}

window.draw = () => {
  background(0)

  // Draw line f(x)
  strokeWeight(2)
  stroke(255, 255, 0)
  let x1 = map(xmin, xmin, xmax, 0, width)
  let y1 = map(f(xmin), ymin, ymax, height, 0)
  let x2 = map(xmax, xmin, xmax, 0, width)
  let y2 = map(f(xmax), ymin, ymax, height, 0)
  line(x1, y1, x2, y2)

  // Draw line for current weights
  stroke(255, 0, 0)
  strokeWeight(4)
  const weights = ptron.weights
  x1 = xmin
  y1 = (-weights[2] - weights[0] * x1) / weights[1]
  x2 = xmax
  y2 = (-weights[2] - weights[0] * x2) / weights[1]

  x1 = map(x1, xmin, xmax, 0, width)
  y1 = map(y1, ymin, ymax, height, 0)
  x2 = map(x2, xmin, xmax, 0, width)
  y2 = map(y2, ymin, ymax, height, 0)
  line(x1, y1, x2, y2)

  // Train the perceptron with one training point at a time
  for (let i = 0; i < batchSize; i++) {
    ptron.train(trainingSet[count].input, trainingSet[count].output)
    count = (count + 1) % trainingSet.length
  }
  total++

  updatePoints(total)
  updateAccuracy(ptron.calculateAccuracy(testSet))

  for (let i = 0; i < pointsToShow; i++) {
    let index = (count + i) % trainingSet.length
    const tVal = trainingSet[index]
    stroke(255)
    strokeWeight(1)
    fill(255)

    const guess = ptron.feedForward(tVal.input)
    if (tVal.output > 0) noFill()
    if (guess > 0 && tVal.output < 0) fill(255, 0, 0)
    if ((guess > 0 && tVal.output < 0) || (guess < 0 && tVal.output > 0)) {
      stroke(255, 0, 0)
    }

    const { input } = tVal
    const x = map(input[0], xmin, xmax, 0, width)
    const y = map(input[1], ymin, ymax, height, 0)
    ellipse(x, y, 6, 6)
  }
}
