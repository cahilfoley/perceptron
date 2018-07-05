class Perceptron {
  constructor(dimensions, learningRate) {
    this._dimensions = dimensions
    this._weights = new Array(dimensions)
    this._learningRate = learningRate
    this._initializeWeights()
  }

  _activate(sum) {
    return sum > 0 ? 1 : -1
  }

  _initializeWeights() {
    for (let i = 0; i < this._dimensions; i++) {
      this._weights[i] = Math.random()
    }
  }

  _weightInputs(inputs) {
    return this._weights.map((weight, i) => weight * inputs[i])
  }

  calculateAccuracy(testSet) {
    let correctGuesses = 0
    for (let i = 0; i < testSet.length; i++) {
      const guess = this.feedForward(testSet[i].input)
      if (guess === testSet[i].output) correctGuesses++
    }
    return correctGuesses / testSet.length
  }

  feedForward(inputs) {
    const weightedInputs = this._weightInputs(inputs)
    const weightedTotal = weightedInputs.reduce((total, val) => total + val)
    return this._activate(weightedTotal)
  }

  train(inputs, desired) {
    // First we guess
    const guess = this.feedForward(inputs)

    // Calculate the error (Error = desired output - guess)
    const error = desired - guess

    // Adjust the weights based on the learningRate * error * input
    this._weights = this._weights.map(
      (w, i) => w + this._learningRate * error * inputs[i]
    )
  }

  get weights() {
    return this._weights
  }
}

export default Perceptron
