'use strict';

var ndarray2d = require('./ndarray2d');

module.exports = network;

function network(data, inputSize, hiddenSize, outputSize, alpha) {
    alpha = alpha || 1;
    var cols = inputSize + outputSize;
    var rows = data.length / cols;

    var syn0 = createSynapse(inputSize, hiddenSize);
    var syn1 = createSynapse(hiddenSize, outputSize);

    var l0 = createInputLayer(data, rows, cols, inputSize);
    var l1 = createLayer(rows, hiddenSize);
    var l2 = createLayer(rows, outputSize);
    var y = ndarray2d(data, rows, outputSize, cols, 1, inputSize);

    for (var i = 0; i < 60000; i++) {
        feedforward(l0, l1, syn0);
        feedforward(l1, l2, syn1);

        l2.error.sub(l2.outputs, y);
        backpropagate(l2, l1, syn1, alpha);

        l1.error.product(l2.delta, syn1.weightsT);
        backpropagate(l1, l0, syn0, alpha);

        if ((i + 1) % 5000 === 0) console.log('err %d after %d iter', l2.error.mean().toFixed(10), i + 1);
    }
}

function feedforward(layer, next, synapse) {
    next.outputs.product(layer.outputs, synapse.weights).sigmoideq();
}

function backpropagate(layer, prev, synapse, alpha) {
    layer.delta.dsigmoid(layer.outputs).muleq(layer.error);
    synapse.delta.product(prev.outputsT, layer.delta).mulseq(alpha);
    synapse.weights.subeq(synapse.delta);
}

function createLayer(rows, cols) {
    var obj = {};
    obj.outputs = ndarray2d(new Float32Array(rows * cols), rows);
    obj.outputsT = obj.outputs.transpose();
    obj.error = ndarray2d(new Float32Array(rows * cols), rows);
    obj.delta = ndarray2d(new Float32Array(rows * cols), rows);
    return obj;
}

function createInputLayer(data, rows, cols, inputSize) {
    var obj = {};
    obj.outputs = ndarray2d(data, rows, inputSize, cols);
    obj.outputsT = obj.outputs.transpose();
    return obj;
}

function createSynapse(rows, cols) {
    var obj = {};
    obj.weights = ndarray2d(new Float32Array(rows * cols), rows).random(-1, 1);
    obj.weightsT = obj.weights.transpose();
    obj.delta = ndarray2d(new Float32Array(rows * cols), rows);
    return obj;
}
