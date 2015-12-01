'use strict';

var ndarray2d = require('./ndarray2d');

module.exports = network;

function network(data, inputSize, hiddenSize, outputSize, alpha) {
    var cols = inputSize + outputSize;
    var rows = data.length / cols;

    var syn0 = createSynapse(inputSize, hiddenSize);
    var syn1 = createSynapse(hiddenSize, outputSize);

    var l0 = createInputLayer(data, rows, cols, inputSize);
    var l1 = createLayer(rows, hiddenSize);
    var l2 = createLayer(rows, outputSize);
    var y = ndarray2d(data, rows, outputSize, cols, 1, inputSize);

    for (var i = 0; i <= 60000; i++) {
        l1.values.product(l0.values, syn0.values).sigmoid(l1.values);
        l2.values.product(l1.values, syn1.values).sigmoid(l2.values);

        l2.error.sub(l2.values, y);
        updateDelta(l2);

        l1.error.product(l2.delta, syn1.t);
        updateDelta(l1);

        updateWeights(syn1, l1, l2, alpha);
        updateWeights(syn0, l0, l1, alpha);

        if (i % 5000 === 0) console.log('err ' + i + ': ' + l2.error.mean());
    }
}

function updateWeights(synapse, layer1, layer2, alpha) {
    synapse.delta.product(layer1.t, layer2.delta);

    for (var i = 0; i < synapse.delta.data.length; i++) {
        synapse.values.data[i] -= (alpha || 1) * synapse.delta.data[i];
    }
}

function updateDelta(a) {
    for (var i = 0, values = a.values.data; i < values.length; i++) {
        a.delta.data[i] = values[i] * (1 - values[i]) * a.error.data[i];
    }
}

function createLayer(rows, cols) {
    var obj = {};
    obj.values = ndarray2d(new Float32Array(rows * cols), rows, cols);
    obj.error = ndarray2d(new Float32Array(rows * cols), rows, cols);
    obj.delta = ndarray2d(new Float32Array(rows * cols), rows, cols);
    obj.t = obj.values.transpose();
    return obj;
}

function createInputLayer(data, rows, cols, inputSize) {
    var obj = {};
    obj.values = ndarray2d(data, rows, inputSize, rows, cols, 1);
    obj.t = obj.values.transpose();
    return obj;
}

function createSynapse(rows, cols) {
    var obj = {};
    obj.values = ndarray2d(new Float32Array(rows * cols), rows, cols).random(-1, 1);
    obj.delta = ndarray2d(new Float32Array(rows * cols), rows, cols);
    obj.t = obj.values.transpose();
    return obj;
}
