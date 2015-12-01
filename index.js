'use strict';

var ndarray = require('ndarray');

module.exports = network;

function network(input, output, rows, hiddenSize, alpha) {
    var inputSize = input.length / rows;
    var outputSize = output.length / rows;

    var x = ndarray(input, [rows, inputSize]);
    var y = ndarray(output, [rows, outputSize]);

    var syn0 = randSyn(matrix2d(inputSize, hiddenSize));
    var syn0d = matrix2d(inputSize, hiddenSize);

    var syn1 = randSyn(matrix2d(hiddenSize, outputSize));
    var syn1d = matrix2d(hiddenSize, outputSize);
    var syn1t = syn1.transpose(1, 0);

    var l0 = {};
    l0.activation = x;
    l0.transposed = x.transpose(1, 0);

    var l1 = createLayer(rows, hiddenSize);
    var l2 = createLayer(rows, outputSize);

    for (var i = 0; i <= 60000; i++) {
        feedforward(l1, l0, syn0);
        feedforward(l2, l1, syn1);

        sub(l2.error, l2.activation, y);
        updateDelta(l2);

        dotProduct(l1.error, l2.delta, syn1t);
        updateDelta(l1);

        updateWeights(syn1, syn1d, l1, l2, alpha);
        updateWeights(syn0, syn0d, l0, l1, alpha);

        if (i % 5000 === 0) console.log('err ' + i + ': ' + mean(l2.error ));
    }
}

function updateWeights(synapse, deltas, layer1, layer2, alpha) {
    dotProduct(deltas, layer1.transposed, layer2.delta);
    for (var i = 0; i < deltas.data.length; i++) synapse.data[i] -= alpha * deltas.data[i];
}

function updateDelta(a) {
    for (var i = 0, values = a.activation.data; i < values.length; i++) {
        a.delta.data[i] = values[i] * (1 - values[i]) * a.error.data[i];
    }
}

function feedforward(output, input, synapse) {
    dotProduct(output.activation, input.activation, synapse);
    for (var i = 0, data = output.activation.data; i < data.length; i++) data[i] = 1 / (1 + Math.exp(-data[i]));
}

function createLayer(rows, cols) {
    var obj = {};
    var shape = [rows, cols];
    obj.activation = ndarray(new Float32Array(rows * cols), shape);
    obj.error = ndarray(new Float32Array(rows * cols), shape);
    obj.delta = ndarray(new Float32Array(rows * cols), shape);
    obj.transposed = obj.activation.transpose(1, 0);
    return obj;
}

function sub(out, a, b) {
    for (var i = 0; i < a.data.length; i++) out.data[i] = a.data[i] - b.data[i];
}

function matrix2d(rows, cols, data) {
    return ndarray(new Float32Array(data || (rows * cols)), [rows, cols]);
}

function dotProduct(out, a, b) {
    for (var i = 0; i < a.shape[0]; i++) {
        for (var j = 0; j < b.shape[1]; j++) {
            var sum = 0;
            for (var k = 0; k < a.shape[1]; k++) sum += a.get(i, k) * b.get(k, j);
            out.set(i, j, sum);
        }
    }
    return out;
}

function randSyn(a) {
    for (var i = 0; i < a.data.length; i++) a.data[i] = Math.random() * 2 - 1;
    return a;
}

function mean(a) {
    for (var i = 0, sum = 0; i < a.data.length; i++) sum += Math.pow(a.data[i], 2);
    return sum / a.data.length;
}
