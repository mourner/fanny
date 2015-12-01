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

    var l0 = x;
    var l0t = l0.transpose(1, 0);

    var l1 = matrix2d(rows, hiddenSize);
    var l1e = matrix2d(rows, hiddenSize);
    var l1d = matrix2d(rows, hiddenSize);
    var l1t = l1.transpose(1, 0);

    var l2 = matrix2d(rows, outputSize);
    var l2e = matrix2d(rows, outputSize);
    var l2d = matrix2d(rows, outputSize);

    for (var i = 0; i <= 60000; i++) {
        feedforward(l1, l0, syn0);
        feedforward(l2, l1, syn1);

        backpropagate(l2d, l2, sub(l2e, l2, y));
        backpropagate(l1d, l1, dotProduct(l1e, l2d, syn1t));

        updateWeights(syn1, dotProduct(syn1d, l1t, l2d), alpha);
        updateWeights(syn0, dotProduct(syn0d, l0t, l1d), alpha);

        if (i % 5000 === 0) console.log('err ' + i + ': ' + mean(l2e));
    }
}

function sub(out, a, b) {
    for (var i = 0; i < a.data.length; i++) out.data[i] = a.data[i] - b.data[i];
    return out;
}

function updateWeights(synapse, deltas, alpha) {
    for (var i = 0; i < deltas.data.length; i++) synapse.data[i] -= alpha * deltas.data[i];
}

function backpropagate(out, layer, error) {
    for (var i = 0; i < layer.data.length; i++) {
        var v = layer.data[i];
        out.data[i] = v * (1 - v) * error.data[i];
    }
    return out;
}

function feedforward(out, layer, synapse) {
    return sigmoid(out, dotProduct(out, layer, synapse));
}

function matrix2d(rows, cols, data) {
    return ndarray(new Float32Array(data || (rows * cols)), [rows, cols]);
}

function sigmoid(out, a) {
    for (var i = 0; i < a.data.length; i++) out.data[i] = 1 / (1 + Math.exp(-a.data[i]));
    return out;
};

function sigmoidDeriv(out, a) {
    for (var i = 0; i < a.data.length; i++) out.data[i] = a.data[i] * (1 - a.data[i]);
    return out;
};

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
