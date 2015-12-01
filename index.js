'use strict';

var ndarray = require('ndarray');

module.exports = network;

function network(data, inputSize, hiddenSize, outputSize, alpha) {
    var cols = inputSize + outputSize;
    var rows = data.length / cols;

    var syn0 = createSynapse(inputSize, hiddenSize);
    var syn1 = createSynapse(hiddenSize, outputSize);

    var l0 = createInputLayer(data, rows, cols, inputSize);
    var l1 = createLayer(rows, hiddenSize);
    var l2 = createLayer(rows, outputSize);
    var y = ndarray(data, [rows, outputSize], [cols, 1], inputSize);

    for (var i = 0; i <= 60000; i++) {
        feedforward(l1, l0, syn0);
        feedforward(l2, l1, syn1);

        sub(l2.error, l2.values, y);
        updateDelta(l2);

        dotProduct(l1.error, l2.delta, syn1.transposed);
        updateDelta(l1);

        updateWeights(syn1, l1, l2, alpha);
        updateWeights(syn0, l0, l1, alpha);

        if (i % 5000 === 0) console.log('err ' + i + ': ' + mean(l2.error));
    }
}

function updateWeights(synapse, layer1, layer2, alpha) {
    dotProduct(synapse.delta, layer1.transposed, layer2.delta);

    for (var i = 0; i < synapse.delta.data.length; i++) {
        synapse.values.data[i] -= (alpha || 1) * synapse.delta.data[i];
    }
}

function updateDelta(a) {
    for (var i = 0, values = a.values.data; i < values.length; i++) {
        a.delta.data[i] = values[i] * (1 - values[i]) * a.error.data[i];
    }
}

function feedforward(output, input, synapse) {
    dotProduct(output.values, input.values, synapse.values);

    for (var i = 0, data = output.values.data; i < data.length; i++) {
        data[i] = 1 / (1 + Math.exp(-data[i]));
    }
}

function createLayer(rows, cols) {
    var shape = [rows, cols];

    var obj = {};
    obj.values = ndarray(new Float32Array(rows * cols), shape);
    obj.error = ndarray(new Float32Array(rows * cols), shape);
    obj.delta = ndarray(new Float32Array(rows * cols), shape);
    obj.transposed = obj.values.transpose(1, 0);

    return obj;
}

function createInputLayer(data, rows, cols, inputSize) {
    var obj = {};
    obj.values = ndarray(data, [rows, inputSize], [cols, 1]);
    obj.transposed = obj.values.transpose(1, 0);
    return obj;
}

function createSynapse(rows, cols) {
    var shape = [rows, cols];

    var obj = {};
    obj.values = ndarray(new Float32Array(rows * cols), shape);
    obj.delta = ndarray(new Float32Array(rows * cols), shape);
    obj.transposed = obj.values.transpose(1, 0);

    for (var i = 0; i < obj.values.data.length; i++) {
        obj.values.data[i] = Math.random() * 2 - 1;
    }
    return obj;
}

function sub(out, a, b) {
    for (var i = 0; i < a.data.length; i++) {
        out.data[i] = a.data[i] - b.data[i];
    }
}

function dotProduct(out, a, b) {
    for (var i = 0; i < a.shape[0]; i++) {
        for (var j = 0; j < b.shape[1]; j++) {
            var sum = 0;
            for (var k = 0; k < a.shape[1]; k++) {
                sum += a.get(i, k) * b.get(k, j);
            }
            out.set(i, j, sum);
        }
    }
    return out;
}

function mean(a) {
    for (var i = 0, sum = 0; i < a.data.length; i++) {
        sum += Math.pow(a.data[i], 2);
    }
    return sum / a.data.length;
}
