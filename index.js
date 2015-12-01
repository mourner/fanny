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
        l1.values.product(l0.values, syn0.values).sigmoideq();
        l2.values.product(l1.values, syn1.values).sigmoideq();

        l2.error.sub(l2.values, y);
        l2.delta.dsigmoid(l2.values).muleq(l2.error);

        l1.error.product(l2.delta, syn1.t);
        l1.delta.dsigmoid(l1.values).muleq(l1.error);

        syn1.values.subeq(syn1.delta.product(l1.t, l2.delta).mulseq(alpha));
        syn0.values.subeq(syn0.delta.product(l0.t, l1.delta).mulseq(alpha));

        if ((i + 1) % 5000 === 0) {
            console.log('error %d after %d iterations', l2.error.mean().toFixed(10), i + 1);
        }
    }
}

function createLayer(rows, cols) {
    var obj = {};
    obj.values = ndarray2d(new Float32Array(rows * cols), rows);
    obj.error = ndarray2d(new Float32Array(rows * cols), rows);
    obj.delta = ndarray2d(new Float32Array(rows * cols), rows);
    obj.t = obj.values.transpose();
    return obj;
}

function createInputLayer(data, rows, cols, inputSize) {
    var obj = {};
    obj.values = ndarray2d(data, rows, inputSize, cols);
    obj.t = obj.values.transpose();
    return obj;
}

function createSynapse(rows, cols) {
    var obj = {};
    obj.values = ndarray2d(new Float32Array(rows * cols), rows).random(-1, 1);
    obj.delta = ndarray2d(new Float32Array(rows * cols), rows);
    obj.t = obj.values.transpose();
    return obj;
}
