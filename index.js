'use strict';

var ndarray = require('ndarray');
var ops = require('ndarray-ops');

module.exports = network;

function network(data, input, hidden, output) {
    var size = input + output;
    var rows = data.length / size;

    var dataMat = ndarray(data, [rows, size]);

    var x = dataMat.hi(null, input);
    var y = dataMat.lo(null, input);

    var syn0 = randSyn(matrix2d(input, hidden));
    var syn0d = matrix2d(input, hidden);

    var syn1 = randSyn(matrix2d(hidden, output));
    var syn1d = matrix2d(hidden, output);
    var syn1t = syn1.transpose(1, 0);

    var xt = x.transpose(1, 0);

    var l1 = matrix2d(rows, hidden);
    var l1e = matrix2d(rows, hidden);
    var l1d = matrix2d(rows, hidden);
    var l1t = l1.transpose(1, 0);

    var l2 = matrix2d(rows, output);
    var l2e = matrix2d(rows, output);
    var l2d = matrix2d(rows, output);

    for (var i = 0; i < 60000; i++) {
        sigmoid(l1, dotProduct(l1, x, syn0));
        sigmoid(l2, dotProduct(l2, l1, syn1));

        ops.muleq(sigmoidDeriv(l2d, l2), ops.sub(l2e, y, l2));
        if (i % 5000 === 0) console.log(mean(l2e));

        ops.muleq(sigmoidDeriv(l1d, l1), dotProduct(l1e, l2d, syn1t));

        ops.addeq(syn1, dotProduct(syn1d, l1t, l2d));
        ops.addeq(syn0, dotProduct(syn0d, xt, l1d));
    }

    logND(l2);
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
    return ops.subseq(ops.mulseq(ops.random(a), 2), 1);
}

function mean(a) {
    var sum = 0;
    for (var i = 0; i < a.data.length; i++) {
        sum += Math.pow(a.data[i], 2);
    }
    return sum / a.data.length;
}

function logND(a) {
    for (var i = 0; i < a.shape[0]; i++) {
        var str = '';
        for (var j = 0; j < a.shape[1]; j++) {
            str += a.get(i, j) + ' ';
        }
        console.log(str);
    }
}
