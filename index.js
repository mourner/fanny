'use strict';

const ndarray = require('ndarray');
const ops = require('ndarray-ops');

const matrix = (rows, cols, data) => ndarray(new Float32Array(data || (rows * cols)), [rows, cols]);

const sigmoid = (out, a) => {
    for (let i = 0; i < a.data.length; i++) out.data[i] = 1 / (1 + Math.exp(-a.data[i]));
};

const sigmoidDeriv = (out, a) => {
    for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] * (1 - a.data[i]);
};

const dotProduct = (out, a, b) => {
    for (let i = 0; i < a.shape[0]; i++) {
        for (let j = 0, sum = 0; j < b.shape[1]; j++) {
            for (let k = 0; k < a.shape[1]; k++) sum += a.get(i, k) * b.get(k, j);
            out.set(i, j, sum);
        }
    }
}

const x = matrix(4, 3, [
    0, 0, 1,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1
]);

const y = matrix(4, 1, [
    0,
    0,
    1,
    1
]);

const l0 = matrix(x.shape[0], x.shape[1]);
ops.assign(l0, x);
const l0t = l0.transpose(1, 0);
const l1 = matrix(x.shape[0], y.shape[1]);
const l1Err = matrix(x.shape[0], y.shape[1]);
const l1Delta = matrix(x.shape[0], y.shape[1]);

const syn0 = matrix(x.shape[1], y.shape[1]);
ops.random(syn0);
ops.mulseq(syn0, 2);
ops.subseq(syn0, 1);

const syn0Delta = matrix(x.shape[1], y.shape[1]);

for (let i = 0; i < 10000; i++) {
    dotProduct(l1, l0, syn0);
    sigmoid(l1, l1);

    // calculate error
    ops.sub(l1Err, y, l1);

    // calculate delta
    sigmoidDeriv(l1Delta, l1);
    ops.muleq(l1Delta, l1Err);

    // update weights
    dotProduct(syn0Delta, l0t, l1Delta);
    ops.addeq(syn0, syn0Delta);
}

console.log(l1);
