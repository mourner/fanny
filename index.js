'use strict';

const ndarray = require('ndarray');
const ops = require('ndarray-ops');

const matrix = (rows, cols, data) => ndarray(new Float32Array(data || (rows * cols)), [rows, cols]);

const sigmoid = (out, a) => {
    for (let i = 0; i < a.data.length; i++) out.data[i] = 1 / (1 + Math.exp(-a.data[i]));
    return out;
};

const sigmoidDeriv = (out, a) => {
    for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] * (1 - a.data[i]);
    return out;
};

const dotProduct = (out, a, b) => {
    for (var i = 0; i < a.shape[0]; i++) {
        for (var j = 0; j < b.shape[1]; j++) {
            var sum = 0;
            for (var k = 0; k < a.shape[1]; k++) sum += a.get(i, k) * b.get(k, j);
            out.set(i, j, sum);
        }
    }
    return out;
}

const randSyn = (a) => ops.subseq(ops.mulseq(ops.random(a), 2), 1);

const mean = (a) => {
    let sum = 0;
    for (let k of a.data) {
        sum += k * k;
    }
    return sum / a.data.length;
}

const x = matrix(4, 3, [
    0, 0, 1,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1
]);

const y = matrix(4, 1, [
    0,
    1,
    1,
    0
]);

const rows = x.shape[0];
const cols1 = x.shape[1];
const cols2 = y.shape[1];

const syn0 = randSyn(matrix(cols1, rows));
const syn0d = matrix(cols1, rows);

const syn1 = randSyn(matrix(rows, cols2));
const syn1d = matrix(rows, cols2);
const syn1t = syn1.transpose(1, 0);

const xt = x.transpose(1, 0);

const l1 = matrix(rows, rows);
const l1e = matrix(rows, rows);
const l1d = matrix(rows, rows);
const l1t = l1.transpose(1, 0);

const l2 = matrix(rows, cols2);
const l2e = matrix(rows, cols2);
const l2d = matrix(rows, cols2);

console.time('training');
for (let i = 0; i < 60000; i++) {
    sigmoid(l1, dotProduct(l1, x, syn0));
    sigmoid(l2, dotProduct(l2, l1, syn1));

    ops.muleq(sigmoidDeriv(l2d, l2), ops.sub(l2e, y, l2));
    if (i % 5000 === 0) console.log(mean(l2e));

    ops.muleq(sigmoidDeriv(l1d, l1), dotProduct(l1e, l2d, syn1t));

    ops.addeq(syn1, dotProduct(syn1d, l1t, l2d));
    ops.addeq(syn0, dotProduct(syn0d, xt, l1d));
}
console.timeEnd('training');

console.log(l2.data);
