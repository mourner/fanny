'use strict';

var network = require('./');

var data = new Float32Array([
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
]);

console.time('training');

network(data, 2, 4, 1);

console.timeEnd('training');
