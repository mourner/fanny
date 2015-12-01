'use strict';

var network = require('./');

var input = [
    0, 0,
    0, 1,
    1, 0,
    1, 1,
];

var output = [0, 1, 1, 0];

console.time('training');

network(input, output, 4, 4, 1);

console.timeEnd('training');
