'use strict';

var network = require('./');

var data = [
    0, 0, 1, 0,
    0, 1, 1, 1,
    1, 0, 1, 1,
    1, 1, 1, 0
];

console.time('training');

network(data, 3, 3, 1, 10);

console.timeEnd('training');
