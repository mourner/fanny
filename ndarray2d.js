'use strict';

module.exports = ndarray2d;

function ndarray2d(data, rows, cols, stride0, stride1, offset) {
    return new NDArray2D(data, rows, cols, stride0, stride1, offset);
}

function NDArray2D(data, rows, cols, stride0, stride1, offset) {
    this.data = data;

    this.rows = rows;
    this.cols = cols || (data.length / rows);

    this.stride0 = stride0 || this.cols;
    this.stride1 = stride1 || 1;

    this.offset = offset || 0;
}

NDArray2D.prototype = {
    get: function (i, j) {
        return this.data[this.offset + this.stride0 * i + this.stride1 * j];
    },

    set: function (i, j, v) {
        this.data[this.offset + this.stride0 * i + this.stride1 * j] = v;
    },

    transpose: function () {
        return new NDArray2D(this.data, this.cols, this.rows, this.stride1, this.stride0, this.offset);
    },

    product: function (a, b) {
        for (var i = 0; i < a.rows; i++) {
            for (var j = 0; j < b.cols; j++) {
                var sum = 0;
                for (var k = 0; k < a.cols; k++) {
                    sum += a.get(i, k) * b.get(k, j);
                }
                this.set(i, j, sum);
            }
        }
        return this;
    },

    sub: function (a, b) {
        for (var i = 0; i < this.data.length; i++) {
            this.data[i] = a.data[i] - b.data[i];
        }
        return this;
    },

    subeq: function (b) {
        return this.sub(this, b);
    },

    sigmoideq: function () {
        for (var i = 0; i < this.data.length; i++) {
            this.data[i] = 1 / (1 + Math.exp(-this.data[i]));
        }
        return this;
    },

    dsigmoid: function (sigm) {
        for (var i = 0; i < this.data.length; i++) {
            this.data[i] = sigm.data[i] * (1 - sigm.data[i]);
        }
        return this;
    },

    muleq: function (a) {
        for (var i = 0; i < this.data.length; i++) {
            this.data[i] *= a.data[i];
        }
        return this;
    },

    mulseq: function (s) {
        for (var i = 0; i < this.data.length; i++) {
            this.data[i] *= s;
        }
        return this;
    },

    random: function (min, max) {
        min = min || 0;
        max = max || 1;
        for (var i = 0; i < this.data.length; i++) {
            this.data[i] = min + (max - min) * Math.random();
        }
        return this;
    },

    mean: function () {
        for (var i = 0, sum = 0; i < this.data.length; i++) {
            sum += Math.pow(this.data[i], 2);
        }
        return sum / this.data.length;
    },

    toString: function () {
        var str = '';
        for (var i = 0; i < this.rows; i++) {
            for (var j = 0; j < this.cols; j++) {
                str += this.get(i, j) + ' ';
            }
            str += '\n';
        }
        return str;
    }
};
