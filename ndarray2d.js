'use strict';

module.exports = ndarray2d;

function ndarray2d(data, rows, cols) {
    return new NDArray2D(data, rows, cols);
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
    }
};
