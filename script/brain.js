"use strict";

// Glorot Normal (Xavier) weight initialization
function glorotNormal(fanIn, fanOut) {
    const stddev = Math.sqrt(2 / (fanIn + fanOut));
    const w = new Float32Array(fanIn * fanOut);
    for (let i = 0; i < w.length; i++) {
        const u1 = Math.random() || Number.EPSILON;
        const u2 = Math.random() || Number.EPSILON;
        w[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * stddev;
    }
    return w;
}

function leakyRelu(x) {
    return x >= 0 ? x : 0.3 * x;
}

// Minimal feedforward neural network: 5 → 5 (linear) → 4 (leakyReLU) → 3 (leakyReLU) → 2 (linear)
// Weights stored as plain Float32Arrays — no framework overhead, no GPU roundtrips.
class Brain {
    constructor() {
        this.layers = [
            { w: glorotNormal(5, 5), b: new Float32Array(5), inSize: 5, outSize: 5, act: null },
            { w: glorotNormal(5, 4), b: new Float32Array(4), inSize: 5, outSize: 4, act: leakyRelu },
            { w: glorotNormal(4, 3), b: new Float32Array(3), inSize: 4, outSize: 3, act: leakyRelu },
            { w: glorotNormal(3, 2), b: new Float32Array(2), inSize: 3, outSize: 2, act: null },
        ];
    }

    forward(input, useTanh = false) {
        let x = input;
        for (const { w, b, inSize, outSize, act } of this.layers) {
            const y = new Float32Array(outSize);
            for (let i = 0; i < outSize; i++) {
                let sum = b[i];
                for (let j = 0; j < inSize; j++) sum += w[i * inSize + j] * x[j];
                y[i] = act ? act(sum) : sum;
            }
            x = y;
        }
        if (useTanh) {
            for (let i = 0; i < x.length; i++) x[i] = Math.tanh(x[i]);
        }
        return x;
    }

    clone() {
        const copy = new Brain();
        for (let i = 0; i < this.layers.length; i++) {
            copy.layers[i].w.set(this.layers[i].w);
            copy.layers[i].b.set(this.layers[i].b);
        }
        return copy;
    }

    crossover(other) {
        for (let i = 0; i < this.layers.length; i++) {
            const w = this.layers[i].w;
            const ow = other.layers[i].w;
            const mid = Math.floor(w.length / 2);
            for (let j = mid; j < w.length; j++) w[j] = ow[j];

            const b = this.layers[i].b;
            const ob = other.layers[i].b;
            const bmid = Math.floor(b.length / 2);
            for (let j = bmid; j < b.length; j++) b[j] = ob[j];
        }
    }

    mutate(mutationProbability, mutationChange, biasProbability, biasChange) {
        for (const { w, b } of this.layers) {
            for (let j = 0; j < w.length; j++) {
                if (Math.random() < mutationProbability)
                    w[j] += Math.random() * mutationChange - mutationChange / 2;
            }
            for (let j = 0; j < b.length; j++) {
                if (Math.random() < biasProbability)
                    b[j] += Math.random() * biasChange - biasChange / 2;
            }
        }
        return this;
    }
}
