#اگر_غیر_تعریف_شدہ HEX_FASTDIV_H
#تعریف_کرو HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
ساخت fastdiv_اقدار {
    uint32_t mp;
    uint32_t l;
};

ساکن اندرونی ساخت fastdiv_اقدار init_fastdiv_اقدار(uint32_t d) {
    ساخت fastdiv_اقدار نتیجہ = { 0, 0 };
    // compute L = ceil(log2(d));
    جب (نتیجہ.l < 32 && ((uint32_t) 1 << نتیجہ.l) < d) {
        ++(نتیجہ.l);
    }

    نتیجہ.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << نتیجہ.l) - d) / d + 1);
    واپس نتیجہ;
}

ساکن اندرونی uint32_t fastdiv(uint32_t n, ثابت ساخت fastdiv_اقدار * vals) {
    // Compute high 32 bits of n * mp
    ثابت uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    واپس (hi + n) >> vals->l;
}

ساکن اندرونی uint32_t fastmodulo(uint32_t n, uint32_t d, ثابت ساخت fastdiv_اقدار * vals) {
    واپس n - fastdiv(n, vals) * d;
}

#اگر_ختم /* HEX_FASTDIV_H */
