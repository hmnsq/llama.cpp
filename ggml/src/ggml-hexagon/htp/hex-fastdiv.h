#اگر_تعریف_نشده HEX_FASTDIV_H
#تعریف_کن HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
بنية fastdiv_مقادیر {
    uint32_t mp;
    uint32_t l;
};

ساكن مضمن بنية fastdiv_مقادیر init_fastdiv_مقادیر(uint32_t d) {
    بنية fastdiv_مقادیر نتیجه = { 0, 0 };
    // compute L = ceil(log2(d));
    طالما (نتیجه.l < 32 && ((uint32_t) 1 << نتیجه.l) < d) {
        ++(نتیجه.l);
    }

    نتیجه.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << نتیجه.l) - d) / d + 1);
    أرجع نتیجه;
}

ساكن مضمن uint32_t fastdiv(uint32_t n, ثابت بنية fastdiv_مقادیر * vals) {
    // Compute high 32 bits of n * mp
    ثابت uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    أرجع (hi + n) >> vals->l;
}

ساكن مضمن uint32_t fastmodulo(uint32_t n, uint32_t d, ثابت بنية fastdiv_مقادیر * vals) {
    أرجع n - fastdiv(n, vals) * d;
}

#پایان_اگر /* HEX_FASTDIV_H */
