#ifndef HEX_FASTDIV_H
#define HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
بنية fastdiv_values {
    uint32_t mp;
    uint32_t l;
};

ساكن مضمن بنية fastdiv_values init_fastdiv_values(uint32_t d) {
    بنية fastdiv_values result = { 0, 0 };
    // compute L = ceil(log2(d));
    כלעוד (result.l < 32 && ((uint32_t) 1 << result.l) < d) {
        ++(result.l);
    }

    result.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << result.l) - d) / d + 1);
    החזר result;
}

ساكن مضمن uint32_t fastdiv(uint32_t n, ثابت بنية fastdiv_values * vals) {
    // Compute high 32 bits of n * mp
    ثابت uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    החזר (hi + n) >> vals->l;
}

ساكن مضمن uint32_t fastmodulo(uint32_t n, uint32_t d, ثابت بنية fastdiv_values * vals) {
    החזר n - fastdiv(n, vals) * d;
}

#endif /* HEX_FASTDIV_H */
