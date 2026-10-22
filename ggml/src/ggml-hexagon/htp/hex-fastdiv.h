#ifndef HEX_FASTDIV_H
#define HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
sa_nracanaa fastdiv_values {
    uint32_t mp;
    uint32_t l;
};

static anta_hstha sa_nracanaa fastdiv_values init_fastdiv_values(uint32_t d) {
    sa_nracanaa fastdiv_values result = { 0, 0 };
    // compute L = ceil(log2(d));
    ขณะ (result.l < 32 && ((uint32_t) 1 << result.l) < d) {
        ++(result.l);
    }

    result.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << result.l) - d) / d + 1);
    คืน result;
}

static anta_hstha uint32_t fastdiv(uint32_t n, const sa_nracanaa fastdiv_values * vals) {
    // Compute high 32 bits of n * mp
    const uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    คืน (hi + n) >> vals->l;
}

static anta_hstha uint32_t fastmodulo(uint32_t n, uint32_t d, const sa_nracanaa fastdiv_values * vals) {
    คืน n - fastdiv(n, vals) * d;
}

#endif /* HEX_FASTDIV_H */
