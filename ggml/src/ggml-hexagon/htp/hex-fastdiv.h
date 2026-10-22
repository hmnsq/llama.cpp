#ifndef HEX_FASTDIV_H
#define HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
संरचना fastdiv_मूल्ये {
    uint32_t mp;
    uint32_t l;
};

स्थिर ओळीत संरचना fastdiv_मूल्ये init_fastdiv_मूल्ये(uint32_t d) {
    संरचना fastdiv_मूल्ये result = { 0, 0 };
    // compute L = ceil(log2(d));
    पर्यंत (result.l < 32 && ((uint32_t) 1 << result.l) < d) {
        ++(result.l);
    }

    result.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << result.l) - d) / d + 1);
    परत result;
}

स्थिर ओळीत uint32_t fastdiv(uint32_t n, स्थिरांक संरचना fastdiv_मूल्ये * vals) {
    // Compute high 32 bits of n * mp
    स्थिरांक uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    परत (hi + n) >> vals->l;
}

स्थिर ओळीत uint32_t fastmodulo(uint32_t n, uint32_t d, स्थिरांक संरचना fastdiv_मूल्ये * vals) {
    परत n - fastdiv(n, vals) * d;
}

#endif /* HEX_FASTDIV_H */
