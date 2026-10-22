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

स्थिर अन्तःस्थ संरचना fastdiv_मूल्ये init_fastdiv_मूल्ये(uint32_t d) {
    संरचना fastdiv_मूल्ये result = { 0, 0 };
    // compute L = ceil(log2(d));
    यावत् (result.l < 32 && ((uint32_t) 1 << result.l) < d) {
        ++(result.l);
    }

    result.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << result.l) - d) / d + 1);
    प्रत्यागम result;
}

स्थिर अन्तःस्थ uint32_t fastdiv(uint32_t n, ध्रुव संरचना fastdiv_मूल्ये * vals) {
    // Compute high 32 bits of n * mp
    ध्रुव uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    प्रत्यागम (hi + n) >> vals->l;
}

स्थिर अन्तःस्थ uint32_t fastmodulo(uint32_t n, uint32_t d, ध्रुव संरचना fastdiv_मूल्ये * vals) {
    प्रत्यागम n - fastdiv(n, vals) * d;
}

#endif /* HEX_FASTDIV_H */
