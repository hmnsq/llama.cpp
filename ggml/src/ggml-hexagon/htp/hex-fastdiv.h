#ifndef HEX_FASTDIV_H
#define HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
সংরচনা fastdiv_মানসমূহ {
    uint32_t mp;
    uint32_t l;
};

স্থির অন্তঃস্থ সংরচনা fastdiv_মানসমূহ init_fastdiv_মানসমূহ(uint32_t d) {
    সংরচনা fastdiv_মানসমূহ result = { 0, 0 };
    // compute L = ceil(log2(d));
    যাবত্ (result.l < 32 && ((uint32_t) 1 << result.l) < d) {
        ++(result.l);
    }

    result.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << result.l) - d) / d + 1);
    প্রত্যাগম result;
}

স্থির অন্তঃস্থ uint32_t fastdiv(uint32_t n, ধ্রুব সংরচনা fastdiv_মানসমূহ * vals) {
    // Compute high 32 bits of n * mp
    ধ্রুব uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    প্রত্যাগম (hi + n) >> vals->l;
}

স্থির অন্তঃস্থ uint32_t fastmodulo(uint32_t n, uint32_t d, ধ্রুব সংরচনা fastdiv_মানসমূহ * vals) {
    প্রত্যাগম n - fastdiv(n, vals) * d;
}

#endif /* HEX_FASTDIV_H */
