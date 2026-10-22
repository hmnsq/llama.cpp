#ifndef HEX_FASTDIV_H
#define HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ {
    uint32_t mp;
    uint32_t l;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(uint32_t d) {
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ result = { 0, 0 };
    // compute L = ceil(log2(d));
    ଯେତେବେଳେ (result.l < 32 && ((uint32_t) 1 << result.l) < d) {
        ++(result.l);
    }

    result.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << result.l) - d) / d + 1);
    ଫେରନ୍ତୁ result;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t fastdiv(uint32_t n, ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * vals) {
    // Compute high 32 bits of n * mp
    ଅଚଳ uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    ଫେରନ୍ତୁ (hi + n) >> vals->l;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t fastmodulo(uint32_t n, uint32_t d, ଅଚଳ ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ * vals) {
    ଫେରନ୍ତୁ n - fastdiv(n, vals) * d;
}

#endif /* HEX_FASTDIV_H */
