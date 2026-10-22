#ifndef HEX_FASTDIV_H
#define HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
ഘടന fastdiv_മൂല്യങ്ങൾ {
    uint32_t mp;
    uint32_t l;
};

സ്ഥിതം വരിയിൽ ഘടന fastdiv_മൂല്യങ്ങൾ init_fastdiv_മൂല്യങ്ങൾ(uint32_t d) {
    ഘടന fastdiv_മൂല്യങ്ങൾ result = { 0, 0 };
    // compute L = ceil(log2(d));
    ആകുമ്പോൾ (result.l < 32 && ((uint32_t) 1 << result.l) < d) {
        ++(result.l);
    }

    result.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << result.l) - d) / d + 1);
    തിരികെ result;
}

സ്ഥിതം വരിയിൽ uint32_t fastdiv(uint32_t n, സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * vals) {
    // Compute high 32 bits of n * mp
    സ്ഥിരം uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    തിരികെ (hi + n) >> vals->l;
}

സ്ഥിതം വരിയിൽ uint32_t fastmodulo(uint32_t n, uint32_t d, സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * vals) {
    തിരികെ n - fastdiv(n, vals) * d;
}

#endif /* HEX_FASTDIV_H */
