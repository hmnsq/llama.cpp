#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HEX_FASTDIV_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ {
    uint32_t mp;
    uint32_t l;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ init_fastdiv_ਮੁੱਲਾਂ(uint32_t d) {
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ ਨਤੀਜਾ = { 0, 0 };
    // compute L = ceil(log2(d));
    ਜਦੋਂ (ਨਤੀਜਾ.l < 32 && ((uint32_t) 1 << ਨਤੀਜਾ.l) < d) {
        ++(ਨਤੀਜਾ.l);
    }

    ਨਤੀਜਾ.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << ਨਤੀਜਾ.l) - d) / d + 1);
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t fastdiv(uint32_t n, ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * vals) {
    // Compute high 32 bits of n * mp
    ਅਚਲ uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    ਵਾਪਸ (hi + n) >> vals->l;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t fastmodulo(uint32_t n, uint32_t d, ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * vals) {
    ਵਾਪਸ n - fastdiv(n, vals) * d;
}

#ਜੇ_ਅੰਤ /* HEX_FASTDIV_H */
