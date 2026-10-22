#ifndef HEX_FASTDIV_H
#define HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
கட்டமைப்பு fastdiv_மதிப்புகள் {
    uint32_t mp;
    uint32_t l;
};

நிலையான உள்வரி கட்டமைப்பு fastdiv_மதிப்புகள் init_fastdiv_மதிப்புகள்(uint32_t d) {
    கட்டமைப்பு fastdiv_மதிப்புகள் result = { 0, 0 };
    // compute L = ceil(log2(d));
    போது (result.l < 32 && ((uint32_t) 1 << result.l) < d) {
        ++(result.l);
    }

    result.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << result.l) - d) / d + 1);
    திரும்பு result;
}

நிலையான உள்வரி uint32_t fastdiv(uint32_t n, மாறிலி கட்டமைப்பு fastdiv_மதிப்புகள் * vals) {
    // Compute high 32 bits of n * mp
    மாறிலி uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    திரும்பு (hi + n) >> vals->l;
}

நிலையான உள்வரி uint32_t fastmodulo(uint32_t n, uint32_t d, மாறிலி கட்டமைப்பு fastdiv_மதிப்புகள் * vals) {
    திரும்பு n - fastdiv(n, vals) * d;
}

#endif /* HEX_FASTDIV_H */
