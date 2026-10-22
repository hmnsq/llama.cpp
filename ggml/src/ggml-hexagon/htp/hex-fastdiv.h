#अगर_अघोषित HEX_FASTDIV_H
#घोषणा HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
काष्ठा fastdiv_मूल्यावली {
    uint32_t mp;
    uint32_t l;
};

अटल अंतरभूत काष्ठा fastdiv_मूल्यावली init_fastdiv_मूल्यावली(uint32_t d) {
    काष्ठा fastdiv_मूल्यावली परिणाम = { 0, 0 };
    // compute L = ceil(log2(d));
    जबतक (परिणाम.l < 32 && ((uint32_t) 1 << परिणाम.l) < d) {
        ++(परिणाम.l);
    }

    परिणाम.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << परिणाम.l) - d) / d + 1);
    वापस परिणाम;
}

अटल अंतरभूत uint32_t fastdiv(uint32_t n, स्थिर काष्ठा fastdiv_मूल्यावली * vals) {
    // Compute high 32 bits of n * mp
    स्थिर uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    वापस (hi + n) >> vals->l;
}

अटल अंतरभूत uint32_t fastmodulo(uint32_t n, uint32_t d, स्थिर काष्ठा fastdiv_मूल्यावली * vals) {
    वापस n - fastdiv(n, vals) * d;
}

#पूर्ण_अगर /* HEX_FASTDIV_H */
