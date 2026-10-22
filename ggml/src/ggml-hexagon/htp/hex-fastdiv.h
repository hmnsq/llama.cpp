#ifndef HEX_FASTDIV_H
#define HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
સંરચના fastdiv_કિંમતો {
    uint32_t mp;
    uint32_t l;
};

સ્થિર અંતઃરેખ સંરચના fastdiv_કિંમતો init_fastdiv_કિંમતો(uint32_t d) {
    સંરચના fastdiv_કિંમતો result = { 0, 0 };
    // compute L = ceil(log2(d));
    જ્યારે (result.l < 32 && ((uint32_t) 1 << result.l) < d) {
        ++(result.l);
    }

    result.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << result.l) - d) / d + 1);
    પાછા result;
}

સ્થિર અંતઃરેખ uint32_t fastdiv(uint32_t n, અચલ સંરચના fastdiv_કિંમતો * vals) {
    // Compute high 32 bits of n * mp
    અચલ uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    પાછા (hi + n) >> vals->l;
}

સ્થિર અંતઃરેખ uint32_t fastmodulo(uint32_t n, uint32_t d, અચલ સંરચના fastdiv_કિંમતો * vals) {
    પાછા n - fastdiv(n, vals) * d;
}

#endif /* HEX_FASTDIV_H */
