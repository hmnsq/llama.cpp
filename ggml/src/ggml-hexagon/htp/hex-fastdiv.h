#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HEX_FASTDIV_H
#ವ್ಯಾಖ್ಯಾನಿಸು HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು {
    uint32_t mp;
    uint32_t l;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು init_fastdiv_ಮೌಲ್ಯಗಳು(uint32_t d) {
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು ಫಲಿತಾಂಶ = { 0, 0 };
    // compute L = ceil(log2(d));
    ಆಗ (ಫಲಿತಾಂಶ.l < 32 && ((uint32_t) 1 << ಫಲಿತಾಂಶ.l) < d) {
        ++(ಫಲಿತಾಂಶ.l);
    }

    ಫಲಿತಾಂಶ.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << ಫಲಿತಾಂಶ.l) - d) / d + 1);
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t fastdiv(uint32_t n, ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * vals) {
    // Compute high 32 bits of n * mp
    ಸ್ಥಿರಾಂಕ uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    ಹಿಂತಿರುಗಿ (hi + n) >> vals->l;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t fastmodulo(uint32_t n, uint32_t d, ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * vals) {
    ಹಿಂತಿರುಗಿ n - fastdiv(n, vals) * d;
}

#ಆದರೆ_ಅಂತ್ಯ /* HEX_FASTDIV_H */
