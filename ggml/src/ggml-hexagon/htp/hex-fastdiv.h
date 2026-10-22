#నిర్వచితం_కాకపోతే HEX_FASTDIV_H
#నిర్వచించు HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
నిర్మాణం fastdiv_విలువలు {
    uint32_t mp;
    uint32_t l;
};

స్థిర అంతఃవరుస నిర్మాణం fastdiv_విలువలు init_fastdiv_విలువలు(uint32_t d) {
    నిర్మాణం fastdiv_విలువలు ఫలితం = { 0, 0 };
    // compute L = ceil(log2(d));
    అయితే (ఫలితం.l < 32 && ((uint32_t) 1 << ఫలితం.l) < d) {
        ++(ఫలితం.l);
    }

    ఫలితం.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << ఫలితం.l) - d) / d + 1);
    తిరిగి ఫలితం;
}

స్థిర అంతఃవరుస uint32_t fastdiv(uint32_t n, స్థిరరాశి నిర్మాణం fastdiv_విలువలు * vals) {
    // Compute high 32 bits of n * mp
    స్థిరరాశి uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    తిరిగి (hi + n) >> vals->l;
}

స్థిర అంతఃవరుస uint32_t fastmodulo(uint32_t n, uint32_t d, స్థిరరాశి నిర్మాణం fastdiv_విలువలు * vals) {
    తిరిగి n - fastdiv(n, vals) * d;
}

#అయితే_ముగింపు /* HEX_FASTDIV_H */
