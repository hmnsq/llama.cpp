#جيڪڏهن_وصف_نه_ٿيل HEX_FASTDIV_H
#وصف_ڪريو HEX_FASTDIV_H

// See https://gmplib.org/~tege/divcnst-pldi94.pdf figure 4.1.
// Precompute mp (m' in the paper) and L such that division
// can be computed using a multiply (high 32b of 64b result)
// and a shift:
//
// n/d = (mulhi(n, mp) + n) >> L;
بنية fastdiv_قدرون {
    uint32_t mp;
    uint32_t l;
};

ساكن مضمن بنية fastdiv_قدرون init_fastdiv_قدرون(uint32_t d) {
    بنية fastdiv_قدرون نتيجو = { 0, 0 };
    // compute L = ceil(log2(d));
    طالما (نتيجو.l < 32 && ((uint32_t) 1 << نتيجو.l) < d) {
        ++(نتيجو.l);
    }

    نتيجو.mp = (uint32_t) (((uint64_t) 1 << 32) * (((uint64_t) 1 << نتيجو.l) - d) / d + 1);
    أرجع نتيجو;
}

ساكن مضمن uint32_t fastdiv(uint32_t n, ثابت بنية fastdiv_قدرون * vals) {
    // Compute high 32 bits of n * mp
    ثابت uint32_t hi = (uint32_t) (((uint64_t) n * vals->mp) >> 32);  // mulhi(n, mp)
    // add n, apply bit shift
    أرجع (hi + n) >> vals->l;
}

ساكن مضمن uint32_t fastmodulo(uint32_t n, uint32_t d, ثابت بنية fastdiv_قدرون * vals) {
    أرجع n - fastdiv(n, vals) * d;
}

#جيڪڏهن_پورو /* HEX_FASTDIV_H */
