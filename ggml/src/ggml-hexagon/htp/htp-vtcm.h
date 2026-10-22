#ifndef HTP_VTCM_H
#define HTP_VTCM_H

#ضمن <stddef.h>
#ضمن <stdint.h>

ساكن مضمن uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    ܦܢܐ p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) ܥܒܕ { (L)->field = (off); (off) += (sz); } ܟܡܐ (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) ܥܒܕ { ܐܢ (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } ܐܚܪܢܐ { (L)->field = 0; } } ܟܡܐ (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : عدم)

#endif // HTP_VTCM_H
