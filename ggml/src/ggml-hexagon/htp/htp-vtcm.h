#ifndef HTP_VTCM_H
#define HTP_VTCM_H

#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdint.h>

স্থির অন্তঃস্থ uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    প্রত্যাগম p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) কুরু { (L)->field = (off); (off) += (sz); } যাবত্ (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) কুরু { যদি (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } অন্যথা { (L)->field = 0; } } যাবত্ (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : রিক্ত)

#endif // HTP_VTCM_H
