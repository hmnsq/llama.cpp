#ifndef HTP_VTCM_H
#define HTP_VTCM_H

#অন্তর্ভাৱ <stddef.h>
#অন্তর্ভাৱ <stdint.h>

স্থির অন্তঃস্থ uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    ঘূৰাই p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) কৰা { (L)->field = (off); (off) += (sz); } যেতিয়ালৈ (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) কৰা { যদি (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } অন্যথা { (L)->field = 0; } } যেতিয়ালৈ (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : রিক্ত)

#endif // HTP_VTCM_H
