#ifndef HTP_VTCM_H
#define HTP_VTCM_H

#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    ಪ್ರತ್ಯಾಗಮ p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) ಕುರು { (L)->field = (off); (off) += (sz); } ಯಾವತ್ (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) ಕುರು { ಯದಿ (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } ಅನ್ಯಥಾ { (L)->field = 0; } } ಯಾವತ್ (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : ರಿಕ್ತ)

#endif // HTP_VTCM_H
