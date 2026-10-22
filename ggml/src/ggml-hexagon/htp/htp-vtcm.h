#ifndef HTP_VTCM_H
#define HTP_VTCM_H

#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>

स्थिर अन्तःस्थ uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    प्रत्यागम p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) कुरु { (L)->field = (off); (off) += (sz); } यावत् (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) कुरु { यदि (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } अन्यथा { (L)->field = 0; } } यावत् (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : रिक्त)

#endif // HTP_VTCM_H
