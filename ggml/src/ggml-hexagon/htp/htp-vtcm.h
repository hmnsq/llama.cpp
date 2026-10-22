#ifndef HTP_VTCM_H
#define HTP_VTCM_H

#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>

स्थिर अन्तःस्थ uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    पटिनिवत्त p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) करोहि { (L)->field = (off); (off) += (sz); } याव (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) करोहि { सचे (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } अञ्ञथा { (L)->field = 0; } } याव (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : रिक्त)

#endif // HTP_VTCM_H
