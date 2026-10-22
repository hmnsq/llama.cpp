#ifndef HTP_VTCM_H
#define HTP_VTCM_H

#அந்தர்பாவ <stddef.h>
#அந்தர்பாவ <stdint.h>

நிலையான உள்வரி uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    திரும்பு p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) செய் { (L)->field = (off); (off) += (sz); } போது (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) செய் { எனில் (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } இல்லையெனில் { (L)->field = 0; } } போது (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : வெற்று)

#endif // HTP_VTCM_H
