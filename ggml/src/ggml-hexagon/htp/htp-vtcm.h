#ifndef HTP_VTCM_H
#define HTP_VTCM_H

#ଅନ୍ତର୍ଭାଵ <stddef.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    ଫେରନ୍ତୁ p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) କର { (L)->field = (off); (off) += (sz); } ଯେତେବେଳେ (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) କର { ଯଦି (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } ଅନ୍ୟଥା { (L)->field = 0; } } ଯେତେବେଳେ (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : ଖାଲି)

#endif // HTP_VTCM_H
