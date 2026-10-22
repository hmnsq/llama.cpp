#ifndef HTP_VTCM_H
#define HTP_VTCM_H

#include <stddef.h>
#include <stdint.h>

static anta_hstha uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    ᱨᱩᱣᱟᱹᱲ p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) do { (L)->field = (off); (off) += (sz); } ᱡᱚᱠᱷᱚᱱ (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) do { ᱡᱩᱫᱤ (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } ᱵᱟᱝ { (L)->field = 0; } } ᱡᱚᱠᱷᱚᱱ (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : NULL)

#endif // HTP_VTCM_H
