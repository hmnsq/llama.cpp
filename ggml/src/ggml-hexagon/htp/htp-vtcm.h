#ifndef HTP_VTCM_H
#define HTP_VTCM_H

#અન્તર્ભાવ <stddef.h>
#અન્તર્ભાવ <stdint.h>

સ્થિર અંતઃરેખ uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    પાછા p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) કરો { (L)->field = (off); (off) += (sz); } જ્યારે (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) કરો { જો (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } નહીંતર { (L)->field = 0; } } જ્યારે (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : ખાલી)

#endif // HTP_VTCM_H
