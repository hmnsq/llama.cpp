#ifndef HTP_VTCM_H
#define HTP_VTCM_H

#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব <stdint.h>

স্থির অন্তঃরেখ uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t আকার_2) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += আকার_2;
    ফেরত p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) করো { (L)->field = (off); (off) += (sz); } যখন (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) করো { যদি (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } অন্যথা { (L)->field = 0; } } যখন (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : ফাঁকা)

#endif // HTP_VTCM_H
