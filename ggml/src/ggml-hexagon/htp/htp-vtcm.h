#ifndef HTP_VTCM_H
#define HTP_VTCM_H

#شامل <stddef.h>
#شامل <stdint.h>

ایستا درونخطی uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    بازگشت p;
}

#define VTCM_LAYOUT_ALLOC(off, field, sz) انجامبده { (L)->field = (off); (off) += (sz); } وقتی (0)
#define VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) انجامبده { اگر (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } وگرنه { (L)->field = 0; } } وقتی (0)

#define VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#define VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : پوچ)

#endif // HTP_VTCM_H
