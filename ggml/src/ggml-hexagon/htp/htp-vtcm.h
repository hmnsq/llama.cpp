#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_VTCM_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_VTCM_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਸਥਿਰ ਸਤਰਅੰਦਰ uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t ਮਾਪ) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += ਮਾਪ;
    ਵਾਪਸ p;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VTCM_LAYOUT_ALLOC(off, ਖੇਤਰ, sz) ਕਰੋ { (L)->ਖੇਤਰ = (off); (off) += (sz); } ਜਦੋਂ (0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VTCM_LAYOUT_ALLOC_OPTIONAL(off, ਖੇਤਰ, sz, cond) ਕਰੋ { ਜੇ (cond) { VTCM_LAYOUT_ALLOC(off, ਖੇਤਰ, sz); } ਨਹੀਂਤਾਂ { (L)->ਖੇਤਰ = 0; } } ਜਦੋਂ (0)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VTCM_LAYOUT_PTR(type, ਅਧਾਰ, ਵਿੱਥ) ((type *)((uint8_t *)(ਅਧਾਰ) + (ਵਿੱਥ)))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VTCM_LAYOUT_PTR_OPTIONAL(type, ਅਧਾਰ, ਵਿੱਥ, cond) ((cond) ? VTCM_LAYOUT_PTR(type, ਅਧਾਰ, ਵਿੱਥ) : ਨਿਰਾ)

#ਜੇ_ਅੰਤ // HTP_VTCM_H
