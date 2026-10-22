#اگر_غیر_تعریف_شدہ HTP_VTCM_H
#تعریف_کرو HTP_VTCM_H

#شامل <stddef.h>
#شامل <stdint.h>

ساکن اندرونی uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t سائز) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += سائز;
    واپس p;
}

#تعریف_کرو VTCM_LAYOUT_ALLOC(off, میدان, sz) کرو { (L)->میدان = (off); (off) += (sz); } جب (0)
#تعریف_کرو VTCM_LAYOUT_ALLOC_OPTIONAL(off, میدان, sz, cond) کرو { اگر (cond) { VTCM_LAYOUT_ALLOC(off, میدان, sz); } ورنہ { (L)->میدان = 0; } } جب (0)

#تعریف_کرو VTCM_LAYOUT_PTR(type, بنیاد, انحراف) ((type *)((uint8_t *)(بنیاد) + (انحراف)))
#تعریف_کرو VTCM_LAYOUT_PTR_OPTIONAL(type, بنیاد, انحراف, cond) ((cond) ? VTCM_LAYOUT_PTR(type, بنیاد, انحراف) : تہی)

#اگر_ختم // HTP_VTCM_H
