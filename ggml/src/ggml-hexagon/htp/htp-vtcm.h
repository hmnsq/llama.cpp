#اگر_تعریف_نشده HTP_VTCM_H
#تعریف_کن HTP_VTCM_H

#ضمن <stddef.h>
#ضمن <stdint.h>

ساكن مضمن uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t اندازه) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += اندازه;
    أرجع p;
}

#تعریف_کن VTCM_LAYOUT_ALLOC(off, فیلد, sz) افعل { (L)->فیلد = (off); (off) += (sz); } طالما (0)
#تعریف_کن VTCM_LAYOUT_ALLOC_OPTIONAL(off, فیلد, sz, cond) افعل { إذا (cond) { VTCM_LAYOUT_ALLOC(off, فیلد, sz); } وإلا { (L)->فیلد = 0; } } طالما (0)

#تعریف_کن VTCM_LAYOUT_PTR(type, پایه, جابه_جایی) ((type *)((uint8_t *)(پایه) + (جابه_جایی)))
#تعریف_کن VTCM_LAYOUT_PTR_OPTIONAL(type, پایه, جابه_جایی, cond) ((cond) ? VTCM_LAYOUT_PTR(type, پایه, جابه_جایی) : عدم)

#پایان_اگر // HTP_VTCM_H
