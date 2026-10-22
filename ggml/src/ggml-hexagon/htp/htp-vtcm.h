#جے_بنا_تعریف HTP_VTCM_H
#تعریف_کرو HTP_VTCM_H

#ضمن <stddef.h>
#ضمن <stdint.h>

ساكن مضمن uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t ناپ) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += ناپ;
    أرجع p;
}

#تعریف_کرو VTCM_LAYOUT_ALLOC(off, میدان, sz) افعل { (L)->میدان = (off); (off) += (sz); } طالما (0)
#تعریف_کرو VTCM_LAYOUT_ALLOC_OPTIONAL(off, میدان, sz, cond) افعل { إذا (cond) { VTCM_LAYOUT_ALLOC(off, میدان, sz); } وإلا { (L)->میدان = 0; } } طالما (0)

#تعریف_کرو VTCM_LAYOUT_PTR(type, ادھار, وتھ) ((type *)((uint8_t *)(ادھار) + (وتھ)))
#تعریف_کرو VTCM_LAYOUT_PTR_OPTIONAL(type, ادھار, وتھ, cond) ((cond) ? VTCM_LAYOUT_PTR(type, ادھار, وتھ) : عدم)

#جے_انت // HTP_VTCM_H
