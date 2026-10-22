#که_نه_تعریف_شوی HTP_VTCM_H
#تعریف_کړه HTP_VTCM_H

#ضمن <stddef.h>
#ضمن <stdint.h>

ساكن مضمن uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t اندازه) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += اندازه;
    أرجع p;
}

#تعریف_کړه VTCM_LAYOUT_ALLOC(off, ډګر, sz) افعل { (L)->ډګر = (off); (off) += (sz); } طالما (0)
#تعریف_کړه VTCM_LAYOUT_ALLOC_OPTIONAL(off, ډګر, sz, cond) افعل { إذا (cond) { VTCM_LAYOUT_ALLOC(off, ډګر, sz); } وإلا { (L)->ډګر = 0; } } طالما (0)

#تعریف_کړه VTCM_LAYOUT_PTR(type, بنسټ, دننه_واټن) ((type *)((uint8_t *)(بنسټ) + (دننه_واټن)))
#تعریف_کړه VTCM_LAYOUT_PTR_OPTIONAL(type, بنسټ, دننه_واټن, cond) ((cond) ? VTCM_LAYOUT_PTR(type, بنسټ, دننه_واټن) : عدم)

#د_که_پای // HTP_VTCM_H
