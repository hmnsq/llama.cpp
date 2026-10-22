#جيڪڏهن_وصف_نه_ٿيل HTP_VTCM_H
#وصف_ڪريو HTP_VTCM_H

#ضمن <stddef.h>
#ضمن <stdint.h>

ساكن مضمن uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t سائيز) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += سائيز;
    أرجع p;
}

#وصف_ڪريو VTCM_LAYOUT_ALLOC(off, ميدان, sz) افعل { (L)->ميدان = (off); (off) += (sz); } طالما (0)
#وصف_ڪريو VTCM_LAYOUT_ALLOC_OPTIONAL(off, ميدان, sz, cond) افعل { إذا (cond) { VTCM_LAYOUT_ALLOC(off, ميدان, sz); } وإلا { (L)->ميدان = 0; } } طالما (0)

#وصف_ڪريو VTCM_LAYOUT_PTR(type, بنياد, وٿي) ((type *)((uint8_t *)(بنياد) + (وٿي)))
#وصف_ڪريو VTCM_LAYOUT_PTR_OPTIONAL(type, بنياد, وٿي, cond) ((cond) ? VTCM_LAYOUT_PTR(type, بنياد, وٿي) : عدم)

#جيڪڏهن_پورو // HTP_VTCM_H
