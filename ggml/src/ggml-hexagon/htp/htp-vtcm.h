#నిర్వచితం_కాకపోతే HTP_VTCM_H
#నిర్వచించు HTP_VTCM_H

#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdint.h>

స్థిర అంతఃవరుస uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t కొలత) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += కొలత;
    తిరిగి p;
}

#నిర్వచించు VTCM_LAYOUT_ALLOC(off, క్షేత్రం, sz) చేయి { (L)->క్షేత్రం = (off); (off) += (sz); } అయితే (0)
#నిర్వచించు VTCM_LAYOUT_ALLOC_OPTIONAL(off, క్షేత్రం, sz, cond) చేయి { ఒకవేళ (cond) { VTCM_LAYOUT_ALLOC(off, క్షేత్రం, sz); } లేకపోతే { (L)->క్షేత్రం = 0; } } అయితే (0)

#నిర్వచించు VTCM_LAYOUT_PTR(type, పునాది, స్థానభ్రంశం) ((type *)((uint8_t *)(పునాది) + (స్థానభ్రంశం)))
#నిర్వచించు VTCM_LAYOUT_PTR_OPTIONAL(type, పునాది, స్థానభ్రంశం, cond) ((cond) ? VTCM_LAYOUT_PTR(type, పునాది, స్థానభ్రంశం) : ఖాళీ)

#అయితే_ముగింపు // HTP_VTCM_H
