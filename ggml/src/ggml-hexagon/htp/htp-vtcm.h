#अगर_अघोषित HTP_VTCM_H
#घोषणा HTP_VTCM_H

#समावेश <stddef.h>
#समावेश <stdint.h>

अटल अंतरभूत uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, माप_प्रकार आकार) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += आकार;
    वापस p;
}

#घोषणा VTCM_LAYOUT_ALLOC(off, field, sz) करो { (L)->field = (off); (off) += (sz); } जबतक (0)
#घोषणा VTCM_LAYOUT_ALLOC_OPTIONAL(off, field, sz, cond) करो { अगर (cond) { VTCM_LAYOUT_ALLOC(off, field, sz); } अन्यथा { (L)->field = 0; } } जबतक (0)

#घोषणा VTCM_LAYOUT_PTR(type, base, offset) ((type *)((uint8_t *)(base) + (offset)))
#घोषणा VTCM_LAYOUT_PTR_OPTIONAL(type, base, offset, cond) ((cond) ? VTCM_LAYOUT_PTR(type, base, offset) : शून्य)

#पूर्ण_अगर // HTP_VTCM_H
