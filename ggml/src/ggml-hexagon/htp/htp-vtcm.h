#यदि_अपरिभाषितम् HTP_VTCM_H
#परिभाषय HTP_VTCM_H

#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>

स्थिर अन्तःस्थ uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t आकार) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += आकार;
    प्रत्यागम p;
}

#परिभाषय VTCM_LAYOUT_ALLOC(off, क्षेत्रम्, sz) कुरु { (L)->क्षेत्रम् = (off); (off) += (sz); } यावत् (0)
#परिभाषय VTCM_LAYOUT_ALLOC_OPTIONAL(off, क्षेत्रम्, sz, cond) कुरु { यदि (cond) { VTCM_LAYOUT_ALLOC(off, क्षेत्रम्, sz); } अन्यथा { (L)->क्षेत्रम् = 0; } } यावत् (0)

#परिभाषय VTCM_LAYOUT_PTR(type, आधारः, विस्थापनम्) ((type *)((uint8_t *)(आधारः) + (विस्थापनम्)))
#परिभाषय VTCM_LAYOUT_PTR_OPTIONAL(type, आधारः, विस्थापनम्, cond) ((cond) ? VTCM_LAYOUT_PTR(type, आधारः, विस्थापनम्) : रिक्त)

#यदि_अन्तः // HTP_VTCM_H
