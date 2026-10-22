#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_VTCM_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_VTCM_H

#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t ಗಾತ್ರ_2) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += ಗಾತ್ರ_2;
    ಹಿಂತಿರುಗಿ p;
}

#ವ್ಯಾಖ್ಯಾನಿಸು VTCM_LAYOUT_ALLOC(off, ಕ್ಷೇತ್ರ, sz) ಮಾಡು { (L)->ಕ್ಷೇತ್ರ = (off); (off) += (sz); } ಆಗ (0)
#ವ್ಯಾಖ್ಯಾನಿಸು VTCM_LAYOUT_ALLOC_OPTIONAL(off, ಕ್ಷೇತ್ರ, sz, cond) ಮಾಡು { ಆದರೆ (cond) { VTCM_LAYOUT_ALLOC(off, ಕ್ಷೇತ್ರ, sz); } ಇಲ್ಲದಿದ್ದರೆ { (L)->ಕ್ಷೇತ್ರ = 0; } } ಆಗ (0)

#ವ್ಯಾಖ್ಯಾನಿಸು VTCM_LAYOUT_PTR(type, ತಳ, ಸ್ಥಾನಾಂತರ) ((type *)((uint8_t *)(ತಳ) + (ಸ್ಥಾನಾಂತರ)))
#ವ್ಯಾಖ್ಯಾನಿಸು VTCM_LAYOUT_PTR_OPTIONAL(type, ತಳ, ಸ್ಥಾನಾಂತರ, cond) ((cond) ? VTCM_LAYOUT_PTR(type, ತಳ, ಸ್ಥಾನಾಂತರ) : ಖಾಲಿ)

#ಆದರೆ_ಅಂತ್ಯ // HTP_VTCM_H
