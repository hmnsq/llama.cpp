#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_ROPE_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_ROPE_OPS_H

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_ROPE_SPAD_ಖಂಡ  8
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_ROPE_SPAD_NSLOTS 4
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_ROPE_SPAD_NROWS  (HTP_ROPE_SPAD_ಖಂಡ * HTP_ROPE_SPAD_NSLOTS)

ರಚನೆ htp_rope_kernel_params {
    uint32_t n_ಎಳೆಗಳು;
    uint32_t src0_nrows;
    uint32_t src0_nrows_ಪ್ರತಿ_thread;
    uint32_t vtcm_ಗಾತ್ರ;
    uint32_t spad_ಪ್ರತಿ_thread;
    uint32_t theta_cache_ಸ್ಥಾನಾಂತರ;
    uint32_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    uint32_t freq_factors_ಸ್ಥಾನಾಂತರ;
    uint32_t freq_factors_ಗಾತ್ರ;

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne2_ne1;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne1;
};

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#ಇಲ್ಲದಿದ್ದರೆ
_Static_assert(ಗಾತ್ರ(ರಚನೆ htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ htp_rope_vtcm_layout {
    size_t ಒಟ್ಟು_ಬೈಟುಗಳು;
    size_t ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    size_t theta_cache_ಗಾತ್ರ_aligned;
    size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    size_t freq_factors_ಗಾತ್ರ_aligned;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_rope_vtcm_layout_build(
    ರಚನೆ htp_rope_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t n_ಎಳೆಗಳು,
    uint32_t n_freq_factors
) {
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ             = ne00 * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned     = hex_ಸುತ್ತು_ಮೇಲೆ((uint32_t) src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128);
    ಸ್ಥಿರಾಂಕ size_t theta_cache_ಗಾತ್ರ_aligned  = hex_ಸುತ್ತು_ಮೇಲೆ((uint32_t) src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 256);
    ಸ್ಥಿರಾಂಕ size_t freq_factors_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(n_freq_factors * ಗಾತ್ರ(ದಶಮಾಂಶ), 256);

    layout->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned     = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    layout->theta_cache_ಗಾತ್ರ_aligned  = theta_cache_ಗಾತ್ರ_aligned;
    layout->freq_factors_ಗಾತ್ರ_aligned = freq_factors_ಗಾತ್ರ_aligned;
    layout->ಬೈಟುಗಳು_ಪ್ರತಿ_thread          = theta_cache_ಗಾತ್ರ_aligned + HTP_ROPE_SPAD_NROWS * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    layout->ಒಟ್ಟು_ಬೈಟುಗಳು               = layout->ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು + freq_factors_ಗಾತ್ರ_aligned;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint8_t * rope_spad_ಸ್ಥಾನ(uint8_t * ತಳ, uint32_t ಸ್ಥಾನ, size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned) {
    ಹಿಂತಿರುಗಿ ತಳ + (ಸ್ಥಾನ * HTP_ROPE_SPAD_ಖಂಡ) * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
}

#ಆದರೆ_ಅಂತ್ಯ // HTP_ROPE_OPS_H
