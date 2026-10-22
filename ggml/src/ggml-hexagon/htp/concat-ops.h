#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_CONCAT_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_CONCAT_OPS_H

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "gather-rows.h"
#ಅನ್ತರ್ಭಾವ <stdint.h>

ಎಣಿಕೆ htp_concat_kernel_type {
    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
    HTP_CONCAT_KERNEL_REGULAR     = 1,
    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
    HTP_CONCAT_KERNEL_GATHER_ಅಡ್ಡಸಾಲುಗಳು = 3,
};

ರಚನೆ htp_concat_transposed_params {
    uint32_t spad0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t spad1_ಗಾತ್ರ_ಪ್ರತಿ_thread;
};

ರಚನೆ htp_concat_kernel_params {
    uint8_t  kernel_type;
    uint8_t  dim;
    uint8_t  n_ಎಳೆಗಳು;
    uint8_t  pad;

    uint32_t vtcm_ಗಾತ್ರ;

    ಒಕ್ಕೂಟ {
        ರಚನೆ htp_concat_transposed_params transposed;
        ರಚನೆ htp_gather_ಅಡ್ಡಸಾಲುಗಳು_params       gather;
    } u;
};

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#ಇಲ್ಲದಿದ್ದರೆ
_Static_assert(ಗಾತ್ರ(ರಚನೆ htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ htp_concat_transposed_vtcm_layout {
    uint32_t src0_spad_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t src1_spad_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t ಒಟ್ಟು_ಬೈಟುಗಳು;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_concat_transposed_vtcm_layout_build(
    ರಚನೆ htp_concat_transposed_vtcm_layout * layout,
    uint32_t src0_ne0,
    uint32_t src1_ne0,
    uint32_t type_ಗಾತ್ರ,
    uint32_t n_ಎಳೆಗಳು) {

    uint32_t ಖಂಡ_i = (type_ಗಾತ್ರ == 4) ? 32 : 64;
    uint32_t spad1_stride = ಖಂಡ_i * type_ಗಾತ್ರ;
    uint32_t src1_ne0_padded = hex_ಸುತ್ತು_ಮೇಲೆ(src1_ne0, ಖಂಡ_i);
    uint32_t spad0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = hex_ಸುತ್ತು_ಮೇಲೆ(src0_ne0 * type_ಗಾತ್ರ, 128) + src1_ne0_padded * type_ಗಾತ್ರ;

    layout->src0_spad_ಗಾತ್ರ_ಪ್ರತಿ_thread = ಖಂಡ_i * spad0_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು;
    layout->src1_spad_ಗಾತ್ರ_ಪ್ರತಿ_thread = src1_ne0_padded * spad1_stride;
    layout->ಒಟ್ಟು_ಬೈಟುಗಳು = n_ಎಳೆಗಳು * (layout->src0_spad_ಗಾತ್ರ_ಪ್ರತಿ_thread + layout->src1_spad_ಗಾತ್ರ_ಪ್ರತಿ_thread);
}

#ಆದರೆ_ಅಂತ್ಯ // HTP_CONCAT_OPS_H
