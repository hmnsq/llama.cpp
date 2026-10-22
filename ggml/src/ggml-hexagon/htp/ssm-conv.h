#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_SSM_CONV_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_SSM_CONV_H

#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"

ರಚನೆ htp_ssm_conv_kernel_params {
    uint32_t n_ಎಳೆಗಳು;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_tile;

    uint32_t vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread;

    uint32_t vtcm_src0_ಗಾತ್ರ;
    uint32_t vtcm_src1_ಗಾತ್ರ;
    uint32_t vtcm_dst_ಗಾತ್ರ;
    uint32_t vtcm_ಗಾತ್ರ;
};

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#ಇಲ್ಲದಿದ್ದರೆ
_Static_assert(ಗಾತ್ರ(ರಚನೆ htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ_ಅಂತ್ಯ // HTP_SSM_CONV_H
