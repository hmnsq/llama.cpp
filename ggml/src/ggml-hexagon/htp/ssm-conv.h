#ifndef HTP_SSM_CONV_H
#define HTP_SSM_CONV_H

#ಅನ್ತರ್ಭಾವ <stdint.h>

#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"

ಸಂರಚನಾ htp_ssm_conv_kernel_params {
    uint32_t n_threads;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_tile;

    uint32_t vtcm_src0_size_per_thread;
    uint32_t vtcm_src1_size_per_thread;
    uint32_t vtcm_dst_size_per_thread;

    uint32_t vtcm_src0_size;
    uint32_t vtcm_src1_size;
    uint32_t vtcm_dst_size;
    uint32_t vtcm_size;
};

#ಯದಿ defined(__cplusplus)
static_assert(ಪರಿಮಾಣ(ಸಂರಚನಾ htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#ಅನ್ಯಥಾ
_Static_assert(ಪರಿಮಾಣ(ಸಂರಚನಾ htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SSM_CONV_H
