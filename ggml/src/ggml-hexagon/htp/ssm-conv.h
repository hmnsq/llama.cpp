#ifndef HTP_SSM_CONV_H
#define HTP_SSM_CONV_H

#অন্তর্ভাৱ <stdint.h>

#অন্তর্ভাৱ "hex-fastdiv.h"
#অন্তর্ভাৱ "htp-ops.h"

সংরচনা htp_ssm_conv_kernel_params {
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

#যদি defined(__cplusplus)
static_assert(পরিমাণ(সংরচনা htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#অন্যথা
_Static_assert(পরিমাণ(সংরচনা htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SSM_CONV_H
