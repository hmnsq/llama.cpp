#اگر_تعریف_نشده HTP_SSM_CONV_H
#تعریف_کن HTP_SSM_CONV_H

#ضمن <stdint.h>

#ضمن "hex-fastdiv.h"
#ضمن "htp-ops.h"

بنية htp_ssm_conv_kernel_params {
    uint32_t n_نخها;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_tile;

    uint32_t vtcm_src0_اندازه_در_هر_thread;
    uint32_t vtcm_src1_اندازه_در_هر_thread;
    uint32_t vtcm_dst_اندازه_در_هر_thread;

    uint32_t vtcm_src0_اندازه;
    uint32_t vtcm_src1_اندازه;
    uint32_t vtcm_dst_اندازه;
    uint32_t vtcm_اندازه;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#پایان_اگر

#پایان_اگر // HTP_SSM_CONV_H
