#جيڪڏهن_وصف_نه_ٿيل HTP_SSM_CONV_H
#وصف_ڪريو HTP_SSM_CONV_H

#ضمن <stdint.h>

#ضمن "hex-fastdiv.h"
#ضمن "htp-ops.h"

بنية htp_ssm_conv_kernel_params {
    uint32_t n_ڌاڳا;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_tile;

    uint32_t vtcm_src0_سائيز_في_thread;
    uint32_t vtcm_src1_سائيز_في_thread;
    uint32_t vtcm_dst_سائيز_في_thread;

    uint32_t vtcm_src0_سائيز;
    uint32_t vtcm_src1_سائيز;
    uint32_t vtcm_dst_سائيز;
    uint32_t vtcm_سائيز;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#جيڪڏهن_پورو

#جيڪڏهن_پورو // HTP_SSM_CONV_H
