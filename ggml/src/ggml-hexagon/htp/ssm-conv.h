#اگر_غیر_تعریف_شدہ HTP_SSM_CONV_H
#تعریف_کرو HTP_SSM_CONV_H

#شامل <stdint.h>

#شامل "hex-fastdiv.h"
#شامل "htp-ops.h"

ساخت htp_ssm_conv_kernel_params {
    uint32_t n_دھاگے;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_tile;

    uint32_t vtcm_src0_سائز_فی_thread;
    uint32_t vtcm_src1_سائز_فی_thread;
    uint32_t vtcm_dst_سائز_فی_thread;

    uint32_t vtcm_src0_سائز;
    uint32_t vtcm_src1_سائز;
    uint32_t vtcm_dst_سائز;
    uint32_t vtcm_سائز;
};

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#ورنہ
_Static_assert(حجم(ساخت htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#اگر_ختم

#اگر_ختم // HTP_SSM_CONV_H
