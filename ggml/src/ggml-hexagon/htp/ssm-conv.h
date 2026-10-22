#جے_بنا_تعریف HTP_SSM_CONV_H
#تعریف_کرو HTP_SSM_CONV_H

#ضمن <stdint.h>

#ضمن "hex-fastdiv.h"
#ضمن "htp-ops.h"

بنية htp_ssm_conv_kernel_params {
    uint32_t n_دھاگے;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_tile;

    uint32_t vtcm_src0_ناپ_فی_thread;
    uint32_t vtcm_src1_ناپ_فی_thread;
    uint32_t vtcm_dst_ناپ_فی_thread;

    uint32_t vtcm_src0_ناپ;
    uint32_t vtcm_src1_ناپ;
    uint32_t vtcm_dst_ناپ;
    uint32_t vtcm_ناپ;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#جے_انت

#جے_انت // HTP_SSM_CONV_H
