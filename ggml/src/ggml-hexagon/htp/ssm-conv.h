#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_SSM_CONV_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_SSM_CONV_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"

ਬਣਤਰ htp_ssm_conv_kernel_params {
    uint32_t n_ਧਾਗੇ;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_tile;

    uint32_t vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread;

    uint32_t vtcm_src0_ਮਾਪ;
    uint32_t vtcm_src1_ਮਾਪ;
    uint32_t vtcm_dst_ਮਾਪ;
    uint32_t vtcm_ਮਾਪ;
};

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#ਨਹੀਂਤਾਂ
_Static_assert(ਆਕਾਰ(ਬਣਤਰ htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ // HTP_SSM_CONV_H
