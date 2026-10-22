#यदि_अपरिभाषितम् HTP_SSM_CONV_H
#परिभाषय HTP_SSM_CONV_H

#अन्तर्भाव <stdint.h>

#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "htp-ops.h"

संरचना htp_ssm_conv_kernel_params {
    uint32_t n_तन्तवः;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_tile;

    uint32_t vtcm_src0_आकार_प्रत्येकम्_thread;
    uint32_t vtcm_src1_आकार_प्रत्येकम्_thread;
    uint32_t vtcm_dst_आकार_प्रत्येकम्_thread;

    uint32_t vtcm_src0_आकार;
    uint32_t vtcm_src1_आकार;
    uint32_t vtcm_dst_आकार;
    uint32_t vtcm_आकार;
};

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#यदि_अन्तः

#यदि_अन्तः // HTP_SSM_CONV_H
