#నిర్వచితం_కాకపోతే HTP_SSM_CONV_H
#నిర్వచించు HTP_SSM_CONV_H

#అన్తర్భావ <stdint.h>

#అన్తర్భావ "hex-fastdiv.h"
#అన్తర్భావ "htp-ops.h"

నిర్మాణం htp_ssm_conv_kernel_params {
    uint32_t n_దారాలు;
    uint32_t d_conv;
    uint32_t d_inner;
    uint32_t n_t;
    uint32_t n_s;
    uint32_t d_inner_tile;

    uint32_t vtcm_src0_కొలత_చొప్పున_thread;
    uint32_t vtcm_src1_కొలత_చొప్పున_thread;
    uint32_t vtcm_dst_కొలత_చొప్పున_thread;

    uint32_t vtcm_src0_కొలత;
    uint32_t vtcm_src1_కొలత;
    uint32_t vtcm_dst_కొలత;
    uint32_t vtcm_కొలత;
};

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#లేకపోతే
_Static_assert(పరిమాణం(నిర్మాణం htp_ssm_conv_kernel_params) <= 128, "htp_ssm_conv_kernel_params is too large for kernel_params blob");
#అయితే_ముగింపు

#అయితే_ముగింపు // HTP_SSM_CONV_H
