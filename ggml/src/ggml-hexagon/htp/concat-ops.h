#నిర్వచితం_కాకపోతే HTP_CONCAT_OPS_H
#నిర్వచించు HTP_CONCAT_OPS_H

#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "gather-rows.h"
#అన్తర్భావ <stdint.h>

గణన htp_concat_kernel_type {
    HTP_CONCAT_KERNEL_UNSUPPORTED = 0,
    HTP_CONCAT_KERNEL_REGULAR     = 1,
    HTP_CONCAT_KERNEL_TRANSPOSED  = 2,
    HTP_CONCAT_KERNEL_GATHER_అడ్డువరుసలు = 3,
};

నిర్మాణం htp_concat_transposed_params {
    uint32_t spad0_కొలత_చొప్పున_thread;
    uint32_t spad1_కొలత_చొప్పున_thread;
};

నిర్మాణం htp_concat_kernel_params {
    uint8_t  kernel_type;
    uint8_t  dim;
    uint8_t  n_దారాలు;
    uint8_t  pad;

    uint32_t vtcm_కొలత;

    సంఘం {
        నిర్మాణం htp_concat_transposed_params transposed;
        నిర్మాణం htp_gather_అడ్డువరుసలు_params       gather;
    } u;
};

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#లేకపోతే
_Static_assert(పరిమాణం(నిర్మాణం htp_concat_kernel_params) <= 128, "htp_concat_kernel_params is too large for kernel_params blob");
#అయితే_ముగింపు

నిర్మాణం htp_concat_transposed_vtcm_layout {
    uint32_t src0_spad_కొలత_చొప్పున_thread;
    uint32_t src1_spad_కొలత_చొప్పున_thread;
    uint32_t మొత్తం_బైట్లు;
};

స్థిర అంతఃవరుస శూన్య htp_concat_transposed_vtcm_layout_build(
    నిర్మాణం htp_concat_transposed_vtcm_layout * layout,
    uint32_t src0_ne0,
    uint32_t src1_ne0,
    uint32_t type_కొలత,
    uint32_t n_దారాలు) {

    uint32_t ఖండం_i = (type_కొలత == 4) ? 32 : 64;
    uint32_t spad1_stride = ఖండం_i * type_కొలత;
    uint32_t src1_ne0_padded = hex_విడత_పైకి(src1_ne0, ఖండం_i);
    uint32_t spad0_అడ్డువరుస_బైట్లు = hex_విడత_పైకి(src0_ne0 * type_కొలత, 128) + src1_ne0_padded * type_కొలత;

    layout->src0_spad_కొలత_చొప్పున_thread = ఖండం_i * spad0_అడ్డువరుస_బైట్లు;
    layout->src1_spad_కొలత_చొప్పున_thread = src1_ne0_padded * spad1_stride;
    layout->మొత్తం_బైట్లు = n_దారాలు * (layout->src0_spad_కొలత_చొప్పున_thread + layout->src1_spad_కొలత_చొప్పున_thread);
}

#అయితే_ముగింపు // HTP_CONCAT_OPS_H
