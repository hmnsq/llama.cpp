#నిర్వచితం_కాకపోతే HTP_SOFTMAX_OPS_H
#నిర్వచించు HTP_SOFTMAX_OPS_H

#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ "hex-fastdiv.h"
#అన్తర్భావ "hex-common.h"

గణన htp_softmax_kernel_id {
    HTP_SOFTMAX_KERNEL_NOMASK = 0,
    HTP_SOFTMAX_KERNEL_ముసుగు_F32,
    HTP_SOFTMAX_KERNEL_ముసుగు_F16,
    HTP_SOFTMAX_KERNEL_లెక్క,
};

నిర్మాణం htp_softmax_kernel_params {
    uint32_t n_దారాలు;
    uint32_t src0_nrows;
    uint32_t src0_nrows_చొప్పున_thread;
    uint32_t vtcm_కొలత;

    uint32_t vtcm_src0_కొలత_చొప్పున_thread;
    uint32_t vtcm_src1_కొలత_చొప్పున_thread;
    uint32_t vtcm_dst_కొలత_చొప్పున_thread;

    uint32_t src0_అడ్డువరుస_కొలత_aligned;
    uint32_t src1_అడ్డువరుస_కొలత_aligned;
    uint32_t dst_అడ్డువరుస_కొలత_aligned;

    uint32_t src0_spad_half_కొలత;
    uint32_t src1_spad_half_కొలత;
    uint32_t dst_spad_half_కొలత;

    uint32_t n_తల;
    uint32_t n_తల_log2;
    uint32_t use_src1;
    uint32_t use_f16;
    uint32_t kernel_id;

    దశాంశం    scale;
    దశాంశం    max_bias;
    దశాంశం    m0;
    దశాంశం    m1;

    నిర్మాణం fastdiv_విలువలు div_ne01;
    నిర్మాణం fastdiv_విలువలు div_ne02;
    నిర్మాణం fastdiv_విలువలు div_ne12;
    నిర్మాణం fastdiv_విలువలు div_ne13;
};

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#లేకపోతే
_Static_assert(పరిమాణం(నిర్మాణం htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#అయితే_ముగింపు

నిర్మాణం htp_softmax_vtcm_layout {
    size_t మొత్తం_బైట్లు;
    size_t off_src0;
    size_t off_dst;
    size_t off_src1;

    size_t src0_బైట్లు_చొప్పున_thread;
    size_t dst_బైట్లు_చొప్పున_thread;
    size_t src1_బైట్లు_చొప్పున_thread;

    size_t src0_spad_half_కొలత;
    size_t dst_spad_half_కొలత;
    size_t src1_spad_half_కొలత;
};

స్థిర అంతఃవరుస శూన్య htp_softmax_vtcm_layout_build(
    నిర్మాణం htp_softmax_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne10,
    bool use_src1,
    bool use_f16,
    uint32_t n_దారాలు
) {
    size_t src0_అడ్డువరుస_కొలత = ne00 * పరిమాణం(దశాంశం);
    size_t dst_అడ్డువరుస_కొలత  = ne00 * పరిమాణం(దశాంశం);
    size_t src1_అడ్డువరుస_కొలత = use_src1 ? (ne10 * (use_f16 ? 2 : 4)) : 0;

    size_t src0_అడ్డువరుస_కొలత_aligned = hex_విడత_పైకి(src0_అడ్డువరుస_కొలత, 128);
    size_t dst_అడ్డువరుస_కొలత_aligned  = hex_విడత_పైకి(dst_అడ్డువరుస_కొలత,  128);
    size_t src1_అడ్డువరుస_కొలత_aligned = use_src1 ? hex_విడత_పైకి(src1_అడ్డువరుస_కొలత, 128) : 0;

    layout->src0_spad_half_కొలత = src0_అడ్డువరుస_కొలత_aligned;
    layout->dst_spad_half_కొలత  = dst_అడ్డువరుస_కొలత_aligned;
    layout->src1_spad_half_కొలత = src1_అడ్డువరుస_కొలత_aligned;

    // Double buffering: 2 half-buffers per thread
    layout->src0_బైట్లు_చొప్పున_thread = src0_అడ్డువరుస_కొలత_aligned * 2;
    layout->dst_బైట్లు_చొప్పున_thread  = dst_అడ్డువరుస_కొలత_aligned  * 2;
    layout->src1_బైట్లు_చొప్పున_thread = src1_అడ్డువరుస_కొలత_aligned * 2;

    layout->off_src0 = 0;
    layout->off_dst  = layout->off_src0 + layout->src0_బైట్లు_చొప్పున_thread * n_దారాలు;
    layout->off_src1 = layout->off_dst  + layout->dst_బైట్లు_చొప్పున_thread  * n_దారాలు;

    layout->మొత్తం_బైట్లు = layout->off_src1 + layout->src1_బైట్లు_చొప్పున_thread * n_దారాలు;
}

#అయితే_ముగింపు // HTP_SOFTMAX_OPS_H
