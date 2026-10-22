#నిర్వచితం_కాకపోతే HTP_సమితి_అడ్డువరుసలు_OPS_H
#నిర్వచించు HTP_సమితి_అడ్డువరుసలు_OPS_H

#అన్తర్భావ "hex-fastdiv.h"

నిర్మాణం htp_సమితి_అడ్డువరుసలు_kernel_params {
    int32_t  n_దారాలు;
    int32_t  మొత్తం_tasks;
    int32_t  tasks_చొప్పున_thread;
    int32_t  vtcm_కొలత;

    // Fastdiv helpers
    నిర్మాణం fastdiv_విలువలు div_ne11;
    నిర్మాణం fastdiv_విలువలు div_ne12;
    నిర్మాణం fastdiv_విలువలు div_tasks_చొప్పున_thread;
    నిర్మాణం fastdiv_విలువలు div_ne02;
};

నిర్మాణం htp_సమితి_అడ్డువరుసలు_vtcm_layout {
    size_t మొత్తం_బైట్లు;
    size_t off_src0;
    size_t off_dst;

    size_t src0_బైట్లు_చొప్పున_thread;
    size_t dst_బైట్లు_చొప్పున_thread;

    size_t src0_spad_half_కొలత;
    size_t dst_spad_half_కొలత;
};

స్థిర అంతఃవరుస శూన్య htp_సమితి_అడ్డువరుసలు_vtcm_layout_build(
    నిర్మాణం htp_సమితి_అడ్డువరుసలు_vtcm_layout * vtcm_layout,
    పూర్ణాంకం dst_type,
    uint32_t ne00,
    uint32_t n_దారాలు) {

    size_t src0_అడ్డువరుస_కొలత = ne00 * 4;
    size_t dst_అడ్డువరుస_కొలత = 0;
    ఎంపిక (dst_type) {
        స్థితి 0: // HTP_TYPE_F32
            dst_అడ్డువరుస_కొలత = ne00 * 4;
            ఆపు;
        స్థితి 1: // HTP_TYPE_F16
            dst_అడ్డువరుస_కొలత = ne00 * 2;
            ఆపు;
        స్థితి 8: // HTP_TYPE_Q8_0
            dst_అడ్డువరుస_కొలత = (ne00 / 32) * 34;
            ఆపు;
        అప్రమేయ:
            dst_అడ్డువరుస_కొలత = 0;
            ఆపు;
    }

    size_t src0_అడ్డువరుస_కొలత_aligned = (src0_అడ్డువరుస_కొలత + 255) & ~255;
    size_t dst_అడ్డువరుస_కొలత_aligned  = (dst_అడ్డువరుస_కొలత + 255) & ~255;

    vtcm_layout->src0_spad_half_కొలత = src0_అడ్డువరుస_కొలత_aligned;
    vtcm_layout->dst_spad_half_కొలత  = dst_అడ్డువరుస_కొలత_aligned;

    vtcm_layout->src0_బైట్లు_చొప్పున_thread = src0_అడ్డువరుస_కొలత_aligned * 2;
    vtcm_layout->dst_బైట్లు_చొప్పున_thread  = dst_అడ్డువరుస_కొలత_aligned * 2;

    vtcm_layout->off_src0 = 0;
    vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_బైట్లు_చొప్పున_thread * n_దారాలు;
    vtcm_layout->మొత్తం_బైట్లు = vtcm_layout->off_dst + vtcm_layout->dst_బైట్లు_చొప్పున_thread * n_దారాలు;
}

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_సమితి_అడ్డువరుసలు_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#లేకపోతే
_Static_assert(పరిమాణం(నిర్మాణం htp_సమితి_అడ్డువరుసలు_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#అయితే_ముగింపు

#అయితే_ముగింపు // HTP_SET_ROWS_OPS_H
