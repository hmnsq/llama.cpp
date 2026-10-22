#నిర్వచితం_కాకపోతే HTP_ROPE_OPS_H
#నిర్వచించు HTP_ROPE_OPS_H

#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "hex-fastdiv.h"

#నిర్వచించు HTP_ROPE_SPAD_ఖండం  8
#నిర్వచించు HTP_ROPE_SPAD_NSLOTS 4
#నిర్వచించు HTP_ROPE_SPAD_NROWS  (HTP_ROPE_SPAD_ఖండం * HTP_ROPE_SPAD_NSLOTS)

నిర్మాణం htp_rope_kernel_params {
    uint32_t n_దారాలు;
    uint32_t src0_nrows;
    uint32_t src0_nrows_చొప్పున_thread;
    uint32_t vtcm_కొలత;
    uint32_t spad_చొప్పున_thread;
    uint32_t theta_cache_స్థానభ్రంశం;
    uint32_t src0_అడ్డువరుస_కొలత_aligned;
    uint32_t freq_factors_స్థానభ్రంశం;
    uint32_t freq_factors_కొలత;

    నిర్మాణం fastdiv_విలువలు div_ne2_ne1;
    నిర్మాణం fastdiv_విలువలు div_ne1;
};

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#లేకపోతే
_Static_assert(పరిమాణం(నిర్మాణం htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#అయితే_ముగింపు

నిర్మాణం htp_rope_vtcm_layout {
    size_t మొత్తం_బైట్లు;
    size_t బైట్లు_చొప్పున_thread;
    size_t theta_cache_కొలత_aligned;
    size_t src0_అడ్డువరుస_కొలత_aligned;
    size_t freq_factors_కొలత_aligned;
};

స్థిర అంతఃవరుస శూన్య htp_rope_vtcm_layout_build(
    నిర్మాణం htp_rope_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t n_దారాలు,
    uint32_t n_freq_factors
) {
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత             = ne00 * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_aligned     = hex_విడత_పైకి((uint32_t) src0_అడ్డువరుస_కొలత, 128);
    స్థిరరాశి size_t theta_cache_కొలత_aligned  = hex_విడత_పైకి((uint32_t) src0_అడ్డువరుస_కొలత, 256);
    స్థిరరాశి size_t freq_factors_కొలత_aligned = hex_విడత_పైకి(n_freq_factors * పరిమాణం(దశాంశం), 256);

    layout->src0_అడ్డువరుస_కొలత_aligned     = src0_అడ్డువరుస_కొలత_aligned;
    layout->theta_cache_కొలత_aligned  = theta_cache_కొలత_aligned;
    layout->freq_factors_కొలత_aligned = freq_factors_కొలత_aligned;
    layout->బైట్లు_చొప్పున_thread          = theta_cache_కొలత_aligned + HTP_ROPE_SPAD_NROWS * src0_అడ్డువరుస_కొలత_aligned;
    layout->మొత్తం_బైట్లు               = layout->బైట్లు_చొప్పున_thread * n_దారాలు + freq_factors_కొలత_aligned;
}

స్థిర అంతఃవరుస uint8_t * rope_spad_స్థానం(uint8_t * పునాది, uint32_t స్థానం, size_t అడ్డువరుస_కొలత_aligned) {
    తిరిగి పునాది + (స్థానం * HTP_ROPE_SPAD_ఖండం) * అడ్డువరుస_కొలత_aligned;
}

#అయితే_ముగింపు // HTP_ROPE_OPS_H
