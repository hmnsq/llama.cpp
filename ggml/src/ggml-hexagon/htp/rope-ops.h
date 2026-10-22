#جيڪڏهن_وصف_نه_ٿيل HTP_ROPE_OPS_H
#وصف_ڪريو HTP_ROPE_OPS_H

#ضمن "hex-common.h"
#ضمن "hex-fastdiv.h"

#وصف_ڪريو HTP_ROPE_SPAD_بلاڪ  8
#وصف_ڪريو HTP_ROPE_SPAD_NSLOTS 4
#وصف_ڪريو HTP_ROPE_SPAD_NROWS  (HTP_ROPE_SPAD_بلاڪ * HTP_ROPE_SPAD_NSLOTS)

بنية htp_rope_kernel_params {
    uint32_t n_ڌاڳا;
    uint32_t src0_nrows;
    uint32_t src0_nrows_في_thread;
    uint32_t vtcm_سائيز;
    uint32_t spad_في_thread;
    uint32_t theta_cache_وٿي;
    uint32_t src0_صف_سائيز_aligned;
    uint32_t freq_factors_وٿي;
    uint32_t freq_factors_سائيز;

    بنية fastdiv_قدرون div_ne2_ne1;
    بنية fastdiv_قدرون div_ne1;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#جيڪڏهن_پورو

بنية htp_rope_vtcm_layout {
    size_t ڪل_بائيٽون;
    size_t بائيٽون_في_thread;
    size_t theta_cache_سائيز_aligned;
    size_t src0_صف_سائيز_aligned;
    size_t freq_factors_سائيز_aligned;
};

ساكن مضمن فراغ htp_rope_vtcm_layout_build(
    بنية htp_rope_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t n_ڌاڳا,
    uint32_t n_freq_factors
) {
    ثابت size_t src0_صف_سائيز             = ne00 * حجم(عائم);
    ثابت size_t src0_صف_سائيز_aligned     = hex_ڦيرو_مٿي((uint32_t) src0_صف_سائيز, 128);
    ثابت size_t theta_cache_سائيز_aligned  = hex_ڦيرو_مٿي((uint32_t) src0_صف_سائيز, 256);
    ثابت size_t freq_factors_سائيز_aligned = hex_ڦيرو_مٿي(n_freq_factors * حجم(عائم), 256);

    layout->src0_صف_سائيز_aligned     = src0_صف_سائيز_aligned;
    layout->theta_cache_سائيز_aligned  = theta_cache_سائيز_aligned;
    layout->freq_factors_سائيز_aligned = freq_factors_سائيز_aligned;
    layout->بائيٽون_في_thread          = theta_cache_سائيز_aligned + HTP_ROPE_SPAD_NROWS * src0_صف_سائيز_aligned;
    layout->ڪل_بائيٽون               = layout->بائيٽون_في_thread * n_ڌاڳا + freq_factors_سائيز_aligned;
}

ساكن مضمن uint8_t * rope_spad_جڳهه(uint8_t * بنياد, uint32_t جڳهه, size_t صف_سائيز_aligned) {
    أرجع بنياد + (جڳهه * HTP_ROPE_SPAD_بلاڪ) * صف_سائيز_aligned;
}

#جيڪڏهن_پورو // HTP_ROPE_OPS_H
