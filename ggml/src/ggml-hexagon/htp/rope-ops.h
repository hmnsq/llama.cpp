#جے_بنا_تعریف HTP_ROPE_OPS_H
#تعریف_کرو HTP_ROPE_OPS_H

#ضمن "hex-common.h"
#ضمن "hex-fastdiv.h"

#تعریف_کرو HTP_ROPE_SPAD_بلاک  8
#تعریف_کرو HTP_ROPE_SPAD_NSLOTS 4
#تعریف_کرو HTP_ROPE_SPAD_NROWS  (HTP_ROPE_SPAD_بلاک * HTP_ROPE_SPAD_NSLOTS)

بنية htp_rope_kernel_params {
    uint32_t n_دھاگے;
    uint32_t src0_nrows;
    uint32_t src0_nrows_فی_thread;
    uint32_t vtcm_ناپ;
    uint32_t spad_فی_thread;
    uint32_t theta_cache_وتھ;
    uint32_t src0_صف_ناپ_aligned;
    uint32_t freq_factors_وتھ;
    uint32_t freq_factors_ناپ;

    بنية fastdiv_اقدار div_ne2_ne1;
    بنية fastdiv_اقدار div_ne1;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#جے_انت

بنية htp_rope_vtcm_layout {
    size_t کل_بائٹاں;
    size_t بائٹاں_فی_thread;
    size_t theta_cache_ناپ_aligned;
    size_t src0_صف_ناپ_aligned;
    size_t freq_factors_ناپ_aligned;
};

ساكن مضمن فراغ htp_rope_vtcm_layout_build(
    بنية htp_rope_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t n_دھاگے,
    uint32_t n_freq_factors
) {
    ثابت size_t src0_صف_ناپ             = ne00 * حجم(عائم);
    ثابت size_t src0_صف_ناپ_aligned     = hex_گیڑا_اتے((uint32_t) src0_صف_ناپ, 128);
    ثابت size_t theta_cache_ناپ_aligned  = hex_گیڑا_اتے((uint32_t) src0_صف_ناپ, 256);
    ثابت size_t freq_factors_ناپ_aligned = hex_گیڑا_اتے(n_freq_factors * حجم(عائم), 256);

    layout->src0_صف_ناپ_aligned     = src0_صف_ناپ_aligned;
    layout->theta_cache_ناپ_aligned  = theta_cache_ناپ_aligned;
    layout->freq_factors_ناپ_aligned = freq_factors_ناپ_aligned;
    layout->بائٹاں_فی_thread          = theta_cache_ناپ_aligned + HTP_ROPE_SPAD_NROWS * src0_صف_ناپ_aligned;
    layout->کل_بائٹاں               = layout->بائٹاں_فی_thread * n_دھاگے + freq_factors_ناپ_aligned;
}

ساكن مضمن uint8_t * rope_spad_تھاں(uint8_t * ادھار, uint32_t تھاں, size_t صف_ناپ_aligned) {
    أرجع ادھار + (تھاں * HTP_ROPE_SPAD_بلاک) * صف_ناپ_aligned;
}

#جے_انت // HTP_ROPE_OPS_H
