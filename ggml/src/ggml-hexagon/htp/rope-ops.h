#اگر_غیر_تعریف_شدہ HTP_ROPE_OPS_H
#تعریف_کرو HTP_ROPE_OPS_H

#شامل "hex-common.h"
#شامل "hex-fastdiv.h"

#تعریف_کرو HTP_ROPE_SPAD_بلاک  8
#تعریف_کرو HTP_ROPE_SPAD_NSLOTS 4
#تعریف_کرو HTP_ROPE_SPAD_NROWS  (HTP_ROPE_SPAD_بلاک * HTP_ROPE_SPAD_NSLOTS)

ساخت htp_rope_kernel_params {
    uint32_t n_دھاگے;
    uint32_t src0_nrows;
    uint32_t src0_nrows_فی_thread;
    uint32_t vtcm_سائز;
    uint32_t spad_فی_thread;
    uint32_t theta_cache_انحراف;
    uint32_t src0_صف_سائز_aligned;
    uint32_t freq_factors_انحراف;
    uint32_t freq_factors_سائز;

    ساخت fastdiv_اقدار div_ne2_ne1;
    ساخت fastdiv_اقدار div_ne1;
};

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#ورنہ
_Static_assert(حجم(ساخت htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#اگر_ختم

ساخت htp_rope_vtcm_layout {
    size_t کل_بائٹیں;
    size_t بائٹیں_فی_thread;
    size_t theta_cache_سائز_aligned;
    size_t src0_صف_سائز_aligned;
    size_t freq_factors_سائز_aligned;
};

ساکن اندرونی خالی htp_rope_vtcm_layout_build(
    ساخت htp_rope_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t n_دھاگے,
    uint32_t n_freq_factors
) {
    ثابت size_t src0_صف_سائز             = ne00 * حجم(اعشاری);
    ثابت size_t src0_صف_سائز_aligned     = hex_باری_اوپر((uint32_t) src0_صف_سائز, 128);
    ثابت size_t theta_cache_سائز_aligned  = hex_باری_اوپر((uint32_t) src0_صف_سائز, 256);
    ثابت size_t freq_factors_سائز_aligned = hex_باری_اوپر(n_freq_factors * حجم(اعشاری), 256);

    layout->src0_صف_سائز_aligned     = src0_صف_سائز_aligned;
    layout->theta_cache_سائز_aligned  = theta_cache_سائز_aligned;
    layout->freq_factors_سائز_aligned = freq_factors_سائز_aligned;
    layout->بائٹیں_فی_thread          = theta_cache_سائز_aligned + HTP_ROPE_SPAD_NROWS * src0_صف_سائز_aligned;
    layout->کل_بائٹیں               = layout->بائٹیں_فی_thread * n_دھاگے + freq_factors_سائز_aligned;
}

ساکن اندرونی uint8_t * rope_spad_خانہ(uint8_t * بنیاد, uint32_t خانہ, size_t صف_سائز_aligned) {
    واپس بنیاد + (خانہ * HTP_ROPE_SPAD_بلاک) * صف_سائز_aligned;
}

#اگر_ختم // HTP_ROPE_OPS_H
