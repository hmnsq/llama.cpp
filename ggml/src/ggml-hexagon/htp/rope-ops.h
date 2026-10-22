#که_نه_تعریف_شوی HTP_ROPE_OPS_H
#تعریف_کړه HTP_ROPE_OPS_H

#ضمن "hex-common.h"
#ضمن "hex-fastdiv.h"

#تعریف_کړه HTP_ROPE_SPAD_بلاک  8
#تعریف_کړه HTP_ROPE_SPAD_NSLOTS 4
#تعریف_کړه HTP_ROPE_SPAD_NROWS  (HTP_ROPE_SPAD_بلاک * HTP_ROPE_SPAD_NSLOTS)

بنية htp_rope_kernel_params {
    uint32_t n_تارونه;
    uint32_t src0_nrows;
    uint32_t src0_nrows_فی_thread;
    uint32_t vtcm_اندازه;
    uint32_t spad_فی_thread;
    uint32_t theta_cache_دننه_واټن;
    uint32_t src0_لیکه_اندازه_aligned;
    uint32_t freq_factors_دننه_واټن;
    uint32_t freq_factors_اندازه;

    بنية fastdiv_ارزښتونه div_ne2_ne1;
    بنية fastdiv_ارزښتونه div_ne1;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_rope_kernel_params) <= 128, "htp_rope_kernel_params is too large for kernel_params blob");
#د_که_پای

بنية htp_rope_vtcm_layout {
    size_t ټول_بایټونه;
    size_t بایټونه_فی_thread;
    size_t theta_cache_اندازه_aligned;
    size_t src0_لیکه_اندازه_aligned;
    size_t freq_factors_اندازه_aligned;
};

ساكن مضمن فراغ htp_rope_vtcm_layout_build(
    بنية htp_rope_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t n_تارونه,
    uint32_t n_freq_factors
) {
    ثابت size_t src0_لیکه_اندازه             = ne00 * حجم(عائم);
    ثابت size_t src0_لیکه_اندازه_aligned     = hex_پړاو_پورته((uint32_t) src0_لیکه_اندازه, 128);
    ثابت size_t theta_cache_اندازه_aligned  = hex_پړاو_پورته((uint32_t) src0_لیکه_اندازه, 256);
    ثابت size_t freq_factors_اندازه_aligned = hex_پړاو_پورته(n_freq_factors * حجم(عائم), 256);

    layout->src0_لیکه_اندازه_aligned     = src0_لیکه_اندازه_aligned;
    layout->theta_cache_اندازه_aligned  = theta_cache_اندازه_aligned;
    layout->freq_factors_اندازه_aligned = freq_factors_اندازه_aligned;
    layout->بایټونه_فی_thread          = theta_cache_اندازه_aligned + HTP_ROPE_SPAD_NROWS * src0_لیکه_اندازه_aligned;
    layout->ټول_بایټونه               = layout->بایټونه_فی_thread * n_تارونه + freq_factors_اندازه_aligned;
}

ساكن مضمن uint8_t * rope_spad_خونه(uint8_t * بنسټ, uint32_t خونه, size_t لیکه_اندازه_aligned) {
    أرجع بنسټ + (خونه * HTP_ROPE_SPAD_بلاک) * لیکه_اندازه_aligned;
}

#د_که_پای // HTP_ROPE_OPS_H
