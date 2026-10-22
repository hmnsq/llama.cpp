#جے_بنا_تعریف HTP_SOFTMAX_OPS_H
#تعریف_کرو HTP_SOFTMAX_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>
#ضمن <math.h>
#ضمن "hex-fastdiv.h"
#ضمن "hex-common.h"

تعداد htp_softmax_kernel_id {
    HTP_SOFTMAX_KERNEL_NOMASK = 0,
    HTP_SOFTMAX_KERNEL_نقاب_F32,
    HTP_SOFTMAX_KERNEL_نقاب_F16,
    HTP_SOFTMAX_KERNEL_گنتی,
};

بنية htp_softmax_kernel_params {
    uint32_t n_دھاگے;
    uint32_t src0_nrows;
    uint32_t src0_nrows_فی_thread;
    uint32_t vtcm_ناپ;

    uint32_t vtcm_src0_ناپ_فی_thread;
    uint32_t vtcm_src1_ناپ_فی_thread;
    uint32_t vtcm_dst_ناپ_فی_thread;

    uint32_t src0_صف_ناپ_aligned;
    uint32_t src1_صف_ناپ_aligned;
    uint32_t dst_صف_ناپ_aligned;

    uint32_t src0_spad_half_ناپ;
    uint32_t src1_spad_half_ناپ;
    uint32_t dst_spad_half_ناپ;

    uint32_t n_سر;
    uint32_t n_سر_log2;
    uint32_t use_src1;
    uint32_t use_f16;
    uint32_t kernel_id;

    عائم    scale;
    عائم    max_bias;
    عائم    m0;
    عائم    m1;

    بنية fastdiv_اقدار div_ne01;
    بنية fastdiv_اقدار div_ne02;
    بنية fastdiv_اقدار div_ne12;
    بنية fastdiv_اقدار div_ne13;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#جے_انت

بنية htp_softmax_vtcm_layout {
    size_t کل_بائٹاں;
    size_t off_src0;
    size_t off_dst;
    size_t off_src1;

    size_t src0_بائٹاں_فی_thread;
    size_t dst_بائٹاں_فی_thread;
    size_t src1_بائٹاں_فی_thread;

    size_t src0_spad_half_ناپ;
    size_t dst_spad_half_ناپ;
    size_t src1_spad_half_ناپ;
};

ساكن مضمن فراغ htp_softmax_vtcm_layout_build(
    بنية htp_softmax_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne10,
    bool use_src1,
    bool use_f16,
    uint32_t n_دھاگے
) {
    size_t src0_صف_ناپ = ne00 * حجم(عائم);
    size_t dst_صف_ناپ  = ne00 * حجم(عائم);
    size_t src1_صف_ناپ = use_src1 ? (ne10 * (use_f16 ? 2 : 4)) : 0;

    size_t src0_صف_ناپ_aligned = hex_گیڑا_اتے(src0_صف_ناپ, 128);
    size_t dst_صف_ناپ_aligned  = hex_گیڑا_اتے(dst_صف_ناپ,  128);
    size_t src1_صف_ناپ_aligned = use_src1 ? hex_گیڑا_اتے(src1_صف_ناپ, 128) : 0;

    layout->src0_spad_half_ناپ = src0_صف_ناپ_aligned;
    layout->dst_spad_half_ناپ  = dst_صف_ناپ_aligned;
    layout->src1_spad_half_ناپ = src1_صف_ناپ_aligned;

    // Double buffering: 2 half-buffers per thread
    layout->src0_بائٹاں_فی_thread = src0_صف_ناپ_aligned * 2;
    layout->dst_بائٹاں_فی_thread  = dst_صف_ناپ_aligned  * 2;
    layout->src1_بائٹاں_فی_thread = src1_صف_ناپ_aligned * 2;

    layout->off_src0 = 0;
    layout->off_dst  = layout->off_src0 + layout->src0_بائٹاں_فی_thread * n_دھاگے;
    layout->off_src1 = layout->off_dst  + layout->dst_بائٹاں_فی_thread  * n_دھاگے;

    layout->کل_بائٹاں = layout->off_src1 + layout->src1_بائٹاں_فی_thread * n_دھاگے;
}

#جے_انت // HTP_SOFTMAX_OPS_H
