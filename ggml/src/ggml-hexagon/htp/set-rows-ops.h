#جے_بنا_تعریف HTP_سیٹ_صفاں_OPS_H
#تعریف_کرو HTP_سیٹ_صفاں_OPS_H

#ضمن "hex-fastdiv.h"

بنية htp_سیٹ_صفاں_kernel_params {
    int32_t  n_دھاگے;
    int32_t  کل_tasks;
    int32_t  tasks_فی_thread;
    int32_t  vtcm_ناپ;

    // Fastdiv helpers
    بنية fastdiv_اقدار div_ne11;
    بنية fastdiv_اقدار div_ne12;
    بنية fastdiv_اقدار div_tasks_فی_thread;
    بنية fastdiv_اقدار div_ne02;
};

بنية htp_سیٹ_صفاں_vtcm_layout {
    size_t کل_بائٹاں;
    size_t off_src0;
    size_t off_dst;

    size_t src0_بائٹاں_فی_thread;
    size_t dst_بائٹاں_فی_thread;

    size_t src0_spad_half_ناپ;
    size_t dst_spad_half_ناپ;
};

ساكن مضمن فراغ htp_سیٹ_صفاں_vtcm_layout_build(
    بنية htp_سیٹ_صفاں_vtcm_layout * vtcm_layout,
    صحيح dst_type,
    uint32_t ne00,
    uint32_t n_دھاگے) {

    size_t src0_صف_ناپ = ne00 * 4;
    size_t dst_صف_ناپ = 0;
    بدل (dst_type) {
        حالة 0: // HTP_TYPE_F32
            dst_صف_ناپ = ne00 * 4;
            اكسر;
        حالة 1: // HTP_TYPE_F16
            dst_صف_ناپ = ne00 * 2;
            اكسر;
        حالة 8: // HTP_TYPE_Q8_0
            dst_صف_ناپ = (ne00 / 32) * 34;
            اكسر;
        افتراضي:
            dst_صف_ناپ = 0;
            اكسر;
    }

    size_t src0_صف_ناپ_aligned = (src0_صف_ناپ + 255) & ~255;
    size_t dst_صف_ناپ_aligned  = (dst_صف_ناپ + 255) & ~255;

    vtcm_layout->src0_spad_half_ناپ = src0_صف_ناپ_aligned;
    vtcm_layout->dst_spad_half_ناپ  = dst_صف_ناپ_aligned;

    vtcm_layout->src0_بائٹاں_فی_thread = src0_صف_ناپ_aligned * 2;
    vtcm_layout->dst_بائٹاں_فی_thread  = dst_صف_ناپ_aligned * 2;

    vtcm_layout->off_src0 = 0;
    vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_بائٹاں_فی_thread * n_دھاگے;
    vtcm_layout->کل_بائٹاں = vtcm_layout->off_dst + vtcm_layout->dst_بائٹاں_فی_thread * n_دھاگے;
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_سیٹ_صفاں_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_سیٹ_صفاں_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#جے_انت

#جے_انت // HTP_SET_ROWS_OPS_H
