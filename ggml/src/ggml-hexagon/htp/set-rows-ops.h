#جيڪڏهن_وصف_نه_ٿيل HTP_سيٽ_صفون_OPS_H
#وصف_ڪريو HTP_سيٽ_صفون_OPS_H

#ضمن "hex-fastdiv.h"

بنية htp_سيٽ_صفون_kernel_params {
    int32_t  n_ڌاڳا;
    int32_t  ڪل_tasks;
    int32_t  tasks_في_thread;
    int32_t  vtcm_سائيز;

    // Fastdiv helpers
    بنية fastdiv_قدرون div_ne11;
    بنية fastdiv_قدرون div_ne12;
    بنية fastdiv_قدرون div_tasks_في_thread;
    بنية fastdiv_قدرون div_ne02;
};

بنية htp_سيٽ_صفون_vtcm_layout {
    size_t ڪل_بائيٽون;
    size_t off_src0;
    size_t off_dst;

    size_t src0_بائيٽون_في_thread;
    size_t dst_بائيٽون_في_thread;

    size_t src0_spad_half_سائيز;
    size_t dst_spad_half_سائيز;
};

ساكن مضمن فراغ htp_سيٽ_صفون_vtcm_layout_build(
    بنية htp_سيٽ_صفون_vtcm_layout * vtcm_layout,
    صحيح dst_type,
    uint32_t ne00,
    uint32_t n_ڌاڳا) {

    size_t src0_صف_سائيز = ne00 * 4;
    size_t dst_صف_سائيز = 0;
    بدل (dst_type) {
        حالة 0: // HTP_TYPE_F32
            dst_صف_سائيز = ne00 * 4;
            اكسر;
        حالة 1: // HTP_TYPE_F16
            dst_صف_سائيز = ne00 * 2;
            اكسر;
        حالة 8: // HTP_TYPE_Q8_0
            dst_صف_سائيز = (ne00 / 32) * 34;
            اكسر;
        افتراضي:
            dst_صف_سائيز = 0;
            اكسر;
    }

    size_t src0_صف_سائيز_aligned = (src0_صف_سائيز + 255) & ~255;
    size_t dst_صف_سائيز_aligned  = (dst_صف_سائيز + 255) & ~255;

    vtcm_layout->src0_spad_half_سائيز = src0_صف_سائيز_aligned;
    vtcm_layout->dst_spad_half_سائيز  = dst_صف_سائيز_aligned;

    vtcm_layout->src0_بائيٽون_في_thread = src0_صف_سائيز_aligned * 2;
    vtcm_layout->dst_بائيٽون_في_thread  = dst_صف_سائيز_aligned * 2;

    vtcm_layout->off_src0 = 0;
    vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_بائيٽون_في_thread * n_ڌاڳا;
    vtcm_layout->ڪل_بائيٽون = vtcm_layout->off_dst + vtcm_layout->dst_بائيٽون_في_thread * n_ڌاڳا;
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_سيٽ_صفون_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_سيٽ_صفون_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#جيڪڏهن_پورو

#جيڪڏهن_پورو // HTP_SET_ROWS_OPS_H
