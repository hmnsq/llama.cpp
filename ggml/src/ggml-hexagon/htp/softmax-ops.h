#جيڪڏهن_وصف_نه_ٿيل HTP_SOFTMAX_OPS_H
#وصف_ڪريو HTP_SOFTMAX_OPS_H

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
    HTP_SOFTMAX_KERNEL_ڳڻپ,
};

بنية htp_softmax_kernel_params {
    uint32_t n_ڌاڳا;
    uint32_t src0_nrows;
    uint32_t src0_nrows_في_thread;
    uint32_t vtcm_سائيز;

    uint32_t vtcm_src0_سائيز_في_thread;
    uint32_t vtcm_src1_سائيز_في_thread;
    uint32_t vtcm_dst_سائيز_في_thread;

    uint32_t src0_صف_سائيز_aligned;
    uint32_t src1_صف_سائيز_aligned;
    uint32_t dst_صف_سائيز_aligned;

    uint32_t src0_spad_half_سائيز;
    uint32_t src1_spad_half_سائيز;
    uint32_t dst_spad_half_سائيز;

    uint32_t n_سر;
    uint32_t n_سر_log2;
    uint32_t use_src1;
    uint32_t use_f16;
    uint32_t kernel_id;

    عائم    scale;
    عائم    max_bias;
    عائم    m0;
    عائم    m1;

    بنية fastdiv_قدرون div_ne01;
    بنية fastdiv_قدرون div_ne02;
    بنية fastdiv_قدرون div_ne12;
    بنية fastdiv_قدرون div_ne13;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#جيڪڏهن_پورو

بنية htp_softmax_vtcm_layout {
    size_t ڪل_بائيٽون;
    size_t off_src0;
    size_t off_dst;
    size_t off_src1;

    size_t src0_بائيٽون_في_thread;
    size_t dst_بائيٽون_في_thread;
    size_t src1_بائيٽون_في_thread;

    size_t src0_spad_half_سائيز;
    size_t dst_spad_half_سائيز;
    size_t src1_spad_half_سائيز;
};

ساكن مضمن فراغ htp_softmax_vtcm_layout_build(
    بنية htp_softmax_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne10,
    bool use_src1,
    bool use_f16,
    uint32_t n_ڌاڳا
) {
    size_t src0_صف_سائيز = ne00 * حجم(عائم);
    size_t dst_صف_سائيز  = ne00 * حجم(عائم);
    size_t src1_صف_سائيز = use_src1 ? (ne10 * (use_f16 ? 2 : 4)) : 0;

    size_t src0_صف_سائيز_aligned = hex_ڦيرو_مٿي(src0_صف_سائيز, 128);
    size_t dst_صف_سائيز_aligned  = hex_ڦيرو_مٿي(dst_صف_سائيز,  128);
    size_t src1_صف_سائيز_aligned = use_src1 ? hex_ڦيرو_مٿي(src1_صف_سائيز, 128) : 0;

    layout->src0_spad_half_سائيز = src0_صف_سائيز_aligned;
    layout->dst_spad_half_سائيز  = dst_صف_سائيز_aligned;
    layout->src1_spad_half_سائيز = src1_صف_سائيز_aligned;

    // Double buffering: 2 half-buffers per thread
    layout->src0_بائيٽون_في_thread = src0_صف_سائيز_aligned * 2;
    layout->dst_بائيٽون_في_thread  = dst_صف_سائيز_aligned  * 2;
    layout->src1_بائيٽون_في_thread = src1_صف_سائيز_aligned * 2;

    layout->off_src0 = 0;
    layout->off_dst  = layout->off_src0 + layout->src0_بائيٽون_في_thread * n_ڌاڳا;
    layout->off_src1 = layout->off_dst  + layout->dst_بائيٽون_في_thread  * n_ڌاڳا;

    layout->ڪل_بائيٽون = layout->off_src1 + layout->src1_بائيٽون_في_thread * n_ڌاڳا;
}

#جيڪڏهن_پورو // HTP_SOFTMAX_OPS_H
