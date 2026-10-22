#اگر_غیر_تعریف_شدہ HTP_SOFTMAX_OPS_H
#تعریف_کرو HTP_SOFTMAX_OPS_H

#شامل <stdint.h>
#شامل <stddef.h>
#شامل <stdbool.h>
#شامل <math.h>
#شامل "hex-fastdiv.h"
#شامل "hex-common.h"

شمار htp_softmax_kernel_id {
    HTP_SOFTMAX_KERNEL_NOMASK = 0,
    HTP_SOFTMAX_KERNEL_نقاب_F32,
    HTP_SOFTMAX_KERNEL_نقاب_F16,
    HTP_SOFTMAX_KERNEL_گنتی,
};

ساخت htp_softmax_kernel_params {
    uint32_t n_دھاگے;
    uint32_t src0_nrows;
    uint32_t src0_nrows_فی_thread;
    uint32_t vtcm_سائز;

    uint32_t vtcm_src0_سائز_فی_thread;
    uint32_t vtcm_src1_سائز_فی_thread;
    uint32_t vtcm_dst_سائز_فی_thread;

    uint32_t src0_صف_سائز_aligned;
    uint32_t src1_صف_سائز_aligned;
    uint32_t dst_صف_سائز_aligned;

    uint32_t src0_spad_half_سائز;
    uint32_t src1_spad_half_سائز;
    uint32_t dst_spad_half_سائز;

    uint32_t n_سر;
    uint32_t n_سر_log2;
    uint32_t use_src1;
    uint32_t use_f16;
    uint32_t kernel_id;

    اعشاری    scale;
    اعشاری    max_bias;
    اعشاری    m0;
    اعشاری    m1;

    ساخت fastdiv_اقدار div_ne01;
    ساخت fastdiv_اقدار div_ne02;
    ساخت fastdiv_اقدار div_ne12;
    ساخت fastdiv_اقدار div_ne13;
};

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#ورنہ
_Static_assert(حجم(ساخت htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#اگر_ختم

ساخت htp_softmax_vtcm_layout {
    size_t کل_بائٹیں;
    size_t off_src0;
    size_t off_dst;
    size_t off_src1;

    size_t src0_بائٹیں_فی_thread;
    size_t dst_بائٹیں_فی_thread;
    size_t src1_بائٹیں_فی_thread;

    size_t src0_spad_half_سائز;
    size_t dst_spad_half_سائز;
    size_t src1_spad_half_سائز;
};

ساکن اندرونی خالی htp_softmax_vtcm_layout_build(
    ساخت htp_softmax_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne10,
    bool use_src1,
    bool use_f16,
    uint32_t n_دھاگے
) {
    size_t src0_صف_سائز = ne00 * حجم(اعشاری);
    size_t dst_صف_سائز  = ne00 * حجم(اعشاری);
    size_t src1_صف_سائز = use_src1 ? (ne10 * (use_f16 ? 2 : 4)) : 0;

    size_t src0_صف_سائز_aligned = hex_باری_اوپر(src0_صف_سائز, 128);
    size_t dst_صف_سائز_aligned  = hex_باری_اوپر(dst_صف_سائز,  128);
    size_t src1_صف_سائز_aligned = use_src1 ? hex_باری_اوپر(src1_صف_سائز, 128) : 0;

    layout->src0_spad_half_سائز = src0_صف_سائز_aligned;
    layout->dst_spad_half_سائز  = dst_صف_سائز_aligned;
    layout->src1_spad_half_سائز = src1_صف_سائز_aligned;

    // Double buffering: 2 half-buffers per thread
    layout->src0_بائٹیں_فی_thread = src0_صف_سائز_aligned * 2;
    layout->dst_بائٹیں_فی_thread  = dst_صف_سائز_aligned  * 2;
    layout->src1_بائٹیں_فی_thread = src1_صف_سائز_aligned * 2;

    layout->off_src0 = 0;
    layout->off_dst  = layout->off_src0 + layout->src0_بائٹیں_فی_thread * n_دھاگے;
    layout->off_src1 = layout->off_dst  + layout->dst_بائٹیں_فی_thread  * n_دھاگے;

    layout->کل_بائٹیں = layout->off_src1 + layout->src1_بائٹیں_فی_thread * n_دھاگے;
}

#اگر_ختم // HTP_SOFTMAX_OPS_H
