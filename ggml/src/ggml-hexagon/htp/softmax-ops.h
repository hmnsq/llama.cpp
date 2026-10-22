#اگر_تعریف_نشده HTP_SOFTMAX_OPS_H
#تعریف_کن HTP_SOFTMAX_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>
#ضمن <math.h>
#ضمن "hex-fastdiv.h"
#ضمن "hex-common.h"

تعداد htp_softmax_kernel_id {
    HTP_SOFTMAX_KERNEL_NOMASK = 0,
    HTP_SOFTMAX_KERNEL_ماسک_F32,
    HTP_SOFTMAX_KERNEL_ماسک_F16,
    HTP_SOFTMAX_KERNEL_شمار,
};

بنية htp_softmax_kernel_params {
    uint32_t n_نخها;
    uint32_t src0_nrows;
    uint32_t src0_nrows_در_هر_thread;
    uint32_t vtcm_اندازه;

    uint32_t vtcm_src0_اندازه_در_هر_thread;
    uint32_t vtcm_src1_اندازه_در_هر_thread;
    uint32_t vtcm_dst_اندازه_در_هر_thread;

    uint32_t src0_ردیف_اندازه_aligned;
    uint32_t src1_ردیف_اندازه_aligned;
    uint32_t dst_ردیف_اندازه_aligned;

    uint32_t src0_spad_half_اندازه;
    uint32_t src1_spad_half_اندازه;
    uint32_t dst_spad_half_اندازه;

    uint32_t n_سر;
    uint32_t n_سر_log2;
    uint32_t use_src1;
    uint32_t use_f16;
    uint32_t kernel_id;

    عائم    scale;
    عائم    max_bias;
    عائم    m0;
    عائم    m1;

    بنية fastdiv_مقادیر div_ne01;
    بنية fastdiv_مقادیر div_ne02;
    بنية fastdiv_مقادیر div_ne12;
    بنية fastdiv_مقادیر div_ne13;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_softmax_kernel_params) <= 128, "htp_softmax_kernel_params is too large for kernel_params blob");
#پایان_اگر

بنية htp_softmax_vtcm_layout {
    size_t مجموع_بایتها;
    size_t off_src0;
    size_t off_dst;
    size_t off_src1;

    size_t src0_بایتها_در_هر_thread;
    size_t dst_بایتها_در_هر_thread;
    size_t src1_بایتها_در_هر_thread;

    size_t src0_spad_half_اندازه;
    size_t dst_spad_half_اندازه;
    size_t src1_spad_half_اندازه;
};

ساكن مضمن فراغ htp_softmax_vtcm_layout_build(
    بنية htp_softmax_vtcm_layout * layout,
    uint32_t ne00,
    uint32_t ne10,
    bool use_src1,
    bool use_f16,
    uint32_t n_نخها
) {
    size_t src0_ردیف_اندازه = ne00 * حجم(عائم);
    size_t dst_ردیف_اندازه  = ne00 * حجم(عائم);
    size_t src1_ردیف_اندازه = use_src1 ? (ne10 * (use_f16 ? 2 : 4)) : 0;

    size_t src0_ردیف_اندازه_aligned = hex_دور_به_بالا(src0_ردیف_اندازه, 128);
    size_t dst_ردیف_اندازه_aligned  = hex_دور_به_بالا(dst_ردیف_اندازه,  128);
    size_t src1_ردیف_اندازه_aligned = use_src1 ? hex_دور_به_بالا(src1_ردیف_اندازه, 128) : 0;

    layout->src0_spad_half_اندازه = src0_ردیف_اندازه_aligned;
    layout->dst_spad_half_اندازه  = dst_ردیف_اندازه_aligned;
    layout->src1_spad_half_اندازه = src1_ردیف_اندازه_aligned;

    // Double buffering: 2 half-buffers per thread
    layout->src0_بایتها_در_هر_thread = src0_ردیف_اندازه_aligned * 2;
    layout->dst_بایتها_در_هر_thread  = dst_ردیف_اندازه_aligned  * 2;
    layout->src1_بایتها_در_هر_thread = src1_ردیف_اندازه_aligned * 2;

    layout->off_src0 = 0;
    layout->off_dst  = layout->off_src0 + layout->src0_بایتها_در_هر_thread * n_نخها;
    layout->off_src1 = layout->off_dst  + layout->dst_بایتها_در_هر_thread  * n_نخها;

    layout->مجموع_بایتها = layout->off_src1 + layout->src1_بایتها_در_هر_thread * n_نخها;
}

#پایان_اگر // HTP_SOFTMAX_OPS_H
