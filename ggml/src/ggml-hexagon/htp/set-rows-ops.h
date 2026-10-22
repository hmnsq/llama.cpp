#اگر_تعریف_نشده HTP_مجموعه_ردیفها_OPS_H
#تعریف_کن HTP_مجموعه_ردیفها_OPS_H

#ضمن "hex-fastdiv.h"

بنية htp_مجموعه_ردیفها_kernel_params {
    int32_t  n_نخها;
    int32_t  مجموع_tasks;
    int32_t  tasks_در_هر_thread;
    int32_t  vtcm_اندازه;

    // Fastdiv helpers
    بنية fastdiv_مقادیر div_ne11;
    بنية fastdiv_مقادیر div_ne12;
    بنية fastdiv_مقادیر div_tasks_در_هر_thread;
    بنية fastdiv_مقادیر div_ne02;
};

بنية htp_مجموعه_ردیفها_vtcm_layout {
    size_t مجموع_بایتها;
    size_t off_src0;
    size_t off_dst;

    size_t src0_بایتها_در_هر_thread;
    size_t dst_بایتها_در_هر_thread;

    size_t src0_spad_half_اندازه;
    size_t dst_spad_half_اندازه;
};

ساكن مضمن فراغ htp_مجموعه_ردیفها_vtcm_layout_build(
    بنية htp_مجموعه_ردیفها_vtcm_layout * vtcm_layout,
    صحيح dst_type,
    uint32_t ne00,
    uint32_t n_نخها) {

    size_t src0_ردیف_اندازه = ne00 * 4;
    size_t dst_ردیف_اندازه = 0;
    بدل (dst_type) {
        حالة 0: // HTP_TYPE_F32
            dst_ردیف_اندازه = ne00 * 4;
            اكسر;
        حالة 1: // HTP_TYPE_F16
            dst_ردیف_اندازه = ne00 * 2;
            اكسر;
        حالة 8: // HTP_TYPE_Q8_0
            dst_ردیف_اندازه = (ne00 / 32) * 34;
            اكسر;
        افتراضي:
            dst_ردیف_اندازه = 0;
            اكسر;
    }

    size_t src0_ردیف_اندازه_aligned = (src0_ردیف_اندازه + 255) & ~255;
    size_t dst_ردیف_اندازه_aligned  = (dst_ردیف_اندازه + 255) & ~255;

    vtcm_layout->src0_spad_half_اندازه = src0_ردیف_اندازه_aligned;
    vtcm_layout->dst_spad_half_اندازه  = dst_ردیف_اندازه_aligned;

    vtcm_layout->src0_بایتها_در_هر_thread = src0_ردیف_اندازه_aligned * 2;
    vtcm_layout->dst_بایتها_در_هر_thread  = dst_ردیف_اندازه_aligned * 2;

    vtcm_layout->off_src0 = 0;
    vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_بایتها_در_هر_thread * n_نخها;
    vtcm_layout->مجموع_بایتها = vtcm_layout->off_dst + vtcm_layout->dst_بایتها_در_هر_thread * n_نخها;
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_مجموعه_ردیفها_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_مجموعه_ردیفها_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#پایان_اگر

#پایان_اگر // HTP_SET_ROWS_OPS_H
