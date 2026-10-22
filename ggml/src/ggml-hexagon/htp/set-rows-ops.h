#که_نه_تعریف_شوی HTP_سېټ_لیکې_OPS_H
#تعریف_کړه HTP_سېټ_لیکې_OPS_H

#ضمن "hex-fastdiv.h"

بنية htp_سېټ_لیکې_kernel_params {
    int32_t  n_تارونه;
    int32_t  ټول_tasks;
    int32_t  tasks_فی_thread;
    int32_t  vtcm_اندازه;

    // Fastdiv helpers
    بنية fastdiv_ارزښتونه div_ne11;
    بنية fastdiv_ارزښتونه div_ne12;
    بنية fastdiv_ارزښتونه div_tasks_فی_thread;
    بنية fastdiv_ارزښتونه div_ne02;
};

بنية htp_سېټ_لیکې_vtcm_layout {
    size_t ټول_بایټونه;
    size_t off_src0;
    size_t off_dst;

    size_t src0_بایټونه_فی_thread;
    size_t dst_بایټونه_فی_thread;

    size_t src0_spad_half_اندازه;
    size_t dst_spad_half_اندازه;
};

ساكن مضمن فراغ htp_سېټ_لیکې_vtcm_layout_build(
    بنية htp_سېټ_لیکې_vtcm_layout * vtcm_layout,
    صحيح dst_type,
    uint32_t ne00,
    uint32_t n_تارونه) {

    size_t src0_لیکه_اندازه = ne00 * 4;
    size_t dst_لیکه_اندازه = 0;
    بدل (dst_type) {
        حالة 0: // HTP_TYPE_F32
            dst_لیکه_اندازه = ne00 * 4;
            اكسر;
        حالة 1: // HTP_TYPE_F16
            dst_لیکه_اندازه = ne00 * 2;
            اكسر;
        حالة 8: // HTP_TYPE_Q8_0
            dst_لیکه_اندازه = (ne00 / 32) * 34;
            اكسر;
        افتراضي:
            dst_لیکه_اندازه = 0;
            اكسر;
    }

    size_t src0_لیکه_اندازه_aligned = (src0_لیکه_اندازه + 255) & ~255;
    size_t dst_لیکه_اندازه_aligned  = (dst_لیکه_اندازه + 255) & ~255;

    vtcm_layout->src0_spad_half_اندازه = src0_لیکه_اندازه_aligned;
    vtcm_layout->dst_spad_half_اندازه  = dst_لیکه_اندازه_aligned;

    vtcm_layout->src0_بایټونه_فی_thread = src0_لیکه_اندازه_aligned * 2;
    vtcm_layout->dst_بایټونه_فی_thread  = dst_لیکه_اندازه_aligned * 2;

    vtcm_layout->off_src0 = 0;
    vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_بایټونه_فی_thread * n_تارونه;
    vtcm_layout->ټول_بایټونه = vtcm_layout->off_dst + vtcm_layout->dst_بایټونه_فی_thread * n_تارونه;
}

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_سېټ_لیکې_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_سېټ_لیکې_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#د_که_پای

#د_که_پای // HTP_SET_ROWS_OPS_H
