#اگر_غیر_تعریف_شدہ HTP_سیٹ_صفیں_OPS_H
#تعریف_کرو HTP_سیٹ_صفیں_OPS_H

#شامل "hex-fastdiv.h"

ساخت htp_سیٹ_صفیں_kernel_params {
    int32_t  n_دھاگے;
    int32_t  کل_tasks;
    int32_t  tasks_فی_thread;
    int32_t  vtcm_سائز;

    // Fastdiv helpers
    ساخت fastdiv_اقدار div_ne11;
    ساخت fastdiv_اقدار div_ne12;
    ساخت fastdiv_اقدار div_tasks_فی_thread;
    ساخت fastdiv_اقدار div_ne02;
};

ساخت htp_سیٹ_صفیں_vtcm_layout {
    size_t کل_بائٹیں;
    size_t off_src0;
    size_t off_dst;

    size_t src0_بائٹیں_فی_thread;
    size_t dst_بائٹیں_فی_thread;

    size_t src0_spad_half_سائز;
    size_t dst_spad_half_سائز;
};

ساکن اندرونی خالی htp_سیٹ_صفیں_vtcm_layout_build(
    ساخت htp_سیٹ_صفیں_vtcm_layout * vtcm_layout,
    صحیح dst_type,
    uint32_t ne00,
    uint32_t n_دھاگے) {

    size_t src0_صف_سائز = ne00 * 4;
    size_t dst_صف_سائز = 0;
    انتخاب (dst_type) {
        حالت 0: // HTP_TYPE_F32
            dst_صف_سائز = ne00 * 4;
            توڑو;
        حالت 1: // HTP_TYPE_F16
            dst_صف_سائز = ne00 * 2;
            توڑو;
        حالت 8: // HTP_TYPE_Q8_0
            dst_صف_سائز = (ne00 / 32) * 34;
            توڑو;
        طےشدہ:
            dst_صف_سائز = 0;
            توڑو;
    }

    size_t src0_صف_سائز_aligned = (src0_صف_سائز + 255) & ~255;
    size_t dst_صف_سائز_aligned  = (dst_صف_سائز + 255) & ~255;

    vtcm_layout->src0_spad_half_سائز = src0_صف_سائز_aligned;
    vtcm_layout->dst_spad_half_سائز  = dst_صف_سائز_aligned;

    vtcm_layout->src0_بائٹیں_فی_thread = src0_صف_سائز_aligned * 2;
    vtcm_layout->dst_بائٹیں_فی_thread  = dst_صف_سائز_aligned * 2;

    vtcm_layout->off_src0 = 0;
    vtcm_layout->off_dst  = vtcm_layout->off_src0 + vtcm_layout->src0_بائٹیں_فی_thread * n_دھاگے;
    vtcm_layout->کل_بائٹیں = vtcm_layout->off_dst + vtcm_layout->dst_بائٹیں_فی_thread * n_دھاگے;
}

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_سیٹ_صفیں_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#ورنہ
_Static_assert(حجم(ساخت htp_سیٹ_صفیں_kernel_params) <= 128, "htp_set_rows_kernel_params is too large for kernel_params blob");
#اگر_ختم

#اگر_ختم // HTP_SET_ROWS_OPS_H
