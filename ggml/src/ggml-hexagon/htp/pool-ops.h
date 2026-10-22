#اگر_غیر_تعریف_شدہ HTP_POOL_OPS_H
#تعریف_کرو HTP_POOL_OPS_H

#شامل <stdint.h>
#شامل <stddef.h>
#شامل <stdbool.h>
#شامل <string.h>

#شامل "hex-common.h"

ساخت htp_pool_2d_kernel_params {
    uint32_t src_x;
    uint32_t src_y;
    uint32_t dst_x;
    uint32_t dst_y;
    uint32_t kernel_x;
    uint32_t kernel_y;
    uint32_t stride_x;
    uint32_t stride_y;
    int32_t  pad_x;
    int32_t  pad_y;
    uint32_t src_plane_بائٹیں;
    uint32_t dst_plane_بائٹیں;
    uint32_t src_plane_بائٹیں_aligned;
    uint32_t dst_plane_بائٹیں_aligned;
    uint32_t n_دھاگے;
    uint32_t planes;
    uint32_t pool_op;
    uint32_t fast_راستہ;
    uint32_t narrow_راستہ;
    uint32_t global_راستہ;
    uint32_t بلاک_راستہ;
    uint32_t avg_divide_گنتی;
    uint32_t ox_lo;
    uint32_t ox_hi;
    اعشاری    inv_kernel_area;
};

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#ورنہ
_Static_assert(حجم(ساخت htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#اگر_ختم

ساخت htp_pool_vtcm_layout {
    size_t   کل_بائٹیں;
    size_t   off_src;
    size_t   off_dst;
    size_t   src_بائٹیں_فی_thread;
    size_t   dst_بائٹیں_فی_thread;
    size_t   src_spad_half_سائز;
    size_t   dst_spad_half_سائز;
};

ساکن اندرونی bool htp_pool_solve_layout(
    ساخت htp_pool_vtcm_layout * layout,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t n_دھاگے,
    size_t   vtcm_budget
) {
    // Full-plane double buffering (256 bytes guard space for vector loads)
    ثابت size_t src_plane_بائٹیں   = (size_t) src_x * src_y * حجم(اعشاری);
    ثابت size_t dst_plane_بائٹیں   = (size_t) dst_x * dst_y * حجم(اعشاری);
    ثابت size_t src_plane_aligned = hex_باری_اوپر((uint32_t) src_plane_بائٹیں + 256, 128);
    ثابت size_t dst_plane_aligned = hex_باری_اوپر((uint32_t) dst_plane_بائٹیں, 128);

    ثابت size_t spad_فی_thread = 2 * (src_plane_aligned + dst_plane_aligned);
    اگر (spad_فی_thread * n_دھاگے > vtcm_budget) {
        واپس false;
    }

    layout->src_spad_half_سائز   = src_plane_aligned;
    layout->dst_spad_half_سائز   = dst_plane_aligned;
    layout->src_بائٹیں_فی_thread = 2 * src_plane_aligned;
    layout->dst_بائٹیں_فی_thread = 2 * dst_plane_aligned;
    layout->off_src              = 0;
    layout->off_dst              = layout->src_بائٹیں_فی_thread * n_دھاگے;
    layout->کل_بائٹیں          = layout->off_dst + layout->dst_بائٹیں_فی_thread * n_دھاگے;
    واپس true;
}

#اگر_ختم // HTP_POOL_OPS_H
