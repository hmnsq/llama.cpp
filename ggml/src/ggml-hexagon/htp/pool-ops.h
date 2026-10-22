#اگر_تعریف_نشده HTP_POOL_OPS_H
#تعریف_کن HTP_POOL_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>
#ضمن <string.h>

#ضمن "hex-common.h"

بنية htp_pool_2d_kernel_params {
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
    uint32_t src_plane_بایتها;
    uint32_t dst_plane_بایتها;
    uint32_t src_plane_بایتها_aligned;
    uint32_t dst_plane_بایتها_aligned;
    uint32_t n_نخها;
    uint32_t planes;
    uint32_t pool_op;
    uint32_t fast_مسیر;
    uint32_t narrow_مسیر;
    uint32_t global_مسیر;
    uint32_t بلوک_مسیر;
    uint32_t avg_divide_شمار;
    uint32_t ox_lo;
    uint32_t ox_hi;
    عائم    inv_kernel_area;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#وإلا
_Static_assert(حجم(بنية htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#پایان_اگر

بنية htp_pool_vtcm_layout {
    size_t   مجموع_بایتها;
    size_t   off_src;
    size_t   off_dst;
    size_t   src_بایتها_در_هر_thread;
    size_t   dst_بایتها_در_هر_thread;
    size_t   src_spad_half_اندازه;
    size_t   dst_spad_half_اندازه;
};

ساكن مضمن bool htp_pool_solve_layout(
    بنية htp_pool_vtcm_layout * layout,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t n_نخها,
    size_t   vtcm_budget
) {
    // Full-plane double buffering (256 bytes guard space for vector loads)
    ثابت size_t src_plane_بایتها   = (size_t) src_x * src_y * حجم(عائم);
    ثابت size_t dst_plane_بایتها   = (size_t) dst_x * dst_y * حجم(عائم);
    ثابت size_t src_plane_aligned = hex_دور_به_بالا((uint32_t) src_plane_بایتها + 256, 128);
    ثابت size_t dst_plane_aligned = hex_دور_به_بالا((uint32_t) dst_plane_بایتها, 128);

    ثابت size_t spad_در_هر_thread = 2 * (src_plane_aligned + dst_plane_aligned);
    إذا (spad_در_هر_thread * n_نخها > vtcm_budget) {
        أرجع false;
    }

    layout->src_spad_half_اندازه   = src_plane_aligned;
    layout->dst_spad_half_اندازه   = dst_plane_aligned;
    layout->src_بایتها_در_هر_thread = 2 * src_plane_aligned;
    layout->dst_بایتها_در_هر_thread = 2 * dst_plane_aligned;
    layout->off_src              = 0;
    layout->off_dst              = layout->src_بایتها_در_هر_thread * n_نخها;
    layout->مجموع_بایتها          = layout->off_dst + layout->dst_بایتها_در_هر_thread * n_نخها;
    أرجع true;
}

#پایان_اگر // HTP_POOL_OPS_H
