#که_نه_تعریف_شوی HTP_POOL_OPS_H
#تعریف_کړه HTP_POOL_OPS_H

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
    uint32_t src_plane_بایټونه;
    uint32_t dst_plane_بایټونه;
    uint32_t src_plane_بایټونه_aligned;
    uint32_t dst_plane_بایټونه_aligned;
    uint32_t n_تارونه;
    uint32_t planes;
    uint32_t pool_op;
    uint32_t fast_لار;
    uint32_t narrow_لار;
    uint32_t global_لار;
    uint32_t بلاک_لار;
    uint32_t avg_divide_شمېر;
    uint32_t ox_lo;
    uint32_t ox_hi;
    عائم    inv_kernel_area;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#وإلا
_Static_assert(حجم(بنية htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#د_که_پای

بنية htp_pool_vtcm_layout {
    size_t   ټول_بایټونه;
    size_t   off_src;
    size_t   off_dst;
    size_t   src_بایټونه_فی_thread;
    size_t   dst_بایټونه_فی_thread;
    size_t   src_spad_half_اندازه;
    size_t   dst_spad_half_اندازه;
};

ساكن مضمن bool htp_pool_solve_layout(
    بنية htp_pool_vtcm_layout * layout,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t n_تارونه,
    size_t   vtcm_budget
) {
    // Full-plane double buffering (256 bytes guard space for vector loads)
    ثابت size_t src_plane_بایټونه   = (size_t) src_x * src_y * حجم(عائم);
    ثابت size_t dst_plane_بایټونه   = (size_t) dst_x * dst_y * حجم(عائم);
    ثابت size_t src_plane_aligned = hex_پړاو_پورته((uint32_t) src_plane_بایټونه + 256, 128);
    ثابت size_t dst_plane_aligned = hex_پړاو_پورته((uint32_t) dst_plane_بایټونه, 128);

    ثابت size_t spad_فی_thread = 2 * (src_plane_aligned + dst_plane_aligned);
    إذا (spad_فی_thread * n_تارونه > vtcm_budget) {
        أرجع false;
    }

    layout->src_spad_half_اندازه   = src_plane_aligned;
    layout->dst_spad_half_اندازه   = dst_plane_aligned;
    layout->src_بایټونه_فی_thread = 2 * src_plane_aligned;
    layout->dst_بایټونه_فی_thread = 2 * dst_plane_aligned;
    layout->off_src              = 0;
    layout->off_dst              = layout->src_بایټونه_فی_thread * n_تارونه;
    layout->ټول_بایټونه          = layout->off_dst + layout->dst_بایټونه_فی_thread * n_تارونه;
    أرجع true;
}

#د_که_پای // HTP_POOL_OPS_H
