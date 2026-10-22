#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_POOL_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_POOL_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"

ਬਣਤਰ htp_pool_2d_kernel_params {
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
    uint32_t src_plane_ਬਾਈਟਾਂ;
    uint32_t dst_plane_ਬਾਈਟਾਂ;
    uint32_t src_plane_ਬਾਈਟਾਂ_aligned;
    uint32_t dst_plane_ਬਾਈਟਾਂ_aligned;
    uint32_t n_ਧਾਗੇ;
    uint32_t planes;
    uint32_t pool_op;
    uint32_t fast_ਰਸਤਾ;
    uint32_t narrow_ਰਸਤਾ;
    uint32_t global_ਰਸਤਾ;
    uint32_t ਖੰਡ_ਰਸਤਾ;
    uint32_t avg_divide_ਗਿਣਤੀ;
    uint32_t ox_lo;
    uint32_t ox_hi;
    ਦਸ਼ਮਲਵ    inv_kernel_area;
};

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#ਨਹੀਂਤਾਂ
_Static_assert(ਆਕਾਰ(ਬਣਤਰ htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#ਜੇ_ਅੰਤ

ਬਣਤਰ htp_pool_vtcm_layout {
    size_t   ਕੁੱਲ_ਬਾਈਟਾਂ;
    size_t   off_src;
    size_t   off_dst;
    size_t   src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    size_t   dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;
    size_t   src_spad_half_ਮਾਪ;
    size_t   dst_spad_half_ਮਾਪ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_pool_solve_layout(
    ਬਣਤਰ htp_pool_vtcm_layout * layout,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t n_ਧਾਗੇ,
    size_t   vtcm_budget
) {
    // Full-plane double buffering (256 bytes guard space for vector loads)
    ਅਚਲ size_t src_plane_ਬਾਈਟਾਂ   = (size_t) src_x * src_y * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t dst_plane_ਬਾਈਟਾਂ   = (size_t) dst_x * dst_y * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t src_plane_aligned = hex_ਗੇੜਾ_ਉੱਪਰ((uint32_t) src_plane_ਬਾਈਟਾਂ + 256, 128);
    ਅਚਲ size_t dst_plane_aligned = hex_ਗੇੜਾ_ਉੱਪਰ((uint32_t) dst_plane_ਬਾਈਟਾਂ, 128);

    ਅਚਲ size_t spad_ਪ੍ਰਤੀ_thread = 2 * (src_plane_aligned + dst_plane_aligned);
    ਜੇ (spad_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ > vtcm_budget) {
        ਵਾਪਸ false;
    }

    layout->src_spad_half_ਮਾਪ   = src_plane_aligned;
    layout->dst_spad_half_ਮਾਪ   = dst_plane_aligned;
    layout->src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = 2 * src_plane_aligned;
    layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread = 2 * dst_plane_aligned;
    layout->off_src              = 0;
    layout->off_dst              = layout->src_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
    layout->ਕੁੱਲ_ਬਾਈਟਾਂ          = layout->off_dst + layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
    ਵਾਪਸ true;
}

#ਜੇ_ਅੰਤ // HTP_POOL_OPS_H
