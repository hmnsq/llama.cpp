#ifndef HTP_POOL_OPS_H
#define HTP_POOL_OPS_H

#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <stddef.h>
#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <string.h>

#અન્તર્ભાવ "hex-common.h"

સંરચના htp_pool_2d_kernel_params {
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
    uint32_t src_plane_bytes;
    uint32_t dst_plane_bytes;
    uint32_t src_plane_bytes_aligned;
    uint32_t dst_plane_bytes_aligned;
    uint32_t n_threads;
    uint32_t planes;
    uint32_t pool_op;
    uint32_t fast_path;
    uint32_t narrow_path;
    uint32_t global_path;
    uint32_t block_path;
    uint32_t avg_divide_ગણતરી;
    uint32_t ox_lo;
    uint32_t ox_hi;
    દશાંશ    inv_kernel_area;
};

#જો defined(__cplusplus)
static_assert(કદ(સંરચના htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#નહીંતર
_Static_assert(કદ(સંરચના htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#endif

સંરચના htp_pool_vtcm_layout {
    size_t   total_bytes;
    size_t   off_src;
    size_t   off_dst;
    size_t   src_bytes_per_thread;
    size_t   dst_bytes_per_thread;
    size_t   src_spad_half_size;
    size_t   dst_spad_half_size;
};

સ્થિર અંતઃરેખ bool htp_pool_solve_layout(
    સંરચના htp_pool_vtcm_layout * layout,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t n_threads,
    size_t   vtcm_budget
) {
    // Full-plane double buffering (256 bytes guard space for vector loads)
    અચલ size_t src_plane_bytes   = (size_t) src_x * src_y * કદ(દશાંશ);
    અચલ size_t dst_plane_bytes   = (size_t) dst_x * dst_y * કદ(દશાંશ);
    અચલ size_t src_plane_aligned = hex_round_up((uint32_t) src_plane_bytes + 256, 128);
    અચલ size_t dst_plane_aligned = hex_round_up((uint32_t) dst_plane_bytes, 128);

    અચલ size_t spad_per_thread = 2 * (src_plane_aligned + dst_plane_aligned);
    જો (spad_per_thread * n_threads > vtcm_budget) {
        પાછા false;
    }

    layout->src_spad_half_size   = src_plane_aligned;
    layout->dst_spad_half_size   = dst_plane_aligned;
    layout->src_bytes_per_thread = 2 * src_plane_aligned;
    layout->dst_bytes_per_thread = 2 * dst_plane_aligned;
    layout->off_src              = 0;
    layout->off_dst              = layout->src_bytes_per_thread * n_threads;
    layout->total_bytes          = layout->off_dst + layout->dst_bytes_per_thread * n_threads;
    પાછા true;
}

#endif // HTP_POOL_OPS_H
