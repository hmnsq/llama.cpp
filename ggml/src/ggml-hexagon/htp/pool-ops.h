#ifndef HTP_POOL_OPS_H
#define HTP_POOL_OPS_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "hex-common.h"

संरचना htp_pool_2d_kernel_params {
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
    uint32_t avg_divide_गिनती;
    uint32_t ox_lo;
    uint32_t ox_hi;
    भिन्न    inv_kernel_area;
};

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#endif

संरचना htp_pool_vtcm_layout {
    size_t   total_bytes;
    size_t   off_src;
    size_t   off_dst;
    size_t   src_bytes_per_thread;
    size_t   dst_bytes_per_thread;
    size_t   src_spad_half_size;
    size_t   dst_spad_half_size;
};

स्थिर अन्तःस्थ bool htp_pool_solve_layout(
    संरचना htp_pool_vtcm_layout * layout,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t n_threads,
    size_t   vtcm_budget
) {
    // Full-plane double buffering (256 bytes guard space for vector loads)
    ध्रुव size_t src_plane_bytes   = (size_t) src_x * src_y * परिमाण(भिन्न);
    ध्रुव size_t dst_plane_bytes   = (size_t) dst_x * dst_y * परिमाण(भिन्न);
    ध्रुव size_t src_plane_aligned = hex_round_up((uint32_t) src_plane_bytes + 256, 128);
    ध्रुव size_t dst_plane_aligned = hex_round_up((uint32_t) dst_plane_bytes, 128);

    ध्रुव size_t spad_per_thread = 2 * (src_plane_aligned + dst_plane_aligned);
    यदि (spad_per_thread * n_threads > vtcm_budget) {
        प्रत्यागम false;
    }

    layout->src_spad_half_size   = src_plane_aligned;
    layout->dst_spad_half_size   = dst_plane_aligned;
    layout->src_bytes_per_thread = 2 * src_plane_aligned;
    layout->dst_bytes_per_thread = 2 * dst_plane_aligned;
    layout->off_src              = 0;
    layout->off_dst              = layout->src_bytes_per_thread * n_threads;
    layout->total_bytes          = layout->off_dst + layout->dst_bytes_per_thread * n_threads;
    प्रत्यागम true;
}

#endif // HTP_POOL_OPS_H
