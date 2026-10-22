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
    uint32_t avg_divide_मोजणी;
    uint32_t ox_lo;
    uint32_t ox_hi;
    दशांश    inv_kernel_area;
};

#जर defined(__cplusplus)
static_assert(आकार(संरचना htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#अन्यथा
_Static_assert(आकार(संरचना htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#endif

संरचना htp_pool_vtcm_layout {
    माप_प्रकार   total_bytes;
    माप_प्रकार   off_src;
    माप_प्रकार   off_dst;
    माप_प्रकार   src_bytes_per_thread;
    माप_प्रकार   dst_bytes_per_thread;
    माप_प्रकार   src_spad_half_size;
    माप_प्रकार   dst_spad_half_size;
};

स्थिर ओळीत bool htp_pool_solve_layout(
    संरचना htp_pool_vtcm_layout * layout,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t n_threads,
    माप_प्रकार   vtcm_budget
) {
    // Full-plane double buffering (256 bytes guard space for vector loads)
    स्थिरांक माप_प्रकार src_plane_bytes   = (माप_प्रकार) src_x * src_y * आकार(दशांश);
    स्थिरांक माप_प्रकार dst_plane_bytes   = (माप_प्रकार) dst_x * dst_y * आकार(दशांश);
    स्थिरांक माप_प्रकार src_plane_aligned = hex_round_up((uint32_t) src_plane_bytes + 256, 128);
    स्थिरांक माप_प्रकार dst_plane_aligned = hex_round_up((uint32_t) dst_plane_bytes, 128);

    स्थिरांक माप_प्रकार spad_per_thread = 2 * (src_plane_aligned + dst_plane_aligned);
    जर (spad_per_thread * n_threads > vtcm_budget) {
        परत false;
    }

    layout->src_spad_half_size   = src_plane_aligned;
    layout->dst_spad_half_size   = dst_plane_aligned;
    layout->src_bytes_per_thread = 2 * src_plane_aligned;
    layout->dst_bytes_per_thread = 2 * dst_plane_aligned;
    layout->off_src              = 0;
    layout->off_dst              = layout->src_bytes_per_thread * n_threads;
    layout->total_bytes          = layout->off_dst + layout->dst_bytes_per_thread * n_threads;
    परत true;
}

#endif // HTP_POOL_OPS_H
