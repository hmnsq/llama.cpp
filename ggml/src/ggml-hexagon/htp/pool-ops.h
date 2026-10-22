#अगर_अघोषित HTP_POOL_OPS_H
#घोषणा HTP_POOL_OPS_H

#समावेश <stdint.h>
#समावेश <stddef.h>
#समावेश <stdbool.h>
#समावेश <string.h>

#समावेश "hex-common.h"

काष्ठा htp_pool_2d_kernel_params {
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
    भग्न    inv_kernel_area;
};

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#अन्यथा
_Static_assert(माप(काष्ठा htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#पूर्ण_अगर

काष्ठा htp_pool_vtcm_layout {
    माप_प्रकार   योग_bytes;
    माप_प्रकार   off_src;
    माप_प्रकार   off_dst;
    माप_प्रकार   src_bytes_per_thread;
    माप_प्रकार   dst_bytes_per_thread;
    माप_प्रकार   src_spad_half_आकार;
    माप_प्रकार   dst_spad_half_आकार;
};

अटल अंतरभूत bool htp_pool_solve_layout(
    काष्ठा htp_pool_vtcm_layout * layout,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t n_threads,
    माप_प्रकार   vtcm_budget
) {
    // Full-plane double buffering (256 bytes guard space for vector loads)
    स्थिर माप_प्रकार src_plane_bytes   = (माप_प्रकार) src_x * src_y * माप(भग्न);
    स्थिर माप_प्रकार dst_plane_bytes   = (माप_प्रकार) dst_x * dst_y * माप(भग्न);
    स्थिर माप_प्रकार src_plane_aligned = hex_round_up((uint32_t) src_plane_bytes + 256, 128);
    स्थिर माप_प्रकार dst_plane_aligned = hex_round_up((uint32_t) dst_plane_bytes, 128);

    स्थिर माप_प्रकार spad_per_thread = 2 * (src_plane_aligned + dst_plane_aligned);
    अगर (spad_per_thread * n_threads > vtcm_budget) {
        वापस false;
    }

    layout->src_spad_half_आकार   = src_plane_aligned;
    layout->dst_spad_half_आकार   = dst_plane_aligned;
    layout->src_bytes_per_thread = 2 * src_plane_aligned;
    layout->dst_bytes_per_thread = 2 * dst_plane_aligned;
    layout->off_src              = 0;
    layout->off_dst              = layout->src_bytes_per_thread * n_threads;
    layout->योग_bytes          = layout->off_dst + layout->dst_bytes_per_thread * n_threads;
    वापस true;
}

#पूर्ण_अगर // HTP_POOL_OPS_H
