#यदि_अपरिभाषितम् HTP_POOL_OPS_H
#परिभाषय HTP_POOL_OPS_H

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
    uint32_t src_plane_अष्टकानि;
    uint32_t dst_plane_अष्टकानि;
    uint32_t src_plane_अष्टकानि_aligned;
    uint32_t dst_plane_अष्टकानि_aligned;
    uint32_t n_तन्तवः;
    uint32_t planes;
    uint32_t pool_op;
    uint32_t fast_मार्गः;
    uint32_t narrow_मार्गः;
    uint32_t global_मार्गः;
    uint32_t खण्डम्_मार्गः;
    uint32_t avg_divide_गणनम्;
    uint32_t ox_lo;
    uint32_t ox_hi;
    भिन्न    inv_kernel_area;
};

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#यदि_अन्तः

संरचना htp_pool_vtcm_layout {
    size_t   योग_अष्टकानि;
    size_t   off_src;
    size_t   off_dst;
    size_t   src_अष्टकानि_प्रत्येकम्_thread;
    size_t   dst_अष्टकानि_प्रत्येकम्_thread;
    size_t   src_spad_half_आकार;
    size_t   dst_spad_half_आकार;
};

स्थिर अन्तःस्थ bool htp_pool_solve_layout(
    संरचना htp_pool_vtcm_layout * layout,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t n_तन्तवः,
    size_t   vtcm_budget
) {
    // Full-plane double buffering (256 bytes guard space for vector loads)
    ध्रुव size_t src_plane_अष्टकानि   = (size_t) src_x * src_y * परिमाण(भिन्न);
    ध्रुव size_t dst_plane_अष्टकानि   = (size_t) dst_x * dst_y * परिमाण(भिन्न);
    ध्रुव size_t src_plane_aligned = hex_आवर्तनम्_ऊर्ध्वम्((uint32_t) src_plane_अष्टकानि + 256, 128);
    ध्रुव size_t dst_plane_aligned = hex_आवर्तनम्_ऊर्ध्वम्((uint32_t) dst_plane_अष्टकानि, 128);

    ध्रुव size_t spad_प्रत्येकम्_thread = 2 * (src_plane_aligned + dst_plane_aligned);
    यदि (spad_प्रत्येकम्_thread * n_तन्तवः > vtcm_budget) {
        प्रत्यागम false;
    }

    layout->src_spad_half_आकार   = src_plane_aligned;
    layout->dst_spad_half_आकार   = dst_plane_aligned;
    layout->src_अष्टकानि_प्रत्येकम्_thread = 2 * src_plane_aligned;
    layout->dst_अष्टकानि_प्रत्येकम्_thread = 2 * dst_plane_aligned;
    layout->off_src              = 0;
    layout->off_dst              = layout->src_अष्टकानि_प्रत्येकम्_thread * n_तन्तवः;
    layout->योग_अष्टकानि          = layout->off_dst + layout->dst_अष्टकानि_प्रत्येकम्_thread * n_तन्तवः;
    प्रत्यागम true;
}

#यदि_अन्तः // HTP_POOL_OPS_H
