#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_POOL_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_POOL_OPS_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "hex-common.h"

ರಚನೆ htp_pool_2d_kernel_params {
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
    uint32_t src_plane_ಬೈಟುಗಳು;
    uint32_t dst_plane_ಬೈಟುಗಳು;
    uint32_t src_plane_ಬೈಟುಗಳು_aligned;
    uint32_t dst_plane_ಬೈಟುಗಳು_aligned;
    uint32_t n_ಎಳೆಗಳು;
    uint32_t planes;
    uint32_t pool_op;
    uint32_t fast_ಮಾರ್ಗ;
    uint32_t narrow_ಮಾರ್ಗ;
    uint32_t global_ಮಾರ್ಗ;
    uint32_t ಖಂಡ_ಮಾರ್ಗ;
    uint32_t avg_divide_ಎಣಿಕೆ;
    uint32_t ox_lo;
    uint32_t ox_hi;
    ದಶಮಾಂಶ    inv_kernel_area;
};

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#ಇಲ್ಲದಿದ್ದರೆ
_Static_assert(ಗಾತ್ರ(ರಚನೆ htp_pool_2d_kernel_params) <= 128, "htp_pool_2d_kernel_params is too large");
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ htp_pool_vtcm_layout {
    size_t   ಒಟ್ಟು_ಬೈಟುಗಳು;
    size_t   off_src;
    size_t   off_dst;
    size_t   src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    size_t   dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    size_t   src_spad_half_ಗಾತ್ರ;
    size_t   dst_spad_half_ಗಾತ್ರ;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_pool_solve_layout(
    ರಚನೆ htp_pool_vtcm_layout * layout,
    uint32_t src_x,
    uint32_t src_y,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t n_ಎಳೆಗಳು,
    size_t   vtcm_budget
) {
    // Full-plane double buffering (256 bytes guard space for vector loads)
    ಸ್ಥಿರಾಂಕ size_t src_plane_ಬೈಟುಗಳು   = (size_t) src_x * src_y * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t dst_plane_ಬೈಟುಗಳು   = (size_t) dst_x * dst_y * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src_plane_aligned = hex_ಸುತ್ತು_ಮೇಲೆ((uint32_t) src_plane_ಬೈಟುಗಳು + 256, 128);
    ಸ್ಥಿರಾಂಕ size_t dst_plane_aligned = hex_ಸುತ್ತು_ಮೇಲೆ((uint32_t) dst_plane_ಬೈಟುಗಳು, 128);

    ಸ್ಥಿರಾಂಕ size_t spad_ಪ್ರತಿ_thread = 2 * (src_plane_aligned + dst_plane_aligned);
    ಆದರೆ (spad_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು > vtcm_budget) {
        ಹಿಂತಿರುಗಿ false;
    }

    layout->src_spad_half_ಗಾತ್ರ   = src_plane_aligned;
    layout->dst_spad_half_ಗಾತ್ರ   = dst_plane_aligned;
    layout->src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = 2 * src_plane_aligned;
    layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = 2 * dst_plane_aligned;
    layout->off_src              = 0;
    layout->off_dst              = layout->src_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
    layout->ಒಟ್ಟು_ಬೈಟುಗಳು          = layout->off_dst + layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
    ಹಿಂತಿರುಗಿ true;
}

#ಆದರೆ_ಅಂತ್ಯ // HTP_POOL_OPS_H
