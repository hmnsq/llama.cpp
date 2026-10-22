#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_OPS_H

#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"

ಎಣಿಕೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_type {
    HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_SAMETYPE = 0,
    HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_TILED,
    HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_FLAT,
};

ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_params {
    int32_t  n_ಎಳೆಗಳು;
    int32_t  kernel_type;
    int32_t  ಒಟ್ಟು_tasks;
    int32_t  tasks_ಪ್ರತಿ_thread;
    int32_t  vtcm_ಗಾತ್ರ;

    // Fastdiv helpers
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne10;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne10_ne11;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne02;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne03;
};

ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout {
    size_t ಒಟ್ಟು_ಬೈಟುಗಳು;
    size_t off_src1;
    size_t src1_ಬೈಟುಗಳು;
    size_t off_src0;
    size_t off_dst;

    size_t src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    size_t dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;

    size_t src0_spad_half_ಗಾತ್ರ;
    size_t dst_spad_half_ಗಾತ್ರ;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_get_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout_build(
    ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout * L,
    ಪೂರ್ಣಾಂಕ kernel_type,
    ಪೂರ್ಣಾಂಕ type,
    uint32_t ne00,
    uint32_t ಒಟ್ಟು_tasks,
    ಪೂರ್ಣಾಂಕ idx_type,
    uint32_t n_ಎಳೆಗಳು) {

    memset(L, 0, ಗಾತ್ರ(*L));

    ಸ್ಥಿರಾಂಕ size_t elem_ಗಾತ್ರ = (idx_type == HTP_TYPE_I64) ? ಗಾತ್ರ(int64_t) : ಗಾತ್ರ(int32_t);
    ಸ್ಥಿರಾಂಕ size_t src1_raw_ಬೈಟುಗಳು = ಒಟ್ಟು_tasks * elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t src1_ಬೈಟುಗಳು = ((src1_raw_ಬೈಟುಗಳು + 127) & ~127) + 128;

    L->src1_ಬೈಟುಗಳು = src1_ಬೈಟುಗಳು;
    ಸ್ಥಿರಾಂಕ size_t src1_alloc_ಬೈಟುಗಳು = (src1_ಬೈಟುಗಳು + 255) & ~255;

    ಆದರೆ (kernel_type != HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_SAMETYPE) {
        size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = 0;
        ಆದರೆ (kernel_type == HTP_GET_ಅಡ್ಡಸಾಲುಗಳು_KERNEL_TILED) {
            ಸ್ಥಿರಾಂಕ size_t tile_stride = htp_tensor_get_tile_padded_ಗಾತ್ರ(type);
            ಸ್ಥಿರಾಂಕ uint32_t n_k_tiles = ne00 / htp_tensor_get_tile_k(type);
            src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = n_k_tiles > 0 ? (n_k_tiles * tile_stride) : tile_stride;
        } ಇಲ್ಲದಿದ್ದರೆ {
            src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_tensor_get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(type, ne00);
        }

        L->src0_spad_half_ಗಾತ್ರ   = (src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ + 255) & ~255;
        L->dst_spad_half_ಗಾತ್ರ    = (ne00 * ಗಾತ್ರ(ದಶಮಾಂಶ) + 255) & ~255;
        L->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = 2 * L->src0_spad_half_ಗಾತ್ರ;
        L->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread  = 2 * L->dst_spad_half_ಗಾತ್ರ;
    }

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_alloc_ಬೈಟುಗಳು);
    VTCM_LAYOUT_ALLOC(off, off_src0, L->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು);
    VTCM_LAYOUT_ALLOC(off, off_dst,  L->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread  * n_ಎಳೆಗಳು);
    L->ಒಟ್ಟು_ಬೈಟುಗಳು = off;
}

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#ಇಲ್ಲದಿದ್ದರೆ
_Static_assert(ಗಾತ್ರ(ರಚನೆ htp_get_ಅಡ್ಡಸಾಲುಗಳು_kernel_params) <= 128, "htp_get_rows_kernel_params is too large for kernel_params blob");
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ_ಅಂತ್ಯ // HTP_GET_ROWS_OPS_H
