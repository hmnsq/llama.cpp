#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_ದ್ವಿಮಾನ_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_ದ್ವಿಮಾನ_OPS_H

#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"

ಎಣಿಕೆ htp_ದ್ವಿಮಾನ_kernel_type {
    HTP_ದ್ವಿಮಾನ_KERNEL_SAME_SHAPE = 0,
    HTP_ದ್ವಿಮಾನ_KERNEL_ಅಡ್ಡಸಾಲು_BCAST,
    HTP_ದ್ವಿಮಾನ_KERNEL_SCALAR,
    HTP_ದ್ವಿಮಾನ_KERNEL_ಕೂಡಿಸು_ID,
    HTP_ದ್ವಿಮಾನ_KERNEL_COMPLEX,
    HTP_ದ್ವಿಮಾನ_KERNEL_REPEAT,
    HTP_ದ್ವಿಮಾನ_KERNEL_CHUNKED,
};

ರಚನೆ htp_ದ್ವಿಮಾನ_kernel_params {
    uint32_t kernel_type;
    uint32_t n_ಎಳೆಗಳು;
    uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್;

    uint32_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    uint32_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    uint32_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    uint32_t src1_ಗಾತ್ರ;
    uint32_t src2_ಗಾತ್ರ;
    uint32_t vtcm_ಗಾತ್ರ;

    uint32_t chunk_ಗಾತ್ರ;
    uint32_t chunk_ಬೈಟುಗಳು;
    uint32_t is_scalar;

    uint8_t  ವಿಭಜಿಸು_at_ne01;
    uint8_t  ವಿಭಜಿಸು_at_ne02;
    uint8_t  src1_strided;
    uint8_t  pad;
};

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_ದ್ವಿಮಾನ_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

ರಚನೆ ggml_tensor;
ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_ದ್ವಿಮಾನ_strides_ಸುರಕ್ಷಿತ(
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * src1,
    ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * dst,
    size_t elem_ಗಾತ್ರ
) {
    ಆದರೆ ((src0->ne[0] != 1 && src0->nb[0] != elem_ಗಾತ್ರ) ||
        (src1->ne[0] != 1 && src1->nb[0] != elem_ಗಾತ್ರ) ||
        (dst->ne[0]  != 1 && dst->nb[0]  != elem_ಗಾತ್ರ)) {
        ಹಿಂತಿರುಗಿ false;
    }
    ಆದರೆ ((src0->ne[1] > 1 && src0->nb[1] < src0->ne[0] * elem_ಗಾತ್ರ) ||
        (src1->ne[1] > 1 && src1->nb[1] < src1->ne[0] * elem_ಗಾತ್ರ) ||
        (dst->ne[1]  > 1 && dst->nb[1]  < dst->ne[0]  * elem_ಗಾತ್ರ)) {
        ಹಿಂತಿರುಗಿ false;
    }
    ಹಿಂತಿರುಗಿ true;
}
#ಇಲ್ಲದಿದ್ದರೆ
_Static_assert(ಗಾತ್ರ(ರಚನೆ htp_ದ್ವಿಮಾನ_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ htp_ದ್ವಿಮಾನ_vtcm_layout {
    size_t ಒಟ್ಟು_ಬೈಟುಗಳು;
    size_t off_src0;
    size_t off_src1;
    size_t off_src2;
    size_t off_dst;

    size_t src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    size_t src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    size_t dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;

    size_t src0_spad_half_ಗಾತ್ರ;
    size_t src1_spad_half_ಗಾತ್ರ;
    size_t dst_spad_half_ಗಾತ್ರ;

    size_t src1_ಗಾತ್ರ;
    size_t src2_ಗಾತ್ರ;
    uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_ದ್ವಿಮಾನ_vtcm_layout_build(
    ರಚನೆ htp_ದ್ವಿಮಾನ_vtcm_layout * L,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ದ್ವಿಮಾನ_kernel_params * kparams,
    size_t vtcm_ಗಾತ್ರ
) {
    memset(L, 0, ಗಾತ್ರ(*L));

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = kparams->n_ಎಳೆಗಳು;
    ಆದರೆ (n_ಎಳೆಗಳು == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    ಆದರೆ (kparams->kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_CHUNKED) {
        ಸ್ಥಿರಾಂಕ size_t chunk_ಬೈಟುಗಳು = kparams->chunk_ಬೈಟುಗಳು;
        ಆದರೆ (chunk_ಬೈಟುಗಳು == 0) {
            ಹಿಂತಿರುಗಿ;
        }

        L->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = 2 * chunk_ಬೈಟುಗಳು;
        L->src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = kparams->is_scalar ? 0 : (2 * chunk_ಬೈಟುಗಳು);
        L->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread  = 2 * chunk_ಬೈಟುಗಳು;

        L->src0_spad_half_ಗಾತ್ರ = chunk_ಬೈಟುಗಳು;
        L->src1_spad_half_ಗಾತ್ರ = kparams->is_scalar ? 0 : chunk_ಬೈಟುಗಳು;
        L->dst_spad_half_ಗಾತ್ರ  = chunk_ಬೈಟುಗಳು;

        L->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್ = 1;
        L->src1_ಗಾತ್ರ = 0;

        ಸ್ಥಿರಾಂಕ size_t src0_ಒಟ್ಟು = n_ಎಳೆಗಳು * L->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
        ಸ್ಥಿರಾಂಕ size_t src1_ಒಟ್ಟು = kparams->is_scalar ? 128 : (n_ಎಳೆಗಳು * L->src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread);
        ಸ್ಥಿರಾಂಕ size_t dst_ಒಟ್ಟು  = n_ಎಳೆಗಳು * L->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;

        size_t off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_ಒಟ್ಟು);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_ಒಟ್ಟು);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_ಒಟ್ಟು);

        ಆದರೆ (off > vtcm_ಗಾತ್ರ) {
            ಹಿಂತಿರುಗಿ;
        }

        L->ಒಟ್ಟು_ಬೈಟುಗಳು = off;
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ size_t spad_ಅಡ್ಡಸಾಲು_ಒಟ್ಟು = (kparams->kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + kparams->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + kparams->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned)
        : 2 * (kparams->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + kparams->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);

    ಸ್ಥಿರಾಂಕ size_t static_vtcm_ಗಾತ್ರ = kparams->src1_ಗಾತ್ರ + kparams->src2_ಗಾತ್ರ;
    ಆದರೆ (spad_ಅಡ್ಡಸಾಲು_ಒಟ್ಟು == 0 || vtcm_ಗಾತ್ರ < static_vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್ = (vtcm_ಗಾತ್ರ - static_vtcm_ಗಾತ್ರ) / (n_ಎಳೆಗಳು * spad_ಅಡ್ಡಸಾಲು_ಒಟ್ಟು);
    ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್ == 0) {
        ಹಿಂತಿರುಗಿ;
    }

    L->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್ = (uint32_t) ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್;
    L->src1_ಗಾತ್ರ = kparams->src1_ಗಾತ್ರ;
    L->src2_ಗಾತ್ರ = kparams->src2_ಗಾತ್ರ;

    L->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್ * 2 * kparams->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    L->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread  = ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್ * 2 * kparams->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    L->src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread = (kparams->kernel_type == HTP_ದ್ವಿಮಾನ_KERNEL_SAME_SHAPE)
        ? ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಬಫರ್ * 2 * kparams->src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned
        : 0;

    L->src0_spad_half_ಗಾತ್ರ = L->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread / 2;
    L->src1_spad_half_ಗಾತ್ರ = L->src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread / 2;
    L->dst_spad_half_ಗಾತ್ರ  = L->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread / 2;

    ಸ್ಥಿರಾಂಕ size_t src0_ಒಟ್ಟು = n_ಎಳೆಗಳು * L->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ size_t src1_ಒಟ್ಟು = (kparams->src1_ಗಾತ್ರ > 0)
        ? kparams->src1_ಗಾತ್ರ
        : n_ಎಳೆಗಳು * L->src1_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ size_t src2_ಒಟ್ಟು = kparams->src2_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_ಒಟ್ಟು  = n_ಎಳೆಗಳು * L->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_ಒಟ್ಟು);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_ಒಟ್ಟು);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_ಒಟ್ಟು);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_ಒಟ್ಟು);

    L->ಒಟ್ಟು_ಬೈಟುಗಳು = off;
}

#ಆದರೆ_ಅಂತ್ಯ
