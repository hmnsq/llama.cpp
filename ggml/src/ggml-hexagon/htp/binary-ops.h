#यदि_अपरिभाषितम् HTP_द्विमान_OPS_H
#परिभाषय HTP_द्विमान_OPS_H

#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-vtcm.h"

गणना htp_द्विमान_kernel_type {
    HTP_द्विमान_KERNEL_SAME_SHAPE = 0,
    HTP_द्विमान_KERNEL_ROW_BCAST,
    HTP_द्विमान_KERNEL_SCALAR,
    HTP_द्विमान_KERNEL_सङ्कलय_ID,
    HTP_द्विमान_KERNEL_COMPLEX,
    HTP_द्विमान_KERNEL_REPEAT,
    HTP_द्विमान_KERNEL_CHUNKED,
};

संरचना htp_द्विमान_kernel_params {
    uint32_t kernel_type;
    uint32_t n_तन्तवः;
    uint32_t पङ्क्तिकाः_प्रत्येकम्_बफर;

    uint32_t src0_row_आकार_aligned;
    uint32_t src1_row_आकार_aligned;
    uint32_t dst_row_आकार_aligned;

    uint32_t src1_आकार;
    uint32_t src2_आकार;
    uint32_t vtcm_आकार;

    uint32_t chunk_आकार;
    uint32_t chunk_अष्टकानि;
    uint32_t is_scalar;

    uint8_t  विभज_at_ne01;
    uint8_t  विभज_at_ne02;
    uint8_t  src1_strided;
    uint8_t  pad;
};

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_द्विमान_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");

संरचना ggml_tensor;
स्थिर अन्तःस्थ bool htp_द्विमान_strides_safe(
    ध्रुव संरचना ggml_tensor * src0,
    ध्रुव संरचना ggml_tensor * src1,
    ध्रुव संरचना ggml_tensor * dst,
    size_t elem_आकार
) {
    यदि ((src0->ne[0] != 1 && src0->nb[0] != elem_आकार) ||
        (src1->ne[0] != 1 && src1->nb[0] != elem_आकार) ||
        (dst->ne[0]  != 1 && dst->nb[0]  != elem_आकार)) {
        प्रत्यागम false;
    }
    यदि ((src0->ne[1] > 1 && src0->nb[1] < src0->ne[0] * elem_आकार) ||
        (src1->ne[1] > 1 && src1->nb[1] < src1->ne[0] * elem_आकार) ||
        (dst->ne[1]  > 1 && dst->nb[1]  < dst->ne[0]  * elem_आकार)) {
        प्रत्यागम false;
    }
    प्रत्यागम true;
}
#अन्यथा
_Static_assert(परिमाण(संरचना htp_द्विमान_kernel_params) <= 128, "htp_binary_kernel_params is too large for kernel_params blob");
#यदि_अन्तः

संरचना htp_द्विमान_vtcm_layout {
    size_t योग_अष्टकानि;
    size_t off_src0;
    size_t off_src1;
    size_t off_src2;
    size_t off_dst;

    size_t src0_अष्टकानि_प्रत्येकम्_thread;
    size_t src1_अष्टकानि_प्रत्येकम्_thread;
    size_t dst_अष्टकानि_प्रत्येकम्_thread;

    size_t src0_spad_half_आकार;
    size_t src1_spad_half_आकार;
    size_t dst_spad_half_आकार;

    size_t src1_आकार;
    size_t src2_आकार;
    uint32_t पङ्क्तिकाः_प्रत्येकम्_बफर;
};

स्थिर अन्तःस्थ शून्य htp_द्विमान_vtcm_layout_build(
    संरचना htp_द्विमान_vtcm_layout * L,
    ध्रुव संरचना htp_द्विमान_kernel_params * kparams,
    size_t vtcm_आकार
) {
    स्मृतिं_पूरय(L, 0, परिमाण(*L));

    ध्रुव uint32_t n_तन्तवः = kparams->n_तन्तवः;
    यदि (n_तन्तवः == 0) {
        प्रत्यागम;
    }

    यदि (kparams->kernel_type == HTP_द्विमान_KERNEL_CHUNKED) {
        ध्रुव size_t chunk_अष्टकानि = kparams->chunk_अष्टकानि;
        यदि (chunk_अष्टकानि == 0) {
            प्रत्यागम;
        }

        L->src0_अष्टकानि_प्रत्येकम्_thread = 2 * chunk_अष्टकानि;
        L->src1_अष्टकानि_प्रत्येकम्_thread = kparams->is_scalar ? 0 : (2 * chunk_अष्टकानि);
        L->dst_अष्टकानि_प्रत्येकम्_thread  = 2 * chunk_अष्टकानि;

        L->src0_spad_half_आकार = chunk_अष्टकानि;
        L->src1_spad_half_आकार = kparams->is_scalar ? 0 : chunk_अष्टकानि;
        L->dst_spad_half_आकार  = chunk_अष्टकानि;

        L->पङ्क्तिकाः_प्रत्येकम्_बफर = 1;
        L->src1_आकार = 0;

        ध्रुव size_t src0_योग = n_तन्तवः * L->src0_अष्टकानि_प्रत्येकम्_thread;
        ध्रुव size_t src1_योग = kparams->is_scalar ? 128 : (n_तन्तवः * L->src1_अष्टकानि_प्रत्येकम्_thread);
        ध्रुव size_t dst_योग  = n_तन्तवः * L->dst_अष्टकानि_प्रत्येकम्_thread;

        size_t off = 0;
        VTCM_LAYOUT_ALLOC(off, off_src0, src0_योग);
        VTCM_LAYOUT_ALLOC(off, off_src1, src1_योग);
        VTCM_LAYOUT_ALLOC(off, off_dst,  dst_योग);

        यदि (off > vtcm_आकार) {
            प्रत्यागम;
        }

        L->योग_अष्टकानि = off;
        प्रत्यागम;
    }

    ध्रुव size_t spad_row_योग = (kparams->kernel_type == HTP_द्विमान_KERNEL_SAME_SHAPE)
        ? 2 * (kparams->src0_row_आकार_aligned + kparams->src1_row_आकार_aligned + kparams->dst_row_आकार_aligned)
        : 2 * (kparams->src0_row_आकार_aligned + kparams->dst_row_आकार_aligned);

    ध्रुव size_t static_vtcm_आकार = kparams->src1_आकार + kparams->src2_आकार;
    यदि (spad_row_योग == 0 || vtcm_आकार < static_vtcm_आकार) {
        प्रत्यागम;
    }

    ध्रुव size_t पङ्क्तिकाः_प्रत्येकम्_बफर = (vtcm_आकार - static_vtcm_आकार) / (n_तन्तवः * spad_row_योग);
    यदि (पङ्क्तिकाः_प्रत्येकम्_बफर == 0) {
        प्रत्यागम;
    }

    L->पङ्क्तिकाः_प्रत्येकम्_बफर = (uint32_t) पङ्क्तिकाः_प्रत्येकम्_बफर;
    L->src1_आकार = kparams->src1_आकार;
    L->src2_आकार = kparams->src2_आकार;

    L->src0_अष्टकानि_प्रत्येकम्_thread = पङ्क्तिकाः_प्रत्येकम्_बफर * 2 * kparams->src0_row_आकार_aligned;
    L->dst_अष्टकानि_प्रत्येकम्_thread  = पङ्क्तिकाः_प्रत्येकम्_बफर * 2 * kparams->dst_row_आकार_aligned;
    L->src1_अष्टकानि_प्रत्येकम्_thread = (kparams->kernel_type == HTP_द्विमान_KERNEL_SAME_SHAPE)
        ? पङ्क्तिकाः_प्रत्येकम्_बफर * 2 * kparams->src1_row_आकार_aligned
        : 0;

    L->src0_spad_half_आकार = L->src0_अष्टकानि_प्रत्येकम्_thread / 2;
    L->src1_spad_half_आकार = L->src1_अष्टकानि_प्रत्येकम्_thread / 2;
    L->dst_spad_half_आकार  = L->dst_अष्टकानि_प्रत्येकम्_thread / 2;

    ध्रुव size_t src0_योग = n_तन्तवः * L->src0_अष्टकानि_प्रत्येकम्_thread;
    ध्रुव size_t src1_योग = (kparams->src1_आकार > 0)
        ? kparams->src1_आकार
        : n_तन्तवः * L->src1_अष्टकानि_प्रत्येकम्_thread;
    ध्रुव size_t src2_योग = kparams->src2_आकार;
    ध्रुव size_t dst_योग  = n_तन्तवः * L->dst_अष्टकानि_प्रत्येकम्_thread;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_src0, src0_योग);
    VTCM_LAYOUT_ALLOC(off, off_src1, src1_योग);
    VTCM_LAYOUT_ALLOC(off, off_src2, src2_योग);
    VTCM_LAYOUT_ALLOC(off, off_dst,  dst_योग);

    L->योग_अष्टकानि = off;
}

#यदि_अन्तः
