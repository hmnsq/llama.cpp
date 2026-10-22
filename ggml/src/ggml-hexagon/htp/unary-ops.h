#यदि_अपरिभाषितम् HTP_UNARY_OPS_H
#परिभाषय HTP_UNARY_OPS_H

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "htp-ops.h"

// Op-specific struct for precomputed unary params
संरचना htp_unary_kernel_params {
    uint32_t  n_तन्तवः;
    uint32_t  col_tile;
    uint32_t  vtcm_row_प्रत्येकम्_thread;
    uint32_t  खण्डम्;
    uint32_t  प्रसारणम्_weight;

    uint32_t  vtcm_src0_आकार_प्रत्येकम्_thread;
    uint32_t  vtcm_src1_आकार_प्रत्येकम्_thread;
    uint32_t  vtcm_dst_आकार_प्रत्येकम्_thread;

    uint32_t  vtcm_src0_आकार;
    uint32_t  vtcm_src1_आकार;
    uint32_t  vtcm_dst_आकार;

    uint32_t  src0_row_आकार_aligned;
    uint32_t  src1_row_आकार_aligned;
    uint32_t  dst_row_आकार_aligned;

    uint32_t  vtcm_आकार;

    // Fastdiv helpers
    संरचना fastdiv_मानानि div_ne01;
    संरचना fastdiv_मानानि div_ne02;
    संरचना fastdiv_मानानि div_ne012;
    संरचना fastdiv_मानानि div_tpr;
};

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#यदि_अन्तः

स्थिर अन्तःस्थ bool htp_op_is_unary(uint32_t opcode) {
    विकल्प (opcode) {
        पक्ष HTP_OP_CLAMP:
        पक्ष HTP_OP_LEAKY_RELU:
        पक्ष HTP_OP_NORM:
        पक्ष HTP_OP_RMS_NORM:
        पक्ष HTP_OP_RMS_NORM_MUL:
        पक्ष HTP_OP_SCALE:
        पक्ष HTP_OP_SQR:
        पक्ष HTP_OP_SQRT:
        पक्ष HTP_OP_UNARY_NEG:
        पक्ष HTP_OP_UNARY_EXP:
        पक्ष HTP_OP_UNARY_SIGMOID:
        पक्ष HTP_OP_UNARY_SILU:
        पक्ष HTP_OP_UNARY_GELU:
        पक्ष HTP_OP_UNARY_GELU_शीघ्र:
        पक्ष HTP_OP_UNARY_GELU_ERF:
        पक्ष HTP_OP_UNARY_SOFTPLUS:
        पक्ष HTP_OP_UNARY_TANH:
        पक्ष HTP_OP_UNARY_ABS:
        पक्ष HTP_OP_UNARY_LOG:
        पक्ष HTP_OP_UNARY_RELU:
        पक्ष HTP_OP_UNARY_चरणः:
        पक्ष HTP_OP_L2_NORM:
        पक्ष HTP_OP_TRI:
        पक्ष HTP_OP_HADAMARD:
            प्रत्यागम true;
        स्वभाव:
            प्रत्यागम false;
    }
}

संरचना htp_unary_vtcm_layout {
    size_t योग_अष्टकानि;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_अष्टकानि;
    size_t src1_अष्टकानि;
    size_t dst_अष्टकानि;
};

स्थिर अन्तःस्थ शून्य htp_unary_vtcm_layout_build(
    संरचना htp_unary_vtcm_layout * L,
    uint32_t op,
    uint32_t ne00,
    uint32_t ne10,
    uint32_t ne11,
    bool प्रसारणम्_weight,
    uint32_t n_तन्तवः,
    size_t vtcm_आकार,
    size_t elem_आकार,
    uint32_t * बहिर्गमनम्_col_tile,
    uint32_t * बहिर्गमनम्_vtcm_row_प्रत्येकम्_thread
) {
    ध्रुव size_t dst_elem_आकार      = (op == HTP_OP_HADAMARD) ? परिमाण(भिन्न) : elem_आकार;
    ध्रुव size_t src0_दत्त_row_आकार = ne00 * elem_आकार;
    ध्रुव size_t dst_दत्त_row_आकार  = ne10 * dst_elem_आकार;

    ध्रुव size_t src0_row_आकार_aligned = hex_आवर्तनम्_ऊर्ध्वम्(src0_दत्त_row_आकार, 128);
    ध्रुव size_t dst_row_आकार_aligned  = hex_आवर्तनम्_ऊर्ध्वम्(dst_दत्त_row_आकार,  128);

    size_t src1_row_आकार_aligned = 0;
    यदि (op == HTP_OP_RMS_NORM_MUL) {
        // RMS_NORM_MUL fusion is F32-only; its weight tensor is always F32.
        ध्रुव size_t src1_दत्त_row_आकार = ne11 * परिमाण(भिन्न);
        src1_row_आकार_aligned = hex_आवर्तनम्_ऊर्ध्वम्(src1_दत्त_row_आकार, 128);
    }

    size_t vtcm_आकार_प्रत्येकम्_row = 0;
    size_t vtcm_row_प्रत्येकम्_thread = 0;

    यदि (op == HTP_OP_RMS_NORM_MUL) {
        यदि (प्रसारणम्_weight) {
            size_t उपलब्धम्_vtcm = vtcm_आकार;
            size_t src1_vtcm_योग = n_तन्तवः * src1_row_आकार_aligned;
            यदि (उपलब्धम्_vtcm > src1_vtcm_योग) {
                उपलब्धम्_vtcm -= src1_vtcm_योग;
            } अन्यथा {
                उपलब्धम्_vtcm = 0;
            }
            vtcm_आकार_प्रत्येकम्_row = 2 * (src0_row_आकार_aligned + dst_row_आकार_aligned);
            vtcm_row_प्रत्येकम्_thread = उपलब्धम्_vtcm / (n_तन्तवः * vtcm_आकार_प्रत्येकम्_row);
        } अन्यथा {
            vtcm_आकार_प्रत्येकम्_row = 2 * (src0_row_आकार_aligned + dst_row_आकार_aligned + src1_row_आकार_aligned);
            vtcm_row_प्रत्येकम्_thread = vtcm_आकार / (n_तन्तवः * vtcm_आकार_प्रत्येकम्_row);
        }
    } अन्यथा {
        vtcm_आकार_प्रत्येकम्_row   = 2 * (src0_row_आकार_aligned + dst_row_आकार_aligned);
        vtcm_row_प्रत्येकम्_thread = vtcm_आकार / (n_तन्तवः * vtcm_आकार_प्रत्येकम्_row);
    }

    ध्रुव bool is_reduction = (op == HTP_OP_NORM || op == HTP_OP_RMS_NORM ||
                               op == HTP_OP_RMS_NORM_MUL || op == HTP_OP_L2_NORM ||
                               op == HTP_OP_HADAMARD);
    // The tiled fallback path below only has F32 task functions (unary_task_f32_tiled_*);
    // F16 has no tiled kernels, so it must stay on the row-block path like reduction ops.
    // NOTE: if F16 ends up with vtcm_row_per_thread == 0 here (row too large for the VTCM
    // budget), execute_op_unary() will see BLOCK == 0 and skip computation for that op
    // (logged via FARF(ERROR, ...)) since there is no F16 tiled fallback. This is a known
    // limitation; supporting it would require adding F16 tiled kernels.
    ध्रुव bool is_f16 = (elem_आकार == परिमाण(_Float16));
    uint32_t col_tile = 0;

    यदि (vtcm_row_प्रत्येकम्_thread == 0 && !is_reduction && !is_f16) {
        ध्रुव size_t प्रत्येकम्_thread_budget = vtcm_आकार / n_तन्तवः;
        ध्रुव size_t col_tile_अष्टकानि = hex_align_अधस्तात्(प्रत्येकम्_thread_budget / 4, 128);
        col_tile = (uint32_t) (col_tile_अष्टकानि / elem_आकार);

        L->src0_अष्टकानि = col_tile_अष्टकानि * 2;
        L->dst_अष्टकानि  = col_tile_अष्टकानि * 2;
        L->src1_अष्टकानि = 0;
    } अन्यथा {
        L->src0_अष्टकानि = src0_row_आकार_aligned * vtcm_row_प्रत्येकम्_thread * 2;
        L->dst_अष्टकानि  = dst_row_आकार_aligned * vtcm_row_प्रत्येकम्_thread * 2;
        यदि (op == HTP_OP_RMS_NORM_MUL) {
            यदि (प्रसारणम्_weight) {
                L->src1_अष्टकानि = src1_row_आकार_aligned;
            } अन्यथा {
                L->src1_अष्टकानि = src1_row_आकार_aligned * vtcm_row_प्रत्येकम्_thread * 2;
            }
        } अन्यथा {
            L->src1_अष्टकानि = 0;
        }
    }

    L->off_src0 = 0;
    यदि (op == HTP_OP_RMS_NORM_MUL) {
        L->off_src1 = L->off_src0 + L->src0_अष्टकानि * n_तन्तवः;
        L->off_dst  = L->off_src1 + L->src1_अष्टकानि * n_तन्तवः;
    } अन्यथा {
        L->off_src1 = 0;
        L->off_dst  = L->off_src0 + L->src0_अष्टकानि * n_तन्तवः;
    }

    L->योग_अष्टकानि = L->off_dst + L->dst_अष्टकानि * n_तन्तवः;

    *बहिर्गमनम्_col_tile = col_tile;
    *बहिर्गमनम्_vtcm_row_प्रत्येकम्_thread = vtcm_row_प्रत्येकम्_thread;
}

#यदि_अन्तः /* HTP_UNARY_OPS_H */
