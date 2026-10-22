#अगर_अघोषित HTP_UNARY_OPS_H
#घोषणा HTP_UNARY_OPS_H

#समावेश "hex-common.h"
#समावेश "htp-ops.h"

// Op-specific struct for precomputed unary params
काष्ठा htp_unary_kernel_params {
    uint32_t  n_threads;
    uint32_t  col_tile;
    uint32_t  vtcm_row_per_thread;
    uint32_t  block;
    uint32_t  broadcast_weight;

    uint32_t  vtcm_src0_आकार_per_thread;
    uint32_t  vtcm_src1_आकार_per_thread;
    uint32_t  vtcm_dst_आकार_per_thread;

    uint32_t  vtcm_src0_आकार;
    uint32_t  vtcm_src1_आकार;
    uint32_t  vtcm_dst_आकार;

    uint32_t  src0_row_आकार_aligned;
    uint32_t  src1_row_आकार_aligned;
    uint32_t  dst_row_आकार_aligned;

    uint32_t  vtcm_आकार;

    // Fastdiv helpers
    काष्ठा fastdiv_मूल्यावली div_ne01;
    काष्ठा fastdiv_मूल्यावली div_ne02;
    काष्ठा fastdiv_मूल्यावली div_ne012;
    काष्ठा fastdiv_मूल्यावली div_tpr;
};

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(माप(काष्ठा htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#पूर्ण_अगर

अटल अंतरभूत bool htp_op_is_unary(uint32_t opcode) {
    चयन (opcode) {
        हाल HTP_OP_CLAMP:
        हाल HTP_OP_LEAKY_RELU:
        हाल HTP_OP_NORM:
        हाल HTP_OP_RMS_NORM:
        हाल HTP_OP_RMS_NORM_MUL:
        हाल HTP_OP_SCALE:
        हाल HTP_OP_SQR:
        हाल HTP_OP_SQRT:
        हाल HTP_OP_UNARY_NEG:
        हाल HTP_OP_UNARY_EXP:
        हाल HTP_OP_UNARY_SIGMOID:
        हाल HTP_OP_UNARY_SILU:
        हाल HTP_OP_UNARY_GELU:
        हाल HTP_OP_UNARY_GELU_QUICK:
        हाल HTP_OP_UNARY_GELU_ERF:
        हाल HTP_OP_UNARY_SOFTPLUS:
        हाल HTP_OP_UNARY_TANH:
        हाल HTP_OP_UNARY_ABS:
        हाल HTP_OP_UNARY_LOG:
        हाल HTP_OP_UNARY_RELU:
        हाल HTP_OP_UNARY_STEP:
        हाल HTP_OP_L2_NORM:
        हाल HTP_OP_TRI:
        हाल HTP_OP_HADAMARD:
            वापस true;
        शेष:
            वापस false;
    }
}

काष्ठा htp_unary_vtcm_layout {
    माप_प्रकार योग_bytes;
    माप_प्रकार off_src0;
    माप_प्रकार off_src1;
    माप_प्रकार off_dst;

    माप_प्रकार src0_bytes;
    माप_प्रकार src1_bytes;
    माप_प्रकार dst_bytes;
};

अटल अंतरभूत व्योम htp_unary_vtcm_layout_build(
    काष्ठा htp_unary_vtcm_layout * L,
    uint32_t op,
    uint32_t ne00,
    uint32_t ne10,
    uint32_t ne11,
    bool broadcast_weight,
    uint32_t n_threads,
    माप_प्रकार vtcm_आकार,
    माप_प्रकार elem_आकार,
    uint32_t * out_col_tile,
    uint32_t * out_vtcm_row_per_thread
) {
    स्थिर माप_प्रकार dst_elem_आकार      = (op == HTP_OP_HADAMARD) ? माप(भग्न) : elem_आकार;
    स्थिर माप_प्रकार src0_तथ्य_row_आकार = ne00 * elem_आकार;
    स्थिर माप_प्रकार dst_तथ्य_row_आकार  = ne10 * dst_elem_आकार;

    स्थिर माप_प्रकार src0_row_आकार_aligned = hex_round_up(src0_तथ्य_row_आकार, 128);
    स्थिर माप_प्रकार dst_row_आकार_aligned  = hex_round_up(dst_तथ्य_row_आकार,  128);

    माप_प्रकार src1_row_आकार_aligned = 0;
    अगर (op == HTP_OP_RMS_NORM_MUL) {
        // RMS_NORM_MUL fusion is F32-only; its weight tensor is always F32.
        स्थिर माप_प्रकार src1_तथ्य_row_आकार = ne11 * माप(भग्न);
        src1_row_आकार_aligned = hex_round_up(src1_तथ्य_row_आकार, 128);
    }

    माप_प्रकार vtcm_आकार_per_row = 0;
    माप_प्रकार vtcm_row_per_thread = 0;

    अगर (op == HTP_OP_RMS_NORM_MUL) {
        अगर (broadcast_weight) {
            माप_प्रकार available_vtcm = vtcm_आकार;
            माप_प्रकार src1_vtcm_योग = n_threads * src1_row_आकार_aligned;
            अगर (available_vtcm > src1_vtcm_योग) {
                available_vtcm -= src1_vtcm_योग;
            } अन्यथा {
                available_vtcm = 0;
            }
            vtcm_आकार_per_row = 2 * (src0_row_आकार_aligned + dst_row_आकार_aligned);
            vtcm_row_per_thread = available_vtcm / (n_threads * vtcm_आकार_per_row);
        } अन्यथा {
            vtcm_आकार_per_row = 2 * (src0_row_आकार_aligned + dst_row_आकार_aligned + src1_row_आकार_aligned);
            vtcm_row_per_thread = vtcm_आकार / (n_threads * vtcm_आकार_per_row);
        }
    } अन्यथा {
        vtcm_आकार_per_row   = 2 * (src0_row_आकार_aligned + dst_row_आकार_aligned);
        vtcm_row_per_thread = vtcm_आकार / (n_threads * vtcm_आकार_per_row);
    }

    स्थिर bool is_reduction = (op == HTP_OP_NORM || op == HTP_OP_RMS_NORM ||
                               op == HTP_OP_RMS_NORM_MUL || op == HTP_OP_L2_NORM ||
                               op == HTP_OP_HADAMARD);
    // The tiled fallback path below only has F32 task functions (unary_task_f32_tiled_*);
    // F16 has no tiled kernels, so it must stay on the row-block path like reduction ops.
    // NOTE: if F16 ends up with vtcm_row_per_thread == 0 here (row too large for the VTCM
    // budget), execute_op_unary() will see BLOCK == 0 and skip computation for that op
    // (logged via FARF(ERROR, ...)) since there is no F16 tiled fallback. This is a known
    // limitation; supporting it would require adding F16 tiled kernels.
    स्थिर bool is_f16 = (elem_आकार == माप(_Float16));
    uint32_t col_tile = 0;

    अगर (vtcm_row_per_thread == 0 && !is_reduction && !is_f16) {
        स्थिर माप_प्रकार per_thread_budget = vtcm_आकार / n_threads;
        स्थिर माप_प्रकार col_tile_bytes = hex_align_down(per_thread_budget / 4, 128);
        col_tile = (uint32_t) (col_tile_bytes / elem_आकार);

        L->src0_bytes = col_tile_bytes * 2;
        L->dst_bytes  = col_tile_bytes * 2;
        L->src1_bytes = 0;
    } अन्यथा {
        L->src0_bytes = src0_row_आकार_aligned * vtcm_row_per_thread * 2;
        L->dst_bytes  = dst_row_आकार_aligned * vtcm_row_per_thread * 2;
        अगर (op == HTP_OP_RMS_NORM_MUL) {
            अगर (broadcast_weight) {
                L->src1_bytes = src1_row_आकार_aligned;
            } अन्यथा {
                L->src1_bytes = src1_row_आकार_aligned * vtcm_row_per_thread * 2;
            }
        } अन्यथा {
            L->src1_bytes = 0;
        }
    }

    L->off_src0 = 0;
    अगर (op == HTP_OP_RMS_NORM_MUL) {
        L->off_src1 = L->off_src0 + L->src0_bytes * n_threads;
        L->off_dst  = L->off_src1 + L->src1_bytes * n_threads;
    } अन्यथा {
        L->off_src1 = 0;
        L->off_dst  = L->off_src0 + L->src0_bytes * n_threads;
    }

    L->योग_bytes = L->off_dst + L->dst_bytes * n_threads;

    *out_col_tile = col_tile;
    *out_vtcm_row_per_thread = vtcm_row_per_thread;
}

#पूर्ण_अगर /* HTP_UNARY_OPS_H */
