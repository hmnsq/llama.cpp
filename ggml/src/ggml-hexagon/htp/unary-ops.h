#ifndef HTP_UNARY_OPS_H
#define HTP_UNARY_OPS_H

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "htp-ops.h"

// Op-specific struct for precomputed unary params
संरचना htp_unary_kernel_params {
    uint32_t  n_threads;
    uint32_t  col_tile;
    uint32_t  vtcm_row_per_thread;
    uint32_t  block;
    uint32_t  broadcast_weight;

    uint32_t  vtcm_src0_size_per_thread;
    uint32_t  vtcm_src1_size_per_thread;
    uint32_t  vtcm_dst_size_per_thread;

    uint32_t  vtcm_src0_size;
    uint32_t  vtcm_src1_size;
    uint32_t  vtcm_dst_size;

    uint32_t  src0_row_size_aligned;
    uint32_t  src1_row_size_aligned;
    uint32_t  dst_row_size_aligned;

    uint32_t  vtcm_size;

    // Fastdiv helpers
    संरचना fastdiv_मानानि div_ne01;
    संरचना fastdiv_मानानि div_ne02;
    संरचना fastdiv_मानानि div_ne012;
    संरचना fastdiv_मानानि div_tpr;
};

#जइ defined(__cplusplus)
static_assert(परिमाण(संरचना htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#अण्णहा
_Static_assert(परिमाण(संरचना htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#endif

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
        पक्ष HTP_OP_UNARY_GELU_QUICK:
        पक्ष HTP_OP_UNARY_GELU_ERF:
        पक्ष HTP_OP_UNARY_SOFTPLUS:
        पक्ष HTP_OP_UNARY_TANH:
        पक्ष HTP_OP_UNARY_ABS:
        पक्ष HTP_OP_UNARY_LOG:
        पक्ष HTP_OP_UNARY_RELU:
        पक्ष HTP_OP_UNARY_STEP:
        पक्ष HTP_OP_L2_NORM:
        पक्ष HTP_OP_TRI:
        पक्ष HTP_OP_HADAMARD:
            प्रत्यागम true;
        स्वभाव:
            प्रत्यागम false;
    }
}

संरचना htp_unary_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_bytes;
    size_t src1_bytes;
    size_t dst_bytes;
};

स्थिर अन्तःस्थ शून्य htp_unary_vtcm_layout_build(
    संरचना htp_unary_vtcm_layout * L,
    uint32_t op,
    uint32_t ne00,
    uint32_t ne10,
    uint32_t ne11,
    bool broadcast_weight,
    uint32_t n_threads,
    size_t vtcm_size,
    size_t elem_size,
    uint32_t * out_col_tile,
    uint32_t * out_vtcm_row_per_thread
) {
    ध्रुव size_t dst_elem_size      = (op == HTP_OP_HADAMARD) ? परिमाण(भिन्न) : elem_size;
    ध्रुव size_t src0_data_row_size = ne00 * elem_size;
    ध्रुव size_t dst_data_row_size  = ne10 * dst_elem_size;

    ध्रुव size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    ध्रुव size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    size_t src1_row_size_aligned = 0;
    जइ (op == HTP_OP_RMS_NORM_MUL) {
        // RMS_NORM_MUL fusion is F32-only; its weight tensor is always F32.
        ध्रुव size_t src1_data_row_size = ne11 * परिमाण(भिन्न);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
    }

    size_t vtcm_size_per_row = 0;
    size_t vtcm_row_per_thread = 0;

    जइ (op == HTP_OP_RMS_NORM_MUL) {
        जइ (broadcast_weight) {
            size_t available_vtcm = vtcm_size;
            size_t src1_vtcm_total = n_threads * src1_row_size_aligned;
            जइ (available_vtcm > src1_vtcm_total) {
                available_vtcm -= src1_vtcm_total;
            } अण्णहा {
                available_vtcm = 0;
            }
            vtcm_size_per_row = 2 * (src0_row_size_aligned + dst_row_size_aligned);
            vtcm_row_per_thread = available_vtcm / (n_threads * vtcm_size_per_row);
        } अण्णहा {
            vtcm_size_per_row = 2 * (src0_row_size_aligned + dst_row_size_aligned + src1_row_size_aligned);
            vtcm_row_per_thread = vtcm_size / (n_threads * vtcm_size_per_row);
        }
    } अण्णहा {
        vtcm_size_per_row   = 2 * (src0_row_size_aligned + dst_row_size_aligned);
        vtcm_row_per_thread = vtcm_size / (n_threads * vtcm_size_per_row);
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
    ध्रुव bool is_f16 = (elem_size == परिमाण(_Float16));
    uint32_t col_tile = 0;

    जइ (vtcm_row_per_thread == 0 && !is_reduction && !is_f16) {
        ध्रुव size_t per_thread_budget = vtcm_size / n_threads;
        ध्रुव size_t col_tile_bytes = hex_align_down(per_thread_budget / 4, 128);
        col_tile = (uint32_t) (col_tile_bytes / elem_size);

        L->src0_bytes = col_tile_bytes * 2;
        L->dst_bytes  = col_tile_bytes * 2;
        L->src1_bytes = 0;
    } अण्णहा {
        L->src0_bytes = src0_row_size_aligned * vtcm_row_per_thread * 2;
        L->dst_bytes  = dst_row_size_aligned * vtcm_row_per_thread * 2;
        जइ (op == HTP_OP_RMS_NORM_MUL) {
            जइ (broadcast_weight) {
                L->src1_bytes = src1_row_size_aligned;
            } अण्णहा {
                L->src1_bytes = src1_row_size_aligned * vtcm_row_per_thread * 2;
            }
        } अण्णहा {
            L->src1_bytes = 0;
        }
    }

    L->off_src0 = 0;
    जइ (op == HTP_OP_RMS_NORM_MUL) {
        L->off_src1 = L->off_src0 + L->src0_bytes * n_threads;
        L->off_dst  = L->off_src1 + L->src1_bytes * n_threads;
    } अण्णहा {
        L->off_src1 = 0;
        L->off_dst  = L->off_src0 + L->src0_bytes * n_threads;
    }

    L->total_bytes = L->off_dst + L->dst_bytes * n_threads;

    *out_col_tile = col_tile;
    *out_vtcm_row_per_thread = vtcm_row_per_thread;
}

#endif /* HTP_UNARY_OPS_H */
