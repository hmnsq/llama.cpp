#ifndef HTP_UNARY_OPS_H
#define HTP_UNARY_OPS_H

#અન્તર્ભાવ "hex-common.h"
#અન્તર્ભાવ "htp-ops.h"

// Op-specific struct for precomputed unary params
સંરચના htp_unary_kernel_params {
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
    સંરચના fastdiv_કિંમતો div_ne01;
    સંરચના fastdiv_કિંમતો div_ne02;
    સંરચના fastdiv_કિંમતો div_ne012;
    સંરચના fastdiv_કિંમતો div_tpr;
};

#જો defined(__cplusplus)
static_assert(કદ(સંરચના htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#નહીંતર
_Static_assert(કદ(સંરચના htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#endif

સ્થિર અંતઃરેખ bool htp_op_is_unary(uint32_t opcode) {
    પસંદગી (opcode) {
        સ્થિતિ HTP_OP_CLAMP:
        સ્થિતિ HTP_OP_LEAKY_RELU:
        સ્થિતિ HTP_OP_NORM:
        સ્થિતિ HTP_OP_RMS_NORM:
        સ્થિતિ HTP_OP_RMS_NORM_MUL:
        સ્થિતિ HTP_OP_SCALE:
        સ્થિતિ HTP_OP_SQR:
        સ્થિતિ HTP_OP_SQRT:
        સ્થિતિ HTP_OP_UNARY_NEG:
        સ્થિતિ HTP_OP_UNARY_EXP:
        સ્થિતિ HTP_OP_UNARY_SIGMOID:
        સ્થિતિ HTP_OP_UNARY_SILU:
        સ્થિતિ HTP_OP_UNARY_GELU:
        સ્થિતિ HTP_OP_UNARY_GELU_QUICK:
        સ્થિતિ HTP_OP_UNARY_GELU_ERF:
        સ્થિતિ HTP_OP_UNARY_SOFTPLUS:
        સ્થિતિ HTP_OP_UNARY_TANH:
        સ્થિતિ HTP_OP_UNARY_ABS:
        સ્થિતિ HTP_OP_UNARY_LOG:
        સ્થિતિ HTP_OP_UNARY_RELU:
        સ્થિતિ HTP_OP_UNARY_STEP:
        સ્થિતિ HTP_OP_L2_NORM:
        સ્થિતિ HTP_OP_TRI:
        સ્થિતિ HTP_OP_HADAMARD:
            પાછા true;
        મૂળભૂત:
            પાછા false;
    }
}

સંરચના htp_unary_vtcm_layout {
    size_t total_bytes;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_bytes;
    size_t src1_bytes;
    size_t dst_bytes;
};

સ્થિર અંતઃરેખ શૂન્ય htp_unary_vtcm_layout_build(
    સંરચના htp_unary_vtcm_layout * L,
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
    અચલ size_t dst_elem_size      = (op == HTP_OP_HADAMARD) ? કદ(દશાંશ) : elem_size;
    અચલ size_t src0_data_row_size = ne00 * elem_size;
    અચલ size_t dst_data_row_size  = ne10 * dst_elem_size;

    અચલ size_t src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    અચલ size_t dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    size_t src1_row_size_aligned = 0;
    જો (op == HTP_OP_RMS_NORM_MUL) {
        // RMS_NORM_MUL fusion is F32-only; its weight tensor is always F32.
        અચલ size_t src1_data_row_size = ne11 * કદ(દશાંશ);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
    }

    size_t vtcm_size_per_row = 0;
    size_t vtcm_row_per_thread = 0;

    જો (op == HTP_OP_RMS_NORM_MUL) {
        જો (broadcast_weight) {
            size_t available_vtcm = vtcm_size;
            size_t src1_vtcm_total = n_threads * src1_row_size_aligned;
            જો (available_vtcm > src1_vtcm_total) {
                available_vtcm -= src1_vtcm_total;
            } નહીંતર {
                available_vtcm = 0;
            }
            vtcm_size_per_row = 2 * (src0_row_size_aligned + dst_row_size_aligned);
            vtcm_row_per_thread = available_vtcm / (n_threads * vtcm_size_per_row);
        } નહીંતર {
            vtcm_size_per_row = 2 * (src0_row_size_aligned + dst_row_size_aligned + src1_row_size_aligned);
            vtcm_row_per_thread = vtcm_size / (n_threads * vtcm_size_per_row);
        }
    } નહીંતર {
        vtcm_size_per_row   = 2 * (src0_row_size_aligned + dst_row_size_aligned);
        vtcm_row_per_thread = vtcm_size / (n_threads * vtcm_size_per_row);
    }

    અચલ bool is_reduction = (op == HTP_OP_NORM || op == HTP_OP_RMS_NORM ||
                               op == HTP_OP_RMS_NORM_MUL || op == HTP_OP_L2_NORM ||
                               op == HTP_OP_HADAMARD);
    // The tiled fallback path below only has F32 task functions (unary_task_f32_tiled_*);
    // F16 has no tiled kernels, so it must stay on the row-block path like reduction ops.
    // NOTE: if F16 ends up with vtcm_row_per_thread == 0 here (row too large for the VTCM
    // budget), execute_op_unary() will see BLOCK == 0 and skip computation for that op
    // (logged via FARF(ERROR, ...)) since there is no F16 tiled fallback. This is a known
    // limitation; supporting it would require adding F16 tiled kernels.
    અચલ bool is_f16 = (elem_size == કદ(_Float16));
    uint32_t col_tile = 0;

    જો (vtcm_row_per_thread == 0 && !is_reduction && !is_f16) {
        અચલ size_t per_thread_budget = vtcm_size / n_threads;
        અચલ size_t col_tile_bytes = hex_align_down(per_thread_budget / 4, 128);
        col_tile = (uint32_t) (col_tile_bytes / elem_size);

        L->src0_bytes = col_tile_bytes * 2;
        L->dst_bytes  = col_tile_bytes * 2;
        L->src1_bytes = 0;
    } નહીંતર {
        L->src0_bytes = src0_row_size_aligned * vtcm_row_per_thread * 2;
        L->dst_bytes  = dst_row_size_aligned * vtcm_row_per_thread * 2;
        જો (op == HTP_OP_RMS_NORM_MUL) {
            જો (broadcast_weight) {
                L->src1_bytes = src1_row_size_aligned;
            } નહીંતર {
                L->src1_bytes = src1_row_size_aligned * vtcm_row_per_thread * 2;
            }
        } નહીંતર {
            L->src1_bytes = 0;
        }
    }

    L->off_src0 = 0;
    જો (op == HTP_OP_RMS_NORM_MUL) {
        L->off_src1 = L->off_src0 + L->src0_bytes * n_threads;
        L->off_dst  = L->off_src1 + L->src1_bytes * n_threads;
    } નહીંતર {
        L->off_src1 = 0;
        L->off_dst  = L->off_src0 + L->src0_bytes * n_threads;
    }

    L->total_bytes = L->off_dst + L->dst_bytes * n_threads;

    *out_col_tile = col_tile;
    *out_vtcm_row_per_thread = vtcm_row_per_thread;
}

#endif /* HTP_UNARY_OPS_H */
