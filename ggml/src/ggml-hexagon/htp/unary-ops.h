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
    संरचना fastdiv_मूल्ये div_ne01;
    संरचना fastdiv_मूल्ये div_ne02;
    संरचना fastdiv_मूल्ये div_ne012;
    संरचना fastdiv_मूल्ये div_tpr;
};

#जर defined(__cplusplus)
static_assert(आकार(संरचना htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(आकार(संरचना htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#endif

स्थिर ओळीत bool htp_op_is_unary(uint32_t opcode) {
    निवड (opcode) {
        स्थिती HTP_OP_CLAMP:
        स्थिती HTP_OP_LEAKY_RELU:
        स्थिती HTP_OP_NORM:
        स्थिती HTP_OP_RMS_NORM:
        स्थिती HTP_OP_RMS_NORM_MUL:
        स्थिती HTP_OP_SCALE:
        स्थिती HTP_OP_SQR:
        स्थिती HTP_OP_SQRT:
        स्थिती HTP_OP_UNARY_NEG:
        स्थिती HTP_OP_UNARY_EXP:
        स्थिती HTP_OP_UNARY_SIGMOID:
        स्थिती HTP_OP_UNARY_SILU:
        स्थिती HTP_OP_UNARY_GELU:
        स्थिती HTP_OP_UNARY_GELU_QUICK:
        स्थिती HTP_OP_UNARY_GELU_ERF:
        स्थिती HTP_OP_UNARY_SOFTPLUS:
        स्थिती HTP_OP_UNARY_TANH:
        स्थिती HTP_OP_UNARY_ABS:
        स्थिती HTP_OP_UNARY_LOG:
        स्थिती HTP_OP_UNARY_RELU:
        स्थिती HTP_OP_UNARY_STEP:
        स्थिती HTP_OP_L2_NORM:
        स्थिती HTP_OP_TRI:
        स्थिती HTP_OP_HADAMARD:
            परत true;
        पूर्वनिर्धारित:
            परत false;
    }
}

संरचना htp_unary_vtcm_layout {
    माप_प्रकार total_bytes;
    माप_प्रकार off_src0;
    माप_प्रकार off_src1;
    माप_प्रकार off_dst;

    माप_प्रकार src0_bytes;
    माप_प्रकार src1_bytes;
    माप_प्रकार dst_bytes;
};

स्थिर ओळीत शून्य htp_unary_vtcm_layout_build(
    संरचना htp_unary_vtcm_layout * L,
    uint32_t op,
    uint32_t ne00,
    uint32_t ne10,
    uint32_t ne11,
    bool broadcast_weight,
    uint32_t n_threads,
    माप_प्रकार vtcm_size,
    माप_प्रकार elem_size,
    uint32_t * out_col_tile,
    uint32_t * out_vtcm_row_per_thread
) {
    स्थिरांक माप_प्रकार dst_elem_size      = (op == HTP_OP_HADAMARD) ? आकार(दशांश) : elem_size;
    स्थिरांक माप_प्रकार src0_data_row_size = ne00 * elem_size;
    स्थिरांक माप_प्रकार dst_data_row_size  = ne10 * dst_elem_size;

    स्थिरांक माप_प्रकार src0_row_size_aligned = hex_round_up(src0_data_row_size, 128);
    स्थिरांक माप_प्रकार dst_row_size_aligned  = hex_round_up(dst_data_row_size,  128);

    माप_प्रकार src1_row_size_aligned = 0;
    जर (op == HTP_OP_RMS_NORM_MUL) {
        // RMS_NORM_MUL fusion is F32-only; its weight tensor is always F32.
        स्थिरांक माप_प्रकार src1_data_row_size = ne11 * आकार(दशांश);
        src1_row_size_aligned = hex_round_up(src1_data_row_size, 128);
    }

    माप_प्रकार vtcm_size_per_row = 0;
    माप_प्रकार vtcm_row_per_thread = 0;

    जर (op == HTP_OP_RMS_NORM_MUL) {
        जर (broadcast_weight) {
            माप_प्रकार available_vtcm = vtcm_size;
            माप_प्रकार src1_vtcm_total = n_threads * src1_row_size_aligned;
            जर (available_vtcm > src1_vtcm_total) {
                available_vtcm -= src1_vtcm_total;
            } अन्यथा {
                available_vtcm = 0;
            }
            vtcm_size_per_row = 2 * (src0_row_size_aligned + dst_row_size_aligned);
            vtcm_row_per_thread = available_vtcm / (n_threads * vtcm_size_per_row);
        } अन्यथा {
            vtcm_size_per_row = 2 * (src0_row_size_aligned + dst_row_size_aligned + src1_row_size_aligned);
            vtcm_row_per_thread = vtcm_size / (n_threads * vtcm_size_per_row);
        }
    } अन्यथा {
        vtcm_size_per_row   = 2 * (src0_row_size_aligned + dst_row_size_aligned);
        vtcm_row_per_thread = vtcm_size / (n_threads * vtcm_size_per_row);
    }

    स्थिरांक bool is_reduction = (op == HTP_OP_NORM || op == HTP_OP_RMS_NORM ||
                               op == HTP_OP_RMS_NORM_MUL || op == HTP_OP_L2_NORM ||
                               op == HTP_OP_HADAMARD);
    // The tiled fallback path below only has F32 task functions (unary_task_f32_tiled_*);
    // F16 has no tiled kernels, so it must stay on the row-block path like reduction ops.
    // NOTE: if F16 ends up with vtcm_row_per_thread == 0 here (row too large for the VTCM
    // budget), execute_op_unary() will see BLOCK == 0 and skip computation for that op
    // (logged via FARF(ERROR, ...)) since there is no F16 tiled fallback. This is a known
    // limitation; supporting it would require adding F16 tiled kernels.
    स्थिरांक bool is_f16 = (elem_size == आकार(_Float16));
    uint32_t col_tile = 0;

    जर (vtcm_row_per_thread == 0 && !is_reduction && !is_f16) {
        स्थिरांक माप_प्रकार per_thread_budget = vtcm_size / n_threads;
        स्थिरांक माप_प्रकार col_tile_bytes = hex_align_down(per_thread_budget / 4, 128);
        col_tile = (uint32_t) (col_tile_bytes / elem_size);

        L->src0_bytes = col_tile_bytes * 2;
        L->dst_bytes  = col_tile_bytes * 2;
        L->src1_bytes = 0;
    } अन्यथा {
        L->src0_bytes = src0_row_size_aligned * vtcm_row_per_thread * 2;
        L->dst_bytes  = dst_row_size_aligned * vtcm_row_per_thread * 2;
        जर (op == HTP_OP_RMS_NORM_MUL) {
            जर (broadcast_weight) {
                L->src1_bytes = src1_row_size_aligned;
            } अन्यथा {
                L->src1_bytes = src1_row_size_aligned * vtcm_row_per_thread * 2;
            }
        } अन्यथा {
            L->src1_bytes = 0;
        }
    }

    L->off_src0 = 0;
    जर (op == HTP_OP_RMS_NORM_MUL) {
        L->off_src1 = L->off_src0 + L->src0_bytes * n_threads;
        L->off_dst  = L->off_src1 + L->src1_bytes * n_threads;
    } अन्यथा {
        L->off_src1 = 0;
        L->off_dst  = L->off_src0 + L->src0_bytes * n_threads;
    }

    L->total_bytes = L->off_dst + L->dst_bytes * n_threads;

    *out_col_tile = col_tile;
    *out_vtcm_row_per_thread = vtcm_row_per_thread;
}

#endif /* HTP_UNARY_OPS_H */
