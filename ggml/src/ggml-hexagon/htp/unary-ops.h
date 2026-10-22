#ifndef HTP_UNARY_OPS_H
#define HTP_UNARY_OPS_H

#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "htp-ops.h"

// Op-specific struct for precomputed unary params
কাঠামো htp_unary_kernel_params {
    uint32_t  n_threads;
    uint32_t  col_tile;
    uint32_t  vtcm_row_per_thread;
    uint32_t  block;
    uint32_t  broadcast_weight;

    uint32_t  vtcm_src0_আকার_per_thread;
    uint32_t  vtcm_src1_আকার_per_thread;
    uint32_t  vtcm_dst_আকার_per_thread;

    uint32_t  vtcm_src0_আকার;
    uint32_t  vtcm_src1_আকার;
    uint32_t  vtcm_dst_আকার;

    uint32_t  src0_row_আকার_aligned;
    uint32_t  src1_row_আকার_aligned;
    uint32_t  dst_row_আকার_aligned;

    uint32_t  vtcm_আকার;

    // Fastdiv helpers
    কাঠামো fastdiv_মানসমূহ div_ne01;
    কাঠামো fastdiv_মানসমূহ div_ne02;
    কাঠামো fastdiv_মানসমূহ div_ne012;
    কাঠামো fastdiv_মানসমূহ div_tpr;
};

#যদি defined(__cplusplus)
static_assert(আকার(কাঠামো htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#অন্যথা
_Static_assert(আকার(কাঠামো htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#endif

স্থির অন্তঃরেখ bool htp_op_is_unary(uint32_t opcode) {
    নির্বাচন (opcode) {
        ক্ষেত্র HTP_OP_CLAMP:
        ক্ষেত্র HTP_OP_LEAKY_RELU:
        ক্ষেত্র HTP_OP_NORM:
        ক্ষেত্র HTP_OP_RMS_NORM:
        ক্ষেত্র HTP_OP_RMS_NORM_MUL:
        ক্ষেত্র HTP_OP_SCALE:
        ক্ষেত্র HTP_OP_SQR:
        ক্ষেত্র HTP_OP_SQRT:
        ক্ষেত্র HTP_OP_UNARY_NEG:
        ক্ষেত্র HTP_OP_UNARY_EXP:
        ক্ষেত্র HTP_OP_UNARY_SIGMOID:
        ক্ষেত্র HTP_OP_UNARY_SILU:
        ক্ষেত্র HTP_OP_UNARY_GELU:
        ক্ষেত্র HTP_OP_UNARY_GELU_QUICK:
        ক্ষেত্র HTP_OP_UNARY_GELU_ERF:
        ক্ষেত্র HTP_OP_UNARY_SOFTPLUS:
        ক্ষেত্র HTP_OP_UNARY_TANH:
        ক্ষেত্র HTP_OP_UNARY_ABS:
        ক্ষেত্র HTP_OP_UNARY_LOG:
        ক্ষেত্র HTP_OP_UNARY_RELU:
        ক্ষেত্র HTP_OP_UNARY_STEP:
        ক্ষেত্র HTP_OP_L2_NORM:
        ক্ষেত্র HTP_OP_TRI:
        ক্ষেত্র HTP_OP_HADAMARD:
            ফেরত true;
        পূর্বনির্ধারিত:
            ফেরত false;
    }
}

কাঠামো htp_unary_vtcm_layout {
    size_t যোগ_bytes;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_bytes;
    size_t src1_bytes;
    size_t dst_bytes;
};

স্থির অন্তঃরেখ শূন্য htp_unary_vtcm_layout_build(
    কাঠামো htp_unary_vtcm_layout * L,
    uint32_t op,
    uint32_t ne00,
    uint32_t ne10,
    uint32_t ne11,
    bool broadcast_weight,
    uint32_t n_threads,
    size_t vtcm_আকার,
    size_t elem_আকার,
    uint32_t * out_col_tile,
    uint32_t * out_vtcm_row_per_thread
) {
    ধ্রুবক size_t dst_elem_আকার      = (op == HTP_OP_HADAMARD) ? আকার(দশমিক) : elem_আকার;
    ধ্রুবক size_t src0_তথ্য_row_আকার = ne00 * elem_আকার;
    ধ্রুবক size_t dst_তথ্য_row_আকার  = ne10 * dst_elem_আকার;

    ধ্রুবক size_t src0_row_আকার_aligned = hex_round_up(src0_তথ্য_row_আকার, 128);
    ধ্রুবক size_t dst_row_আকার_aligned  = hex_round_up(dst_তথ্য_row_আকার,  128);

    size_t src1_row_আকার_aligned = 0;
    যদি (op == HTP_OP_RMS_NORM_MUL) {
        // RMS_NORM_MUL fusion is F32-only; its weight tensor is always F32.
        ধ্রুবক size_t src1_তথ্য_row_আকার = ne11 * আকার(দশমিক);
        src1_row_আকার_aligned = hex_round_up(src1_তথ্য_row_আকার, 128);
    }

    size_t vtcm_আকার_per_row = 0;
    size_t vtcm_row_per_thread = 0;

    যদি (op == HTP_OP_RMS_NORM_MUL) {
        যদি (broadcast_weight) {
            size_t available_vtcm = vtcm_আকার;
            size_t src1_vtcm_যোগ = n_threads * src1_row_আকার_aligned;
            যদি (available_vtcm > src1_vtcm_যোগ) {
                available_vtcm -= src1_vtcm_যোগ;
            } অন্যথা {
                available_vtcm = 0;
            }
            vtcm_আকার_per_row = 2 * (src0_row_আকার_aligned + dst_row_আকার_aligned);
            vtcm_row_per_thread = available_vtcm / (n_threads * vtcm_আকার_per_row);
        } অন্যথা {
            vtcm_আকার_per_row = 2 * (src0_row_আকার_aligned + dst_row_আকার_aligned + src1_row_আকার_aligned);
            vtcm_row_per_thread = vtcm_আকার / (n_threads * vtcm_আকার_per_row);
        }
    } অন্যথা {
        vtcm_আকার_per_row   = 2 * (src0_row_আকার_aligned + dst_row_আকার_aligned);
        vtcm_row_per_thread = vtcm_আকার / (n_threads * vtcm_আকার_per_row);
    }

    ধ্রুবক bool is_reduction = (op == HTP_OP_NORM || op == HTP_OP_RMS_NORM ||
                               op == HTP_OP_RMS_NORM_MUL || op == HTP_OP_L2_NORM ||
                               op == HTP_OP_HADAMARD);
    // The tiled fallback path below only has F32 task functions (unary_task_f32_tiled_*);
    // F16 has no tiled kernels, so it must stay on the row-block path like reduction ops.
    // NOTE: if F16 ends up with vtcm_row_per_thread == 0 here (row too large for the VTCM
    // budget), execute_op_unary() will see BLOCK == 0 and skip computation for that op
    // (logged via FARF(ERROR, ...)) since there is no F16 tiled fallback. This is a known
    // limitation; supporting it would require adding F16 tiled kernels.
    ধ্রুবক bool is_f16 = (elem_আকার == আকার(_Float16));
    uint32_t col_tile = 0;

    যদি (vtcm_row_per_thread == 0 && !is_reduction && !is_f16) {
        ধ্রুবক size_t per_thread_budget = vtcm_আকার / n_threads;
        ধ্রুবক size_t col_tile_bytes = hex_align_down(per_thread_budget / 4, 128);
        col_tile = (uint32_t) (col_tile_bytes / elem_আকার);

        L->src0_bytes = col_tile_bytes * 2;
        L->dst_bytes  = col_tile_bytes * 2;
        L->src1_bytes = 0;
    } অন্যথা {
        L->src0_bytes = src0_row_আকার_aligned * vtcm_row_per_thread * 2;
        L->dst_bytes  = dst_row_আকার_aligned * vtcm_row_per_thread * 2;
        যদি (op == HTP_OP_RMS_NORM_MUL) {
            যদি (broadcast_weight) {
                L->src1_bytes = src1_row_আকার_aligned;
            } অন্যথা {
                L->src1_bytes = src1_row_আকার_aligned * vtcm_row_per_thread * 2;
            }
        } অন্যথা {
            L->src1_bytes = 0;
        }
    }

    L->off_src0 = 0;
    যদি (op == HTP_OP_RMS_NORM_MUL) {
        L->off_src1 = L->off_src0 + L->src0_bytes * n_threads;
        L->off_dst  = L->off_src1 + L->src1_bytes * n_threads;
    } অন্যথা {
        L->off_src1 = 0;
        L->off_dst  = L->off_src0 + L->src0_bytes * n_threads;
    }

    L->যোগ_bytes = L->off_dst + L->dst_bytes * n_threads;

    *out_col_tile = col_tile;
    *out_vtcm_row_per_thread = vtcm_row_per_thread;
}

#endif /* HTP_UNARY_OPS_H */
