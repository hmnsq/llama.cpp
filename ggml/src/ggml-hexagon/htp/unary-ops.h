#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_UNARY_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_UNARY_OPS_H

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"

// Op-specific struct for precomputed unary params
ರಚನೆ htp_unary_kernel_params {
    uint32_t  n_ಎಳೆಗಳು;
    uint32_t  ಲಂಬಸಾಲು_tile;
    uint32_t  vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread;
    uint32_t  ಖಂಡ;
    uint32_t  ಪ್ರಸಾರ_ತೂಕ;

    uint32_t  vtcm_src0_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t  vtcm_src1_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    uint32_t  vtcm_dst_ಗಾತ್ರ_ಪ್ರತಿ_thread;

    uint32_t  vtcm_src0_ಗಾತ್ರ;
    uint32_t  vtcm_src1_ಗಾತ್ರ;
    uint32_t  vtcm_dst_ಗಾತ್ರ;

    uint32_t  src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    uint32_t  src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    uint32_t  dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    uint32_t  vtcm_ಗಾತ್ರ;

    // Fastdiv helpers
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne01;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne02;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne012;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_tpr;
};

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#ಇಲ್ಲದಿದ್ದರೆ
_Static_assert(ಗಾತ್ರ(ರಚನೆ htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_op_is_unary(uint32_t opcode) {
    ಆಯ್ಕೆ (opcode) {
        ಸ್ಥಿತಿ HTP_OP_CLAMP:
        ಸ್ಥಿತಿ HTP_OP_LEAKY_RELU:
        ಸ್ಥಿತಿ HTP_OP_NORM:
        ಸ್ಥಿತಿ HTP_OP_RMS_NORM:
        ಸ್ಥಿತಿ HTP_OP_RMS_NORM_MUL:
        ಸ್ಥಿತಿ HTP_OP_SCALE:
        ಸ್ಥಿತಿ HTP_OP_SQR:
        ಸ್ಥಿತಿ HTP_OP_SQRT:
        ಸ್ಥಿತಿ HTP_OP_UNARY_NEG:
        ಸ್ಥಿತಿ HTP_OP_UNARY_EXP:
        ಸ್ಥಿತಿ HTP_OP_UNARY_SIGMOID:
        ಸ್ಥಿತಿ HTP_OP_UNARY_SILU:
        ಸ್ಥಿತಿ HTP_OP_UNARY_GELU:
        ಸ್ಥಿತಿ HTP_OP_UNARY_GELU_ಶೀಘ್ರ:
        ಸ್ಥಿತಿ HTP_OP_UNARY_GELU_ERF:
        ಸ್ಥಿತಿ HTP_OP_UNARY_SOFTPLUS:
        ಸ್ಥಿತಿ HTP_OP_UNARY_TANH:
        ಸ್ಥಿತಿ HTP_OP_UNARY_ABS:
        ಸ್ಥಿತಿ HTP_OP_UNARY_LOG:
        ಸ್ಥಿತಿ HTP_OP_UNARY_RELU:
        ಸ್ಥಿತಿ HTP_OP_UNARY_ಹೆಜ್ಜೆ:
        ಸ್ಥಿತಿ HTP_OP_L2_NORM:
        ಸ್ಥಿತಿ HTP_OP_TRI:
        ಸ್ಥಿತಿ HTP_OP_HADAMARD:
            ಹಿಂತಿರುಗಿ true;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ false;
    }
}

ರಚನೆ htp_unary_vtcm_layout {
    size_t ಒಟ್ಟು_ಬೈಟುಗಳು;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_ಬೈಟುಗಳು;
    size_t src1_ಬೈಟುಗಳು;
    size_t dst_ಬೈಟುಗಳು;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_unary_vtcm_layout_build(
    ರಚನೆ htp_unary_vtcm_layout * L,
    uint32_t op,
    uint32_t ne00,
    uint32_t ne10,
    uint32_t ne11,
    bool ಪ್ರಸಾರ_ತೂಕ,
    uint32_t n_ಎಳೆಗಳು,
    size_t vtcm_ಗಾತ್ರ,
    size_t elem_ಗಾತ್ರ,
    uint32_t * ಹೊರಹೋಗು_ಲಂಬಸಾಲು_tile,
    uint32_t * ಹೊರಹೋಗು_vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread
) {
    ಸ್ಥಿರಾಂಕ size_t dst_elem_ಗಾತ್ರ      = (op == HTP_OP_HADAMARD) ? ಗಾತ್ರ(ದಶಮಾಂಶ) : elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ne00 * elem_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = ne10 * dst_elem_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(src0_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128);
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned  = hex_ಸುತ್ತು_ಮೇಲೆ(dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,  128);

    size_t src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = 0;
    ಆದರೆ (op == HTP_OP_RMS_NORM_MUL) {
        // RMS_NORM_MUL fusion is F32-only; its weight tensor is always F32.
        ಸ್ಥಿರಾಂಕ size_t src1_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ne11 * ಗಾತ್ರ(ದಶಮಾಂಶ);
        src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = hex_ಸುತ್ತು_ಮೇಲೆ(src1_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128);
    }

    size_t vtcm_ಗಾತ್ರ_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = 0;
    size_t vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread = 0;

    ಆದರೆ (op == HTP_OP_RMS_NORM_MUL) {
        ಆದರೆ (ಪ್ರಸಾರ_ತೂಕ) {
            size_t ಲಭ್ಯ_vtcm = vtcm_ಗಾತ್ರ;
            size_t src1_vtcm_ಒಟ್ಟು = n_ಎಳೆಗಳು * src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
            ಆದರೆ (ಲಭ್ಯ_vtcm > src1_vtcm_ಒಟ್ಟು) {
                ಲಭ್ಯ_vtcm -= src1_vtcm_ಒಟ್ಟು;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಲಭ್ಯ_vtcm = 0;
            }
            vtcm_ಗಾತ್ರ_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = 2 * (src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
            vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread = ಲಭ್ಯ_vtcm / (n_ಎಳೆಗಳು * vtcm_ಗಾತ್ರ_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು);
        } ಇಲ್ಲದಿದ್ದರೆ {
            vtcm_ಗಾತ್ರ_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = 2 * (src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
            vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread = vtcm_ಗಾತ್ರ / (n_ಎಳೆಗಳು * vtcm_ಗಾತ್ರ_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        vtcm_ಗಾತ್ರ_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು   = 2 * (src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned + dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread = vtcm_ಗಾತ್ರ / (n_ಎಳೆಗಳು * vtcm_ಗಾತ್ರ_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು);
    }

    ಸ್ಥಿರಾಂಕ bool is_reduction = (op == HTP_OP_NORM || op == HTP_OP_RMS_NORM ||
                               op == HTP_OP_RMS_NORM_MUL || op == HTP_OP_L2_NORM ||
                               op == HTP_OP_HADAMARD);
    // The tiled fallback path below only has F32 task functions (unary_task_f32_tiled_*);
    // F16 has no tiled kernels, so it must stay on the row-block path like reduction ops.
    // NOTE: if F16 ends up with vtcm_row_per_thread == 0 here (row too large for the VTCM
    // budget), execute_op_unary() will see BLOCK == 0 and skip computation for that op
    // (logged via FARF(ERROR, ...)) since there is no F16 tiled fallback. This is a known
    // limitation; supporting it would require adding F16 tiled kernels.
    ಸ್ಥಿರಾಂಕ bool is_f16 = (elem_ಗಾತ್ರ == ಗಾತ್ರ(_Float16));
    uint32_t ಲಂಬಸಾಲು_tile = 0;

    ಆದರೆ (vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread == 0 && !is_reduction && !is_f16) {
        ಸ್ಥಿರಾಂಕ size_t ಪ್ರತಿ_thread_budget = vtcm_ಗಾತ್ರ / n_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ size_t ಲಂಬಸಾಲು_tile_ಬೈಟುಗಳು = hex_align_ಕೆಳಕ್ಕೆ(ಪ್ರತಿ_thread_budget / 4, 128);
        ಲಂಬಸಾಲು_tile = (uint32_t) (ಲಂಬಸಾಲು_tile_ಬೈಟುಗಳು / elem_ಗಾತ್ರ);

        L->src0_ಬೈಟುಗಳು = ಲಂಬಸಾಲು_tile_ಬೈಟುಗಳು * 2;
        L->dst_ಬೈಟುಗಳು  = ಲಂಬಸಾಲು_tile_ಬೈಟುಗಳು * 2;
        L->src1_ಬೈಟುಗಳು = 0;
    } ಇಲ್ಲದಿದ್ದರೆ {
        L->src0_ಬೈಟುಗಳು = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned * vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread * 2;
        L->dst_ಬೈಟುಗಳು  = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned * vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread * 2;
        ಆದರೆ (op == HTP_OP_RMS_NORM_MUL) {
            ಆದರೆ (ಪ್ರಸಾರ_ತೂಕ) {
                L->src1_ಬೈಟುಗಳು = src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
            } ಇಲ್ಲದಿದ್ದರೆ {
                L->src1_ಬೈಟುಗಳು = src1_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned * vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread * 2;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            L->src1_ಬೈಟುಗಳು = 0;
        }
    }

    L->off_src0 = 0;
    ಆದರೆ (op == HTP_OP_RMS_NORM_MUL) {
        L->off_src1 = L->off_src0 + L->src0_ಬೈಟುಗಳು * n_ಎಳೆಗಳು;
        L->off_dst  = L->off_src1 + L->src1_ಬೈಟುಗಳು * n_ಎಳೆಗಳು;
    } ಇಲ್ಲದಿದ್ದರೆ {
        L->off_src1 = 0;
        L->off_dst  = L->off_src0 + L->src0_ಬೈಟುಗಳು * n_ಎಳೆಗಳು;
    }

    L->ಒಟ್ಟು_ಬೈಟುಗಳು = L->off_dst + L->dst_ಬೈಟುಗಳು * n_ಎಳೆಗಳು;

    *ಹೊರಹೋಗು_ಲಂಬಸಾಲು_tile = ಲಂಬಸಾಲು_tile;
    *ಹೊರಹೋಗು_vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread = vtcm_ಅಡ್ಡಸಾಲು_ಪ್ರತಿ_thread;
}

#ಆದರೆ_ಅಂತ್ಯ /* HTP_UNARY_OPS_H */
