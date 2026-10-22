#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_UNARY_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_UNARY_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"

// Op-specific struct for precomputed unary params
ਬਣਤਰ htp_unary_kernel_params {
    uint32_t  n_ਧਾਗੇ;
    uint32_t  ਖਾਨਾ_tile;
    uint32_t  vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread;
    uint32_t  ਖੰਡ;
    uint32_t  ਪ੍ਰਸਾਰਣ_ਭਾਰ;

    uint32_t  vtcm_src0_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t  vtcm_src1_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    uint32_t  vtcm_dst_ਮਾਪ_ਪ੍ਰਤੀ_thread;

    uint32_t  vtcm_src0_ਮਾਪ;
    uint32_t  vtcm_src1_ਮਾਪ;
    uint32_t  vtcm_dst_ਮਾਪ;

    uint32_t  src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    uint32_t  src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
    uint32_t  dst_ਪੰਕਤੀ_ਮਾਪ_aligned;

    uint32_t  vtcm_ਮਾਪ;

    // Fastdiv helpers
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne01;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne02;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne012;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_tpr;
};

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#ਨਹੀਂਤਾਂ
_Static_assert(ਆਕਾਰ(ਬਣਤਰ htp_unary_kernel_params) <= 128, "htp_unary_kernel_params is too large for kernel_params blob");
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_op_is_unary(uint32_t opcode) {
    ਚੋਣ (opcode) {
        ਹਾਲਤ HTP_OP_CLAMP:
        ਹਾਲਤ HTP_OP_LEAKY_RELU:
        ਹਾਲਤ HTP_OP_NORM:
        ਹਾਲਤ HTP_OP_RMS_NORM:
        ਹਾਲਤ HTP_OP_RMS_NORM_MUL:
        ਹਾਲਤ HTP_OP_SCALE:
        ਹਾਲਤ HTP_OP_SQR:
        ਹਾਲਤ HTP_OP_SQRT:
        ਹਾਲਤ HTP_OP_UNARY_NEG:
        ਹਾਲਤ HTP_OP_UNARY_EXP:
        ਹਾਲਤ HTP_OP_UNARY_SIGMOID:
        ਹਾਲਤ HTP_OP_UNARY_SILU:
        ਹਾਲਤ HTP_OP_UNARY_GELU:
        ਹਾਲਤ HTP_OP_UNARY_GELU_ਤੇਜ਼:
        ਹਾਲਤ HTP_OP_UNARY_GELU_ERF:
        ਹਾਲਤ HTP_OP_UNARY_SOFTPLUS:
        ਹਾਲਤ HTP_OP_UNARY_TANH:
        ਹਾਲਤ HTP_OP_UNARY_ABS:
        ਹਾਲਤ HTP_OP_UNARY_LOG:
        ਹਾਲਤ HTP_OP_UNARY_RELU:
        ਹਾਲਤ HTP_OP_UNARY_ਕਦਮ:
        ਹਾਲਤ HTP_OP_L2_NORM:
        ਹਾਲਤ HTP_OP_TRI:
        ਹਾਲਤ HTP_OP_HADAMARD:
            ਵਾਪਸ true;
        ਮੂਲ:
            ਵਾਪਸ false;
    }
}

ਬਣਤਰ htp_unary_vtcm_layout {
    size_t ਕੁੱਲ_ਬਾਈਟਾਂ;
    size_t off_src0;
    size_t off_src1;
    size_t off_dst;

    size_t src0_ਬਾਈਟਾਂ;
    size_t src1_ਬਾਈਟਾਂ;
    size_t dst_ਬਾਈਟਾਂ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_unary_vtcm_layout_build(
    ਬਣਤਰ htp_unary_vtcm_layout * L,
    uint32_t op,
    uint32_t ne00,
    uint32_t ne10,
    uint32_t ne11,
    bool ਪ੍ਰਸਾਰਣ_ਭਾਰ,
    uint32_t n_ਧਾਗੇ,
    size_t vtcm_ਮਾਪ,
    size_t elem_ਮਾਪ,
    uint32_t * ਬਾਹਰਲਾ_ਖਾਨਾ_tile,
    uint32_t * ਬਾਹਰਲਾ_vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread
) {
    ਅਚਲ size_t dst_elem_ਮਾਪ      = (op == HTP_OP_HADAMARD) ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) : elem_ਮਾਪ;
    ਅਚਲ size_t src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ = ne00 * elem_ਮਾਪ;
    ਅਚਲ size_t dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ  = ne10 * dst_elem_ਮਾਪ;

    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(src0_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, 128);
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ_aligned  = hex_ਗੇੜਾ_ਉੱਪਰ(dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ,  128);

    size_t src1_ਪੰਕਤੀ_ਮਾਪ_aligned = 0;
    ਜੇ (op == HTP_OP_RMS_NORM_MUL) {
        // RMS_NORM_MUL fusion is F32-only; its weight tensor is always F32.
        ਅਚਲ size_t src1_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ = ne11 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        src1_ਪੰਕਤੀ_ਮਾਪ_aligned = hex_ਗੇੜਾ_ਉੱਪਰ(src1_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, 128);
    }

    size_t vtcm_ਮਾਪ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = 0;
    size_t vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread = 0;

    ਜੇ (op == HTP_OP_RMS_NORM_MUL) {
        ਜੇ (ਪ੍ਰਸਾਰਣ_ਭਾਰ) {
            size_t ਉਪਲਬਧ_vtcm = vtcm_ਮਾਪ;
            size_t src1_vtcm_ਕੁੱਲ = n_ਧਾਗੇ * src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
            ਜੇ (ਉਪਲਬਧ_vtcm > src1_vtcm_ਕੁੱਲ) {
                ਉਪਲਬਧ_vtcm -= src1_vtcm_ਕੁੱਲ;
            } ਨਹੀਂਤਾਂ {
                ਉਪਲਬਧ_vtcm = 0;
            }
            vtcm_ਮਾਪ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = 2 * (src0_ਪੰਕਤੀ_ਮਾਪ_aligned + dst_ਪੰਕਤੀ_ਮਾਪ_aligned);
            vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread = ਉਪਲਬਧ_vtcm / (n_ਧਾਗੇ * vtcm_ਮਾਪ_ਪ੍ਰਤੀ_ਪੰਕਤੀ);
        } ਨਹੀਂਤਾਂ {
            vtcm_ਮਾਪ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = 2 * (src0_ਪੰਕਤੀ_ਮਾਪ_aligned + dst_ਪੰਕਤੀ_ਮਾਪ_aligned + src1_ਪੰਕਤੀ_ਮਾਪ_aligned);
            vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread = vtcm_ਮਾਪ / (n_ਧਾਗੇ * vtcm_ਮਾਪ_ਪ੍ਰਤੀ_ਪੰਕਤੀ);
        }
    } ਨਹੀਂਤਾਂ {
        vtcm_ਮਾਪ_ਪ੍ਰਤੀ_ਪੰਕਤੀ   = 2 * (src0_ਪੰਕਤੀ_ਮਾਪ_aligned + dst_ਪੰਕਤੀ_ਮਾਪ_aligned);
        vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread = vtcm_ਮਾਪ / (n_ਧਾਗੇ * vtcm_ਮਾਪ_ਪ੍ਰਤੀ_ਪੰਕਤੀ);
    }

    ਅਚਲ bool is_reduction = (op == HTP_OP_NORM || op == HTP_OP_RMS_NORM ||
                               op == HTP_OP_RMS_NORM_MUL || op == HTP_OP_L2_NORM ||
                               op == HTP_OP_HADAMARD);
    // The tiled fallback path below only has F32 task functions (unary_task_f32_tiled_*);
    // F16 has no tiled kernels, so it must stay on the row-block path like reduction ops.
    // NOTE: if F16 ends up with vtcm_row_per_thread == 0 here (row too large for the VTCM
    // budget), execute_op_unary() will see BLOCK == 0 and skip computation for that op
    // (logged via FARF(ERROR, ...)) since there is no F16 tiled fallback. This is a known
    // limitation; supporting it would require adding F16 tiled kernels.
    ਅਚਲ bool is_f16 = (elem_ਮਾਪ == ਆਕਾਰ(_Float16));
    uint32_t ਖਾਨਾ_tile = 0;

    ਜੇ (vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread == 0 && !is_reduction && !is_f16) {
        ਅਚਲ size_t ਪ੍ਰਤੀ_thread_budget = vtcm_ਮਾਪ / n_ਧਾਗੇ;
        ਅਚਲ size_t ਖਾਨਾ_tile_ਬਾਈਟਾਂ = hex_align_ਥੱਲੇ(ਪ੍ਰਤੀ_thread_budget / 4, 128);
        ਖਾਨਾ_tile = (uint32_t) (ਖਾਨਾ_tile_ਬਾਈਟਾਂ / elem_ਮਾਪ);

        L->src0_ਬਾਈਟਾਂ = ਖਾਨਾ_tile_ਬਾਈਟਾਂ * 2;
        L->dst_ਬਾਈਟਾਂ  = ਖਾਨਾ_tile_ਬਾਈਟਾਂ * 2;
        L->src1_ਬਾਈਟਾਂ = 0;
    } ਨਹੀਂਤਾਂ {
        L->src0_ਬਾਈਟਾਂ = src0_ਪੰਕਤੀ_ਮਾਪ_aligned * vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread * 2;
        L->dst_ਬਾਈਟਾਂ  = dst_ਪੰਕਤੀ_ਮਾਪ_aligned * vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread * 2;
        ਜੇ (op == HTP_OP_RMS_NORM_MUL) {
            ਜੇ (ਪ੍ਰਸਾਰਣ_ਭਾਰ) {
                L->src1_ਬਾਈਟਾਂ = src1_ਪੰਕਤੀ_ਮਾਪ_aligned;
            } ਨਹੀਂਤਾਂ {
                L->src1_ਬਾਈਟਾਂ = src1_ਪੰਕਤੀ_ਮਾਪ_aligned * vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread * 2;
            }
        } ਨਹੀਂਤਾਂ {
            L->src1_ਬਾਈਟਾਂ = 0;
        }
    }

    L->off_src0 = 0;
    ਜੇ (op == HTP_OP_RMS_NORM_MUL) {
        L->off_src1 = L->off_src0 + L->src0_ਬਾਈਟਾਂ * n_ਧਾਗੇ;
        L->off_dst  = L->off_src1 + L->src1_ਬਾਈਟਾਂ * n_ਧਾਗੇ;
    } ਨਹੀਂਤਾਂ {
        L->off_src1 = 0;
        L->off_dst  = L->off_src0 + L->src0_ਬਾਈਟਾਂ * n_ਧਾਗੇ;
    }

    L->ਕੁੱਲ_ਬਾਈਟਾਂ = L->off_dst + L->dst_ਬਾਈਟਾਂ * n_ਧਾਗੇ;

    *ਬਾਹਰਲਾ_ਖਾਨਾ_tile = ਖਾਨਾ_tile;
    *ਬਾਹਰਲਾ_vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread = vtcm_ਪੰਕਤੀ_ਪ੍ਰਤੀ_thread;
}

#ਜੇ_ਅੰਤ /* HTP_UNARY_OPS_H */
