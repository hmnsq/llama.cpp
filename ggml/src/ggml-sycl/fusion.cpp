#ಅನ್ತರ್ಭಾವ "fusion.hpp"
#ಅನ್ತರ್ಭಾವ "binbcast.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_op_mul_mat_glu_fused().
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_should_fuse_mul_mat_glu(ಸ್ಥಿರಾಂಕ ggml_tensor * gate, ಸ್ಥಿರಾಂಕ ggml_tensor * ಮೇಲೆ,
                                              ಸ್ಥಿರಾಂಕ ggml_tensor * glu) {
    // the fused epilogue implements these two; the rest fall back to the standalone GLU kernels
    ಸ್ಥಿರಾಂಕ ggml_glu_op glu_op = ggml_get_glu_op(glu);
    ಆದರೆ (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // the kernel always treats src[0] as the activated operand and src[1] as the multiplier
    ಆದರೆ (ggml_get_op_params_i32(glu, 1) /* swapped */) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * wu  = ಮೇಲೆ->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * wg  = gate->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * act = ಮೇಲೆ->src[1];

    // one activation and one output indexing must serve both weights; the block types
    // may differ, since the plain-layout fused kernel runs each operand's own vec_dot
    // (different types then imply different byte strides, so only the shape must agree)
    ಆದರೆ (!ggml_are_same_shape(wu, wg)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (act != gate->src[1]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // fused K-quant GEMVs walk whole QK_K super-blocks: the reorder kernel covers same-type
    // q4_K, the plain-layout kernel covers q5_K / iq4_xs pairs incl. mixed gate/up types
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ reorder_pair = wu->type == GGML_TYPE_Q4_K && wg->type == GGML_TYPE_Q4_K;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ plain_pair   = (wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_IQ4_XS) &&
                            (wg->type == GGML_TYPE_Q5_K || wg->type == GGML_TYPE_IQ4_XS);
    // the ESIMD kernel covers same-type q2_K..q6_K and q8_0, single column only (as unfused DMMV)
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_ESIMD
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ esimd_pair   = g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
                            (wu->type == GGML_TYPE_Q2_K || wu->type == GGML_TYPE_Q3_K || wu->type == GGML_TYPE_Q4_K ||
                             wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_Q6_K || wu->type == GGML_TYPE_Q8_0);
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ esimd_pair   = ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
    // q8_0 rows are walked in 32-wide blocks, so they only need to be a multiple of QK8_0
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_align = esimd_pair && wu->type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
    ಆದರೆ ((!reorder_pair && !plain_pair && !esimd_pair) || wu->ne[0] % ಅಡ್ಡಸಾಲು_align != 0) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // one 2D reorder-layout matrix in, a plain column stride out: no broadcast or padding
    ಆದರೆ (!ggml_is_contiguous(wu) || !ggml_is_contiguous(wg) || !ggml_is_contiguous(act) ||
        !ggml_is_contiguous(glu)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (act->ne[2] != 1 || act->ne[3] != 1 || wu->ne[2] != 1 || wu->ne[3] != 1) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // the kernel writes rows [0, wu->ne[1]) of each glu column, strided by glu->ne[0]
    ಆದರೆ (glu->ne[0] != wu->ne[1] || glu->ne[1] != act->ne[1]) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // mat-vec only: one column per decoded token, up to the batch the reorder kernels cover
    ಆದರೆ (act->ne[1] > MMVQ_MAX_BATCH_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// the unary ops the fused unary chains in element_wise.cpp have a functor for
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_fused_unary_has_kernel(ggml_unary_op unary_op) {
    ಹಿಂತಿರುಗಿ unary_op == GGML_UNARY_OP_SILU || unary_op == GGML_UNARY_OP_SIGMOID ||
           unary_op == GGML_UNARY_OP_SOFTPLUS;
}

ತಾರ್ಕಿಕ ggml_sycl_can_fuse(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, std::initializer_list<ಎಣಿಕೆ ggml_op> ops,
                        std::initializer_list<ಎಣಿಕೆ ggml_unary_op> unary_ops) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
    ಸ್ಥಿರಾಂಕ size_t ಅಂಕೆ_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == ಅಂಕೆ_unary);
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (!g_ggml_sycl_enable_fusion) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // gate and up are siblings, not a chain, so ggml_can_fuse cannot express this: use the
    // subgraph form with the GLU as the only materialised output.
    ಆದರೆ (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_MUL_MAT &&
        ops.begin()[2] == GGML_OP_GLU) {
        ಆದರೆ (!ggml_can_fuse_subgraph(cgraph, ಗಂಟು_idx, ops, { ಗಂಟು_idx + 2 })) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ggml_tensor * glu  = cgraph->nodes[ಗಂಟು_idx + 2];
        ಸ್ಥಿರಾಂಕ ggml_tensor * gate = glu->src[0];
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಮೇಲೆ   = glu->src[1];

        // don't assume which of the two mat-muls is the gate; infer it from the GLU's operands
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ok = (gate == cgraph->nodes[ಗಂಟು_idx] && ಮೇಲೆ == cgraph->nodes[ಗಂಟು_idx + 1]) ||
                        (gate == cgraph->nodes[ಗಂಟು_idx + 1] && ಮೇಲೆ == cgraph->nodes[ಗಂಟು_idx]);
        ಆದರೆ (!ok) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಹಿಂತಿರುಗಿ ggml_sycl_should_fuse_mul_mat_glu(gate, ಮೇಲೆ, glu);
    }

    ಆದರೆ (!ggml_can_fuse(cgraph, ಗಂಟು_idx, ops)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        ಆದರೆ (ops.size() == 3 && ops.begin()[2] != GGML_OP_ಕೂಡಿಸು) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ggml_tensor * rms_norm = cgraph->nodes[ಗಂಟು_idx];
        ಸ್ಥಿರಾಂಕ ggml_tensor * mul      = cgraph->nodes[ಗಂಟು_idx + 1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2      = ops.size() == 3 ? cgraph->nodes[ಗಂಟು_idx + 2] : ಶೂನ್ಯಸೂಚಕ;

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        ಆದರೆ (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // if rms norm is the B operand, then we don't handle broadcast
        ಆದರೆ (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        // the fused kernel indexes the weight as mul[col], so it must span ncols contiguously
        ಆದರೆ (mul_w->ne[0] != rms_norm->ne[0] || mul_w->nb[0] != ggml_type_ಗಾತ್ರ(mul_w->type)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (!ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(mul->src[0]) || !ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(mul->src[1])) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (ಕೂಡಿಸು_2 != ಶೂನ್ಯಸೂಚಕ) {
            ಆದರೆ (ಕೂಡಿಸು_2->src[0]->type != GGML_TYPE_F32 ||
                ಕೂಡಿಸು_2->src[1]->type != GGML_TYPE_F32 ||
                ಕೂಡಿಸು_2->type != GGML_TYPE_F32) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            // the fused kernel indexes the residual as add[col] and does not broadcast it
            ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_w = (ಕೂಡಿಸು_2->src[0] == mul) ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];
            ಆದರೆ (!ggml_are_same_shape(ಕೂಡಿಸು_w, ಕೂಡಿಸು_2)) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಆದರೆ (!ggml_is_contiguous(ಕೂಡಿಸು_2->src[0]) || !ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(ಕೂಡಿಸು_2->src[1])) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಆದರೆ (ops.size() == 2 && ops.begin()[0] == GGML_OP_ಕೂಡಿಸು && ops.begin()[1] == GGML_OP_ಕೂಡಿಸು) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * add0 = cgraph->nodes[ಗಂಟು_idx];
        ಸ್ಥಿರಾಂಕ ggml_tensor * add1 = cgraph->nodes[ಗಂಟು_idx + 1];
        // ggml_can_fuse already guarantees add1 consumes add0 and that add0 has a single use.
        // Keep the CUDA association: the running sum is src0 of the next ADD so the fused
        // float fold matches two sequential add() launches.
        ಆದರೆ (add1->src[0] != add0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ggml_tensor * c = add1->src[1];
        ಆದರೆ (!ggml_sycl_ಕೂಡಿಸು_kernel_supports(add0->src[0]->type, add0->src[1]->type, add0->type) ||
            !ggml_sycl_ಕೂಡಿಸು_kernel_supports(add0->type, c->type, add1->type)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಆದರೆ (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * unary = cgraph->nodes[ಗಂಟು_idx];
        ಸ್ಥಿರಾಂಕ ggml_tensor * mul   = cgraph->nodes[ಗಂಟು_idx + 1];

        ಸ್ಥಿರಾಂಕ ggml_unary_op unary_op = ggml_get_unary_op(unary);
        ಆದರೆ (unary_op != unary_ops.begin()[0]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        ಆದರೆ (other->type != unary->type) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // one row stride per source comes from nb[1], so rows must be contiguous and equally
        // shaped; the destination is written flat, so it must be fully contiguous
        ಆದರೆ (!ggml_is_contiguous_1(unary->src[0]) || !ggml_is_contiguous_1(other) ||
            !ggml_are_same_shape(other, unary) || !ggml_is_contiguous(mul)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        ಆದರೆ (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಆದರೆ (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * ssm_conv = cgraph->nodes[ಗಂಟು_idx];
        ಸ್ಥಿರಾಂಕ ggml_tensor * silu     = cgraph->nodes[ಗಂಟು_idx + 1];

        ಆದರೆ (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        ಆದರೆ (!ggml_is_contiguous(silu)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    // ADD(bias) + UNARY + MUL(scale): the delta-net alpha gate, softplus(alpha + dt) * a.
    // The broadcast is what stops the same-shape UNARY + MUL branch above firing past one token.
    ಆದರೆ (ops.size() == 3 && ops.begin()[0] == GGML_OP_ಕೂಡಿಸು && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2   = cgraph->nodes[ಗಂಟು_idx];
        ಸ್ಥಿರಾಂಕ ggml_tensor * unary = cgraph->nodes[ಗಂಟು_idx + 1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * mul   = cgraph->nodes[ಗಂಟು_idx + 2];

        ಸ್ಥಿರಾಂಕ ggml_unary_op unary_op = ggml_get_unary_op(unary);
        ಆದರೆ (unary_op != unary_ops.begin()[0]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // ggml_can_fuse() has already pinned the chain: unary consumes add, mul consumes
        // unary, add and unary have one use each, and all three have the same shape
        ಸ್ಥಿರಾಂಕ ggml_tensor * a     = ಕೂಡಿಸು_2->src[0];
        ಸ್ಥಿರಾಂಕ ggml_tensor * bias  = ಕೂಡಿಸು_2->src[1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * scale = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];

        ಆದರೆ (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // the activation and the destination are indexed flat
        ಆದರೆ (!ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(a, mul)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // the kernel reads the bias and the scale as v[col], so each must be a single
        // contiguous row spanning ne0
        ಆದರೆ (bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] ||
            ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        ಆದರೆ (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಆದರೆ (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ಕೂಡಿಸು &&
        ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * ssm_conv = cgraph->nodes[ಗಂಟು_idx];
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2      = cgraph->nodes[ಗಂಟು_idx + 1];
        ಸ್ಥಿರಾಂಕ ggml_tensor * silu     = cgraph->nodes[ಗಂಟು_idx + 2];

        ಆದರೆ (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (ssm_conv->type != GGML_TYPE_F32 || ಕೂಡಿಸು_2->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        ಆದರೆ (!ggml_is_contiguous(silu)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias
        ಸ್ಥಿರಾಂಕ ggml_tensor * bias = (ಕೂಡಿಸು_2->src[0] == ssm_conv) ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];
        ಆದರೆ (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಆದರೆ (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * rms_norm = cgraph->nodes[ಗಂಟು_idx];
        ಸ್ಥಿರಾಂಕ ggml_tensor * scale    = cgraph->nodes[ಗಂಟು_idx + 1];
        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        ಆದರೆ (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        // the fused kernel reads/writes rows flat like the unfused pair
        ಆದರೆ (!ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(rms_norm) || !ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(scale)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}
