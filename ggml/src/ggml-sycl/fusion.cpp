#ಅನ್ತರ್ಭಾವ "fusion.hpp"
#ಅನ್ತರ್ಭಾವ "binbcast.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_op_mul_mat_glu_fused().
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_should_fuse_mul_mat_glu(ಧ್ರುವ ggml_tensor * gate, ಧ್ರುವ ggml_tensor * up,
                                              ಧ್ರುವ ggml_tensor * glu) {
    // the fused epilogue implements these two; the rest fall back to the standalone GLU kernels
    ಧ್ರುವ ggml_glu_op glu_op = ggml_get_glu_op(glu);
    ಯದಿ (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // the kernel always treats src[0] as the activated operand and src[1] as the multiplier
    ಯದಿ (ggml_get_op_params_i32(glu, 1) /* swapped */) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಧ್ರುವ ggml_tensor * wu  = up->src[0];
    ಧ್ರುವ ggml_tensor * wg  = gate->src[0];
    ಧ್ರುವ ggml_tensor * act = up->src[1];

    // one activation and one output indexing must serve both weights; the block types
    // may differ, since the plain-layout fused kernel runs each operand's own vec_dot
    // (different types then imply different byte strides, so only the shape must agree)
    ಯದಿ (!ggml_are_same_shape(wu, wg)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (act != gate->src[1]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // fused K-quant GEMVs walk whole QK_K super-blocks: the reorder kernel covers same-type
    // q4_K, the plain-layout kernel covers q5_K / iq4_xs pairs incl. mixed gate/up types
    ಧ್ರುವ ತಾರ್ಕಿಕ reorder_pair = wu->type == GGML_TYPE_Q4_K && wg->type == GGML_TYPE_Q4_K;
    ಧ್ರುವ ತಾರ್ಕಿಕ plain_pair   = (wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_IQ4_XS) &&
                            (wg->type == GGML_TYPE_Q5_K || wg->type == GGML_TYPE_IQ4_XS);
    // the ESIMD kernel covers same-type q2_K..q6_K and q8_0, single column only (as unfused DMMV)
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    ಧ್ರುವ ತಾರ್ಕಿಕ esimd_pair   = g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
                            (wu->type == GGML_TYPE_Q2_K || wu->type == GGML_TYPE_Q3_K || wu->type == GGML_TYPE_Q4_K ||
                             wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_Q6_K || wu->type == GGML_TYPE_Q8_0);
#ಅನ್ಯಥಾ
    ಧ್ರುವ ತಾರ್ಕಿಕ esimd_pair   = ಅಸತ್ಯ;
#endif
    // q8_0 rows are walked in 32-wide blocks, so they only need to be a multiple of QK8_0
    ಧ್ರುವ int64_t row_align = esimd_pair && wu->type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
    ಯದಿ ((!reorder_pair && !plain_pair && !esimd_pair) || wu->ne[0] % row_align != 0) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // one 2D reorder-layout matrix in, a plain column stride out: no broadcast or padding
    ಯದಿ (!ggml_is_contiguous(wu) || !ggml_is_contiguous(wg) || !ggml_is_contiguous(act) ||
        !ggml_is_contiguous(glu)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಯದಿ (act->ne[2] != 1 || act->ne[3] != 1 || wu->ne[2] != 1 || wu->ne[3] != 1) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    // the kernel writes rows [0, wu->ne[1]) of each glu column, strided by glu->ne[0]
    ಯದಿ (glu->ne[0] != wu->ne[1] || glu->ne[1] != act->ne[1]) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    // mat-vec only: one column per decoded token, up to the batch the reorder kernels cover
    ಯದಿ (act->ne[1] > MMVQ_MAX_BATCH_SIZE) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}

// the unary ops the fused unary chains in element_wise.cpp have a functor for
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_fused_unary_has_kernel(ggml_unary_op unary_op) {
    ಪ್ರತ್ಯಾಗಮ unary_op == GGML_UNARY_OP_SILU || unary_op == GGML_UNARY_OP_SIGMOID ||
           unary_op == GGML_UNARY_OP_SOFTPLUS;
}

ತಾರ್ಕಿಕ ggml_sycl_can_fuse(ಧ್ರುವ ggml_cgraph * cgraph, ಪೂರ್ಣಾಙ್ಕ ಗಂಟು_idx, std::initializer_list<ಗಣನಾ ggml_op> ops,
                        std::initializer_list<ಗಣನಾ ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    ಧ್ರುವ size_t ಅಂಕೆ_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == ಅಂಕೆ_unary);
#endif

    ಯದಿ (!g_ggml_sycl_enable_fusion) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    // gate and up are siblings, not a chain, so ggml_can_fuse cannot express this: use the
    // subgraph form with the GLU as the only materialised output.
    ಯದಿ (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_MUL_MAT &&
        ops.begin()[2] == GGML_OP_GLU) {
        ಯದಿ (!ggml_can_fuse_subgraph(cgraph, ಗಂಟು_idx, ops, { ಗಂಟು_idx + 2 })) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ggml_tensor * glu  = cgraph->nodes[ಗಂಟು_idx + 2];
        ಧ್ರುವ ggml_tensor * gate = glu->src[0];
        ಧ್ರುವ ggml_tensor * up   = glu->src[1];

        // don't assume which of the two mat-muls is the gate; infer it from the GLU's operands
        ಧ್ರುವ ತಾರ್ಕಿಕ ok = (gate == cgraph->nodes[ಗಂಟು_idx] && up == cgraph->nodes[ಗಂಟು_idx + 1]) ||
                        (gate == cgraph->nodes[ಗಂಟು_idx + 1] && up == cgraph->nodes[ಗಂಟು_idx]);
        ಯದಿ (!ok) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಪ್ರತ್ಯಾಗಮ ggml_sycl_should_fuse_mul_mat_glu(gate, up, glu);
    }

    ಯದಿ (!ggml_can_fuse(cgraph, ಗಂಟು_idx, ops)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }

    ಯದಿ ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        ಯದಿ (ops.size() == 3 && ops.begin()[2] != GGML_OP_ADD) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ggml_tensor * rms_norm = cgraph->nodes[ಗಂಟು_idx];
        ಧ್ರುವ ggml_tensor * mul      = cgraph->nodes[ಗಂಟು_idx + 1];
        ಧ್ರುವ ggml_tensor * add      = ops.size() == 3 ? cgraph->nodes[ಗಂಟು_idx + 2] : nullptr;

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        ಯದಿ (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // if rms norm is the B operand, then we don't handle broadcast
        ಯದಿ (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        // the fused kernel indexes the weight as mul[col], so it must span ncols contiguously
        ಯದಿ (mul_w->ne[0] != rms_norm->ne[0] || mul_w->nb[0] != ggml_type_size(mul_w->type)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (add != nullptr) {
            ಯದಿ (add->src[0]->type != GGML_TYPE_F32 ||
                add->src[1]->type != GGML_TYPE_F32 ||
                add->type != GGML_TYPE_F32) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            // the fused kernel indexes the residual as add[col] and does not broadcast it
            ಧ್ರುವ ggml_tensor * add_w = (add->src[0] == mul) ? add->src[1] : add->src[0];
            ಯದಿ (!ggml_are_same_shape(add_w, add)) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಯದಿ (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1])) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಯದಿ (ops.size() == 2 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_ADD) {
        ಧ್ರುವ ggml_tensor * add0 = cgraph->nodes[ಗಂಟು_idx];
        ಧ್ರುವ ggml_tensor * add1 = cgraph->nodes[ಗಂಟು_idx + 1];
        // ggml_can_fuse already guarantees add1 consumes add0 and that add0 has a single use.
        // Keep the CUDA association: the running sum is src0 of the next ADD so the fused
        // float fold matches two sequential add() launches.
        ಯದಿ (add1->src[0] != add0) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ggml_tensor * c = add1->src[1];
        ಯದಿ (!ggml_sycl_add_kernel_supports(add0->src[0]->type, add0->src[1]->type, add0->type) ||
            !ggml_sycl_add_kernel_supports(add0->type, c->type, add1->type)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಯದಿ (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        ಧ್ರುವ ggml_tensor * unary = cgraph->nodes[ಗಂಟು_idx];
        ಧ್ರುವ ggml_tensor * mul   = cgraph->nodes[ಗಂಟು_idx + 1];

        ಧ್ರುವ ggml_unary_op unary_op = ggml_get_unary_op(unary);
        ಯದಿ (unary_op != unary_ops.begin()[0]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        ಯದಿ (other->type != unary->type) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // one row stride per source comes from nb[1], so rows must be contiguous and equally
        // shaped; the destination is written flat, so it must be fully contiguous
        ಯದಿ (!ggml_is_contiguous_1(unary->src[0]) || !ggml_is_contiguous_1(other) ||
            !ggml_are_same_shape(other, unary) || !ggml_is_contiguous(mul)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        ಯದಿ (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಯದಿ (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        ಧ್ರುವ ggml_tensor * ssm_conv = cgraph->nodes[ಗಂಟು_idx];
        ಧ್ರುವ ggml_tensor * silu     = cgraph->nodes[ಗಂಟು_idx + 1];

        ಯದಿ (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        ಯದಿ (!ggml_is_contiguous(silu)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    // ADD(bias) + UNARY + MUL(scale): the delta-net alpha gate, softplus(alpha + dt) * a.
    // The broadcast is what stops the same-shape UNARY + MUL branch above firing past one token.
    ಯದಿ (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1) {
        ಧ್ರುವ ggml_tensor * add   = cgraph->nodes[ಗಂಟು_idx];
        ಧ್ರುವ ggml_tensor * unary = cgraph->nodes[ಗಂಟು_idx + 1];
        ಧ್ರುವ ggml_tensor * mul   = cgraph->nodes[ಗಂಟು_idx + 2];

        ಧ್ರುವ ggml_unary_op unary_op = ggml_get_unary_op(unary);
        ಯದಿ (unary_op != unary_ops.begin()[0]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಯದಿ (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // ggml_can_fuse() has already pinned the chain: unary consumes add, mul consumes
        // unary, add and unary have one use each, and all three have the same shape
        ಧ್ರುವ ggml_tensor * a     = add->src[0];
        ಧ್ರುವ ggml_tensor * bias  = add->src[1];
        ಧ್ರುವ ggml_tensor * scale = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];

        ಯದಿ (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // the activation and the destination are indexed flat
        ಯದಿ (!ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(a, mul)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // the kernel reads the bias and the scale as v[col], so each must be a single
        // contiguous row spanning ne0
        ಯದಿ (bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] ||
            ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        ಯದಿ (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಯದಿ (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD &&
        ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        ಧ್ರುವ ggml_tensor * ssm_conv = cgraph->nodes[ಗಂಟು_idx];
        ಧ್ರುವ ggml_tensor * add      = cgraph->nodes[ಗಂಟು_idx + 1];
        ಧ್ರುವ ggml_tensor * silu     = cgraph->nodes[ಗಂಟು_idx + 2];

        ಯದಿ (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        ಯದಿ (!ggml_is_contiguous(silu)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias
        ಧ್ರುವ ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        ಯದಿ (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಯದಿ (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    ಯದಿ (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        ಧ್ರುವ ggml_tensor * rms_norm = cgraph->nodes[ಗಂಟು_idx];
        ಧ್ರುವ ggml_tensor * scale    = cgraph->nodes[ಗಂಟು_idx + 1];
        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        ಯದಿ (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        // the fused kernel reads/writes rows flat like the unfused pair
        ಯದಿ (!ggml_is_contiguous_rows(rms_norm) || !ggml_is_contiguous_rows(scale)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}
