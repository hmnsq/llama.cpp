#অন্তর্ভাব "fusion.hpp"
#অন্তর্ভাব "binbcast.hpp"

#অন্তর্ভাব <algorithm>

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_op_mul_mat_glu_fused().
স্থির যৌক্তিক ggml_sycl_should_fuse_mul_mat_glu(ধ্রুবক ggml_tensor * gate, ধ্রুবক ggml_tensor * up,
                                              ধ্রুবক ggml_tensor * glu) {
    // the fused epilogue implements these two; the rest fall back to the standalone GLU kernels
    ধ্রুবক ggml_glu_op glu_op = ggml_get_glu_op(glu);
    যদি (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        ফেরত মিথ্যা;
    }

    // the kernel always treats src[0] as the activated operand and src[1] as the multiplier
    যদি (ggml_get_op_params_i32(glu, 1) /* swapped */) {
        ফেরত মিথ্যা;
    }

    ধ্রুবক ggml_tensor * wu  = up->src[0];
    ধ্রুবক ggml_tensor * wg  = gate->src[0];
    ধ্রুবক ggml_tensor * act = up->src[1];

    // one activation and one output indexing must serve both weights; the block types
    // may differ, since the plain-layout fused kernel runs each operand's own vec_dot
    // (different types then imply different byte strides, so only the shape must agree)
    যদি (!ggml_are_same_shape(wu, wg)) {
        ফেরত মিথ্যা;
    }
    যদি (act != gate->src[1]) {
        ফেরত মিথ্যা;
    }

    // fused K-quant GEMVs walk whole QK_K super-blocks: the reorder kernel covers same-type
    // q4_K, the plain-layout kernel covers q5_K / iq4_xs pairs incl. mixed gate/up types
    ধ্রুবক যৌক্তিক reorder_pair = wu->type == GGML_TYPE_Q4_K && wg->type == GGML_TYPE_Q4_K;
    ধ্রুবক যৌক্তিক plain_pair   = (wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_IQ4_XS) &&
                            (wg->type == GGML_TYPE_Q5_K || wg->type == GGML_TYPE_IQ4_XS);
    // the ESIMD kernel covers same-type q2_K..q6_K and q8_0, single column only (as unfused DMMV)
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    ধ্রুবক যৌক্তিক esimd_pair   = g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
                            (wu->type == GGML_TYPE_Q2_K || wu->type == GGML_TYPE_Q3_K || wu->type == GGML_TYPE_Q4_K ||
                             wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_Q6_K || wu->type == GGML_TYPE_Q8_0);
#অন্যথা
    ধ্রুবক যৌক্তিক esimd_pair   = মিথ্যা;
#endif
    // q8_0 rows are walked in 32-wide blocks, so they only need to be a multiple of QK8_0
    ধ্রুবক int64_t row_align = esimd_pair && wu->type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
    যদি ((!reorder_pair && !plain_pair && !esimd_pair) || wu->ne[0] % row_align != 0) {
        ফেরত মিথ্যা;
    }

    // one 2D reorder-layout matrix in, a plain column stride out: no broadcast or padding
    যদি (!ggml_is_contiguous(wu) || !ggml_is_contiguous(wg) || !ggml_is_contiguous(act) ||
        !ggml_is_contiguous(glu)) {
        ফেরত মিথ্যা;
    }
    যদি (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        ফেরত মিথ্যা;
    }
    যদি (act->ne[2] != 1 || act->ne[3] != 1 || wu->ne[2] != 1 || wu->ne[3] != 1) {
        ফেরত মিথ্যা;
    }
    // the kernel writes rows [0, wu->ne[1]) of each glu column, strided by glu->ne[0]
    যদি (glu->ne[0] != wu->ne[1] || glu->ne[1] != act->ne[1]) {
        ফেরত মিথ্যা;
    }
    // mat-vec only: one column per decoded token, up to the batch the reorder kernels cover
    যদি (act->ne[1] > MMVQ_MAX_BATCH_আকার) {
        ফেরত মিথ্যা;
    }

    ফেরত সত্য;
}

// the unary ops the fused unary chains in element_wise.cpp have a functor for
স্থির যৌক্তিক ggml_sycl_fused_unary_has_kernel(ggml_unary_op unary_op) {
    ফেরত unary_op == GGML_UNARY_OP_SILU || unary_op == GGML_UNARY_OP_SIGMOID ||
           unary_op == GGML_UNARY_OP_SOFTPLUS;
}

যৌক্তিক ggml_sycl_can_fuse(ধ্রুবক ggml_cgraph * cgraph, পূর্ণসংখ্যা গ্রন্থি_idx, std::initializer_list<গণনা ggml_op> ops,
                        std::initializer_list<গণনা ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    ধ্রুবক size_t অঙ্ক_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == অঙ্ক_unary);
#endif

    যদি (!g_ggml_sycl_enable_fusion) {
        ফেরত মিথ্যা;
    }

    // gate and up are siblings, not a chain, so ggml_can_fuse cannot express this: use the
    // subgraph form with the GLU as the only materialised output.
    যদি (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_MUL_MAT &&
        ops.begin()[2] == GGML_OP_GLU) {
        যদি (!ggml_can_fuse_subgraph(cgraph, গ্রন্থি_idx, ops, { গ্রন্থি_idx + 2 })) {
            ফেরত মিথ্যা;
        }

        ধ্রুবক ggml_tensor * glu  = cgraph->nodes[গ্রন্থি_idx + 2];
        ধ্রুবক ggml_tensor * gate = glu->src[0];
        ধ্রুবক ggml_tensor * up   = glu->src[1];

        // don't assume which of the two mat-muls is the gate; infer it from the GLU's operands
        ধ্রুবক যৌক্তিক ok = (gate == cgraph->nodes[গ্রন্থি_idx] && up == cgraph->nodes[গ্রন্থি_idx + 1]) ||
                        (gate == cgraph->nodes[গ্রন্থি_idx + 1] && up == cgraph->nodes[গ্রন্থি_idx]);
        যদি (!ok) {
            ফেরত মিথ্যা;
        }

        ফেরত ggml_sycl_should_fuse_mul_mat_glu(gate, up, glu);
    }

    যদি (!ggml_can_fuse(cgraph, গ্রন্থি_idx, ops)) {
        ফেরত মিথ্যা;
    }

    যদি ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        যদি (ops.size() == 3 && ops.begin()[2] != GGML_OP_ADD) {
            ফেরত মিথ্যা;
        }

        ধ্রুবক ggml_tensor * rms_norm = cgraph->nodes[গ্রন্থি_idx];
        ধ্রুবক ggml_tensor * mul      = cgraph->nodes[গ্রন্থি_idx + 1];
        ধ্রুবক ggml_tensor * add      = ops.size() == 3 ? cgraph->nodes[গ্রন্থি_idx + 2] : nullptr;

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        যদি (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            ফেরত মিথ্যা;
        }

        // if rms norm is the B operand, then we don't handle broadcast
        যদি (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            ফেরত মিথ্যা;
        }

        ধ্রুবক ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        // the fused kernel indexes the weight as mul[col], so it must span ncols contiguously
        যদি (mul_w->ne[0] != rms_norm->ne[0] || mul_w->nb[0] != ggml_type_আকার(mul_w->type)) {
            ফেরত মিথ্যা;
        }

        যদি (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            ফেরত মিথ্যা;
        }

        যদি (add != nullptr) {
            যদি (add->src[0]->type != GGML_TYPE_F32 ||
                add->src[1]->type != GGML_TYPE_F32 ||
                add->type != GGML_TYPE_F32) {
                ফেরত মিথ্যা;
            }

            // the fused kernel indexes the residual as add[col] and does not broadcast it
            ধ্রুবক ggml_tensor * add_w = (add->src[0] == mul) ? add->src[1] : add->src[0];
            যদি (!ggml_are_same_shape(add_w, add)) {
                ফেরত মিথ্যা;
            }

            যদি (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1])) {
                ফেরত মিথ্যা;
            }
        }

        ফেরত সত্য;
    }

    যদি (ops.size() == 2 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_ADD) {
        ধ্রুবক ggml_tensor * add0 = cgraph->nodes[গ্রন্থি_idx];
        ধ্রুবক ggml_tensor * add1 = cgraph->nodes[গ্রন্থি_idx + 1];
        // ggml_can_fuse already guarantees add1 consumes add0 and that add0 has a single use.
        // Keep the CUDA association: the running sum is src0 of the next ADD so the fused
        // float fold matches two sequential add() launches.
        যদি (add1->src[0] != add0) {
            ফেরত মিথ্যা;
        }

        ধ্রুবক ggml_tensor * c = add1->src[1];
        যদি (!ggml_sycl_add_kernel_supports(add0->src[0]->type, add0->src[1]->type, add0->type) ||
            !ggml_sycl_add_kernel_supports(add0->type, c->type, add1->type)) {
            ফেরত মিথ্যা;
        }

        ফেরত সত্য;
    }

    যদি (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        ধ্রুবক ggml_tensor * unary = cgraph->nodes[গ্রন্থি_idx];
        ধ্রুবক ggml_tensor * mul   = cgraph->nodes[গ্রন্থি_idx + 1];

        ধ্রুবক ggml_unary_op unary_op = ggml_get_unary_op(unary);
        যদি (unary_op != unary_ops.begin()[0]) {
            ফেরত মিথ্যা;
        }

        যদি (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            ফেরত মিথ্যা;
        }

        যদি (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            ফেরত মিথ্যা;
        }

        ধ্রুবক ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        যদি (other->type != unary->type) {
            ফেরত মিথ্যা;
        }

        // one row stride per source comes from nb[1], so rows must be contiguous and equally
        // shaped; the destination is written flat, so it must be fully contiguous
        যদি (!ggml_is_contiguous_1(unary->src[0]) || !ggml_is_contiguous_1(other) ||
            !ggml_are_same_shape(other, unary) || !ggml_is_contiguous(mul)) {
            ফেরত মিথ্যা;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        যদি (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            ফেরত মিথ্যা;
        }

        ফেরত সত্য;
    }

    যদি (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        ধ্রুবক ggml_tensor * ssm_conv = cgraph->nodes[গ্রন্থি_idx];
        ধ্রুবক ggml_tensor * silu     = cgraph->nodes[গ্রন্থি_idx + 1];

        যদি (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            ফেরত মিথ্যা;
        }
        যদি (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            ফেরত মিথ্যা;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        যদি (!ggml_is_contiguous(silu)) {
            ফেরত মিথ্যা;
        }

        ফেরত সত্য;
    }

    // ADD(bias) + UNARY + MUL(scale): the delta-net alpha gate, softplus(alpha + dt) * a.
    // The broadcast is what stops the same-shape UNARY + MUL branch above firing past one token.
    যদি (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1) {
        ধ্রুবক ggml_tensor * add   = cgraph->nodes[গ্রন্থি_idx];
        ধ্রুবক ggml_tensor * unary = cgraph->nodes[গ্রন্থি_idx + 1];
        ধ্রুবক ggml_tensor * mul   = cgraph->nodes[গ্রন্থি_idx + 2];

        ধ্রুবক ggml_unary_op unary_op = ggml_get_unary_op(unary);
        যদি (unary_op != unary_ops.begin()[0]) {
            ফেরত মিথ্যা;
        }

        যদি (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            ফেরত মিথ্যা;
        }

        // ggml_can_fuse() has already pinned the chain: unary consumes add, mul consumes
        // unary, add and unary have one use each, and all three have the same shape
        ধ্রুবক ggml_tensor * a     = add->src[0];
        ধ্রুবক ggml_tensor * bias  = add->src[1];
        ধ্রুবক ggml_tensor * scale = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];

        যদি (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            ফেরত মিথ্যা;
        }

        // the activation and the destination are indexed flat
        যদি (!ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(a, mul)) {
            ফেরত মিথ্যা;
        }

        // the kernel reads the bias and the scale as v[col], so each must be a single
        // contiguous row spanning ne0
        যদি (bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] ||
            ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
            ফেরত মিথ্যা;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        যদি (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            ফেরত মিথ্যা;
        }

        ফেরত সত্য;
    }

    যদি (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD &&
        ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        ধ্রুবক ggml_tensor * ssm_conv = cgraph->nodes[গ্রন্থি_idx];
        ধ্রুবক ggml_tensor * add      = cgraph->nodes[গ্রন্থি_idx + 1];
        ধ্রুবক ggml_tensor * silu     = cgraph->nodes[গ্রন্থি_idx + 2];

        যদি (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            ফেরত মিথ্যা;
        }
        যদি (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            ফেরত মিথ্যা;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        যদি (!ggml_is_contiguous(silu)) {
            ফেরত মিথ্যা;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias
        ধ্রুবক ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        যদি (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            ফেরত মিথ্যা;
        }
        যদি (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            ফেরত মিথ্যা;
        }

        ফেরত সত্য;
    }

    যদি (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        ধ্রুবক ggml_tensor * rms_norm = cgraph->nodes[গ্রন্থি_idx];
        ধ্রুবক ggml_tensor * scale    = cgraph->nodes[গ্রন্থি_idx + 1];
        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        যদি (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            ফেরত মিথ্যা;
        }
        // the fused kernel reads/writes rows flat like the unfused pair
        যদি (!ggml_is_contiguous_rows(rms_norm) || !ggml_is_contiguous_rows(scale)) {
            ফেরত মিথ্যা;
        }
        ফেরত সত্য;
    }
    ফেরত মিথ্যা;
}
