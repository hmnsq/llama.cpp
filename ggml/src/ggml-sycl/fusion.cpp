#ଅନ୍ତର୍ଭାଵ "fusion.hpp"
#ଅନ୍ତର୍ଭାଵ "binbcast.hpp"

#ଅନ୍ତର୍ଭାଵ <algorithm>

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_op_mul_mat_glu_fused().
ସ୍ଥିର ତାର୍କିକ ggml_sycl_should_fuse_mul_mat_glu(ଅଚଳ ggml_tensor * gate, ଅଚଳ ggml_tensor * up,
                                              ଅଚଳ ggml_tensor * glu) {
    // the fused epilogue implements these two; the rest fall back to the standalone GLU kernels
    ଅଚଳ ggml_glu_op glu_op = ggml_get_glu_op(glu);
    ଯଦି (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // the kernel always treats src[0] as the activated operand and src[1] as the multiplier
    ଯଦି (ggml_get_op_params_i32(glu, 1) /* swapped */) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ ggml_tensor * wu  = up->src[0];
    ଅଚଳ ggml_tensor * wg  = gate->src[0];
    ଅଚଳ ggml_tensor * act = up->src[1];

    // one activation and one output indexing must serve both weights; the block types
    // may differ, since the plain-layout fused kernel runs each operand's own vec_dot
    // (different types then imply different byte strides, so only the shape must agree)
    ଯଦି (!ggml_are_same_shape(wu, wg)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (act != gate->src[1]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // fused K-quant GEMVs walk whole QK_K super-blocks: the reorder kernel covers same-type
    // q4_K, the plain-layout kernel covers q5_K / iq4_xs pairs incl. mixed gate/up types
    ଅଚଳ ତାର୍କିକ reorder_pair = wu->type == GGML_TYPE_Q4_K && wg->type == GGML_TYPE_Q4_K;
    ଅଚଳ ତାର୍କିକ plain_pair   = (wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_IQ4_XS) &&
                            (wg->type == GGML_TYPE_Q5_K || wg->type == GGML_TYPE_IQ4_XS);
    // the ESIMD kernel covers same-type q2_K..q6_K and q8_0, single column only (as unfused DMMV)
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    ଅଚଳ ତାର୍କିକ esimd_pair   = g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
                            (wu->type == GGML_TYPE_Q2_K || wu->type == GGML_TYPE_Q3_K || wu->type == GGML_TYPE_Q4_K ||
                             wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_Q6_K || wu->type == GGML_TYPE_Q8_0);
#ଅନ୍ୟଥା
    ଅଚଳ ତାର୍କିକ esimd_pair   = ମିଥ୍ୟା;
#endif
    // q8_0 rows are walked in 32-wide blocks, so they only need to be a multiple of QK8_0
    ଅଚଳ int64_t row_align = esimd_pair && wu->type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
    ଯଦି ((!reorder_pair && !plain_pair && !esimd_pair) || wu->ne[0] % row_align != 0) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // one 2D reorder-layout matrix in, a plain column stride out: no broadcast or padding
    ଯଦି (!ggml_is_contiguous(wu) || !ggml_is_contiguous(wg) || !ggml_is_contiguous(act) ||
        !ggml_is_contiguous(glu)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (act->ne[2] != 1 || act->ne[3] != 1 || wu->ne[2] != 1 || wu->ne[3] != 1) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    // the kernel writes rows [0, wu->ne[1]) of each glu column, strided by glu->ne[0]
    ଯଦି (glu->ne[0] != wu->ne[1] || glu->ne[1] != act->ne[1]) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    // mat-vec only: one column per decoded token, up to the batch the reorder kernels cover
    ଯଦି (act->ne[1] > MMVQ_MAX_BATCH_SIZE) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// the unary ops the fused unary chains in element_wise.cpp have a functor for
ସ୍ଥିର ତାର୍କିକ ggml_sycl_fused_unary_has_kernel(ggml_unary_op unary_op) {
    ଫେରନ୍ତୁ unary_op == GGML_UNARY_OP_SILU || unary_op == GGML_UNARY_OP_SIGMOID ||
           unary_op == GGML_UNARY_OP_SOFTPLUS;
}

ତାର୍କିକ ggml_sycl_can_fuse(ଅଚଳ ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx, std::initializer_list<ଗଣନା ggml_op> ops,
                        std::initializer_list<ଗଣନା ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    ଅଚଳ size_t ଅଙ୍କ_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == ଅଙ୍କ_unary);
#endif

    ଯଦି (!g_ggml_sycl_enable_fusion) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // gate and up are siblings, not a chain, so ggml_can_fuse cannot express this: use the
    // subgraph form with the GLU as the only materialised output.
    ଯଦି (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_MUL_MAT &&
        ops.begin()[2] == GGML_OP_GLU) {
        ଯଦି (!ggml_can_fuse_subgraph(cgraph, ଗଣ୍ଠି_idx, ops, { ଗଣ୍ଠି_idx + 2 })) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ggml_tensor * glu  = cgraph->nodes[ଗଣ୍ଠି_idx + 2];
        ଅଚଳ ggml_tensor * gate = glu->src[0];
        ଅଚଳ ggml_tensor * up   = glu->src[1];

        // don't assume which of the two mat-muls is the gate; infer it from the GLU's operands
        ଅଚଳ ତାର୍କିକ ok = (gate == cgraph->nodes[ଗଣ୍ଠି_idx] && up == cgraph->nodes[ଗଣ୍ଠି_idx + 1]) ||
                        (gate == cgraph->nodes[ଗଣ୍ଠି_idx + 1] && up == cgraph->nodes[ଗଣ୍ଠି_idx]);
        ଯଦି (!ok) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଫେରନ୍ତୁ ggml_sycl_should_fuse_mul_mat_glu(gate, up, glu);
    }

    ଯଦି (!ggml_can_fuse(cgraph, ଗଣ୍ଠି_idx, ops)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଯଦି ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        ଯଦି (ops.size() == 3 && ops.begin()[2] != GGML_OP_ADD) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ggml_tensor * rms_norm = cgraph->nodes[ଗଣ୍ଠି_idx];
        ଅଚଳ ggml_tensor * mul      = cgraph->nodes[ଗଣ୍ଠି_idx + 1];
        ଅଚଳ ggml_tensor * add      = ops.size() == 3 ? cgraph->nodes[ଗଣ୍ଠି_idx + 2] : nullptr;

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        ଯଦି (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        // if rms norm is the B operand, then we don't handle broadcast
        ଯଦି (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        // the fused kernel indexes the weight as mul[col], so it must span ncols contiguously
        ଯଦି (mul_w->ne[0] != rms_norm->ne[0] || mul_w->nb[0] != ggml_type_size(mul_w->type)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (add != nullptr) {
            ଯଦି (add->src[0]->type != GGML_TYPE_F32 ||
                add->src[1]->type != GGML_TYPE_F32 ||
                add->type != GGML_TYPE_F32) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            // the fused kernel indexes the residual as add[col] and does not broadcast it
            ଅଚଳ ggml_tensor * add_w = (add->src[0] == mul) ? add->src[1] : add->src[0];
            ଯଦି (!ggml_are_same_shape(add_w, add)) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଯଦି (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1])) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଯଦି (ops.size() == 2 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_ADD) {
        ଅଚଳ ggml_tensor * add0 = cgraph->nodes[ଗଣ୍ଠି_idx];
        ଅଚଳ ggml_tensor * add1 = cgraph->nodes[ଗଣ୍ଠି_idx + 1];
        // ggml_can_fuse already guarantees add1 consumes add0 and that add0 has a single use.
        // Keep the CUDA association: the running sum is src0 of the next ADD so the fused
        // float fold matches two sequential add() launches.
        ଯଦି (add1->src[0] != add0) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ggml_tensor * c = add1->src[1];
        ଯଦି (!ggml_sycl_add_kernel_supports(add0->src[0]->type, add0->src[1]->type, add0->type) ||
            !ggml_sycl_add_kernel_supports(add0->type, c->type, add1->type)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଯଦି (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        ଅଚଳ ggml_tensor * unary = cgraph->nodes[ଗଣ୍ଠି_idx];
        ଅଚଳ ggml_tensor * mul   = cgraph->nodes[ଗଣ୍ଠି_idx + 1];

        ଅଚଳ ggml_unary_op unary_op = ggml_get_unary_op(unary);
        ଯଦି (unary_op != unary_ops.begin()[0]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        ଯଦି (other->type != unary->type) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        // one row stride per source comes from nb[1], so rows must be contiguous and equally
        // shaped; the destination is written flat, so it must be fully contiguous
        ଯଦି (!ggml_is_contiguous_1(unary->src[0]) || !ggml_is_contiguous_1(other) ||
            !ggml_are_same_shape(other, unary) || !ggml_is_contiguous(mul)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        ଯଦି (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଯଦି (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        ଅଚଳ ggml_tensor * ssm_conv = cgraph->nodes[ଗଣ୍ଠି_idx];
        ଅଚଳ ggml_tensor * silu     = cgraph->nodes[ଗଣ୍ଠି_idx + 1];

        ଯଦି (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        ଯଦି (!ggml_is_contiguous(silu)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    // ADD(bias) + UNARY + MUL(scale): the delta-net alpha gate, softplus(alpha + dt) * a.
    // The broadcast is what stops the same-shape UNARY + MUL branch above firing past one token.
    ଯଦି (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1) {
        ଅଚଳ ggml_tensor * add   = cgraph->nodes[ଗଣ୍ଠି_idx];
        ଅଚଳ ggml_tensor * unary = cgraph->nodes[ଗଣ୍ଠି_idx + 1];
        ଅଚଳ ggml_tensor * mul   = cgraph->nodes[ଗଣ୍ଠି_idx + 2];

        ଅଚଳ ggml_unary_op unary_op = ggml_get_unary_op(unary);
        ଯଦି (unary_op != unary_ops.begin()[0]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଯଦି (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        // ggml_can_fuse() has already pinned the chain: unary consumes add, mul consumes
        // unary, add and unary have one use each, and all three have the same shape
        ଅଚଳ ggml_tensor * a     = add->src[0];
        ଅଚଳ ggml_tensor * bias  = add->src[1];
        ଅଚଳ ggml_tensor * scale = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];

        ଯଦି (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        // the activation and the destination are indexed flat
        ଯଦି (!ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(a, mul)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        // the kernel reads the bias and the scale as v[col], so each must be a single
        // contiguous row spanning ne0
        ଯଦି (bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] ||
            ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        ଯଦି (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଯଦି (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD &&
        ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        ଅଚଳ ggml_tensor * ssm_conv = cgraph->nodes[ଗଣ୍ଠି_idx];
        ଅଚଳ ggml_tensor * add      = cgraph->nodes[ଗଣ୍ଠି_idx + 1];
        ଅଚଳ ggml_tensor * silu     = cgraph->nodes[ଗଣ୍ଠି_idx + 2];

        ଯଦି (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        ଯଦି (!ggml_is_contiguous(silu)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias
        ଅଚଳ ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        ଯଦି (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଯଦି (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଯଦି (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        ଅଚଳ ggml_tensor * rms_norm = cgraph->nodes[ଗଣ୍ଠି_idx];
        ଅଚଳ ggml_tensor * scale    = cgraph->nodes[ଗଣ୍ଠି_idx + 1];
        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        ଯଦି (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        // the fused kernel reads/writes rows flat like the unfused pair
        ଯଦି (!ggml_is_contiguous_rows(rms_norm) || !ggml_is_contiguous_rows(scale)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}
