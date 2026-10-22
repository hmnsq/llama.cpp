#समावेश "fusion.hpp"
#समावेश "binbcast.hpp"

#समावेश <algorithm>

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_op_mul_mat_glu_fused().
अटल bool ggml_sycl_should_fuse_mul_mat_glu(स्थिर ggml_tensor * gate, स्थिर ggml_tensor * up,
                                              स्थिर ggml_tensor * glu) {
    // the fused epilogue implements these two; the rest fall back to the standalone GLU kernels
    स्थिर ggml_glu_op glu_op = ggml_get_glu_op(glu);
    अगर (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        वापस false;
    }

    // the kernel always treats src[0] as the activated operand and src[1] as the multiplier
    अगर (ggml_get_op_params_i32(glu, 1) /* swapped */) {
        वापस false;
    }

    स्थिर ggml_tensor * wu  = up->src[0];
    स्थिर ggml_tensor * wg  = gate->src[0];
    स्थिर ggml_tensor * act = up->src[1];

    // one activation and one output indexing must serve both weights; the block types
    // may differ, since the plain-layout fused kernel runs each operand's own vec_dot
    // (different types then imply different byte strides, so only the shape must agree)
    अगर (!ggml_are_same_shape(wu, wg)) {
        वापस false;
    }
    अगर (act != gate->src[1]) {
        वापस false;
    }

    // fused K-quant GEMVs walk whole QK_K super-blocks: the reorder kernel covers same-type
    // q4_K, the plain-layout kernel covers q5_K / iq4_xs pairs incl. mixed gate/up types
    स्थिर bool reorder_pair = wu->type == GGML_TYPE_Q4_K && wg->type == GGML_TYPE_Q4_K;
    स्थिर bool plain_pair   = (wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_IQ4_XS) &&
                            (wg->type == GGML_TYPE_Q5_K || wg->type == GGML_TYPE_IQ4_XS);
    // the ESIMD kernel covers same-type q2_K..q6_K and q8_0, single column only (as unfused DMMV)
#अगर_घोषित GGML_SYCL_DMMV_HAS_ESIMD
    स्थिर bool esimd_pair   = g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
                            (wu->type == GGML_TYPE_Q2_K || wu->type == GGML_TYPE_Q3_K || wu->type == GGML_TYPE_Q4_K ||
                             wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_Q6_K || wu->type == GGML_TYPE_Q8_0);
#अन्यथा
    स्थिर bool esimd_pair   = false;
#पूर्ण_अगर
    // q8_0 rows are walked in 32-wide blocks, so they only need to be a multiple of QK8_0
    स्थिर int64_t row_align = esimd_pair && wu->type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
    अगर ((!reorder_pair && !plain_pair && !esimd_pair) || wu->ne[0] % row_align != 0) {
        वापस false;
    }

    // one 2D reorder-layout matrix in, a plain column stride out: no broadcast or padding
    अगर (!ggml_is_contiguous(wu) || !ggml_is_contiguous(wg) || !ggml_is_contiguous(act) ||
        !ggml_is_contiguous(glu)) {
        वापस false;
    }
    अगर (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        वापस false;
    }
    अगर (act->ne[2] != 1 || act->ne[3] != 1 || wu->ne[2] != 1 || wu->ne[3] != 1) {
        वापस false;
    }
    // the kernel writes rows [0, wu->ne[1]) of each glu column, strided by glu->ne[0]
    अगर (glu->ne[0] != wu->ne[1] || glu->ne[1] != act->ne[1]) {
        वापस false;
    }
    // mat-vec only: one column per decoded token, up to the batch the reorder kernels cover
    अगर (act->ne[1] > MMVQ_MAX_BATCH_आकार) {
        वापस false;
    }

    वापस true;
}

// the unary ops the fused unary chains in element_wise.cpp have a functor for
अटल bool ggml_sycl_fused_unary_has_kernel(ggml_unary_op unary_op) {
    वापस unary_op == GGML_UNARY_OP_SILU || unary_op == GGML_UNARY_OP_SIGMOID ||
           unary_op == GGML_UNARY_OP_SOFTPLUS;
}

bool ggml_sycl_can_fuse(स्थिर ggml_cgraph * cgraph, पूर्णांक गाँठ_idx, std::initializer_list<क्रमागत ggml_op> ops,
                        std::initializer_list<क्रमागत ggml_unary_op> unary_ops) {
#अगर_अघोषित न_संशोधन
    स्थिर माप_प्रकार अंक_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == अंक_unary);
#पूर्ण_अगर

    अगर (!g_ggml_sycl_enable_fusion) {
        वापस false;
    }

    // gate and up are siblings, not a chain, so ggml_can_fuse cannot express this: use the
    // subgraph form with the GLU as the only materialised output.
    अगर (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_MUL_MAT &&
        ops.begin()[2] == GGML_OP_GLU) {
        अगर (!ggml_can_fuse_subgraph(cgraph, गाँठ_idx, ops, { गाँठ_idx + 2 })) {
            वापस false;
        }

        स्थिर ggml_tensor * glu  = cgraph->nodes[गाँठ_idx + 2];
        स्थिर ggml_tensor * gate = glu->src[0];
        स्थिर ggml_tensor * up   = glu->src[1];

        // don't assume which of the two mat-muls is the gate; infer it from the GLU's operands
        स्थिर bool ok = (gate == cgraph->nodes[गाँठ_idx] && up == cgraph->nodes[गाँठ_idx + 1]) ||
                        (gate == cgraph->nodes[गाँठ_idx + 1] && up == cgraph->nodes[गाँठ_idx]);
        अगर (!ok) {
            वापस false;
        }

        वापस ggml_sycl_should_fuse_mul_mat_glu(gate, up, glu);
    }

    अगर (!ggml_can_fuse(cgraph, गाँठ_idx, ops)) {
        वापस false;
    }

    अगर ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        अगर (ops.size() == 3 && ops.begin()[2] != GGML_OP_ADD) {
            वापस false;
        }

        स्थिर ggml_tensor * rms_norm = cgraph->nodes[गाँठ_idx];
        स्थिर ggml_tensor * mul      = cgraph->nodes[गाँठ_idx + 1];
        स्थिर ggml_tensor * add      = ops.size() == 3 ? cgraph->nodes[गाँठ_idx + 2] : nullptr;

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        अगर (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            वापस false;
        }

        // if rms norm is the B operand, then we don't handle broadcast
        अगर (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            वापस false;
        }

        स्थिर ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        // the fused kernel indexes the weight as mul[col], so it must span ncols contiguously
        अगर (mul_w->ne[0] != rms_norm->ne[0] || mul_w->nb[0] != ggml_type_आकार(mul_w->type)) {
            वापस false;
        }

        अगर (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            वापस false;
        }

        अगर (add != nullptr) {
            अगर (add->src[0]->type != GGML_TYPE_F32 ||
                add->src[1]->type != GGML_TYPE_F32 ||
                add->type != GGML_TYPE_F32) {
                वापस false;
            }

            // the fused kernel indexes the residual as add[col] and does not broadcast it
            स्थिर ggml_tensor * add_w = (add->src[0] == mul) ? add->src[1] : add->src[0];
            अगर (!ggml_are_same_shape(add_w, add)) {
                वापस false;
            }

            अगर (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1])) {
                वापस false;
            }
        }

        वापस true;
    }

    अगर (ops.size() == 2 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_ADD) {
        स्थिर ggml_tensor * add0 = cgraph->nodes[गाँठ_idx];
        स्थिर ggml_tensor * add1 = cgraph->nodes[गाँठ_idx + 1];
        // ggml_can_fuse already guarantees add1 consumes add0 and that add0 has a single use.
        // Keep the CUDA association: the running sum is src0 of the next ADD so the fused
        // float fold matches two sequential add() launches.
        अगर (add1->src[0] != add0) {
            वापस false;
        }

        स्थिर ggml_tensor * c = add1->src[1];
        अगर (!ggml_sycl_add_kernel_supports(add0->src[0]->type, add0->src[1]->type, add0->type) ||
            !ggml_sycl_add_kernel_supports(add0->type, c->type, add1->type)) {
            वापस false;
        }

        वापस true;
    }

    अगर (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        स्थिर ggml_tensor * unary = cgraph->nodes[गाँठ_idx];
        स्थिर ggml_tensor * mul   = cgraph->nodes[गाँठ_idx + 1];

        स्थिर ggml_unary_op unary_op = ggml_get_unary_op(unary);
        अगर (unary_op != unary_ops.begin()[0]) {
            वापस false;
        }

        अगर (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            वापस false;
        }

        अगर (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            वापस false;
        }

        स्थिर ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        अगर (other->type != unary->type) {
            वापस false;
        }

        // one row stride per source comes from nb[1], so rows must be contiguous and equally
        // shaped; the destination is written flat, so it must be fully contiguous
        अगर (!ggml_is_contiguous_1(unary->src[0]) || !ggml_is_contiguous_1(other) ||
            !ggml_are_same_shape(other, unary) || !ggml_is_contiguous(mul)) {
            वापस false;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        अगर (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            वापस false;
        }

        वापस true;
    }

    अगर (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        स्थिर ggml_tensor * ssm_conv = cgraph->nodes[गाँठ_idx];
        स्थिर ggml_tensor * silu     = cgraph->nodes[गाँठ_idx + 1];

        अगर (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            वापस false;
        }
        अगर (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            वापस false;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        अगर (!ggml_is_contiguous(silu)) {
            वापस false;
        }

        वापस true;
    }

    // ADD(bias) + UNARY + MUL(scale): the delta-net alpha gate, softplus(alpha + dt) * a.
    // The broadcast is what stops the same-shape UNARY + MUL branch above firing past one token.
    अगर (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1) {
        स्थिर ggml_tensor * add   = cgraph->nodes[गाँठ_idx];
        स्थिर ggml_tensor * unary = cgraph->nodes[गाँठ_idx + 1];
        स्थिर ggml_tensor * mul   = cgraph->nodes[गाँठ_idx + 2];

        स्थिर ggml_unary_op unary_op = ggml_get_unary_op(unary);
        अगर (unary_op != unary_ops.begin()[0]) {
            वापस false;
        }

        अगर (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            वापस false;
        }

        // ggml_can_fuse() has already pinned the chain: unary consumes add, mul consumes
        // unary, add and unary have one use each, and all three have the same shape
        स्थिर ggml_tensor * a     = add->src[0];
        स्थिर ggml_tensor * bias  = add->src[1];
        स्थिर ggml_tensor * scale = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];

        अगर (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            वापस false;
        }

        // the activation and the destination are indexed flat
        अगर (!ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(a, mul)) {
            वापस false;
        }

        // the kernel reads the bias and the scale as v[col], so each must be a single
        // contiguous row spanning ne0
        अगर (bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] ||
            ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
            वापस false;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        अगर (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            वापस false;
        }

        वापस true;
    }

    अगर (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD &&
        ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        स्थिर ggml_tensor * ssm_conv = cgraph->nodes[गाँठ_idx];
        स्थिर ggml_tensor * add      = cgraph->nodes[गाँठ_idx + 1];
        स्थिर ggml_tensor * silu     = cgraph->nodes[गाँठ_idx + 2];

        अगर (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            वापस false;
        }
        अगर (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            वापस false;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        अगर (!ggml_is_contiguous(silu)) {
            वापस false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias
        स्थिर ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        अगर (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            वापस false;
        }
        अगर (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            वापस false;
        }

        वापस true;
    }

    अगर (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        स्थिर ggml_tensor * rms_norm = cgraph->nodes[गाँठ_idx];
        स्थिर ggml_tensor * scale    = cgraph->nodes[गाँठ_idx + 1];
        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        अगर (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            वापस false;
        }
        // the fused kernel reads/writes rows flat like the unfused pair
        अगर (!ggml_is_contiguous_rows(rms_norm) || !ggml_is_contiguous_rows(scale)) {
            वापस false;
        }
        वापस true;
    }
    वापस false;
}
