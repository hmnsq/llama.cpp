#ਅਨ੍ਤਰ੍ਭਾਵ "fusion.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "binbcast.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_op_mul_mat_glu_fused().
ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_should_fuse_mul_mat_glu(ਅਚਲ ggml_tensor * gate, ਅਚਲ ggml_tensor * ਉੱਪਰ,
                                              ਅਚਲ ggml_tensor * glu) {
    // the fused epilogue implements these two; the rest fall back to the standalone GLU kernels
    ਅਚਲ ggml_glu_op glu_op = ggml_get_glu_op(glu);
    ਜੇ (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        ਵਾਪਸ ਝੂਠ;
    }

    // the kernel always treats src[0] as the activated operand and src[1] as the multiplier
    ਜੇ (ggml_get_op_params_i32(glu, 1) /* swapped */) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ ggml_tensor * wu  = ਉੱਪਰ->src[0];
    ਅਚਲ ggml_tensor * wg  = gate->src[0];
    ਅਚਲ ggml_tensor * act = ਉੱਪਰ->src[1];

    // one activation and one output indexing must serve both weights; the block types
    // may differ, since the plain-layout fused kernel runs each operand's own vec_dot
    // (different types then imply different byte strides, so only the shape must agree)
    ਜੇ (!ggml_are_same_shape(wu, wg)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (act != gate->src[1]) {
        ਵਾਪਸ ਝੂਠ;
    }

    // fused K-quant GEMVs walk whole QK_K super-blocks: the reorder kernel covers same-type
    // q4_K, the plain-layout kernel covers q5_K / iq4_xs pairs incl. mixed gate/up types
    ਅਚਲ ਤਾਰਕਿਕ reorder_pair = wu->type == GGML_TYPE_Q4_K && wg->type == GGML_TYPE_Q4_K;
    ਅਚਲ ਤਾਰਕਿਕ plain_pair   = (wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_IQ4_XS) &&
                            (wg->type == GGML_TYPE_Q5_K || wg->type == GGML_TYPE_IQ4_XS);
    // the ESIMD kernel covers same-type q2_K..q6_K and q8_0, single column only (as unfused DMMV)
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_ESIMD
    ਅਚਲ ਤਾਰਕਿਕ esimd_pair   = g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
                            (wu->type == GGML_TYPE_Q2_K || wu->type == GGML_TYPE_Q3_K || wu->type == GGML_TYPE_Q4_K ||
                             wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_Q6_K || wu->type == GGML_TYPE_Q8_0);
#ਨਹੀਂਤਾਂ
    ਅਚਲ ਤਾਰਕਿਕ esimd_pair   = ਝੂਠ;
#ਜੇ_ਅੰਤ
    // q8_0 rows are walked in 32-wide blocks, so they only need to be a multiple of QK8_0
    ਅਚਲ int64_t ਪੰਕਤੀ_align = esimd_pair && wu->type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
    ਜੇ ((!reorder_pair && !plain_pair && !esimd_pair) || wu->ne[0] % ਪੰਕਤੀ_align != 0) {
        ਵਾਪਸ ਝੂਠ;
    }

    // one 2D reorder-layout matrix in, a plain column stride out: no broadcast or padding
    ਜੇ (!ggml_is_contiguous(wu) || !ggml_is_contiguous(wg) || !ggml_is_contiguous(act) ||
        !ggml_is_contiguous(glu)) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (act->ne[2] != 1 || act->ne[3] != 1 || wu->ne[2] != 1 || wu->ne[3] != 1) {
        ਵਾਪਸ ਝੂਠ;
    }
    // the kernel writes rows [0, wu->ne[1]) of each glu column, strided by glu->ne[0]
    ਜੇ (glu->ne[0] != wu->ne[1] || glu->ne[1] != act->ne[1]) {
        ਵਾਪਸ ਝੂਠ;
    }
    // mat-vec only: one column per decoded token, up to the batch the reorder kernels cover
    ਜੇ (act->ne[1] > MMVQ_MAX_BATCH_ਮਾਪ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
}

// the unary ops the fused unary chains in element_wise.cpp have a functor for
ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_fused_unary_has_kernel(ggml_unary_op unary_op) {
    ਵਾਪਸ unary_op == GGML_UNARY_OP_SILU || unary_op == GGML_UNARY_OP_SIGMOID ||
           unary_op == GGML_UNARY_OP_SOFTPLUS;
}

ਤਾਰਕਿਕ ggml_sycl_can_fuse(ਅਚਲ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, std::initializer_list<ਗਿਣਤੀ ggml_op> ops,
                        std::initializer_list<ਗਿਣਤੀ ggml_unary_op> unary_ops) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
    ਅਚਲ size_t ਅੰਕ_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == ਅੰਕ_unary);
#ਜੇ_ਅੰਤ

    ਜੇ (!g_ggml_sycl_enable_fusion) {
        ਵਾਪਸ ਝੂਠ;
    }

    // gate and up are siblings, not a chain, so ggml_can_fuse cannot express this: use the
    // subgraph form with the GLU as the only materialised output.
    ਜੇ (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_MUL_MAT &&
        ops.begin()[2] == GGML_OP_GLU) {
        ਜੇ (!ggml_can_fuse_subgraph(cgraph, ਗੰਢ_idx, ops, { ਗੰਢ_idx + 2 })) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ggml_tensor * glu  = cgraph->nodes[ਗੰਢ_idx + 2];
        ਅਚਲ ggml_tensor * gate = glu->src[0];
        ਅਚਲ ggml_tensor * ਉੱਪਰ   = glu->src[1];

        // don't assume which of the two mat-muls is the gate; infer it from the GLU's operands
        ਅਚਲ ਤਾਰਕਿਕ ok = (gate == cgraph->nodes[ਗੰਢ_idx] && ਉੱਪਰ == cgraph->nodes[ਗੰਢ_idx + 1]) ||
                        (gate == cgraph->nodes[ਗੰਢ_idx + 1] && ਉੱਪਰ == cgraph->nodes[ਗੰਢ_idx]);
        ਜੇ (!ok) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਵਾਪਸ ggml_sycl_should_fuse_mul_mat_glu(gate, ਉੱਪਰ, glu);
    }

    ਜੇ (!ggml_can_fuse(cgraph, ਗੰਢ_idx, ops)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        ਜੇ (ops.size() == 3 && ops.begin()[2] != GGML_OP_ਜੋੜੋ) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ggml_tensor * rms_norm = cgraph->nodes[ਗੰਢ_idx];
        ਅਚਲ ggml_tensor * mul      = cgraph->nodes[ਗੰਢ_idx + 1];
        ਅਚਲ ggml_tensor * ਜੋੜੋ_2      = ops.size() == 3 ? cgraph->nodes[ਗੰਢ_idx + 2] : ਖਾਲੀ_ਸੂਚਕ;

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        ਜੇ (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }

        // if rms norm is the B operand, then we don't handle broadcast
        ਜੇ (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        // the fused kernel indexes the weight as mul[col], so it must span ncols contiguously
        ਜੇ (mul_w->ne[0] != rms_norm->ne[0] || mul_w->nb[0] != ggml_type_ਮਾਪ(mul_w->type)) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (!ggml_is_contiguous_ਪੰਕਤੀਆਂ(mul->src[0]) || !ggml_is_contiguous_ਪੰਕਤੀਆਂ(mul->src[1])) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (ਜੋੜੋ_2 != ਖਾਲੀ_ਸੂਚਕ) {
            ਜੇ (ਜੋੜੋ_2->src[0]->type != GGML_TYPE_F32 ||
                ਜੋੜੋ_2->src[1]->type != GGML_TYPE_F32 ||
                ਜੋੜੋ_2->type != GGML_TYPE_F32) {
                ਵਾਪਸ ਝੂਠ;
            }

            // the fused kernel indexes the residual as add[col] and does not broadcast it
            ਅਚਲ ggml_tensor * ਜੋੜੋ_w = (ਜੋੜੋ_2->src[0] == mul) ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];
            ਜੇ (!ggml_are_same_shape(ਜੋੜੋ_w, ਜੋੜੋ_2)) {
                ਵਾਪਸ ਝੂਠ;
            }

            ਜੇ (!ggml_is_contiguous(ਜੋੜੋ_2->src[0]) || !ggml_is_contiguous_ਪੰਕਤੀਆਂ(ਜੋੜੋ_2->src[1])) {
                ਵਾਪਸ ਝੂਠ;
            }
        }

        ਵਾਪਸ ਸੱਚ;
    }

    ਜੇ (ops.size() == 2 && ops.begin()[0] == GGML_OP_ਜੋੜੋ && ops.begin()[1] == GGML_OP_ਜੋੜੋ) {
        ਅਚਲ ggml_tensor * add0 = cgraph->nodes[ਗੰਢ_idx];
        ਅਚਲ ggml_tensor * add1 = cgraph->nodes[ਗੰਢ_idx + 1];
        // ggml_can_fuse already guarantees add1 consumes add0 and that add0 has a single use.
        // Keep the CUDA association: the running sum is src0 of the next ADD so the fused
        // float fold matches two sequential add() launches.
        ਜੇ (add1->src[0] != add0) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ggml_tensor * c = add1->src[1];
        ਜੇ (!ggml_sycl_ਜੋੜੋ_kernel_supports(add0->src[0]->type, add0->src[1]->type, add0->type) ||
            !ggml_sycl_ਜੋੜੋ_kernel_supports(add0->type, c->type, add1->type)) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਵਾਪਸ ਸੱਚ;
    }

    ਜੇ (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        ਅਚਲ ggml_tensor * unary = cgraph->nodes[ਗੰਢ_idx];
        ਅਚਲ ggml_tensor * mul   = cgraph->nodes[ਗੰਢ_idx + 1];

        ਅਚਲ ggml_unary_op unary_op = ggml_get_unary_op(unary);
        ਜੇ (unary_op != unary_ops.begin()[0]) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        ਜੇ (other->type != unary->type) {
            ਵਾਪਸ ਝੂਠ;
        }

        // one row stride per source comes from nb[1], so rows must be contiguous and equally
        // shaped; the destination is written flat, so it must be fully contiguous
        ਜੇ (!ggml_is_contiguous_1(unary->src[0]) || !ggml_is_contiguous_1(other) ||
            !ggml_are_same_shape(other, unary) || !ggml_is_contiguous(mul)) {
            ਵਾਪਸ ਝੂਠ;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        ਜੇ (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਵਾਪਸ ਸੱਚ;
    }

    ਜੇ (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        ਅਚਲ ggml_tensor * ssm_conv = cgraph->nodes[ਗੰਢ_idx];
        ਅਚਲ ggml_tensor * silu     = cgraph->nodes[ਗੰਢ_idx + 1];

        ਜੇ (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        ਜੇ (!ggml_is_contiguous(silu)) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਵਾਪਸ ਸੱਚ;
    }

    // ADD(bias) + UNARY + MUL(scale): the delta-net alpha gate, softplus(alpha + dt) * a.
    // The broadcast is what stops the same-shape UNARY + MUL branch above firing past one token.
    ਜੇ (ops.size() == 3 && ops.begin()[0] == GGML_OP_ਜੋੜੋ && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1) {
        ਅਚਲ ggml_tensor * ਜੋੜੋ_2   = cgraph->nodes[ਗੰਢ_idx];
        ਅਚਲ ggml_tensor * unary = cgraph->nodes[ਗੰਢ_idx + 1];
        ਅਚਲ ggml_tensor * mul   = cgraph->nodes[ਗੰਢ_idx + 2];

        ਅਚਲ ggml_unary_op unary_op = ggml_get_unary_op(unary);
        ਜੇ (unary_op != unary_ops.begin()[0]) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਜੇ (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            ਵਾਪਸ ਝੂਠ;
        }

        // ggml_can_fuse() has already pinned the chain: unary consumes add, mul consumes
        // unary, add and unary have one use each, and all three have the same shape
        ਅਚਲ ggml_tensor * a     = ਜੋੜੋ_2->src[0];
        ਅਚਲ ggml_tensor * bias  = ਜੋੜੋ_2->src[1];
        ਅਚਲ ggml_tensor * scale = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];

        ਜੇ (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }

        // the activation and the destination are indexed flat
        ਜੇ (!ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(a, mul)) {
            ਵਾਪਸ ਝੂਠ;
        }

        // the kernel reads the bias and the scale as v[col], so each must be a single
        // contiguous row spanning ne0
        ਜੇ (bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] ||
            ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
            ਵਾਪਸ ਝੂਠ;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        ਜੇ (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਵਾਪਸ ਸੱਚ;
    }

    ਜੇ (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ਜੋੜੋ &&
        ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        ਅਚਲ ggml_tensor * ssm_conv = cgraph->nodes[ਗੰਢ_idx];
        ਅਚਲ ggml_tensor * ਜੋੜੋ_2      = cgraph->nodes[ਗੰਢ_idx + 1];
        ਅਚਲ ggml_tensor * silu     = cgraph->nodes[ਗੰਢ_idx + 2];

        ਜੇ (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (ssm_conv->type != GGML_TYPE_F32 || ਜੋੜੋ_2->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        ਜੇ (!ggml_is_contiguous(silu)) {
            ਵਾਪਸ ਝੂਠ;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias
        ਅਚਲ ggml_tensor * bias = (ਜੋੜੋ_2->src[0] == ssm_conv) ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];
        ਜੇ (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਵਾਪਸ ਸੱਚ;
    }

    ਜੇ (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        ਅਚਲ ggml_tensor * rms_norm = cgraph->nodes[ਗੰਢ_idx];
        ਅਚਲ ggml_tensor * scale    = cgraph->nodes[ਗੰਢ_idx + 1];
        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        ਜੇ (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }
        // the fused kernel reads/writes rows flat like the unfused pair
        ਜੇ (!ggml_is_contiguous_ਪੰਕਤੀਆਂ(rms_norm) || !ggml_is_contiguous_ਪੰਕਤੀਆਂ(scale)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਵਾਪਸ ਸੱਚ;
    }
    ਵਾਪਸ ਝੂਠ;
}
