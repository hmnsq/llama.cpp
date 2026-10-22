#అన్తర్భావ "fusion.hpp"
#అన్తర్భావ "binbcast.hpp"

#అన్తర్భావ <algorithm>

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_op_mul_mat_glu_fused().
స్థిర తార్కిక ggml_sycl_should_fuse_mul_mat_glu(స్థిరరాశి ggml_tensor * gate, స్థిరరాశి ggml_tensor * పైకి,
                                              స్థిరరాశి ggml_tensor * glu) {
    // the fused epilogue implements these two; the rest fall back to the standalone GLU kernels
    స్థిరరాశి ggml_glu_op glu_op = ggml_get_glu_op(glu);
    ఒకవేళ (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        తిరిగి అబద్ధం;
    }

    // the kernel always treats src[0] as the activated operand and src[1] as the multiplier
    ఒకవేళ (ggml_get_op_params_i32(glu, 1) /* swapped */) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి ggml_tensor * wu  = పైకి->src[0];
    స్థిరరాశి ggml_tensor * wg  = gate->src[0];
    స్థిరరాశి ggml_tensor * act = పైకి->src[1];

    // one activation and one output indexing must serve both weights; the block types
    // may differ, since the plain-layout fused kernel runs each operand's own vec_dot
    // (different types then imply different byte strides, so only the shape must agree)
    ఒకవేళ (!ggml_are_same_shape(wu, wg)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (act != gate->src[1]) {
        తిరిగి అబద్ధం;
    }

    // fused K-quant GEMVs walk whole QK_K super-blocks: the reorder kernel covers same-type
    // q4_K, the plain-layout kernel covers q5_K / iq4_xs pairs incl. mixed gate/up types
    స్థిరరాశి తార్కిక reorder_pair = wu->type == GGML_TYPE_Q4_K && wg->type == GGML_TYPE_Q4_K;
    స్థిరరాశి తార్కిక plain_pair   = (wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_IQ4_XS) &&
                            (wg->type == GGML_TYPE_Q5_K || wg->type == GGML_TYPE_IQ4_XS);
    // the ESIMD kernel covers same-type q2_K..q6_K and q8_0, single column only (as unfused DMMV)
#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_ESIMD
    స్థిరరాశి తార్కిక esimd_pair   = g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
                            (wu->type == GGML_TYPE_Q2_K || wu->type == GGML_TYPE_Q3_K || wu->type == GGML_TYPE_Q4_K ||
                             wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_Q6_K || wu->type == GGML_TYPE_Q8_0);
#లేకపోతే
    స్థిరరాశి తార్కిక esimd_pair   = అబద్ధం;
#అయితే_ముగింపు
    // q8_0 rows are walked in 32-wide blocks, so they only need to be a multiple of QK8_0
    స్థిరరాశి int64_t అడ్డువరుస_align = esimd_pair && wu->type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
    ఒకవేళ ((!reorder_pair && !plain_pair && !esimd_pair) || wu->ne[0] % అడ్డువరుస_align != 0) {
        తిరిగి అబద్ధం;
    }

    // one 2D reorder-layout matrix in, a plain column stride out: no broadcast or padding
    ఒకవేళ (!ggml_is_contiguous(wu) || !ggml_is_contiguous(wg) || !ggml_is_contiguous(act) ||
        !ggml_is_contiguous(glu)) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (act->ne[2] != 1 || act->ne[3] != 1 || wu->ne[2] != 1 || wu->ne[3] != 1) {
        తిరిగి అబద్ధం;
    }
    // the kernel writes rows [0, wu->ne[1]) of each glu column, strided by glu->ne[0]
    ఒకవేళ (glu->ne[0] != wu->ne[1] || glu->ne[1] != act->ne[1]) {
        తిరిగి అబద్ధం;
    }
    // mat-vec only: one column per decoded token, up to the batch the reorder kernels cover
    ఒకవేళ (act->ne[1] > MMVQ_MAX_BATCH_కొలత) {
        తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
}

// the unary ops the fused unary chains in element_wise.cpp have a functor for
స్థిర తార్కిక ggml_sycl_fused_unary_has_kernel(ggml_unary_op unary_op) {
    తిరిగి unary_op == GGML_UNARY_OP_SILU || unary_op == GGML_UNARY_OP_SIGMOID ||
           unary_op == GGML_UNARY_OP_SOFTPLUS;
}

తార్కిక ggml_sycl_can_fuse(స్థిరరాశి ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, std::initializer_list<గణన ggml_op> ops,
                        std::initializer_list<గణన ggml_unary_op> unary_ops) {
#నిర్వచితం_కాకపోతే NDEBUG
    స్థిరరాశి size_t అంకె_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == అంకె_unary);
#అయితే_ముగింపు

    ఒకవేళ (!g_ggml_sycl_enable_fusion) {
        తిరిగి అబద్ధం;
    }

    // gate and up are siblings, not a chain, so ggml_can_fuse cannot express this: use the
    // subgraph form with the GLU as the only materialised output.
    ఒకవేళ (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_MUL_MAT &&
        ops.begin()[2] == GGML_OP_GLU) {
        ఒకవేళ (!ggml_can_fuse_subgraph(cgraph, కణుపు_idx, ops, { కణుపు_idx + 2 })) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి ggml_tensor * glu  = cgraph->nodes[కణుపు_idx + 2];
        స్థిరరాశి ggml_tensor * gate = glu->src[0];
        స్థిరరాశి ggml_tensor * పైకి   = glu->src[1];

        // don't assume which of the two mat-muls is the gate; infer it from the GLU's operands
        స్థిరరాశి తార్కిక ok = (gate == cgraph->nodes[కణుపు_idx] && పైకి == cgraph->nodes[కణుపు_idx + 1]) ||
                        (gate == cgraph->nodes[కణుపు_idx + 1] && పైకి == cgraph->nodes[కణుపు_idx]);
        ఒకవేళ (!ok) {
            తిరిగి అబద్ధం;
        }

        తిరిగి ggml_sycl_should_fuse_mul_mat_glu(gate, పైకి, glu);
    }

    ఒకవేళ (!ggml_can_fuse(cgraph, కణుపు_idx, ops)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        ఒకవేళ (ops.size() == 3 && ops.begin()[2] != GGML_OP_కూడు) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి ggml_tensor * rms_norm = cgraph->nodes[కణుపు_idx];
        స్థిరరాశి ggml_tensor * mul      = cgraph->nodes[కణుపు_idx + 1];
        స్థిరరాశి ggml_tensor * కూడు_2      = ops.size() == 3 ? cgraph->nodes[కణుపు_idx + 2] : శూన్యసూచిక;

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        ఒకవేళ (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }

        // if rms norm is the B operand, then we don't handle broadcast
        ఒకవేళ (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        // the fused kernel indexes the weight as mul[col], so it must span ncols contiguously
        ఒకవేళ (mul_w->ne[0] != rms_norm->ne[0] || mul_w->nb[0] != ggml_type_కొలత(mul_w->type)) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (!ggml_is_contiguous_అడ్డువరుసలు(mul->src[0]) || !ggml_is_contiguous_అడ్డువరుసలు(mul->src[1])) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (కూడు_2 != శూన్యసూచిక) {
            ఒకవేళ (కూడు_2->src[0]->type != GGML_TYPE_F32 ||
                కూడు_2->src[1]->type != GGML_TYPE_F32 ||
                కూడు_2->type != GGML_TYPE_F32) {
                తిరిగి అబద్ధం;
            }

            // the fused kernel indexes the residual as add[col] and does not broadcast it
            స్థిరరాశి ggml_tensor * కూడు_w = (కూడు_2->src[0] == mul) ? కూడు_2->src[1] : కూడు_2->src[0];
            ఒకవేళ (!ggml_are_same_shape(కూడు_w, కూడు_2)) {
                తిరిగి అబద్ధం;
            }

            ఒకవేళ (!ggml_is_contiguous(కూడు_2->src[0]) || !ggml_is_contiguous_అడ్డువరుసలు(కూడు_2->src[1])) {
                తిరిగి అబద్ధం;
            }
        }

        తిరిగి నిజం;
    }

    ఒకవేళ (ops.size() == 2 && ops.begin()[0] == GGML_OP_కూడు && ops.begin()[1] == GGML_OP_కూడు) {
        స్థిరరాశి ggml_tensor * add0 = cgraph->nodes[కణుపు_idx];
        స్థిరరాశి ggml_tensor * add1 = cgraph->nodes[కణుపు_idx + 1];
        // ggml_can_fuse already guarantees add1 consumes add0 and that add0 has a single use.
        // Keep the CUDA association: the running sum is src0 of the next ADD so the fused
        // float fold matches two sequential add() launches.
        ఒకవేళ (add1->src[0] != add0) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి ggml_tensor * c = add1->src[1];
        ఒకవేళ (!ggml_sycl_కూడు_kernel_supports(add0->src[0]->type, add0->src[1]->type, add0->type) ||
            !ggml_sycl_కూడు_kernel_supports(add0->type, c->type, add1->type)) {
            తిరిగి అబద్ధం;
        }

        తిరిగి నిజం;
    }

    ఒకవేళ (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        స్థిరరాశి ggml_tensor * unary = cgraph->nodes[కణుపు_idx];
        స్థిరరాశి ggml_tensor * mul   = cgraph->nodes[కణుపు_idx + 1];

        స్థిరరాశి ggml_unary_op unary_op = ggml_get_unary_op(unary);
        ఒకవేళ (unary_op != unary_ops.begin()[0]) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        ఒకవేళ (other->type != unary->type) {
            తిరిగి అబద్ధం;
        }

        // one row stride per source comes from nb[1], so rows must be contiguous and equally
        // shaped; the destination is written flat, so it must be fully contiguous
        ఒకవేళ (!ggml_is_contiguous_1(unary->src[0]) || !ggml_is_contiguous_1(other) ||
            !ggml_are_same_shape(other, unary) || !ggml_is_contiguous(mul)) {
            తిరిగి అబద్ధం;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        ఒకవేళ (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            తిరిగి అబద్ధం;
        }

        తిరిగి నిజం;
    }

    ఒకవేళ (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        స్థిరరాశి ggml_tensor * ssm_conv = cgraph->nodes[కణుపు_idx];
        స్థిరరాశి ggml_tensor * silu     = cgraph->nodes[కణుపు_idx + 1];

        ఒకవేళ (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        ఒకవేళ (!ggml_is_contiguous(silu)) {
            తిరిగి అబద్ధం;
        }

        తిరిగి నిజం;
    }

    // ADD(bias) + UNARY + MUL(scale): the delta-net alpha gate, softplus(alpha + dt) * a.
    // The broadcast is what stops the same-shape UNARY + MUL branch above firing past one token.
    ఒకవేళ (ops.size() == 3 && ops.begin()[0] == GGML_OP_కూడు && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1) {
        స్థిరరాశి ggml_tensor * కూడు_2   = cgraph->nodes[కణుపు_idx];
        స్థిరరాశి ggml_tensor * unary = cgraph->nodes[కణుపు_idx + 1];
        స్థిరరాశి ggml_tensor * mul   = cgraph->nodes[కణుపు_idx + 2];

        స్థిరరాశి ggml_unary_op unary_op = ggml_get_unary_op(unary);
        ఒకవేళ (unary_op != unary_ops.begin()[0]) {
            తిరిగి అబద్ధం;
        }

        ఒకవేళ (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            తిరిగి అబద్ధం;
        }

        // ggml_can_fuse() has already pinned the chain: unary consumes add, mul consumes
        // unary, add and unary have one use each, and all three have the same shape
        స్థిరరాశి ggml_tensor * a     = కూడు_2->src[0];
        స్థిరరాశి ggml_tensor * bias  = కూడు_2->src[1];
        స్థిరరాశి ggml_tensor * scale = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];

        ఒకవేళ (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }

        // the activation and the destination are indexed flat
        ఒకవేళ (!ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(a, mul)) {
            తిరిగి అబద్ధం;
        }

        // the kernel reads the bias and the scale as v[col], so each must be a single
        // contiguous row spanning ne0
        ఒకవేళ (bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] ||
            ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
            తిరిగి అబద్ధం;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        ఒకవేళ (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            తిరిగి అబద్ధం;
        }

        తిరిగి నిజం;
    }

    ఒకవేళ (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_కూడు &&
        ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        స్థిరరాశి ggml_tensor * ssm_conv = cgraph->nodes[కణుపు_idx];
        స్థిరరాశి ggml_tensor * కూడు_2      = cgraph->nodes[కణుపు_idx + 1];
        స్థిరరాశి ggml_tensor * silu     = cgraph->nodes[కణుపు_idx + 2];

        ఒకవేళ (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (ssm_conv->type != GGML_TYPE_F32 || కూడు_2->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        ఒకవేళ (!ggml_is_contiguous(silu)) {
            తిరిగి అబద్ధం;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias
        స్థిరరాశి ggml_tensor * bias = (కూడు_2->src[0] == ssm_conv) ? కూడు_2->src[1] : కూడు_2->src[0];
        ఒకవేళ (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            తిరిగి అబద్ధం;
        }

        తిరిగి నిజం;
    }

    ఒకవేళ (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        స్థిరరాశి ggml_tensor * rms_norm = cgraph->nodes[కణుపు_idx];
        స్థిరరాశి ggml_tensor * scale    = cgraph->nodes[కణుపు_idx + 1];
        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        ఒకవేళ (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }
        // the fused kernel reads/writes rows flat like the unfused pair
        ఒకవేళ (!ggml_is_contiguous_అడ్డువరుసలు(rms_norm) || !ggml_is_contiguous_అడ్డువరుసలు(scale)) {
            తిరిగి అబద్ధం;
        }
        తిరిగి నిజం;
    }
    తిరిగి అబద్ధం;
}
