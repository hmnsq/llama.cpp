#شامل "fusion.hpp"
#شامل "binbcast.hpp"

#شامل <algorithm>

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_op_mul_mat_glu_fused().
ایستا منطقی ggml_sycl_should_fuse_mul_mat_glu(ثابت ggml_tensor * gate, ثابت ggml_tensor * up,
                                              ثابت ggml_tensor * glu) {
    // the fused epilogue implements these two; the rest fall back to the standalone GLU kernels
    ثابت ggml_glu_op glu_op = ggml_get_glu_op(glu);
    اگر (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        بازگشت نادرست;
    }

    // the kernel always treats src[0] as the activated operand and src[1] as the multiplier
    اگر (ggml_get_op_params_i32(glu, 1) /* swapped */) {
        بازگشت نادرست;
    }

    ثابت ggml_tensor * wu  = up->src[0];
    ثابت ggml_tensor * wg  = gate->src[0];
    ثابت ggml_tensor * act = up->src[1];

    // one activation and one output indexing must serve both weights; the block types
    // may differ, since the plain-layout fused kernel runs each operand's own vec_dot
    // (different types then imply different byte strides, so only the shape must agree)
    اگر (!ggml_are_same_shape(wu, wg)) {
        بازگشت نادرست;
    }
    اگر (act != gate->src[1]) {
        بازگشت نادرست;
    }

    // fused K-quant GEMVs walk whole QK_K super-blocks: the reorder kernel covers same-type
    // q4_K, the plain-layout kernel covers q5_K / iq4_xs pairs incl. mixed gate/up types
    ثابت منطقی reorder_pair = wu->type == GGML_TYPE_Q4_K && wg->type == GGML_TYPE_Q4_K;
    ثابت منطقی plain_pair   = (wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_IQ4_XS) &&
                            (wg->type == GGML_TYPE_Q5_K || wg->type == GGML_TYPE_IQ4_XS);
    // the ESIMD kernel covers same-type q2_K..q6_K and q8_0, single column only (as unfused DMMV)
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    ثابت منطقی esimd_pair   = g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
                            (wu->type == GGML_TYPE_Q2_K || wu->type == GGML_TYPE_Q3_K || wu->type == GGML_TYPE_Q4_K ||
                             wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_Q6_K || wu->type == GGML_TYPE_Q8_0);
#وگرنه
    ثابت منطقی esimd_pair   = نادرست;
#endif
    // q8_0 rows are walked in 32-wide blocks, so they only need to be a multiple of QK8_0
    ثابت int64_t row_align = esimd_pair && wu->type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
    اگر ((!reorder_pair && !plain_pair && !esimd_pair) || wu->ne[0] % row_align != 0) {
        بازگشت نادرست;
    }

    // one 2D reorder-layout matrix in, a plain column stride out: no broadcast or padding
    اگر (!ggml_is_contiguous(wu) || !ggml_is_contiguous(wg) || !ggml_is_contiguous(act) ||
        !ggml_is_contiguous(glu)) {
        بازگشت نادرست;
    }
    اگر (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        بازگشت نادرست;
    }
    اگر (act->ne[2] != 1 || act->ne[3] != 1 || wu->ne[2] != 1 || wu->ne[3] != 1) {
        بازگشت نادرست;
    }
    // the kernel writes rows [0, wu->ne[1]) of each glu column, strided by glu->ne[0]
    اگر (glu->ne[0] != wu->ne[1] || glu->ne[1] != act->ne[1]) {
        بازگشت نادرست;
    }
    // mat-vec only: one column per decoded token, up to the batch the reorder kernels cover
    اگر (act->ne[1] > MMVQ_MAX_BATCH_SIZE) {
        بازگشت نادرست;
    }

    بازگشت درست;
}

// the unary ops the fused unary chains in element_wise.cpp have a functor for
ایستا منطقی ggml_sycl_fused_unary_has_kernel(ggml_unary_op unary_op) {
    بازگشت unary_op == GGML_UNARY_OP_SILU || unary_op == GGML_UNARY_OP_SIGMOID ||
           unary_op == GGML_UNARY_OP_SOFTPLUS;
}

منطقی ggml_sycl_can_fuse(ثابت ggml_cgraph * cgraph, صحیح گره_idx, std::initializer_list<شمارش ggml_op> ops,
                        std::initializer_list<شمارش ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    ثابت size_t رقم_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == رقم_unary);
#endif

    اگر (!g_ggml_sycl_enable_fusion) {
        بازگشت نادرست;
    }

    // gate and up are siblings, not a chain, so ggml_can_fuse cannot express this: use the
    // subgraph form with the GLU as the only materialised output.
    اگر (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_MUL_MAT &&
        ops.begin()[2] == GGML_OP_GLU) {
        اگر (!ggml_can_fuse_subgraph(cgraph, گره_idx, ops, { گره_idx + 2 })) {
            بازگشت نادرست;
        }

        ثابت ggml_tensor * glu  = cgraph->nodes[گره_idx + 2];
        ثابت ggml_tensor * gate = glu->src[0];
        ثابت ggml_tensor * up   = glu->src[1];

        // don't assume which of the two mat-muls is the gate; infer it from the GLU's operands
        ثابت منطقی ok = (gate == cgraph->nodes[گره_idx] && up == cgraph->nodes[گره_idx + 1]) ||
                        (gate == cgraph->nodes[گره_idx + 1] && up == cgraph->nodes[گره_idx]);
        اگر (!ok) {
            بازگشت نادرست;
        }

        بازگشت ggml_sycl_should_fuse_mul_mat_glu(gate, up, glu);
    }

    اگر (!ggml_can_fuse(cgraph, گره_idx, ops)) {
        بازگشت نادرست;
    }

    اگر ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        اگر (ops.size() == 3 && ops.begin()[2] != GGML_OP_ADD) {
            بازگشت نادرست;
        }

        ثابت ggml_tensor * rms_norm = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * mul      = cgraph->nodes[گره_idx + 1];
        ثابت ggml_tensor * add      = ops.size() == 3 ? cgraph->nodes[گره_idx + 2] : nullptr;

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        اگر (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            بازگشت نادرست;
        }

        // if rms norm is the B operand, then we don't handle broadcast
        اگر (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            بازگشت نادرست;
        }

        ثابت ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        // the fused kernel indexes the weight as mul[col], so it must span ncols contiguously
        اگر (mul_w->ne[0] != rms_norm->ne[0] || mul_w->nb[0] != ggml_type_size(mul_w->type)) {
            بازگشت نادرست;
        }

        اگر (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            بازگشت نادرست;
        }

        اگر (add != nullptr) {
            اگر (add->src[0]->type != GGML_TYPE_F32 ||
                add->src[1]->type != GGML_TYPE_F32 ||
                add->type != GGML_TYPE_F32) {
                بازگشت نادرست;
            }

            // the fused kernel indexes the residual as add[col] and does not broadcast it
            ثابت ggml_tensor * add_w = (add->src[0] == mul) ? add->src[1] : add->src[0];
            اگر (!ggml_are_same_shape(add_w, add)) {
                بازگشت نادرست;
            }

            اگر (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1])) {
                بازگشت نادرست;
            }
        }

        بازگشت درست;
    }

    اگر (ops.size() == 2 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_ADD) {
        ثابت ggml_tensor * add0 = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * add1 = cgraph->nodes[گره_idx + 1];
        // ggml_can_fuse already guarantees add1 consumes add0 and that add0 has a single use.
        // Keep the CUDA association: the running sum is src0 of the next ADD so the fused
        // float fold matches two sequential add() launches.
        اگر (add1->src[0] != add0) {
            بازگشت نادرست;
        }

        ثابت ggml_tensor * c = add1->src[1];
        اگر (!ggml_sycl_add_kernel_supports(add0->src[0]->type, add0->src[1]->type, add0->type) ||
            !ggml_sycl_add_kernel_supports(add0->type, c->type, add1->type)) {
            بازگشت نادرست;
        }

        بازگشت درست;
    }

    اگر (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        ثابت ggml_tensor * unary = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * mul   = cgraph->nodes[گره_idx + 1];

        ثابت ggml_unary_op unary_op = ggml_get_unary_op(unary);
        اگر (unary_op != unary_ops.begin()[0]) {
            بازگشت نادرست;
        }

        اگر (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            بازگشت نادرست;
        }

        اگر (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            بازگشت نادرست;
        }

        ثابت ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        اگر (other->type != unary->type) {
            بازگشت نادرست;
        }

        // one row stride per source comes from nb[1], so rows must be contiguous and equally
        // shaped; the destination is written flat, so it must be fully contiguous
        اگر (!ggml_is_contiguous_1(unary->src[0]) || !ggml_is_contiguous_1(other) ||
            !ggml_are_same_shape(other, unary) || !ggml_is_contiguous(mul)) {
            بازگشت نادرست;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        اگر (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            بازگشت نادرست;
        }

        بازگشت درست;
    }

    اگر (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        ثابت ggml_tensor * ssm_conv = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * silu     = cgraph->nodes[گره_idx + 1];

        اگر (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            بازگشت نادرست;
        }
        اگر (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            بازگشت نادرست;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        اگر (!ggml_is_contiguous(silu)) {
            بازگشت نادرست;
        }

        بازگشت درست;
    }

    // ADD(bias) + UNARY + MUL(scale): the delta-net alpha gate, softplus(alpha + dt) * a.
    // The broadcast is what stops the same-shape UNARY + MUL branch above firing past one token.
    اگر (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1) {
        ثابت ggml_tensor * add   = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * unary = cgraph->nodes[گره_idx + 1];
        ثابت ggml_tensor * mul   = cgraph->nodes[گره_idx + 2];

        ثابت ggml_unary_op unary_op = ggml_get_unary_op(unary);
        اگر (unary_op != unary_ops.begin()[0]) {
            بازگشت نادرست;
        }

        اگر (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            بازگشت نادرست;
        }

        // ggml_can_fuse() has already pinned the chain: unary consumes add, mul consumes
        // unary, add and unary have one use each, and all three have the same shape
        ثابت ggml_tensor * a     = add->src[0];
        ثابت ggml_tensor * bias  = add->src[1];
        ثابت ggml_tensor * scale = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];

        اگر (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            بازگشت نادرست;
        }

        // the activation and the destination are indexed flat
        اگر (!ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(a, mul)) {
            بازگشت نادرست;
        }

        // the kernel reads the bias and the scale as v[col], so each must be a single
        // contiguous row spanning ne0
        اگر (bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] ||
            ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
            بازگشت نادرست;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        اگر (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            بازگشت نادرست;
        }

        بازگشت درست;
    }

    اگر (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD &&
        ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        ثابت ggml_tensor * ssm_conv = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * add      = cgraph->nodes[گره_idx + 1];
        ثابت ggml_tensor * silu     = cgraph->nodes[گره_idx + 2];

        اگر (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            بازگشت نادرست;
        }
        اگر (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            بازگشت نادرست;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        اگر (!ggml_is_contiguous(silu)) {
            بازگشت نادرست;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias
        ثابت ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        اگر (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            بازگشت نادرست;
        }
        اگر (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            بازگشت نادرست;
        }

        بازگشت درست;
    }

    اگر (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        ثابت ggml_tensor * rms_norm = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * scale    = cgraph->nodes[گره_idx + 1];
        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        اگر (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            بازگشت نادرست;
        }
        // the fused kernel reads/writes rows flat like the unfused pair
        اگر (!ggml_is_contiguous_rows(rms_norm) || !ggml_is_contiguous_rows(scale)) {
            بازگشت نادرست;
        }
        بازگشت درست;
    }
    بازگشت نادرست;
}
