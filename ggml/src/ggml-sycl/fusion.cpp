#ضمن "fusion.hpp"
#ضمن "binbcast.hpp"

#ضمن <algorithm>

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_op_mul_mat_glu_fused().
ساكن منطقي ggml_sycl_should_fuse_mul_mat_glu(ثابت ggml_tensor * gate, ثابت ggml_tensor * به_بالا,
                                              ثابت ggml_tensor * glu) {
    // the fused epilogue implements these two; the rest fall back to the standalone GLU kernels
    ثابت ggml_glu_op glu_op = ggml_get_glu_op(glu);
    إذا (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        أرجع خطأ;
    }

    // the kernel always treats src[0] as the activated operand and src[1] as the multiplier
    إذا (ggml_get_op_params_i32(glu, 1) /* swapped */) {
        أرجع خطأ;
    }

    ثابت ggml_tensor * wu  = به_بالا->src[0];
    ثابت ggml_tensor * wg  = gate->src[0];
    ثابت ggml_tensor * act = به_بالا->src[1];

    // one activation and one output indexing must serve both weights; the block types
    // may differ, since the plain-layout fused kernel runs each operand's own vec_dot
    // (different types then imply different byte strides, so only the shape must agree)
    إذا (!ggml_are_same_shape(wu, wg)) {
        أرجع خطأ;
    }
    إذا (act != gate->src[1]) {
        أرجع خطأ;
    }

    // fused K-quant GEMVs walk whole QK_K super-blocks: the reorder kernel covers same-type
    // q4_K, the plain-layout kernel covers q5_K / iq4_xs pairs incl. mixed gate/up types
    ثابت منطقي reorder_pair = wu->type == GGML_TYPE_Q4_K && wg->type == GGML_TYPE_Q4_K;
    ثابت منطقي plain_pair   = (wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_IQ4_XS) &&
                            (wg->type == GGML_TYPE_Q5_K || wg->type == GGML_TYPE_IQ4_XS);
    // the ESIMD kernel covers same-type q2_K..q6_K and q8_0, single column only (as unfused DMMV)
#اگر_تعریف_شده GGML_SYCL_DMMV_HAS_ESIMD
    ثابت منطقي esimd_pair   = g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
                            (wu->type == GGML_TYPE_Q2_K || wu->type == GGML_TYPE_Q3_K || wu->type == GGML_TYPE_Q4_K ||
                             wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_Q6_K || wu->type == GGML_TYPE_Q8_0);
#وإلا
    ثابت منطقي esimd_pair   = خطأ;
#پایان_اگر
    // q8_0 rows are walked in 32-wide blocks, so they only need to be a multiple of QK8_0
    ثابت int64_t ردیف_align = esimd_pair && wu->type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
    إذا ((!reorder_pair && !plain_pair && !esimd_pair) || wu->ne[0] % ردیف_align != 0) {
        أرجع خطأ;
    }

    // one 2D reorder-layout matrix in, a plain column stride out: no broadcast or padding
    إذا (!ggml_is_contiguous(wu) || !ggml_is_contiguous(wg) || !ggml_is_contiguous(act) ||
        !ggml_is_contiguous(glu)) {
        أرجع خطأ;
    }
    إذا (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        أرجع خطأ;
    }
    إذا (act->ne[2] != 1 || act->ne[3] != 1 || wu->ne[2] != 1 || wu->ne[3] != 1) {
        أرجع خطأ;
    }
    // the kernel writes rows [0, wu->ne[1]) of each glu column, strided by glu->ne[0]
    إذا (glu->ne[0] != wu->ne[1] || glu->ne[1] != act->ne[1]) {
        أرجع خطأ;
    }
    // mat-vec only: one column per decoded token, up to the batch the reorder kernels cover
    إذا (act->ne[1] > MMVQ_MAX_BATCH_اندازه) {
        أرجع خطأ;
    }

    أرجع درست;
}

// the unary ops the fused unary chains in element_wise.cpp have a functor for
ساكن منطقي ggml_sycl_fused_unary_has_kernel(ggml_unary_op unary_op) {
    أرجع unary_op == GGML_UNARY_OP_SILU || unary_op == GGML_UNARY_OP_SIGMOID ||
           unary_op == GGML_UNARY_OP_SOFTPLUS;
}

منطقي ggml_sycl_can_fuse(ثابت ggml_cgraph * cgraph, صحيح گره_idx, std::initializer_list<تعداد ggml_op> ops,
                        std::initializer_list<تعداد ggml_unary_op> unary_ops) {
#اگر_تعریف_نشده NDEBUG
    ثابت size_t رقم_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == رقم_unary);
#پایان_اگر

    إذا (!g_ggml_sycl_enable_fusion) {
        أرجع خطأ;
    }

    // gate and up are siblings, not a chain, so ggml_can_fuse cannot express this: use the
    // subgraph form with the GLU as the only materialised output.
    إذا (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_MUL_MAT &&
        ops.begin()[2] == GGML_OP_GLU) {
        إذا (!ggml_can_fuse_subgraph(cgraph, گره_idx, ops, { گره_idx + 2 })) {
            أرجع خطأ;
        }

        ثابت ggml_tensor * glu  = cgraph->nodes[گره_idx + 2];
        ثابت ggml_tensor * gate = glu->src[0];
        ثابت ggml_tensor * به_بالا   = glu->src[1];

        // don't assume which of the two mat-muls is the gate; infer it from the GLU's operands
        ثابت منطقي ok = (gate == cgraph->nodes[گره_idx] && به_بالا == cgraph->nodes[گره_idx + 1]) ||
                        (gate == cgraph->nodes[گره_idx + 1] && به_بالا == cgraph->nodes[گره_idx]);
        إذا (!ok) {
            أرجع خطأ;
        }

        أرجع ggml_sycl_should_fuse_mul_mat_glu(gate, به_بالا, glu);
    }

    إذا (!ggml_can_fuse(cgraph, گره_idx, ops)) {
        أرجع خطأ;
    }

    إذا ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        إذا (ops.size() == 3 && ops.begin()[2] != GGML_OP_جمع_کن) {
            أرجع خطأ;
        }

        ثابت ggml_tensor * rms_norm = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * mul      = cgraph->nodes[گره_idx + 1];
        ثابت ggml_tensor * جمع_کن_2      = ops.size() == 3 ? cgraph->nodes[گره_idx + 2] : اشاره_تهی;

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        إذا (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }

        // if rms norm is the B operand, then we don't handle broadcast
        إذا (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            أرجع خطأ;
        }

        ثابت ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        // the fused kernel indexes the weight as mul[col], so it must span ncols contiguously
        إذا (mul_w->ne[0] != rms_norm->ne[0] || mul_w->nb[0] != ggml_type_اندازه(mul_w->type)) {
            أرجع خطأ;
        }

        إذا (!ggml_is_contiguous_ردیفها(mul->src[0]) || !ggml_is_contiguous_ردیفها(mul->src[1])) {
            أرجع خطأ;
        }

        إذا (جمع_کن_2 != اشاره_تهی) {
            إذا (جمع_کن_2->src[0]->type != GGML_TYPE_F32 ||
                جمع_کن_2->src[1]->type != GGML_TYPE_F32 ||
                جمع_کن_2->type != GGML_TYPE_F32) {
                أرجع خطأ;
            }

            // the fused kernel indexes the residual as add[col] and does not broadcast it
            ثابت ggml_tensor * جمع_کن_w = (جمع_کن_2->src[0] == mul) ? جمع_کن_2->src[1] : جمع_کن_2->src[0];
            إذا (!ggml_are_same_shape(جمع_کن_w, جمع_کن_2)) {
                أرجع خطأ;
            }

            إذا (!ggml_is_contiguous(جمع_کن_2->src[0]) || !ggml_is_contiguous_ردیفها(جمع_کن_2->src[1])) {
                أرجع خطأ;
            }
        }

        أرجع درست;
    }

    إذا (ops.size() == 2 && ops.begin()[0] == GGML_OP_جمع_کن && ops.begin()[1] == GGML_OP_جمع_کن) {
        ثابت ggml_tensor * add0 = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * add1 = cgraph->nodes[گره_idx + 1];
        // ggml_can_fuse already guarantees add1 consumes add0 and that add0 has a single use.
        // Keep the CUDA association: the running sum is src0 of the next ADD so the fused
        // float fold matches two sequential add() launches.
        إذا (add1->src[0] != add0) {
            أرجع خطأ;
        }

        ثابت ggml_tensor * c = add1->src[1];
        إذا (!ggml_sycl_جمع_کن_kernel_supports(add0->src[0]->type, add0->src[1]->type, add0->type) ||
            !ggml_sycl_جمع_کن_kernel_supports(add0->type, c->type, add1->type)) {
            أرجع خطأ;
        }

        أرجع درست;
    }

    إذا (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        ثابت ggml_tensor * unary = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * mul   = cgraph->nodes[گره_idx + 1];

        ثابت ggml_unary_op unary_op = ggml_get_unary_op(unary);
        إذا (unary_op != unary_ops.begin()[0]) {
            أرجع خطأ;
        }

        إذا (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            أرجع خطأ;
        }

        إذا (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            أرجع خطأ;
        }

        ثابت ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        إذا (other->type != unary->type) {
            أرجع خطأ;
        }

        // one row stride per source comes from nb[1], so rows must be contiguous and equally
        // shaped; the destination is written flat, so it must be fully contiguous
        إذا (!ggml_is_contiguous_1(unary->src[0]) || !ggml_is_contiguous_1(other) ||
            !ggml_are_same_shape(other, unary) || !ggml_is_contiguous(mul)) {
            أرجع خطأ;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        إذا (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            أرجع خطأ;
        }

        أرجع درست;
    }

    إذا (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        ثابت ggml_tensor * ssm_conv = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * silu     = cgraph->nodes[گره_idx + 1];

        إذا (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            أرجع خطأ;
        }
        إذا (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        إذا (!ggml_is_contiguous(silu)) {
            أرجع خطأ;
        }

        أرجع درست;
    }

    // ADD(bias) + UNARY + MUL(scale): the delta-net alpha gate, softplus(alpha + dt) * a.
    // The broadcast is what stops the same-shape UNARY + MUL branch above firing past one token.
    إذا (ops.size() == 3 && ops.begin()[0] == GGML_OP_جمع_کن && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1) {
        ثابت ggml_tensor * جمع_کن_2   = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * unary = cgraph->nodes[گره_idx + 1];
        ثابت ggml_tensor * mul   = cgraph->nodes[گره_idx + 2];

        ثابت ggml_unary_op unary_op = ggml_get_unary_op(unary);
        إذا (unary_op != unary_ops.begin()[0]) {
            أرجع خطأ;
        }

        إذا (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            أرجع خطأ;
        }

        // ggml_can_fuse() has already pinned the chain: unary consumes add, mul consumes
        // unary, add and unary have one use each, and all three have the same shape
        ثابت ggml_tensor * a     = جمع_کن_2->src[0];
        ثابت ggml_tensor * bias  = جمع_کن_2->src[1];
        ثابت ggml_tensor * scale = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];

        إذا (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }

        // the activation and the destination are indexed flat
        إذا (!ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(a, mul)) {
            أرجع خطأ;
        }

        // the kernel reads the bias and the scale as v[col], so each must be a single
        // contiguous row spanning ne0
        إذا (bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] ||
            ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
            أرجع خطأ;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        إذا (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            أرجع خطأ;
        }

        أرجع درست;
    }

    إذا (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_جمع_کن &&
        ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        ثابت ggml_tensor * ssm_conv = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * جمع_کن_2      = cgraph->nodes[گره_idx + 1];
        ثابت ggml_tensor * silu     = cgraph->nodes[گره_idx + 2];

        إذا (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            أرجع خطأ;
        }
        إذا (ssm_conv->type != GGML_TYPE_F32 || جمع_کن_2->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        إذا (!ggml_is_contiguous(silu)) {
            أرجع خطأ;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias
        ثابت ggml_tensor * bias = (جمع_کن_2->src[0] == ssm_conv) ? جمع_کن_2->src[1] : جمع_کن_2->src[0];
        إذا (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            أرجع خطأ;
        }
        إذا (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            أرجع خطأ;
        }

        أرجع درست;
    }

    إذا (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        ثابت ggml_tensor * rms_norm = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * scale    = cgraph->nodes[گره_idx + 1];
        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        إذا (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }
        // the fused kernel reads/writes rows flat like the unfused pair
        إذا (!ggml_is_contiguous_ردیفها(rms_norm) || !ggml_is_contiguous_ردیفها(scale)) {
            أرجع خطأ;
        }
        أرجع درست;
    }
    أرجع خطأ;
}
