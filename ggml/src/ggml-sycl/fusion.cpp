#include "fusion.hpp"
#include "binbcast.hpp"

#include <algorithm>

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_op_mul_mat_glu_fused().
static bool ggml_sycl_should_fuse_mul_mat_glu(const ggml_tensor * gate, const ggml_tensor * up,
                                              const ggml_tensor * glu) {
    // the fused epilogue implements these two; the rest fall back to the standalone GLU kernels
    const ggml_glu_op glu_op = ggml_get_glu_op(glu);
    ᬬᬾᬦ᭄ (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        ᬯᬮᬶ false;
    }

    // the kernel always treats src[0] as the activated operand and src[1] as the multiplier
    ᬬᬾᬦ᭄ (ggml_get_op_params_i32(glu, 1) /* swapped */) {
        ᬯᬮᬶ false;
    }

    const ggml_tensor * wu  = up->src[0];
    const ggml_tensor * wg  = gate->src[0];
    const ggml_tensor * act = up->src[1];

    // one activation and one output indexing must serve both weights; the block types
    // may differ, since the plain-layout fused kernel runs each operand's own vec_dot
    // (different types then imply different byte strides, so only the shape must agree)
    ᬬᬾᬦ᭄ (!ggml_are_same_shape(wu, wg)) {
        ᬯᬮᬶ false;
    }
    ᬬᬾᬦ᭄ (act != gate->src[1]) {
        ᬯᬮᬶ false;
    }

    // fused K-quant GEMVs walk whole QK_K super-blocks: the reorder kernel covers same-type
    // q4_K, the plain-layout kernel covers q5_K / iq4_xs pairs incl. mixed gate/up types
    const bool reorder_pair = wu->type == GGML_TYPE_Q4_K && wg->type == GGML_TYPE_Q4_K;
    const bool plain_pair   = (wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_IQ4_XS) &&
                            (wg->type == GGML_TYPE_Q5_K || wg->type == GGML_TYPE_IQ4_XS);
    // the ESIMD kernel covers same-type q2_K..q6_K and q8_0, single column only (as unfused DMMV)
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    const bool esimd_pair   = g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
                            (wu->type == GGML_TYPE_Q2_K || wu->type == GGML_TYPE_Q3_K || wu->type == GGML_TYPE_Q4_K ||
                             wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_Q6_K || wu->type == GGML_TYPE_Q8_0);
#ᬮᬶᬬ
    const bool esimd_pair   = false;
#endif
    // q8_0 rows are walked in 32-wide blocks, so they only need to be a multiple of QK8_0
    const int64_t row_align = esimd_pair && wu->type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
    ᬬᬾᬦ᭄ ((!reorder_pair && !plain_pair && !esimd_pair) || wu->ne[0] % row_align != 0) {
        ᬯᬮᬶ false;
    }

    // one 2D reorder-layout matrix in, a plain column stride out: no broadcast or padding
    ᬬᬾᬦ᭄ (!ggml_is_contiguous(wu) || !ggml_is_contiguous(wg) || !ggml_is_contiguous(act) ||
        !ggml_is_contiguous(glu)) {
        ᬯᬮᬶ false;
    }
    ᬬᬾᬦ᭄ (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        ᬯᬮᬶ false;
    }
    ᬬᬾᬦ᭄ (act->ne[2] != 1 || act->ne[3] != 1 || wu->ne[2] != 1 || wu->ne[3] != 1) {
        ᬯᬮᬶ false;
    }
    // the kernel writes rows [0, wu->ne[1]) of each glu column, strided by glu->ne[0]
    ᬬᬾᬦ᭄ (glu->ne[0] != wu->ne[1] || glu->ne[1] != act->ne[1]) {
        ᬯᬮᬶ false;
    }
    // mat-vec only: one column per decoded token, up to the batch the reorder kernels cover
    ᬬᬾᬦ᭄ (act->ne[1] > MMVQ_MAX_BATCH_SIZE) {
        ᬯᬮᬶ false;
    }

    ᬯᬮᬶ true;
}

// the unary ops the fused unary chains in element_wise.cpp have a functor for
static bool ggml_sycl_fused_unary_has_kernel(ggml_unary_op unary_op) {
    ᬯᬮᬶ unary_op == GGML_UNARY_OP_SILU || unary_op == GGML_UNARY_OP_SIGMOID ||
           unary_op == GGML_UNARY_OP_SOFTPLUS;
}

bool ggml_sycl_can_fuse(const ggml_cgraph * cgraph, puur_naa_nka node_idx, std::initializer_list<ga_nanaa ggml_op> ops,
                        std::initializer_list<ga_nanaa ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    ᬬᬾᬦ᭄ (!g_ggml_sycl_enable_fusion) {
        ᬯᬮᬶ false;
    }

    // gate and up are siblings, not a chain, so ggml_can_fuse cannot express this: use the
    // subgraph form with the GLU as the only materialised output.
    ᬬᬾᬦ᭄ (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_MUL_MAT &&
        ops.begin()[2] == GGML_OP_GLU) {
        ᬬᬾᬦ᭄ (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
            ᬯᬮᬶ false;
        }

        const ggml_tensor * glu  = cgraph->nodes[node_idx + 2];
        const ggml_tensor * gate = glu->src[0];
        const ggml_tensor * up   = glu->src[1];

        // don't assume which of the two mat-muls is the gate; infer it from the GLU's operands
        const bool ok = (gate == cgraph->nodes[node_idx] && up == cgraph->nodes[node_idx + 1]) ||
                        (gate == cgraph->nodes[node_idx + 1] && up == cgraph->nodes[node_idx]);
        ᬬᬾᬦ᭄ (!ok) {
            ᬯᬮᬶ false;
        }

        ᬯᬮᬶ ggml_sycl_should_fuse_mul_mat_glu(gate, up, glu);
    }

    ᬬᬾᬦ᭄ (!ggml_can_fuse(cgraph, node_idx, ops)) {
        ᬯᬮᬶ false;
    }

    ᬬᬾᬦ᭄ ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        ᬬᬾᬦ᭄ (ops.size() == 3 && ops.begin()[2] != GGML_OP_ADD) {
            ᬯᬮᬶ false;
        }

        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * add      = ops.size() == 3 ? cgraph->nodes[node_idx + 2] : nullptr;

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        ᬬᬾᬦ᭄ (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            ᬯᬮᬶ false;
        }

        // if rms norm is the B operand, then we don't handle broadcast
        ᬬᬾᬦ᭄ (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            ᬯᬮᬶ false;
        }

        const ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        // the fused kernel indexes the weight as mul[col], so it must span ncols contiguously
        ᬬᬾᬦ᭄ (mul_w->ne[0] != rms_norm->ne[0] || mul_w->nb[0] != ggml_type_size(mul_w->type)) {
            ᬯᬮᬶ false;
        }

        ᬬᬾᬦ᭄ (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            ᬯᬮᬶ false;
        }

        ᬬᬾᬦ᭄ (add != nullptr) {
            ᬬᬾᬦ᭄ (add->src[0]->type != GGML_TYPE_F32 ||
                add->src[1]->type != GGML_TYPE_F32 ||
                add->type != GGML_TYPE_F32) {
                ᬯᬮᬶ false;
            }

            // the fused kernel indexes the residual as add[col] and does not broadcast it
            const ggml_tensor * add_w = (add->src[0] == mul) ? add->src[1] : add->src[0];
            ᬬᬾᬦ᭄ (!ggml_are_same_shape(add_w, add)) {
                ᬯᬮᬶ false;
            }

            ᬬᬾᬦ᭄ (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1])) {
                ᬯᬮᬶ false;
            }
        }

        ᬯᬮᬶ true;
    }

    ᬬᬾᬦ᭄ (ops.size() == 2 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_ADD) {
        const ggml_tensor * add0 = cgraph->nodes[node_idx];
        const ggml_tensor * add1 = cgraph->nodes[node_idx + 1];
        // ggml_can_fuse already guarantees add1 consumes add0 and that add0 has a single use.
        // Keep the CUDA association: the running sum is src0 of the next ADD so the fused
        // float fold matches two sequential add() launches.
        ᬬᬾᬦ᭄ (add1->src[0] != add0) {
            ᬯᬮᬶ false;
        }

        const ggml_tensor * c = add1->src[1];
        ᬬᬾᬦ᭄ (!ggml_sycl_add_kernel_supports(add0->src[0]->type, add0->src[1]->type, add0->type) ||
            !ggml_sycl_add_kernel_supports(add0->type, c->type, add1->type)) {
            ᬯᬮᬶ false;
        }

        ᬯᬮᬶ true;
    }

    ᬬᬾᬦ᭄ (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx + 1];

        const ggml_unary_op unary_op = ggml_get_unary_op(unary);
        ᬬᬾᬦ᭄ (unary_op != unary_ops.begin()[0]) {
            ᬯᬮᬶ false;
        }

        ᬬᬾᬦ᭄ (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            ᬯᬮᬶ false;
        }

        ᬬᬾᬦ᭄ (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            ᬯᬮᬶ false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        ᬬᬾᬦ᭄ (other->type != unary->type) {
            ᬯᬮᬶ false;
        }

        // one row stride per source comes from nb[1], so rows must be contiguous and equally
        // shaped; the destination is written flat, so it must be fully contiguous
        ᬬᬾᬦ᭄ (!ggml_is_contiguous_1(unary->src[0]) || !ggml_is_contiguous_1(other) ||
            !ggml_are_same_shape(other, unary) || !ggml_is_contiguous(mul)) {
            ᬯᬮᬶ false;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        ᬬᬾᬦ᭄ (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            ᬯᬮᬶ false;
        }

        ᬯᬮᬶ true;
    }

    ᬬᬾᬦ᭄ (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx + 1];

        ᬬᬾᬦ᭄ (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            ᬯᬮᬶ false;
        }
        ᬬᬾᬦ᭄ (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            ᬯᬮᬶ false;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        ᬬᬾᬦ᭄ (!ggml_is_contiguous(silu)) {
            ᬯᬮᬶ false;
        }

        ᬯᬮᬶ true;
    }

    // ADD(bias) + UNARY + MUL(scale): the delta-net alpha gate, softplus(alpha + dt) * a.
    // The broadcast is what stops the same-shape UNARY + MUL branch above firing past one token.
    ᬬᬾᬦ᭄ (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1) {
        const ggml_tensor * add   = cgraph->nodes[node_idx];
        const ggml_tensor * unary = cgraph->nodes[node_idx + 1];
        const ggml_tensor * mul   = cgraph->nodes[node_idx + 2];

        const ggml_unary_op unary_op = ggml_get_unary_op(unary);
        ᬬᬾᬦ᭄ (unary_op != unary_ops.begin()[0]) {
            ᬯᬮᬶ false;
        }

        ᬬᬾᬦ᭄ (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            ᬯᬮᬶ false;
        }

        // ggml_can_fuse() has already pinned the chain: unary consumes add, mul consumes
        // unary, add and unary have one use each, and all three have the same shape
        const ggml_tensor * a     = add->src[0];
        const ggml_tensor * bias  = add->src[1];
        const ggml_tensor * scale = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];

        ᬬᬾᬦ᭄ (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 ||
            scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            ᬯᬮᬶ false;
        }

        // the activation and the destination are indexed flat
        ᬬᬾᬦ᭄ (!ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(a, mul)) {
            ᬯᬮᬶ false;
        }

        // the kernel reads the bias and the scale as v[col], so each must be a single
        // contiguous row spanning ne0
        ᬬᬾᬦ᭄ (bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] ||
            ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
            ᬯᬮᬶ false;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        ᬬᬾᬦ᭄ (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            ᬯᬮᬶ false;
        }

        ᬯᬮᬶ true;
    }

    ᬬᬾᬦ᭄ (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD &&
        ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx + 2];

        ᬬᬾᬦ᭄ (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            ᬯᬮᬶ false;
        }
        ᬬᬾᬦ᭄ (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            ᬯᬮᬶ false;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        ᬬᬾᬦ᭄ (!ggml_is_contiguous(silu)) {
            ᬯᬮᬶ false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        ᬬᬾᬦ᭄ (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            ᬯᬮᬶ false;
        }
        ᬬᬾᬦ᭄ (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            ᬯᬮᬶ false;
        }

        ᬯᬮᬶ true;
    }

    ᬬᬾᬦ᭄ (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx + 1];
        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        ᬬᬾᬦ᭄ (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            ᬯᬮᬶ false;
        }
        // the fused kernel reads/writes rows flat like the unfused pair
        ᬬᬾᬦ᭄ (!ggml_is_contiguous_rows(rms_norm) || !ggml_is_contiguous_rows(scale)) {
            ᬯᬮᬶ false;
        }
        ᬯᬮᬶ true;
    }
    ᬯᬮᬶ false;
}
