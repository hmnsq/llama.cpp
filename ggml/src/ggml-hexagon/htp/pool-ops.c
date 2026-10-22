#pragma clang diagnostic ignored "-Wunused-variable"

#अन्तर्भाव <float.h>
#अन्तर्भाव <HAP_farf.h>

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "hvx-inverse.h"
#अन्तर्भाव "hvx-types.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "pool-ops.h"

#define HTP_POOL_MAX 0
#define HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
स्थिर शून्य pool_plane_hvx(
    ध्रुव भिन्न * src, भिन्न * dst, ध्रुव संरचना htp_pool_2d_kernel_params * p) {
    ध्रुव bool is_max = (p->pool_op == HTP_POOL_MAX);
    ध्रुव HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ध्रुव HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    अत्थाय (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ध्रुव uint32_t sy = oy * p->kernel_y;
        अत्थाय (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            ध्रुव uint32_t rem = p->dst_x - ox;
            ध्रुव uint32_t nbytes = (rem < VLEN_FP32) ? (rem * परिमाण(भिन्न)) : VLEN;
            HVX_Vector acc = seed;
            अत्थाय (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                ध्रुव भिन्न * row = src + (sy + ky) * p->src_x;
                सचे (p->kernel_x == 2) {
                    ध्रुव HVX_Vector v0 = *(ध्रुव HVX_UVector *) (row + ox * 2);
                    ध्रुव HVX_Vector v1 = *(ध्रुव HVX_UVector *) (row + ox * 2 + VLEN_FP32);
                    ध्रुव HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ध्रुव HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    ध्रुव HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_add_f32_f32(acc, hvx_vec_add_f32_f32(lo, hi));
                } अञ्ञथा सचे (p->kernel_x == 1) {
                    ध्रुव HVX_Vector v = *(ध्रुव HVX_UVector *) (row + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
स्थिर शून्य pool_plane_hvx_narrow(
    ध्रुव भिन्न * src, भिन्न * dst, ध्रुव संरचना htp_pool_2d_kernel_params * p) {
    ध्रुव bool is_max = (p->pool_op == HTP_POOL_MAX);
    ध्रुव HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ध्रुव HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    अत्थाय (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ध्रुव uint32_t sy = oy * p->kernel_y;
        HVX_Vector acc = seed;
        अत्थाय (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            ध्रुव भिन्न * row = src + (sy + ky) * p->src_x;
            ध्रुव HVX_Vector v0 = *(ध्रुव HVX_UVector *) row;
            HVX_Vector v = v0;
            सचे (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                ध्रुव HVX_Vector zero = Q6_V_vsplat_R(0);
                ध्रुव HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                ध्रुव HVX_Vector even = Q6_V_lo_W(deinterleaved);
                ध्रुव HVX_Vector odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_add_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_add_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * परिमाण(भिन्न),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

स्थिर शून्य pool_plane_global(
    ध्रुव भिन्न * src, भिन्न * dst, ध्रुव संरचना htp_pool_2d_kernel_params * p) {
    ध्रुव uint32_t n = p->src_x * p->src_y;
    भिन्न val;
    सचे (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((ध्रुव uint8_t *) src, n);
    } अञ्ञथा {
        val = hvx_reduce_sum_f32((ध्रुव uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, परिमाण(भिन्न), hvx_vec_splat_f32(val));
}

स्थिर शून्य pool_plane_block(
    ध्रुव भिन्न * src, भिन्न * dst, ध्रुव संरचना htp_pool_2d_kernel_params * p) {
    अत्थाय (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        ध्रुव भिन्न * row = src + oy * p->src_x;
        भिन्न * dst_row = dst + oy * p->dst_x;
        अत्थाय (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            ध्रुव uint8_t * block = (ध्रुव uint8_t *) (row + ox * p->kernel_x);
            भिन्न val;
            सचे (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(block, p->kernel_x);
            } अञ्ञथा {
                val = hvx_reduce_sum_f32(block, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_row + ox, परिमाण(भिन्न), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
स्थिर अन्तःस्थ शून्य pool_row_bounds_y(
    ध्रुव संरचना htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    ध्रुव int32_t lo = -(*iy0);
    ध्रुव int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

स्थिर अन्तःस्थ शून्य pool_pixel_boundary_vec(
    ध्रुव भिन्न * src, भिन्न * dst_row, ध्रुव संरचना htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    ध्रुव int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    ध्रुव int32_t kx_lo = MAX(0, -ix0);
    ध्रुव int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    सचे (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_Vector empty_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_row + ox, परिमाण(भिन्न), empty_val);
        पटिनिवत्त;
    }

    ध्रुव uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    ध्रुव HVX_Vector mask_identity = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_Vector acc = mask_identity;
    अत्थाय (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        ध्रुव भिन्न * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        अत्थाय (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            ध्रुव uint32_t k_rem = valid_kx - k;
            ध्रुव uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            ध्रुव HVX_VectorPred q = Q6_Q_vsetq_R(n * परिमाण(भिन्न));
            ध्रुव HVX_Vector raw = *(ध्रुव HVX_UVector *) (row + ix0 + kx_lo + k);
            ध्रुव HVX_Vector v = Q6_V_vmux_QVV(q, raw, mask_identity);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
        }
    }

    HVX_Vector reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_sum_f32(acc);
    सचे (!is_max) {
        HVX_Vector scale_vec;
        सचे (p->avg_divide_गणनम्) {
            ध्रुव uint32_t गणनम् = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((भिन्न) गणनम्));
        } अञ्ञथा {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_row + ox, परिमाण(भिन्न), reduced);
}

स्थिर अन्तःस्थ शून्य pool_row_general_boundary_vec(
    ध्रुव भिन्न * src, भिन्न * dst_row, ध्रुव संरचना htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    अत्थाय (uint32_t ox = ox_start; ox < ox_end; ++ox) {
        pool_pixel_boundary_vec(src, dst_row, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
स्थिर अन्तःस्थ शून्य pool_row_general_vec(
    ध्रुव भिन्न * src, भिन्न * dst_row, ध्रुव संरचना htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    सचे (ox_start >= ox_end) {
        पटिनिवत्त;
    }

    सचे (p->stride_x != 1 && p->stride_x != 2) {
        pool_row_general_boundary_vec(src, dst_row, p, ox_start, ox_end, iy0, ky_lo, ky_hi, is_max);
        पटिनिवत्त;
    }

    ध्रुव HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    ध्रुव HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    अत्थाय (uint32_t ox = ox_start; ox < ox_end; ox += VLEN_FP32) {
        ध्रुव uint32_t rem = ox_end - ox;
        ध्रुव uint32_t nbytes = (rem < VLEN_FP32) ? (rem * परिमाण(भिन्न)) : VLEN;
        ध्रुव int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_Vector acc = seed;

        अत्थाय (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            ध्रुव भिन्न * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            सचे (p->stride_x == 1) {
                अत्थाय (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    ध्रुव HVX_Vector v = *(ध्रुव HVX_UVector *) (row + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            } अञ्ञथा {
                अत्थाय (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    ध्रुव HVX_Vector v0 = *(ध्रुव HVX_UVector *) (row + ix0 + (int32_t) kx);
                    ध्रुव HVX_Vector v1 = *(ध्रुव HVX_UVector *) (row + ix0 + (int32_t) kx + VLEN_FP32);
                    ध्रुव HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    ध्रुव HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_add_f32_f32(acc, lo);
                    सचे (kx + 1 < p->kernel_x) {
                        ध्रुव HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_add_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_row + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

स्थिर शून्य pool_plane_general(
    ध्रुव भिन्न * src, भिन्न * dst, ध्रुव संरचना htp_pool_2d_kernel_params * p) {
    ध्रुव bool is_max = (p->pool_op == HTP_POOL_MAX);

    अत्थाय (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_row_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        भिन्न * dst_row = dst + oy * p->dst_x;

        pool_row_general_boundary_vec(src, dst_row, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_vec(src, dst_row, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_boundary_vec(src, dst_row, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

प्रकारनिर्देश शून्य (*pool_plane_fn_t)(ध्रुव भिन्न * src, भिन्न * dst, ध्रुव संरचना htp_pool_2d_kernel_params * p);

संरचना pool_2d_context {
    संरचना htp_ops_context * octx;
    ध्रुव संरचना htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_threads;
    uint32_t plane_start;
    uint32_t plane_गणनम्;
    uint32_t planes_per_thread;
};

स्थिर शून्य pool_2d_thread(unsigned पूर्णाङ्क nth, unsigned पूर्णाङ्क ith, शून्य * data) {
    संरचना pool_2d_context * ctx = (संरचना pool_2d_context *) data;
    ध्रुव संरचना htp_pool_2d_kernel_params * p = ctx->kparams;
    ध्रुव संरचना htp_tensor * src0 = ctx->octx->src[0];
    ध्रुव संरचना htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    ध्रुव uint32_t planes_per_thread = ctx->planes_per_thread;
    ध्रुव uint32_t first = ctx->plane_start + ith * planes_per_thread;
    ध्रुव uint32_t last = MIN(first + planes_per_thread, ctx->plane_start + ctx->plane_गणनम्);

    सचे (first >= last) {
        पटिनिवत्त;
    }

    संरचना htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_queue * dma_queue = ctx->octx->ctx->dma[ith];

    ध्रुव uint32_t src_spad_half = p->src_plane_bytes_aligned;
    ध्रुव uint32_t dst_spad_half = p->dst_plane_bytes_aligned;
    ध्रुव uint32_t src_bytes_per_thread = 2 * src_spad_half;
    ध्रुव uint32_t dst_bytes_per_thread = 2 * dst_spad_half;
    ध्रुव size_t off_dst = (size_t) ctx->n_threads * src_bytes_per_thread;

    uint8_t * vtcm_base = (uint8_t *) ctx->octx->ctx->vtcm_base;
    uint8_t * src_spad  = vtcm_base + ith * src_bytes_per_thread;
    uint8_t * dst_spad  = vtcm_base + off_dst + ith * dst_bytes_per_thread;

    भिन्न * srcb2[2] = { (भिन्न *) src_spad, (भिन्न *) (src_spad + src_spad_half) };
    भिन्न * dstb2[2] = { (भिन्न *) dst_spad, (भिन्न *) (dst_spad + dst_spad_half) };

    ध्रुव uint32_t total = last - first;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    अत्थाय (uint32_t i = 0; i < total && i < 2; ++i) {
        dma_queue_प्रक्षिप(dma_queue,
                       dma_make_data(dst->data, dstb2[i]),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 0);

        ध्रुव dma_addr_t src_addr = src0->data + (first + i) * p->src_plane_bytes;
        dma_queue_प्रक्षिप(dma_queue,
                       dma_make_data(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_bytes,
                       p->src_plane_bytes, 1);
    }

    अत्थाय (uint32_t i = 0; i < total; ++i) {
        ध्रुव uint32_t plane = first + i;
        ध्रुव uint32_t buf   = i & 1u;
        भिन्न * srcb = srcb2[buf];
        भिन्न * dstb = dstb2[buf];

        dma_queue_निष्कासय(dma_queue); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_queue_निष्कासय(dma_queue); // input for plane i

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        ध्रुव dma_addr_t dst_addr = dst->data + plane * p->dst_plane_bytes;
        dma_queue_प्रक्षिप(dma_queue,
                       dma_make_data(dst_addr, dstb),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 1);

        सचे (i + 2 < total) {
            ध्रुव dma_addr_t next_src_addr = src0->data + (plane + 2) * p->src_plane_bytes;
            dma_queue_प्रक्षिप(dma_queue,
                           dma_make_data(srcb, next_src_addr),
                           src_spad_half, p->src_plane_bytes,
                           p->src_plane_bytes, 1);
        }
    }

    dma_queue_flush(dma_queue);

    FARF(उच्चः_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], first, last);
    (शून्य) nth;
}

पूर्णाङ्क op_pool_2d(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst = octx->dst;
    ध्रुव संरचना htp_pool_2d_kernel_params * p =
        (ध्रुव संरचना htp_pool_2d_kernel_params *) octx->kernel_params;

    सचे (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        पटिनिवत्त HTP_STATUS_NO_SUPPORT;
    }

    uint32_t plane_start = 0;
    uint32_t plane_गणनम् = p->planes;
    सचे (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव uint32_t planes_per_chunk = (p->dst_plane_bytes > 0) ? (HEX_L2_पङ्क्तिः_SIZE / hex_gcd_u32(p->dst_plane_bytes, HEX_L2_पङ्क्तिः_SIZE)) : 1;
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            plane_गणनम्, htp_tensor_mdev_data_aligned(dst) ? planes_per_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        plane_start = range.start;
        plane_गणनम् = range.गणनम्;
    }
    सचे (plane_गणनम् == 0) {
        पटिनिवत्त HTP_STATUS_OK;
    }

    ध्रुव uint32_t n_threads = MIN(p->n_threads, plane_गणनम्);
    सचे (!htp_ops_context_set_n_threads(octx, n_threads)) {
        पटिनिवत्त HTP_STATUS_INVAL_PARAMS;
    }

    ध्रुव uint32_t planes_per_thread = fastdiv(plane_गणनम् + n_threads - 1, &octx->n_threads_div);

    pool_plane_fn_t pool_plane;
    सचे (p->global_path) {
        pool_plane = pool_plane_global;
    } अञ्ञथा सचे (p->block_path) {
        pool_plane = pool_plane_block;
    } अञ्ञथा सचे (p->narrow_path) {
        pool_plane = pool_plane_hvx_narrow;
    } अञ्ञथा सचे (p->fast_path) {
        pool_plane = pool_plane_hvx;
    } अञ्ञथा {
        pool_plane = pool_plane_general;
    }

    संरचना pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_threads = n_threads,
        .plane_start = plane_start,
        .plane_गणनम् = plane_गणनम्,
        .planes_per_thread = planes_per_thread,
    };
    work_queue_run(octx->ctx->work_queue, pool_2d_thread, &ctx, n_threads);
    पटिनिवत्त HTP_STATUS_OK;
}

पूर्णाङ्क op_pool_1d(संरचना htp_ops_context * octx) {
    पटिनिवत्त op_pool_2d(octx);
}
