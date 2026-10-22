#आशय clang diagnostic ignored "-Wunused-variable"

#समावेश <float.h>
#समावेश <HAP_farf.h>

#समावेश "hex-common.h"
#समावेश "dma-queue.h"
#समावेश "hex-profile.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"
#समावेश "hvx-inverse.h"
#समावेश "hvx-types.h"
#समावेश "hvx-utils.h"
#समावेश "pool-ops.h"

#घोषणा HTP_POOL_MAX 0
#घोषणा HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
अटल व्योम pool_plane_hvx(
    स्थिर भग्न * src, भग्न * dst, स्थिर काष्ठा htp_pool_2d_kernel_params * p) {
    स्थिर bool is_max = (p->pool_op == HTP_POOL_MAX);
    स्थिर HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    स्थिर HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-भग्न_उच्च) : Q6_V_vsplat_R(0);

    क्रम (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        स्थिर uint32_t sy = oy * p->kernel_y;
        क्रम (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            स्थिर uint32_t rem = p->dst_x - ox;
            स्थिर uint32_t nbytes = (rem < VLEN_FP32) ? (rem * माप(भग्न)) : VLEN;
            HVX_Vector acc = seed;
            क्रम (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                स्थिर भग्न * row = src + (sy + ky) * p->src_x;
                अगर (p->kernel_x == 2) {
                    स्थिर HVX_Vector v0 = *(स्थिर HVX_UVector *) (row + ox * 2);
                    स्थिर HVX_Vector v1 = *(स्थिर HVX_UVector *) (row + ox * 2 + VLEN_FP32);
                    स्थिर HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    स्थिर HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    स्थिर HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_add_f32_f32(acc, hvx_vec_add_f32_f32(lo, hi));
                } अन्यथा अगर (p->kernel_x == 1) {
                    स्थिर HVX_Vector v = *(स्थिर HVX_UVector *) (row + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
अटल व्योम pool_plane_hvx_narrow(
    स्थिर भग्न * src, भग्न * dst, स्थिर काष्ठा htp_pool_2d_kernel_params * p) {
    स्थिर bool is_max = (p->pool_op == HTP_POOL_MAX);
    स्थिर HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    स्थिर HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-भग्न_उच्च) : Q6_V_vsplat_R(0);

    क्रम (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        स्थिर uint32_t sy = oy * p->kernel_y;
        HVX_Vector acc = seed;
        क्रम (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            स्थिर भग्न * row = src + (sy + ky) * p->src_x;
            स्थिर HVX_Vector v0 = *(स्थिर HVX_UVector *) row;
            HVX_Vector v = v0;
            अगर (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                स्थिर HVX_Vector zero = Q6_V_vsplat_R(0);
                स्थिर HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                स्थिर HVX_Vector even = Q6_V_lo_W(deinterleaved);
                स्थिर HVX_Vector odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_add_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_add_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * माप(भग्न),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

अटल व्योम pool_plane_global(
    स्थिर भग्न * src, भग्न * dst, स्थिर काष्ठा htp_pool_2d_kernel_params * p) {
    स्थिर uint32_t n = p->src_x * p->src_y;
    भग्न val;
    अगर (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((स्थिर uint8_t *) src, n);
    } अन्यथा {
        val = hvx_reduce_योग_f32((स्थिर uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, माप(भग्न), hvx_vec_splat_f32(val));
}

अटल व्योम pool_plane_block(
    स्थिर भग्न * src, भग्न * dst, स्थिर काष्ठा htp_pool_2d_kernel_params * p) {
    क्रम (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        स्थिर भग्न * row = src + oy * p->src_x;
        भग्न * dst_row = dst + oy * p->dst_x;
        क्रम (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            स्थिर uint8_t * block = (स्थिर uint8_t *) (row + ox * p->kernel_x);
            भग्न val;
            अगर (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(block, p->kernel_x);
            } अन्यथा {
                val = hvx_reduce_योग_f32(block, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_row + ox, माप(भग्न), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
अटल अंतरभूत व्योम pool_row_bounds_y(
    स्थिर काष्ठा htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    स्थिर int32_t lo = -(*iy0);
    स्थिर int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

अटल अंतरभूत व्योम pool_pixel_boundary_vec(
    स्थिर भग्न * src, भग्न * dst_row, स्थिर काष्ठा htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    स्थिर int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    स्थिर int32_t kx_lo = MAX(0, -ix0);
    स्थिर int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    अगर (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_Vector empty_val = is_max ? hvx_vec_splat_f32(-भग्न_उच्च) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_row + ox, माप(भग्न), empty_val);
        वापस;
    }

    स्थिर uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    स्थिर HVX_Vector mask_identity = is_max ? hvx_vec_splat_f32(-भग्न_उच्च) : Q6_V_vsplat_R(0);

    HVX_Vector acc = mask_identity;
    क्रम (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        स्थिर भग्न * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        क्रम (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            स्थिर uint32_t k_rem = valid_kx - k;
            स्थिर uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            स्थिर HVX_VectorPred q = Q6_Q_vsetq_R(n * माप(भग्न));
            स्थिर HVX_Vector raw = *(स्थिर HVX_UVector *) (row + ix0 + kx_lo + k);
            स्थिर HVX_Vector v = Q6_V_vmux_QVV(q, raw, mask_identity);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
        }
    }

    HVX_Vector reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_योग_f32(acc);
    अगर (!is_max) {
        HVX_Vector scale_vec;
        अगर (p->avg_divide_गिनती) {
            स्थिर uint32_t गिनती = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((भग्न) गिनती));
        } अन्यथा {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_row + ox, माप(भग्न), reduced);
}

अटल अंतरभूत व्योम pool_row_general_boundary_vec(
    स्थिर भग्न * src, भग्न * dst_row, स्थिर काष्ठा htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    क्रम (uint32_t ox = ox_start; ox < ox_end; ++ox) {
        pool_pixel_boundary_vec(src, dst_row, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
अटल अंतरभूत व्योम pool_row_general_vec(
    स्थिर भग्न * src, भग्न * dst_row, स्थिर काष्ठा htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    अगर (ox_start >= ox_end) {
        वापस;
    }

    अगर (p->stride_x != 1 && p->stride_x != 2) {
        pool_row_general_boundary_vec(src, dst_row, p, ox_start, ox_end, iy0, ky_lo, ky_hi, is_max);
        वापस;
    }

    स्थिर HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    स्थिर HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-भग्न_उच्च) : Q6_V_vsplat_R(0);

    क्रम (uint32_t ox = ox_start; ox < ox_end; ox += VLEN_FP32) {
        स्थिर uint32_t rem = ox_end - ox;
        स्थिर uint32_t nbytes = (rem < VLEN_FP32) ? (rem * माप(भग्न)) : VLEN;
        स्थिर int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_Vector acc = seed;

        क्रम (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            स्थिर भग्न * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            अगर (p->stride_x == 1) {
                क्रम (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    स्थिर HVX_Vector v = *(स्थिर HVX_UVector *) (row + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            } अन्यथा {
                क्रम (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    स्थिर HVX_Vector v0 = *(स्थिर HVX_UVector *) (row + ix0 + (int32_t) kx);
                    स्थिर HVX_Vector v1 = *(स्थिर HVX_UVector *) (row + ix0 + (int32_t) kx + VLEN_FP32);
                    स्थिर HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    स्थिर HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_add_f32_f32(acc, lo);
                    अगर (kx + 1 < p->kernel_x) {
                        स्थिर HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_add_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_row + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

अटल व्योम pool_plane_general(
    स्थिर भग्न * src, भग्न * dst, स्थिर काष्ठा htp_pool_2d_kernel_params * p) {
    स्थिर bool is_max = (p->pool_op == HTP_POOL_MAX);

    क्रम (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_row_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        भग्न * dst_row = dst + oy * p->dst_x;

        pool_row_general_boundary_vec(src, dst_row, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_vec(src, dst_row, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_boundary_vec(src, dst_row, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

प्रकार व्योम (*pool_plane_fn_t)(स्थिर भग्न * src, भग्न * dst, स्थिर काष्ठा htp_pool_2d_kernel_params * p);

काष्ठा pool_2d_context {
    काष्ठा htp_ops_context * octx;
    स्थिर काष्ठा htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_threads;
    uint32_t plane_start;
    uint32_t plane_गिनती;
    uint32_t planes_per_thread;
};

अटल व्योम pool_2d_thread(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा pool_2d_context * ctx = (काष्ठा pool_2d_context *) तथ्य;
    स्थिर काष्ठा htp_pool_2d_kernel_params * p = ctx->kparams;
    स्थिर काष्ठा htp_tensor * src0 = ctx->octx->src[0];
    स्थिर काष्ठा htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    स्थिर uint32_t planes_per_thread = ctx->planes_per_thread;
    स्थिर uint32_t first = ctx->plane_start + ith * planes_per_thread;
    स्थिर uint32_t last = MIN(first + planes_per_thread, ctx->plane_start + ctx->plane_गिनती);

    अगर (first >= last) {
        वापस;
    }

    काष्ठा htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_queue * dma_queue = ctx->octx->ctx->dma[ith];

    स्थिर uint32_t src_spad_half = p->src_plane_bytes_aligned;
    स्थिर uint32_t dst_spad_half = p->dst_plane_bytes_aligned;
    स्थिर uint32_t src_bytes_per_thread = 2 * src_spad_half;
    स्थिर uint32_t dst_bytes_per_thread = 2 * dst_spad_half;
    स्थिर माप_प्रकार off_dst = (माप_प्रकार) ctx->n_threads * src_bytes_per_thread;

    uint8_t * vtcm_base = (uint8_t *) ctx->octx->ctx->vtcm_base;
    uint8_t * src_spad  = vtcm_base + ith * src_bytes_per_thread;
    uint8_t * dst_spad  = vtcm_base + off_dst + ith * dst_bytes_per_thread;

    भग्न * srcb2[2] = { (भग्न *) src_spad, (भग्न *) (src_spad + src_spad_half) };
    भग्न * dstb2[2] = { (भग्न *) dst_spad, (भग्न *) (dst_spad + dst_spad_half) };

    स्थिर uint32_t योग = last - first;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    क्रम (uint32_t i = 0; i < योग && i < 2; ++i) {
        dma_queue_धकेलो(dma_queue,
                       dma_make_तथ्य(dst->तथ्य, dstb2[i]),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 0);

        स्थिर dma_addr_t src_addr = src0->तथ्य + (first + i) * p->src_plane_bytes;
        dma_queue_धकेलो(dma_queue,
                       dma_make_तथ्य(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_bytes,
                       p->src_plane_bytes, 1);
    }

    क्रम (uint32_t i = 0; i < योग; ++i) {
        स्थिर uint32_t plane = first + i;
        स्थिर uint32_t buf   = i & 1u;
        भग्न * srcb = srcb2[buf];
        भग्न * dstb = dstb2[buf];

        dma_queue_निकालो(dma_queue); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_queue_निकालो(dma_queue); // input for plane i

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        स्थिर dma_addr_t dst_addr = dst->तथ्य + plane * p->dst_plane_bytes;
        dma_queue_धकेलो(dma_queue,
                       dma_make_तथ्य(dst_addr, dstb),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 1);

        अगर (i + 2 < योग) {
            स्थिर dma_addr_t अग्र_src_addr = src0->तथ्य + (plane + 2) * p->src_plane_bytes;
            dma_queue_धकेलो(dma_queue,
                           dma_make_तथ्य(srcb, अग्र_src_addr),
                           src_spad_half, p->src_plane_bytes,
                           p->src_plane_bytes, 1);
        }
    }

    dma_queue_flush(dma_queue);

    FARF(ऊपरी_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], first, last);
    (व्योम) nth;
}

पूर्णांक op_pool_2d(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst = octx->dst;
    स्थिर काष्ठा htp_pool_2d_kernel_params * p =
        (स्थिर काष्ठा htp_pool_2d_kernel_params *) octx->kernel_params;

    अगर (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    uint32_t plane_start = 0;
    uint32_t plane_गिनती = p->planes;
    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर uint32_t planes_per_chunk = (p->dst_plane_bytes > 0) ? (HEX_L2_पंक्ति_आकार / hex_gcd_u32(p->dst_plane_bytes, HEX_L2_पंक्ति_आकार)) : 1;
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            plane_गिनती, htp_tensor_mdev_तथ्य_aligned(dst) ? planes_per_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        plane_start = range.start;
        plane_गिनती = range.गिनती;
    }
    अगर (plane_गिनती == 0) {
        वापस HTP_STATUS_OK;
    }

    स्थिर uint32_t n_threads = MIN(p->n_threads, plane_गिनती);
    अगर (!htp_ops_context_set_n_threads(octx, n_threads)) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    स्थिर uint32_t planes_per_thread = fastdiv(plane_गिनती + n_threads - 1, &octx->n_threads_div);

    pool_plane_fn_t pool_plane;
    अगर (p->global_path) {
        pool_plane = pool_plane_global;
    } अन्यथा अगर (p->block_path) {
        pool_plane = pool_plane_block;
    } अन्यथा अगर (p->narrow_path) {
        pool_plane = pool_plane_hvx_narrow;
    } अन्यथा अगर (p->fast_path) {
        pool_plane = pool_plane_hvx;
    } अन्यथा {
        pool_plane = pool_plane_general;
    }

    काष्ठा pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_threads = n_threads,
        .plane_start = plane_start,
        .plane_गिनती = plane_गिनती,
        .planes_per_thread = planes_per_thread,
    };
    work_queue_run(octx->ctx->work_queue, pool_2d_thread, &ctx, n_threads);
    वापस HTP_STATUS_OK;
}

पूर्णांक op_pool_1d(काष्ठा htp_ops_context * octx) {
    वापस op_pool_2d(octx);
}
