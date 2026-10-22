#pragma clang diagnostic ignored "-Wunused-variable"

#અન્તર્ભાવ <float.h>
#અન્તર્ભાવ <HAP_farf.h>

#અન્તર્ભાવ "hex-common.h"
#અન્તર્ભાવ "dma-queue.h"
#અન્તર્ભાવ "hex-profile.h"
#અન્તર્ભાવ "htp-ctx.h"
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "htp-tensor.h"
#અન્તર્ભાવ "hvx-inverse.h"
#અન્તર્ભાવ "hvx-types.h"
#અન્તર્ભાવ "hvx-utils.h"
#અન્તર્ભાવ "pool-ops.h"

#define HTP_POOL_MAX 0
#define HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
સ્થિર શૂન્ય pool_plane_hvx(
    અચલ દશાંશ * src, દશાંશ * dst, અચલ સંરચના htp_pool_2d_kernel_params * p) {
    અચલ bool is_max = (p->pool_op == HTP_POOL_MAX);
    અચલ HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    અચલ HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    માટે (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        અચલ uint32_t sy = oy * p->kernel_y;
        માટે (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            અચલ uint32_t rem = p->dst_x - ox;
            અચલ uint32_t nbytes = (rem < VLEN_FP32) ? (rem * કદ(દશાંશ)) : VLEN;
            HVX_Vector acc = seed;
            માટે (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                અચલ દશાંશ * row = src + (sy + ky) * p->src_x;
                જો (p->kernel_x == 2) {
                    અચલ HVX_Vector v0 = *(અચલ HVX_UVector *) (row + ox * 2);
                    અચલ HVX_Vector v1 = *(અચલ HVX_UVector *) (row + ox * 2 + VLEN_FP32);
                    અચલ HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    અચલ HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    અચલ HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_add_f32_f32(acc, hvx_vec_add_f32_f32(lo, hi));
                } નહીંતર જો (p->kernel_x == 1) {
                    અચલ HVX_Vector v = *(અચલ HVX_UVector *) (row + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
સ્થિર શૂન્ય pool_plane_hvx_narrow(
    અચલ દશાંશ * src, દશાંશ * dst, અચલ સંરચના htp_pool_2d_kernel_params * p) {
    અચલ bool is_max = (p->pool_op == HTP_POOL_MAX);
    અચલ HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    અચલ HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    માટે (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        અચલ uint32_t sy = oy * p->kernel_y;
        HVX_Vector acc = seed;
        માટે (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            અચલ દશાંશ * row = src + (sy + ky) * p->src_x;
            અચલ HVX_Vector v0 = *(અચલ HVX_UVector *) row;
            HVX_Vector v = v0;
            જો (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                અચલ HVX_Vector zero = Q6_V_vsplat_R(0);
                અચલ HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                અચલ HVX_Vector even = Q6_V_lo_W(deinterleaved);
                અચલ HVX_Vector odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_add_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_add_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * કદ(દશાંશ),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

સ્થિર શૂન્ય pool_plane_global(
    અચલ દશાંશ * src, દશાંશ * dst, અચલ સંરચના htp_pool_2d_kernel_params * p) {
    અચલ uint32_t n = p->src_x * p->src_y;
    દશાંશ val;
    જો (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((અચલ uint8_t *) src, n);
    } નહીંતર {
        val = hvx_reduce_sum_f32((અચલ uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, કદ(દશાંશ), hvx_vec_splat_f32(val));
}

સ્થિર શૂન્ય pool_plane_block(
    અચલ દશાંશ * src, દશાંશ * dst, અચલ સંરચના htp_pool_2d_kernel_params * p) {
    માટે (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        અચલ દશાંશ * row = src + oy * p->src_x;
        દશાંશ * dst_row = dst + oy * p->dst_x;
        માટે (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            અચલ uint8_t * block = (અચલ uint8_t *) (row + ox * p->kernel_x);
            દશાંશ val;
            જો (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(block, p->kernel_x);
            } નહીંતર {
                val = hvx_reduce_sum_f32(block, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_row + ox, કદ(દશાંશ), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
સ્થિર અંતઃરેખ શૂન્ય pool_row_bounds_y(
    અચલ સંરચના htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    અચલ int32_t lo = -(*iy0);
    અચલ int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

સ્થિર અંતઃરેખ શૂન્ય pool_pixel_boundary_vec(
    અચલ દશાંશ * src, દશાંશ * dst_row, અચલ સંરચના htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    અચલ int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    અચલ int32_t kx_lo = MAX(0, -ix0);
    અચલ int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    જો (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_Vector empty_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_row + ox, કદ(દશાંશ), empty_val);
        પાછા;
    }

    અચલ uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    અચલ HVX_Vector mask_identity = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_Vector acc = mask_identity;
    માટે (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        અચલ દશાંશ * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        માટે (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            અચલ uint32_t k_rem = valid_kx - k;
            અચલ uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            અચલ HVX_VectorPred q = Q6_Q_vsetq_R(n * કદ(દશાંશ));
            અચલ HVX_Vector raw = *(અચલ HVX_UVector *) (row + ix0 + kx_lo + k);
            અચલ HVX_Vector v = Q6_V_vmux_QVV(q, raw, mask_identity);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
        }
    }

    HVX_Vector reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_sum_f32(acc);
    જો (!is_max) {
        HVX_Vector scale_vec;
        જો (p->avg_divide_ગણતરી) {
            અચલ uint32_t ગણતરી = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((દશાંશ) ગણતરી));
        } નહીંતર {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_row + ox, કદ(દશાંશ), reduced);
}

સ્થિર અંતઃરેખ શૂન્ય pool_row_general_boundary_vec(
    અચલ દશાંશ * src, દશાંશ * dst_row, અચલ સંરચના htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    માટે (uint32_t ox = ox_start; ox < ox_end; ++ox) {
        pool_pixel_boundary_vec(src, dst_row, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
સ્થિર અંતઃરેખ શૂન્ય pool_row_general_vec(
    અચલ દશાંશ * src, દશાંશ * dst_row, અચલ સંરચના htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    જો (ox_start >= ox_end) {
        પાછા;
    }

    જો (p->stride_x != 1 && p->stride_x != 2) {
        pool_row_general_boundary_vec(src, dst_row, p, ox_start, ox_end, iy0, ky_lo, ky_hi, is_max);
        પાછા;
    }

    અચલ HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    અચલ HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    માટે (uint32_t ox = ox_start; ox < ox_end; ox += VLEN_FP32) {
        અચલ uint32_t rem = ox_end - ox;
        અચલ uint32_t nbytes = (rem < VLEN_FP32) ? (rem * કદ(દશાંશ)) : VLEN;
        અચલ int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_Vector acc = seed;

        માટે (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            અચલ દશાંશ * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            જો (p->stride_x == 1) {
                માટે (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    અચલ HVX_Vector v = *(અચલ HVX_UVector *) (row + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            } નહીંતર {
                માટે (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    અચલ HVX_Vector v0 = *(અચલ HVX_UVector *) (row + ix0 + (int32_t) kx);
                    અચલ HVX_Vector v1 = *(અચલ HVX_UVector *) (row + ix0 + (int32_t) kx + VLEN_FP32);
                    અચલ HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    અચલ HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_add_f32_f32(acc, lo);
                    જો (kx + 1 < p->kernel_x) {
                        અચલ HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_add_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_row + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

સ્થિર શૂન્ય pool_plane_general(
    અચલ દશાંશ * src, દશાંશ * dst, અચલ સંરચના htp_pool_2d_kernel_params * p) {
    અચલ bool is_max = (p->pool_op == HTP_POOL_MAX);

    માટે (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_row_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        દશાંશ * dst_row = dst + oy * p->dst_x;

        pool_row_general_boundary_vec(src, dst_row, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_vec(src, dst_row, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_boundary_vec(src, dst_row, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

પ્રકારવ્યાખ્યા શૂન્ય (*pool_plane_fn_t)(અચલ દશાંશ * src, દશાંશ * dst, અચલ સંરચના htp_pool_2d_kernel_params * p);

સંરચના pool_2d_context {
    સંરચના htp_ops_context * octx;
    અચલ સંરચના htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_threads;
    uint32_t plane_start;
    uint32_t plane_ગણતરી;
    uint32_t planes_per_thread;
};

સ્થિર શૂન્ય pool_2d_thread(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    સંરચના pool_2d_context * ctx = (સંરચના pool_2d_context *) data;
    અચલ સંરચના htp_pool_2d_kernel_params * p = ctx->kparams;
    અચલ સંરચના htp_tensor * src0 = ctx->octx->src[0];
    અચલ સંરચના htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    અચલ uint32_t planes_per_thread = ctx->planes_per_thread;
    અચલ uint32_t first = ctx->plane_start + ith * planes_per_thread;
    અચલ uint32_t last = MIN(first + planes_per_thread, ctx->plane_start + ctx->plane_ગણતરી);

    જો (first >= last) {
        પાછા;
    }

    સંરચના htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_queue * dma_queue = ctx->octx->ctx->dma[ith];

    અચલ uint32_t src_spad_half = p->src_plane_bytes_aligned;
    અચલ uint32_t dst_spad_half = p->dst_plane_bytes_aligned;
    અચલ uint32_t src_bytes_per_thread = 2 * src_spad_half;
    અચલ uint32_t dst_bytes_per_thread = 2 * dst_spad_half;
    અચલ size_t off_dst = (size_t) ctx->n_threads * src_bytes_per_thread;

    uint8_t * vtcm_base = (uint8_t *) ctx->octx->ctx->vtcm_base;
    uint8_t * src_spad  = vtcm_base + ith * src_bytes_per_thread;
    uint8_t * dst_spad  = vtcm_base + off_dst + ith * dst_bytes_per_thread;

    દશાંશ * srcb2[2] = { (દશાંશ *) src_spad, (દશાંશ *) (src_spad + src_spad_half) };
    દશાંશ * dstb2[2] = { (દશાંશ *) dst_spad, (દશાંશ *) (dst_spad + dst_spad_half) };

    અચલ uint32_t total = last - first;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    માટે (uint32_t i = 0; i < total && i < 2; ++i) {
        dma_queue_ધકેલો(dma_queue,
                       dma_make_data(dst->data, dstb2[i]),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 0);

        અચલ dma_addr_t src_addr = src0->data + (first + i) * p->src_plane_bytes;
        dma_queue_ધકેલો(dma_queue,
                       dma_make_data(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_bytes,
                       p->src_plane_bytes, 1);
    }

    માટે (uint32_t i = 0; i < total; ++i) {
        અચલ uint32_t plane = first + i;
        અચલ uint32_t buf   = i & 1u;
        દશાંશ * srcb = srcb2[buf];
        દશાંશ * dstb = dstb2[buf];

        dma_queue_કાઢો(dma_queue); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_queue_કાઢો(dma_queue); // input for plane i

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        અચલ dma_addr_t dst_addr = dst->data + plane * p->dst_plane_bytes;
        dma_queue_ધકેલો(dma_queue,
                       dma_make_data(dst_addr, dstb),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 1);

        જો (i + 2 < total) {
            અચલ dma_addr_t next_src_addr = src0->data + (plane + 2) * p->src_plane_bytes;
            dma_queue_ધકેલો(dma_queue,
                           dma_make_data(srcb, next_src_addr),
                           src_spad_half, p->src_plane_bytes,
                           p->src_plane_bytes, 1);
        }
    }

    dma_queue_flush(dma_queue);

    FARF(ઉપલું_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], first, last);
    (શૂન્ય) nth;
}

પૂર્ણાંક op_pool_2d(સંરચના htp_ops_context * octx) {
    અચલ સંરચના htp_tensor * src0 = octx->src[0];
    અચલ સંરચના htp_tensor * dst = octx->dst;
    અચલ સંરચના htp_pool_2d_kernel_params * p =
        (અચલ સંરચના htp_pool_2d_kernel_params *) octx->kernel_params;

    જો (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    uint32_t plane_start = 0;
    uint32_t plane_ગણતરી = p->planes;
    જો (octx->ctx->mdev.ગણતરી > 1) {
        અચલ uint32_t planes_per_chunk = (p->dst_plane_bytes > 0) ? (HEX_L2_લીટી_SIZE / hex_gcd_u32(p->dst_plane_bytes, HEX_L2_લીટી_SIZE)) : 1;
        અચલ સંરચના htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            plane_ગણતરી, htp_tensor_mdev_data_aligned(dst) ? planes_per_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.ગણતરી, &octx->ctx->mdev.ગણતરી_div);
        plane_start = range.start;
        plane_ગણતરી = range.ગણતરી;
    }
    જો (plane_ગણતરી == 0) {
        પાછા HTP_STATUS_OK;
    }

    અચલ uint32_t n_threads = MIN(p->n_threads, plane_ગણતરી);
    જો (!htp_ops_context_set_n_threads(octx, n_threads)) {
        પાછા HTP_STATUS_INVAL_PARAMS;
    }

    અચલ uint32_t planes_per_thread = fastdiv(plane_ગણતરી + n_threads - 1, &octx->n_threads_div);

    pool_plane_fn_t pool_plane;
    જો (p->global_path) {
        pool_plane = pool_plane_global;
    } નહીંતર જો (p->block_path) {
        pool_plane = pool_plane_block;
    } નહીંતર જો (p->narrow_path) {
        pool_plane = pool_plane_hvx_narrow;
    } નહીંતર જો (p->fast_path) {
        pool_plane = pool_plane_hvx;
    } નહીંતર {
        pool_plane = pool_plane_general;
    }

    સંરચના pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_threads = n_threads,
        .plane_start = plane_start,
        .plane_ગણતરી = plane_ગણતરી,
        .planes_per_thread = planes_per_thread,
    };
    work_queue_run(octx->ctx->work_queue, pool_2d_thread, &ctx, n_threads);
    પાછા HTP_STATUS_OK;
}

પૂર્ણાંક op_pool_1d(સંરચના htp_ops_context * octx) {
    પાછા op_pool_2d(octx);
}
