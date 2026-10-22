#pragma clang diagnostic ignored "-Wunused-variable"

#അന്തര്ഭാവ <float.h>
#അന്തര്ഭാവ <HAP_farf.h>

#അന്തര്ഭാവ "hex-common.h"
#അന്തര്ഭാവ "dma-queue.h"
#അന്തര്ഭാവ "hex-profile.h"
#അന്തര്ഭാവ "htp-ctx.h"
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "htp-tensor.h"
#അന്തര്ഭാവ "hvx-inverse.h"
#അന്തര്ഭാവ "hvx-types.h"
#അന്തര്ഭാവ "hvx-utils.h"
#അന്തര്ഭാവ "pool-ops.h"

#define HTP_POOL_MAX 0
#define HTP_POOL_AVG 1

// Fast path: exact non-overlapping tiling (stride == kernel, no padding), kernel_x in {1,2}.
// Every window is guaranteed fully in-bounds, so this never needs boundary clamping.
സ്ഥിതം ശൂന്യം pool_plane_hvx(
    സ്ഥിരം ദശാംശം * src, ദശാംശം * dst, സ്ഥിരം ഘടന htp_pool_2d_kernel_params * p) {
    സ്ഥിരം bool is_max = (p->pool_op == HTP_POOL_MAX);
    സ്ഥിരം HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    സ്ഥിരം HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    വേണ്ടി (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        സ്ഥിരം uint32_t sy = oy * p->kernel_y;
        വേണ്ടി (uint32_t ox = 0; ox < p->dst_x; ox += VLEN_FP32) {
            സ്ഥിരം uint32_t rem = p->dst_x - ox;
            സ്ഥിരം uint32_t nbytes = (rem < VLEN_FP32) ? (rem * വലുപ്പം(ദശാംശം)) : VLEN;
            HVX_Vector acc = seed;
            വേണ്ടി (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
                സ്ഥിരം ദശാംശം * row = src + (sy + ky) * p->src_x;
                എങ്കിൽ (p->kernel_x == 2) {
                    സ്ഥിരം HVX_Vector v0 = *(സ്ഥിരം HVX_UVector *) (row + ox * 2);
                    സ്ഥിരം HVX_Vector v1 = *(സ്ഥിരം HVX_UVector *) (row + ox * 2 + VLEN_FP32);
                    സ്ഥിരം HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    സ്ഥിരം HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    സ്ഥിരം HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, Q6_Vsf_vmax_VsfVsf(lo, hi))
                                 : hvx_vec_add_f32_f32(acc, hvx_vec_add_f32_f32(lo, hi));
                } അല്ലെങ്കിൽ എങ്കിൽ (p->kernel_x == 1) {
                    സ്ഥിരം HVX_Vector v = *(സ്ഥിരം HVX_UVector *) (row + ox);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            }
            hvx_vec_store_u(dst + oy * p->dst_x + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
        }
    }
}

// Narrow exact-tiling path. The input is staged in VTCM with one vector of
// guard space, so full-width loads are safe even when src_x is below 32.
സ്ഥിതം ശൂന്യം pool_plane_hvx_narrow(
    സ്ഥിരം ദശാംശം * src, ദശാംശം * dst, സ്ഥിരം ഘടന htp_pool_2d_kernel_params * p) {
    സ്ഥിരം bool is_max = (p->pool_op == HTP_POOL_MAX);
    സ്ഥിരം HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    സ്ഥിരം HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    വേണ്ടി (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        സ്ഥിരം uint32_t sy = oy * p->kernel_y;
        HVX_Vector acc = seed;
        വേണ്ടി (uint32_t ky = 0; ky < p->kernel_y; ++ky) {
            സ്ഥിരം ദശാംശം * row = src + (sy + ky) * p->src_x;
            സ്ഥിരം HVX_Vector v0 = *(സ്ഥിരം HVX_UVector *) row;
            HVX_Vector v = v0;
            എങ്കിൽ (p->kernel_x == 2) {
                // The second vector is zero because a narrow row has fewer
                // than 32 input elements. The low lanes still contain the
                // complete even/odd pairs needed by the output.
                സ്ഥിരം HVX_Vector zero = Q6_V_vsplat_R(0);
                സ്ഥിരം HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(zero, v0, -4);
                സ്ഥിരം HVX_Vector even = Q6_V_lo_W(deinterleaved);
                സ്ഥിരം HVX_Vector odd  = Q6_V_hi_W(deinterleaved);
                v = is_max ? Q6_Vsf_vmax_VsfVsf(even, odd)
                           : hvx_vec_add_f32_f32(even, odd);
            }
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v)
                         : hvx_vec_add_f32_f32(acc, v);
        }
        hvx_vec_store_u(dst + oy * p->dst_x, p->dst_x * വലുപ്പം(ദശാംശം),
                        is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

സ്ഥിതം ശൂന്യം pool_plane_global(
    സ്ഥിരം ദശാംശം * src, ദശാംശം * dst, സ്ഥിരം ഘടന htp_pool_2d_kernel_params * p) {
    സ്ഥിരം uint32_t n = p->src_x * p->src_y;
    ദശാംശം val;
    എങ്കിൽ (p->pool_op == HTP_POOL_MAX) {
        val = hvx_reduce_max_f32((സ്ഥിരം uint8_t *) src, n);
    } അല്ലെങ്കിൽ {
        val = hvx_reduce_sum_f32((സ്ഥിരം uint8_t *) src, n) * p->inv_kernel_area;
    }
    hvx_vec_store_u(dst, വലുപ്പം(ദശാംശം), hvx_vec_splat_f32(val));
}

സ്ഥിതം ശൂന്യം pool_plane_block(
    സ്ഥിരം ദശാംശം * src, ദശാംശം * dst, സ്ഥിരം ഘടന htp_pool_2d_kernel_params * p) {
    വേണ്ടി (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        സ്ഥിരം ദശാംശം * row = src + oy * p->src_x;
        ദശാംശം * dst_row = dst + oy * p->dst_x;
        വേണ്ടി (uint32_t ox = 0; ox < p->dst_x; ++ox) {
            സ്ഥിരം uint8_t * block = (സ്ഥിരം uint8_t *) (row + ox * p->kernel_x);
            ദശാംശം val;
            എങ്കിൽ (p->pool_op == HTP_POOL_MAX) {
                val = hvx_reduce_max_f32(block, p->kernel_x);
            } അല്ലെങ്കിൽ {
                val = hvx_reduce_sum_f32(block, p->kernel_x) * p->inv_kernel_area;
            }
            hvx_vec_store_u(dst_row + ox, വലുപ്പം(ദശാംശം), hvx_vec_splat_f32(val));
        }
    }
}

// General path: arbitrary kernel/stride/padding

// Vertical padding is uniform across a whole output row (iy0 depends only on oy, not ox),
// so it collapses to one valid-ky range per row instead of a per-element check.
സ്ഥിതം വരിയിൽ ശൂന്യം pool_row_bounds_y(
    സ്ഥിരം ഘടന htp_pool_2d_kernel_params * p, uint32_t oy, int32_t * iy0, uint32_t * ky_lo, uint32_t * ky_hi) {
    *iy0 = (int32_t) (oy * p->stride_y) - p->pad_y;
    സ്ഥിരം int32_t lo = -(*iy0);
    സ്ഥിരം int32_t hi = (int32_t) p->src_y - *iy0;
    *ky_lo = (uint32_t) MAX(0, lo);
    *ky_hi = (uint32_t) MAX(0, MIN((int32_t) p->kernel_y, hi));
}

സ്ഥിതം വരിയിൽ ശൂന്യം pool_pixel_boundary_vec(
    സ്ഥിരം ദശാംശം * src, ദശാംശം * dst_row, സ്ഥിരം ഘടന htp_pool_2d_kernel_params * p,
    uint32_t ox, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    സ്ഥിരം int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
    സ്ഥിരം int32_t kx_lo = MAX(0, -ix0);
    സ്ഥിരം int32_t kx_hi = MIN((int32_t) p->kernel_x, (int32_t) p->src_x - ix0);

    എങ്കിൽ (kx_lo >= kx_hi || ky_lo >= ky_hi) {
        HVX_Vector empty_val = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);
        hvx_vec_store_u(dst_row + ox, വലുപ്പം(ദശാംശം), empty_val);
        തിരികെ;
    }

    സ്ഥിരം uint32_t valid_kx = (uint32_t) (kx_hi - kx_lo);
    സ്ഥിരം HVX_Vector mask_identity = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    HVX_Vector acc = mask_identity;
    വേണ്ടി (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
        സ്ഥിരം ദശാംശം * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
        വേണ്ടി (uint32_t k = 0; k < valid_kx; k += VLEN_FP32) {
            സ്ഥിരം uint32_t k_rem = valid_kx - k;
            സ്ഥിരം uint32_t n = (k_rem < VLEN_FP32) ? k_rem : VLEN_FP32;
            സ്ഥിരം HVX_VectorPred q = Q6_Q_vsetq_R(n * വലുപ്പം(ദശാംശം));
            സ്ഥിരം HVX_Vector raw = *(സ്ഥിരം HVX_UVector *) (row + ix0 + kx_lo + k);
            സ്ഥിരം HVX_Vector v = Q6_V_vmux_QVV(q, raw, mask_identity);
            acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
        }
    }

    HVX_Vector reduced = is_max ? hvx_vec_reduce_max_f32(acc) : hvx_vec_reduce_sum_f32(acc);
    എങ്കിൽ (!is_max) {
        HVX_Vector scale_vec;
        എങ്കിൽ (p->avg_divide_എണ്ണം) {
            സ്ഥിരം uint32_t എണ്ണം = (ky_hi - ky_lo) * valid_kx;
            scale_vec = hvx_vec_inverse_f32(hvx_vec_splat_f32((ദശാംശം) എണ്ണം));
        } അല്ലെങ്കിൽ {
            scale_vec = hvx_vec_splat_f32(p->inv_kernel_area);
        }
        reduced = hvx_vec_mul_f32_f32(reduced, scale_vec);
    }
    hvx_vec_store_u(dst_row + ox, വലുപ്പം(ദശാംശം), reduced);
}

സ്ഥിതം വരിയിൽ ശൂന്യം pool_row_general_boundary_vec(
    സ്ഥിരം ദശാംശം * src, ദശാംശം * dst_row, സ്ഥിരം ഘടന htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    വേണ്ടി (uint32_t ox = ox_start; ox < ox_end; ++ox) {
        pool_pixel_boundary_vec(src, dst_row, p, ox, iy0, ky_lo, ky_hi, is_max);
    }
}

// Vectorized interior loop.
സ്ഥിതം വരിയിൽ ശൂന്യം pool_row_general_vec(
    സ്ഥിരം ദശാംശം * src, ദശാംശം * dst_row, സ്ഥിരം ഘടന htp_pool_2d_kernel_params * p,
    uint32_t ox_start, uint32_t ox_end, int32_t iy0, uint32_t ky_lo, uint32_t ky_hi, bool is_max) {
    എങ്കിൽ (ox_start >= ox_end) {
        തിരികെ;
    }

    എങ്കിൽ (p->stride_x != 1 && p->stride_x != 2) {
        pool_row_general_boundary_vec(src, dst_row, p, ox_start, ox_end, iy0, ky_lo, ky_hi, is_max);
        തിരികെ;
    }

    സ്ഥിരം HVX_Vector scale = hvx_vec_splat_f32(p->inv_kernel_area);
    സ്ഥിരം HVX_Vector seed  = is_max ? hvx_vec_splat_f32(-FLT_MAX) : Q6_V_vsplat_R(0);

    വേണ്ടി (uint32_t ox = ox_start; ox < ox_end; ox += VLEN_FP32) {
        സ്ഥിരം uint32_t rem = ox_end - ox;
        സ്ഥിരം uint32_t nbytes = (rem < VLEN_FP32) ? (rem * വലുപ്പം(ദശാംശം)) : VLEN;
        സ്ഥിരം int32_t ix0 = (int32_t) (ox * p->stride_x) - p->pad_x;
        HVX_Vector acc = seed;

        വേണ്ടി (uint32_t ky = ky_lo; ky < ky_hi; ++ky) {
            സ്ഥിരം ദശാംശം * row = src + (uint32_t) (iy0 + (int32_t) ky) * p->src_x;
            എങ്കിൽ (p->stride_x == 1) {
                വേണ്ടി (uint32_t kx = 0; kx < p->kernel_x; ++kx) {
                    സ്ഥിരം HVX_Vector v = *(സ്ഥിരം HVX_UVector *) (row + ix0 + (int32_t) kx);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, v) : hvx_vec_add_f32_f32(acc, v);
                }
            } അല്ലെങ്കിൽ {
                വേണ്ടി (uint32_t kx = 0; kx < p->kernel_x; kx += 2) {
                    സ്ഥിരം HVX_Vector v0 = *(സ്ഥിരം HVX_UVector *) (row + ix0 + (int32_t) kx);
                    സ്ഥിരം HVX_Vector v1 = *(സ്ഥിരം HVX_UVector *) (row + ix0 + (int32_t) kx + VLEN_FP32);
                    സ്ഥിരം HVX_VectorPair deinterleaved = Q6_W_vdeal_VVR(v1, v0, -4);
                    സ്ഥിരം HVX_Vector lo = Q6_V_lo_W(deinterleaved);
                    acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, lo) : hvx_vec_add_f32_f32(acc, lo);
                    എങ്കിൽ (kx + 1 < p->kernel_x) {
                        സ്ഥിരം HVX_Vector hi = Q6_V_hi_W(deinterleaved);
                        acc = is_max ? Q6_Vsf_vmax_VsfVsf(acc, hi) : hvx_vec_add_f32_f32(acc, hi);
                    }
                }
            }
        }
        hvx_vec_store_u(dst_row + ox, nbytes, is_max ? acc : hvx_vec_mul_f32_f32(acc, scale));
    }
}

സ്ഥിതം ശൂന്യം pool_plane_general(
    സ്ഥിരം ദശാംശം * src, ദശാംശം * dst, സ്ഥിരം ഘടന htp_pool_2d_kernel_params * p) {
    സ്ഥിരം bool is_max = (p->pool_op == HTP_POOL_MAX);

    വേണ്ടി (uint32_t oy = 0; oy < p->dst_y; ++oy) {
        int32_t iy0;
        uint32_t ky_lo, ky_hi;
        pool_row_bounds_y(p, oy, &iy0, &ky_lo, &ky_hi);
        ദശാംശം * dst_row = dst + oy * p->dst_x;

        pool_row_general_boundary_vec(src, dst_row, p, 0, p->ox_lo, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_vec(src, dst_row, p, p->ox_lo, p->ox_hi, iy0, ky_lo, ky_hi, is_max);
        pool_row_general_boundary_vec(src, dst_row, p, p->ox_hi, p->dst_x, iy0, ky_lo, ky_hi, is_max);
    }
}

തരനിർവചനം ശൂന്യം (*pool_plane_fn_t)(സ്ഥിരം ദശാംശം * src, ദശാംശം * dst, സ്ഥിരം ഘടന htp_pool_2d_kernel_params * p);

ഘടന pool_2d_context {
    ഘടന htp_ops_context * octx;
    സ്ഥിരം ഘടന htp_pool_2d_kernel_params * kparams;
    pool_plane_fn_t pool_plane;
    uint32_t n_threads;
    uint32_t plane_start;
    uint32_t plane_എണ്ണം;
    uint32_t planes_per_thread;
};

സ്ഥിതം ശൂന്യം pool_2d_thread(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    ഘടന pool_2d_context * ctx = (ഘടന pool_2d_context *) data;
    സ്ഥിരം ഘടന htp_pool_2d_kernel_params * p = ctx->kparams;
    സ്ഥിരം ഘടന htp_tensor * src0 = ctx->octx->src[0];
    സ്ഥിരം ഘടന htp_tensor * dst = ctx->octx->dst;
    pool_plane_fn_t pool_plane = ctx->pool_plane;
    സ്ഥിരം uint32_t planes_per_thread = ctx->planes_per_thread;
    സ്ഥിരം uint32_t first = ctx->plane_start + ith * planes_per_thread;
    സ്ഥിരം uint32_t last = MIN(first + planes_per_thread, ctx->plane_start + ctx->plane_എണ്ണം);

    എങ്കിൽ (first >= last) {
        തിരികെ;
    }

    ഘടന htp_thread_trace * tr = &ctx->octx->ctx->trace[ith];
    dma_queue * dma_queue = ctx->octx->ctx->dma[ith];

    സ്ഥിരം uint32_t src_spad_half = p->src_plane_bytes_aligned;
    സ്ഥിരം uint32_t dst_spad_half = p->dst_plane_bytes_aligned;
    സ്ഥിരം uint32_t src_bytes_per_thread = 2 * src_spad_half;
    സ്ഥിരം uint32_t dst_bytes_per_thread = 2 * dst_spad_half;
    സ്ഥിരം size_t off_dst = (size_t) ctx->n_threads * src_bytes_per_thread;

    uint8_t * vtcm_base = (uint8_t *) ctx->octx->ctx->vtcm_base;
    uint8_t * src_spad  = vtcm_base + ith * src_bytes_per_thread;
    uint8_t * dst_spad  = vtcm_base + off_dst + ith * dst_bytes_per_thread;

    ദശാംശം * srcb2[2] = { (ദശാംശം *) src_spad, (ദശാംശം *) (src_spad + src_spad_half) };
    ദശാംശം * dstb2[2] = { (ദശാംശം *) dst_spad, (ദശാംശം *) (dst_spad + dst_spad_half) };

    സ്ഥിരം uint32_t total = last - first;

    // Warm up the pipeline: push up to 2 initial (dummy dst, src) transfer pairs.
    വേണ്ടി (uint32_t i = 0; i < total && i < 2; ++i) {
        dma_queue_തള്ളുക(dma_queue,
                       dma_make_data(dst->data, dstb2[i]),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 0);

        സ്ഥിരം dma_addr_t src_addr = src0->data + (first + i) * p->src_plane_bytes;
        dma_queue_തള്ളുക(dma_queue,
                       dma_make_data(srcb2[i], src_addr),
                       src_spad_half, p->src_plane_bytes,
                       p->src_plane_bytes, 1);
    }

    വേണ്ടി (uint32_t i = 0; i < total; ++i) {
        സ്ഥിരം uint32_t plane = first + i;
        സ്ഥിരം uint32_t buf   = i & 1u;
        ദശാംശം * srcb = srcb2[buf];
        ദശാംശം * dstb = dstb2[buf];

        dma_queue_എടുക്കുക(dma_queue); // dst writeback from plane i - 2 (or dummy on iter 0, 1)
        dma_queue_എടുക്കുക(dma_queue); // input for plane i

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);
        pool_plane(srcb, dstb, p);
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) plane);

        സ്ഥിരം dma_addr_t dst_addr = dst->data + plane * p->dst_plane_bytes;
        dma_queue_തള്ളുക(dma_queue,
                       dma_make_data(dst_addr, dstb),
                       p->dst_plane_bytes, dst_spad_half,
                       p->dst_plane_bytes, 1);

        എങ്കിൽ (i + 2 < total) {
            സ്ഥിരം dma_addr_t next_src_addr = src0->data + (plane + 2) * p->src_plane_bytes;
            dma_queue_തള്ളുക(dma_queue,
                           dma_make_data(srcb, next_src_addr),
                           src_spad_half, p->src_plane_bytes,
                           p->src_plane_bytes, 1);
        }
    }

    dma_queue_flush(dma_queue);

    FARF(മുകളിലെ_2, "pool2d-f32-dma %d/%d: %ux%ux%ux%u -> %ux%ux%ux%u (%u:%u)\n",
         ith, nth, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], first, last);
    (ശൂന്യം) nth;
}

പൂർണ്ണസംഖ്യ op_pool_2d(ഘടന htp_ops_context * octx) {
    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];
    സ്ഥിരം ഘടന htp_tensor * dst = octx->dst;
    സ്ഥിരം ഘടന htp_pool_2d_kernel_params * p =
        (സ്ഥിരം ഘടന htp_pool_2d_kernel_params *) octx->kernel_params;

    എങ്കിൽ (src0->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 ||
        (p->pool_op != HTP_POOL_AVG && p->pool_op != HTP_POOL_MAX)) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    uint32_t plane_start = 0;
    uint32_t plane_എണ്ണം = p->planes;
    എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1) {
        സ്ഥിരം uint32_t planes_per_chunk = (p->dst_plane_bytes > 0) ? (HEX_L2_വരി_SIZE / hex_gcd_u32(p->dst_plane_bytes, HEX_L2_വരി_SIZE)) : 1;
        സ്ഥിരം ഘടന htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            plane_എണ്ണം, htp_tensor_mdev_data_aligned(dst) ? planes_per_chunk : 0,
            octx->ctx->mdev.idx, octx->ctx->mdev.എണ്ണം, &octx->ctx->mdev.എണ്ണം_div);
        plane_start = range.start;
        plane_എണ്ണം = range.എണ്ണം;
    }
    എങ്കിൽ (plane_എണ്ണം == 0) {
        തിരികെ HTP_STATUS_OK;
    }

    സ്ഥിരം uint32_t n_threads = MIN(p->n_threads, plane_എണ്ണം);
    എങ്കിൽ (!htp_ops_context_set_n_threads(octx, n_threads)) {
        തിരികെ HTP_STATUS_INVAL_PARAMS;
    }

    സ്ഥിരം uint32_t planes_per_thread = fastdiv(plane_എണ്ണം + n_threads - 1, &octx->n_threads_div);

    pool_plane_fn_t pool_plane;
    എങ്കിൽ (p->global_path) {
        pool_plane = pool_plane_global;
    } അല്ലെങ്കിൽ എങ്കിൽ (p->block_path) {
        pool_plane = pool_plane_block;
    } അല്ലെങ്കിൽ എങ്കിൽ (p->narrow_path) {
        pool_plane = pool_plane_hvx_narrow;
    } അല്ലെങ്കിൽ എങ്കിൽ (p->fast_path) {
        pool_plane = pool_plane_hvx;
    } അല്ലെങ്കിൽ {
        pool_plane = pool_plane_general;
    }

    ഘടന pool_2d_context ctx = {
        .octx = octx,
        .kparams = p,
        .pool_plane = pool_plane,
        .n_threads = n_threads,
        .plane_start = plane_start,
        .plane_എണ്ണം = plane_എണ്ണം,
        .planes_per_thread = planes_per_thread,
    };
    work_queue_run(octx->ctx->work_queue, pool_2d_thread, &ctx, n_threads);
    തിരികെ HTP_STATUS_OK;
}

പൂർണ്ണസംഖ്യ op_pool_1d(ഘടന htp_ops_context * octx) {
    തിരികെ op_pool_2d(octx);
}
