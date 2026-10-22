// Flash attention via oneMKL GEMM (XMX-accelerated).
// Uses column_major::gemm for Q*K^T and S*V matmuls
// with an online softmax SYCL kernel.
//
// All GQA query heads sharing a KV head are batched into single
// GEMM calls, amortizing MKL launch overhead across K and V reuse.
//

#समावेश "common.hpp"
#समावेश "fattn-common.hpp"
#समावेश "convert.hpp"
#समावेश "fattn.hpp"

#समावेश <oneapi/mkl.hpp>
#समावेश <cstdio>
#समावेश <chrono>

#घोषणा MKL_FA_CHUNK_आकार_KV 8192

// Number of query rows processed per tile. The score buffers (KQ_f32, S_f16)
// are sized q_tile_rows * chunk_size, so this bounds their footprint
// regardless of batch size (n_query_rows = n_queries * gqa_ratio). A typical
// single-ubatch prefill (e.g. ubatch 1024 * gqa 8 = 8192 rows) is exactly one
// tile, so it runs with no extra iterations. Larger batches tile and stay
// bounded. Override with GGML_SYCL_MKL_FA_Q_TILE.
#घोषणा MKL_FA_Q_TILE 8192

#घोषणा MKL_FA_WG_आकार 256

using oneapi::mkl::transpose;
using oneapi::mkl::blas::column_major::gemm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Pack all GQA Q heads for one KV head into fp16, applying q_scale.
// Launches one kernel per GQA group — each kernel copies exactly
// n_queries * DKQ elements using the per-group dst offset and
// per-head source stride.
अटल व्योम mkl_fa_pack_q_fp16(
    dpct::queue_ptr stream,
    sycl::half * __restrict dst,
    स्थिर भग्न * __restrict q_src,
    पूर्णांक n_queries, पूर्णांक DKQ,
    पूर्णांक gqa_ratio, पूर्णांक kvh_base_शीर्ष,
    भग्न q_scale, int64_t q_row_stride, int64_t q_शीर्ष_stride,
    int64_t wg_आकार) {

    क्रम (पूर्णांक iqg = 0; iqg < gqa_ratio; iqg++) {
        पूर्णांक     iqh       = kvh_base_शीर्ष + iqg;
        sycl::half * dst_g = dst + (int64_t)iqg * n_queries * DKQ;

        स्थिर int64_t n_elem = (int64_t)n_queries * DKQ;
        स्थिर int64_t wg = ((n_elem + wg_आकार - 1) / wg_आकार) * wg_आकार;

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(wg, wg_आकार),
                [=](sycl::nd_item<1> item) {
                    int64_t e = item.get_global_id(0);
                    अगर (e >= n_elem) वापस;

                    int64_t q = e / DKQ;
                    int64_t d = e - q * DKQ;

                    // Stride-aware source offset: handles permuted,
                    // sliced, or contiguous Q tensor layouts.
                    int64_t src_off = d
                        + q * q_row_stride
                        + (int64_t)iqh * q_शीर्ष_stride;

                    dst_g[e] = sycl::half(
                        q_src[src_off] * q_scale);
                });
        });
    }
}

// Zero-initialize the online softmax state arrays.
// KQ_max → -inf, KQ_sum → 0, VKQ_accum → 0.
// Merged into one kernel to avoid per-array launch overhead.
अटल व्योम mkl_fa_init_softmax_state(
    dpct::queue_ptr stream,
    भग्न * kmax, भग्न * ksum, भग्न * vacc,
    पूर्णांक n_query_rows, पूर्णांक DV, int64_t wg_आकार) {

    स्थिर भग्न    neg_inf   = -1e30f;
    स्थिर int64_t  n_maxsum  = n_query_rows;
    स्थिर int64_t  n_vacc    = (int64_t)n_query_rows * DV;
    स्थिर int64_t  योग     = (n_vacc > n_maxsum) ? n_vacc : n_maxsum;
    स्थिर int64_t  wg = ((योग + wg_आकार - 1) / wg_आकार) * wg_आकार;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_आकार),
            [=](sycl::nd_item<1> item) {
                int64_t i = item.get_global_id(0);
                अगर (i < n_maxsum) {
                    kmax[i] = neg_inf;
                    ksum[i] = 0.0f;
                }
                अगर (i < n_vacc) {
                    vacc[i] = 0.0f;
                }
            });
    });
}

// Online softmax over one KV chunk for a tile of GQA query rows.
// The tile spans absolute rows [q0, q0 + q_rows). Score buffers
// (KQ_f32/S_f16) are indexed RELATIVE to the tile; the persistent state
// (VKQ_accum/KQ_max/KQ_sum) and mask are indexed by ABSOLUTE row.
// One WORK-GROUP per query row (local size = wg_size): work-items stride
// over the chunk so adjacent items touch adjacent elements (coalesced),
// the row max/sum come from group reductions, and the DV-long VKQ
// rescale is spread across the items. Item 0 is the sole writer of
// KQ_max/KQ_sum; its writes are ordered after every other item's reads
// by the second group reduction (a collective). Per-element math is
// identical to the original one-item-per-row kernel: softcap before
// mask, native::exp, -1e30 sentinel, half-precision S. Only the float
// summation order differs (tree vs serial), i.e. last-ulp level.
अटल व्योम mkl_fa_online_softmax_chunk(
    dpct::queue_ptr stream,
    भग्न * __restrict KQ_f32,
    sycl::half * __restrict S_f16,
    भग्न * __restrict KQ_max,
    भग्न * __restrict KQ_योग,
    भग्न * __restrict VKQ_accum,
    पूर्णांक q0, पूर्णांक q_rows, पूर्णांक n_queries, पूर्णांक DV,
    पूर्णांक chunk_आकार, पूर्णांक chunk_start,
    पूर्णांक kvh_शीर्ष,
    स्थिर sycl::half * mask_तथ्य, int64_t mask_शीर्ष_stride,
    int64_t mask_row_stride, पूर्णांक mask_n_heads,
    भग्न logit_softcap, int64_t wg_आकार) {

    // One work-group per query row: exactly q_rows groups of wg_size
    // items. q_rows * wg_size is already a multiple of wg_size, so unlike
    // the one-item-per-row kernels there is no round-up / tail guard.
    स्थिर int64_t wg         = q_rows * wg_आकार;
    स्थिर पूर्णांक     local_आकार = (पूर्णांक) wg_आकार;  // stride in the loops below
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_आकार),
            [=](sycl::nd_item<1> item) {
                स्थिर पूर्णांक local_id = (पूर्णांक)item.get_local_id(0);
                स्थिर पूर्णांक row = (पूर्णांक)item.get_group(0); // tile-relative
                स्थिर पूर्णांक jc_abs    = q0 + row;
                स्थिर पूर्णांक gqa_group = jc_abs / n_queries;
                स्थिर पूर्णांक q_row     = jc_abs % n_queries;
                // Score buffers are tile-local (relative index).
                स्थिर भग्न * __restrict KQ_row = KQ_f32
                    + row * (int64_t)chunk_आकार;
                sycl::half * __restrict S_row = S_f16
                    + row * (int64_t)chunk_आकार;
                // Persistent accumulator is full-sized (absolute index).
                भग्न * __restrict vkq = VKQ_accum
                    + jc_abs * (int64_t)DV;
                स्थिर sycl::half * mask_h = nullptr;
                int64_t m_stride = 0;
                अगर (mask_तथ्य) {
                    पूर्णांक m_शीर्ष = (mask_n_heads > 1)
                        ? (kvh_शीर्ष + gqa_group) : 0;
                    mask_h   = mask_तथ्य + (int64_t)m_शीर्ष * mask_शीर्ष_stride;
                    m_stride = mask_row_stride;
                }
                // Score at chunk offset i — original per-element math.
                स्वतः score = [&](पूर्णांक i) {
                    भग्न s = KQ_row[i];
                    अगर (logit_softcap != 0.0f) {
                        s = logit_softcap * sycl::tanh(s);
                    }
                    अगर (mask_h) {
                        s += (भग्न)mask_h[q_row * m_stride
                            + (chunk_start + i)];
                    }
                    वापस s;
                };
                // Pass 1: strided (coalesced) row-wise local maximum.
                भग्न local_max = -1e30f;
                क्रम (पूर्णांक i = local_id; i < chunk_आकार; i += local_आकार) {
                    भग्न s = score(i);
                    अगर (s > local_max) local_max = s;
                }
                स्थिर भग्न final_local_max = sycl::reduce_over_group(
                    item.get_group(), local_max, sycl::maximum<भग्न>());
                // Rescale previous accumulator by exp(old_max - new_max)
                भग्न old_max = KQ_max[jc_abs];
                भग्न new_max = (old_max > final_local_max) ? old_max : final_local_max;
                भग्न rescale = (old_max < -1e29f) ? 1.0f
                    : sycl::native::exp(old_max - new_max);
                क्रम (पूर्णांक v = local_id; v < DV; v += local_आकार) {
                    vkq[v] *= rescale;
                }
                // Pass 2: softmax numerators, strided; S row written once.
                भग्न local_योग = 0.0f;
                क्रम (पूर्णांक i = local_id; i < chunk_आकार; i += local_आकार) {
                    भग्न s = score(i);
                    भग्न val = sycl::native::exp(s - new_max);
                    S_row[i] = sycl::half(val);
                    local_योग += val;
                }
                स्थिर भग्न योग_योग = sycl::reduce_over_group(
                    item.get_group(), local_योग, sycl::plus<भग्न>());
                अगर (local_id == 0) {
                    KQ_योग[jc_abs] = KQ_योग[jc_abs] * rescale + योग_योग;
                    KQ_max[jc_abs] = new_max;
                }
            });
    });
}

// Write one GQA group's normalized output to its destination head.
अटल व्योम mkl_fa_normalize_शीर्ष(
    dpct::queue_ptr stream,
    भग्न * __restrict dst_batch,
    स्थिर भग्न * __restrict VKQ_accum,
    स्थिर भग्न * __restrict KQ_योग,
    पूर्णांक iqh, पूर्णांक n_queries, पूर्णांक DV, पूर्णांक n_q_heads,
    int64_t src_offset, int64_t wg_आकार) {

    स्थिर int64_t wg = ((n_queries + wg_आकार - 1) / wg_आकार) * wg_आकार;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_आकार),
            [=](sycl::nd_item<1> item) {
                पूर्णांक jc = item.get_global_id(0);
                अगर (jc >= n_queries) वापस;

                पूर्णांक     ksum_idx = (पूर्णांक)(src_offset / DV) + jc;
                भग्न   inv_योग  = 1.0f / KQ_योग[ksum_idx];
                स्थिर भग्न * __restrict src = VKQ_accum
                    + src_offset + jc * (int64_t)DV;
                // Interleaved dst layout (matching TILE):
                // rows alternate between heads, then increment query.
                // offset = (query * n_q_heads + head) * DV
                भग्न * __restrict dst_row = dst_batch
                    + ((int64_t)jc * n_q_heads + iqh) * (int64_t)DV;

                क्रम (पूर्णांक v = 0; v < DV; v++) {
                    dst_row[v] = src[v] * inv_योग;
                }
            });
    });
}

// ---------------------------------------------------------------------------
// Per-chunk dequant
//
// Rather than dequantizing all of K/V up front (footprint scales with
// context), we dequant one KV-head chunk at a time into a dense
// [this_chunk x D] fp16 buffer (row-major, lda = D). The source address of
// element (head=ikvh, row=chunk_start+r, col=c) decomposes into independent
// linear terms head_off(ikvh) + row_off(chunk_start) + (r,c), so slicing a
// chunk is a clean pointer offset in every layout case. The true-Gemma-
// interleave vs padded-seq-view distinction is resolved once when the
// descriptor is built; slicing does not reintroduce it.
// ---------------------------------------------------------------------------
क्रमागत mkl_fa_kv_desc_mode {
    MKL_FA_KV_MODE_F16_DENSE       = 0,
    MKL_FA_KV_MODE_F16_INTERLEAVED = 1,
    MKL_FA_KV_MODE_QUANT_CONTIG    = 2,
    MKL_FA_KV_MODE_QUANT_NC        = 3,
};

काष्ठा mkl_fa_kv_desc {
    स्थिर अक्षर *         तथ्य = nullptr;
    ggml_type            type = GGML_TYPE_F16;
    int64_t              D    = 0;      // ne[0]
    int64_t              nb1  = 0;      // byte stride, seq dim
    int64_t              nb2  = 0;      // byte stride, head dim
    mkl_fa_kv_desc_mode  mode = MKL_FA_KV_MODE_F16_DENSE;
    int64_t              ts   = 0;      // type size (mode 3 base offset)
    int64_t              s01  = 0;      // nc row stride in blocks (mode 3)
    int64_t              s02  = 0;      // nc head stride in blocks (mode 3)
};

अटल mkl_fa_kv_desc mkl_fa_make_desc(स्थिर ggml_tensor * T, bool interleaved, पूर्णांक n_kv_heads) {
    mkl_fa_kv_desc d;
    d.तथ्य = (स्थिर अक्षर *)T->तथ्य;
    d.type = T->type;
    d.D    = T->ne[0];
    d.nb1  = (int64_t)T->nb[1];
    d.nb2  = (int64_t)T->nb[2];
    d.ts   = (int64_t)ggml_type_आकार(T->type);

    अगर (T->type == GGML_TYPE_F16) {
        // MLA's V cache is a 512-wide view of 576-wide K rows. Treat any
        // padded row stride as strided even when there is only one KV head.
        d.mode = interleaved || d.nb1 != d.D * (int64_t)माप(sycl::half)
            ? MKL_FA_KV_MODE_F16_INTERLEAVED : MKL_FA_KV_MODE_F16_DENSE;
    } अन्यथा अगर (ggml_is_contiguously_allocated(T) && !interleaved) {
        d.mode = MKL_FA_KV_MODE_QUANT_CONTIG;
    } अन्यथा {
        d.mode = MKL_FA_KV_MODE_QUANT_NC;
        स्थिर int64_t bs          = (int64_t)ggml_blck_आकार(T->type);
        स्थिर int64_t blk_per_row = T->ne[0] / bs;
        // True Gemma interleave packs heads within a row (nb[2] < ne[1]*nb[1])
        // → reconstruct physical strides. Padded seq-views (nb[2] > ne[1]*nb[1])
        // already have correct physical strides.
        स्थिर bool gemma = interleaved &&
            ((int64_t)T->nb[2] < (int64_t)T->ne[1] * (int64_t)T->nb[1]);
        अगर (gemma) {
            d.s01 = (int64_t)n_kv_heads * blk_per_row;
            d.s02 = blk_per_row;
        } अन्यथा {
            d.s01 = d.nb1 / d.ts;
            d.s02 = d.nb2 / d.ts;
        }
    }
    वापस d;
}

// Dequant one KV-head chunk into a dense [this_chunk x D] fp16 buffer.
अटल व्योम mkl_fa_dequant_chunk(
    dpct::queue_ptr stream, स्थिर mkl_fa_kv_desc & d, ggml_tensor * dst_ctx,
    sycl::half * out, पूर्णांक ikvh, पूर्णांक chunk_start, पूर्णांक this_chunk) {

    स्थिर int64_t D = d.D;
    चयन (d.mode) {
        हाल MKL_FA_KV_MODE_F16_DENSE: {
            स्थिर अक्षर * base = d.तथ्य + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1;
            stream->स_नकल(out, base, (माप_प्रकार)this_chunk * D * माप(sycl::half));
            अवरोध;
        }
        हाल MKL_FA_KV_MODE_F16_INTERLEAVED: {
            स्थिर अक्षर * base = d.तथ्य + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1;
            स्थिर int64_t row_halfs = d.nb1 / (int64_t)माप(sycl::half);
            स्थिर sycl::half * src = (स्थिर sycl::half *)base;
            stream->parallel_for(
                sycl::range<2>((माप_प्रकार)this_chunk, (माप_प्रकार)D),
                [=](sycl::item<2> it) {
                    int64_t r = it.get_id(0);
                    int64_t c = it.get_id(1);
                    out[r * D + c] = src[r * row_halfs + c];
                });
            अवरोध;
        }
        हाल MKL_FA_KV_MODE_QUANT_CONTIG: {
            स्थिर अक्षर * base = d.तथ्य + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1;
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(d.type, dst_ctx);
            to_fp16(base, out, (int64_t)this_chunk * D, stream);
            अवरोध;
        }
        शेष: {  // MKL_FA_KV_MODE_QUANT_NC
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(d.type);
            स्थिर int64_t base_blocks = (int64_t)ikvh * d.s02
                + (int64_t)chunk_start * d.s01;
            स्थिर अक्षर * base = d.तथ्य + base_blocks * d.ts;
            // ne02 = ne03 = 1 → s02/s03 inert; head+chunk offset carried by base.
            to_fp16(base, out, D, this_chunk, 1, 1, d.s01, d.s02, d.s02, stream);
            अवरोध;
        }
    }
}

// ---------------------------------------------------------------------------
// MKL Flash Attention orchestrator
//
// Pipeline: dequantize K/V → for each KV head:
//   pack GQA Q heads → MKL GEMM KQ → online softmax →
//   MKL GEMM VKQ → accumulate → normalize → scatter to dst
// ---------------------------------------------------------------------------
व्योम ggml_sycl_flash_attn_ext_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    स्थिर ggml_tensor * Q    = dst->src[0];
    स्थिर ggml_tensor * K    = dst->src[1];
    स्थिर ggml_tensor * V    = dst->src[2];
    स्थिर ggml_tensor * mask = dst->src[3];
    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    // --- Op params ---
    भग्न scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    स_नकल(&scale,         (स्थिर भग्न *)KQV->op_params + 0, माप(भग्न));
    स_नकल(&max_bias,      (स्थिर भग्न *)KQV->op_params + 1, माप(भग्न));
    स_नकल(&logit_softcap, (स्थिर भग्न *)KQV->op_params + 2, माप(भग्न));

    स्थिर भग्न q_scale = scale;

    // --- Dimensions ---
    स्थिर पूर्णांक DKQ        = (पूर्णांक)K->ne[0];
    स्थिर पूर्णांक DV         = (पूर्णांक)V->ne[0];
    स्थिर पूर्णांक n_queries  = (पूर्णांक)Q->ne[1];
    स्थिर पूर्णांक n_q_heads  = (पूर्णांक)Q->ne[2];
    स्थिर पूर्णांक n_kv_heads = (पूर्णांक)K->ne[2];
    स्थिर पूर्णांक n_batch    = (पूर्णांक)Q->ne[3];
    स्थिर पूर्णांक n_kv       = (पूर्णांक)K->ne[1];
    स्थिर पूर्णांक gqa_ratio  = n_q_heads / n_kv_heads;
    स्थिर पूर्णांक n_query_rows = n_queries * gqa_ratio;

    GGML_ASSERT(n_q_heads % n_kv_heads == 0);
    GGML_ASSERT(max_bias == 0.0f);  // ALiBi not supported
    GGML_ASSERT(Q->ne[3] == K->ne[3] || K->ne[3] == 1);

    स्थिर पूर्णांक chunk_आकार = std::min(MKL_FA_CHUNK_आकार_KV, n_kv);

    // Query rows are processed in tiles of q_tile_rows so the score buffers
    // (KQ_f32/S_f16 = q_tile_rows * chunk_size) stay bounded regardless of
    // batch size. n_query_rows <= Q_TILE is a single tile (no extra work).
    अटल पूर्णांक q_tile_env = ggml_sycl_get_env("GGML_SYCL_MKL_FA_Q_TILE", MKL_FA_Q_TILE);
    स्थिर पूर्णांक q_tile_rows = std::max(1, std::min(q_tile_env, n_query_rows));

    स्थिर int64_t wg_आकार = MKL_FA_WG_आकार;

    // --- Debug output (gated by GGML_SYCL_MKL_FA_DEBUG=1) ---
    अटल पूर्णांक mkl_call_गिनती = 0;
    mkl_call_गिनती++;
    अटल पूर्णांक mkl_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    स्थिर bool do_print = (mkl_debug == 1);

    स्थिर int64_t q_row_stride  = Q->nb[1] / माप(भग्न);
    स्थिर int64_t q_शीर्ष_stride = Q->nb[2] / माप(भग्न);

    // Alias the dequantized buffers only when K and V expose the same values.
    // MLA V is a narrower view of K and needs its own strided dequantization.
    स्थिर bool V_is_K_view = V->ne[0] == K->ne[0] && V->view_src
        && (V->view_src == K || (V->view_src == K->view_src
            && V->view_offs == K->view_offs));

    // Early interleaved detection for debug output.
    // True interleaved detection happens after dequant (nb12_fp16 == nb11_fp16),
    // but we can pre-detect on the original tensor strides.
    स्थिर bool k_early_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]);
    स्थिर bool v_early_interleaved =
        !V_is_K_view && ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]);

    अगर (do_print) {
        GGML_LOG_INFO("[MKL-FA] #%d D=%d DV=%d n_q=%d n_kv=%d "
                "n_qh=%d n_kvh=%d gqa=%d batch=%d K=%s V=%s "
                "chunk=%d buf=%.1fMB%s%s\n",
                mkl_call_गिनती, DKQ, DV, n_queries, n_kv,
                n_q_heads, n_kv_heads, gqa_ratio, n_batch,
                ggml_type_name(K->type), ggml_type_name(V->type),
                chunk_आकार,
                (द्विगुन)((int64_t)n_query_rows * chunk_आकार * माप(भग्न))
                    / (1024.0 * 1024.0),
                k_early_interleaved ? " K_ILV" : "",
                v_early_interleaved ? " V_ILV" : "");
        GGML_LOG_INFO("[MKL-FA] #%d Q-nb1=%lld Q-nb2=%lld "
                "q_rs=%lld q_hs=%lld dst_rs=%lld dst_hs=%lld\n",
                mkl_call_गिनती,
                (दीर्घ दीर्घ)Q->nb[1], (दीर्घ दीर्घ)Q->nb[2],
                (दीर्घ दीर्घ)q_row_stride, (दीर्घ दीर्घ)q_शीर्ष_stride,
                (दीर्घ दीर्घ)(KQV->nb[1] / माप(भग्न)),
                (दीर्घ दीर्घ)(KQV->nb[2] / माप(भग्न)));
    }

    // --- Stream and allocators ---
    dpct::queue_ptr stream = ctx.stream();

#घोषणा MKL_TAKE_TIME(t0)  स्वतः t0 = std::chrono::steady_clock::now()
#घोषणा MKL_ACCUM(acc, t0) करो { अगर (do_print) { \
    acc += (int64_t)std::chrono::duration_cast \
    <std::chrono::microseconds>(std::chrono::steady_clock::now() - (t0)).count(); \
} } जबतक(0)

    int64_t gemm_kq_time_us  = 0;
    int64_t gemm_vkq_time_us = 0;
    int64_t softmax_time_us  = 0;
    int64_t dequant_time_us  = 0;

    MKL_TAKE_TIME(t_deq);

    // --- K/V dequant descriptors ---
    // Dequant is done per-chunk inside the KV loop (footprint independent of
    // context). Output is always dense row-major fp16 [this_chunk x D], lda=D.
    // Interleaved detection: ne[1]*nb[1] != nb[2] means heads are interleaved.
    स्थिर bool k_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
    स्थिर bool v_interleaved =
        ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;

    स्थिर mkl_fa_kv_desc K_desc = mkl_fa_make_desc(K, k_interleaved, n_kv_heads);
    स्थिर mkl_fa_kv_desc V_desc = V_is_K_view
        ? K_desc : mkl_fa_make_desc(V, v_interleaved, n_kv_heads);

    MKL_ACCUM(dequant_time_us, t_deq);

    // --- Resolve mask pointers ---
    int64_t mask_शीर्ष_stride = 0;
    int64_t mask_row_stride  = 0;
    पूर्णांक     mask_n_heads     = 0;

    अगर (mask) {
        // Use actual fp16 device size (2 bytes), NOT sizeof(sycl::half)
        // which may be 4 on the host in oneAPI.
        mask_शीर्ष_stride = mask->nb[2] / 2;
        mask_row_stride  = mask->nb[1] / 2;
        mask_n_heads     = (पूर्णांक)mask->ne[2];
    }

    // --- Allocate intermediates from pool ---
    ggml_sycl_pool & pool = ctx.pool();

    ggml_sycl_pool_alloc<भग्न>      KQ_f32(pool);      // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<sycl::half> S_f16(pool);       // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<भग्न>      VKQ_chunk(pool);   // [q_tile_rows x DV]
    ggml_sycl_pool_alloc<भग्न>      VKQ_accum(pool);   // [n_query_rows x DV] (full)
    ggml_sycl_pool_alloc<भग्न>      KQ_max(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<भग्न>      KQ_योग(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<sycl::half> Q_शीर्ष_f16(pool);  // [n_query_rows x DKQ] (full)
    ggml_sycl_pool_alloc<sycl::half> K_chunk_f16(pool); // [chunk x DKQ] (per-chunk dequant)
    ggml_sycl_pool_alloc<sycl::half> V_chunk_f16(pool); // [chunk x DV] (per-chunk dequant)

    KQ_f32.alloc((माप_प्रकार)q_tile_rows * chunk_आकार);
    S_f16.alloc((माप_प्रकार)q_tile_rows * chunk_आकार);
    VKQ_chunk.alloc((माप_प्रकार)q_tile_rows * DV);
    VKQ_accum.alloc((माप_प्रकार)n_query_rows * DV);
    KQ_max.alloc(n_query_rows);
    KQ_योग.alloc(n_query_rows);
    Q_शीर्ष_f16.alloc((माप_प्रकार)n_query_rows * DKQ);
    K_chunk_f16.alloc((माप_प्रकार)chunk_आकार * DKQ);

    sycl::half * V_chunk_f16_ptr;
    अगर (V_is_K_view) {
        V_chunk_f16_ptr = K_chunk_f16.ptr;   // V aliases K (DV == DKQ)
    } अन्यथा {
        V_chunk_f16.alloc((माप_प्रकार)chunk_आकार * DV);
        V_chunk_f16_ptr = V_chunk_f16.ptr;
    }

    sycl::half * Q_शीर्ष_f16_ptr  = Q_शीर्ष_f16.ptr;
    भग्न      * KQ_f32_ptr      = KQ_f32.ptr;
    sycl::half * S_f16_ptr       = S_f16.ptr;
    भग्न      * VKQ_chunk_ptr   = VKQ_chunk.ptr;
    भग्न      * VKQ_accum_ptr   = VKQ_accum.ptr;
    भग्न      * KQ_max_ptr      = KQ_max.ptr;
    भग्न      * KQ_योग_ptr      = KQ_योग.ptr;
    sycl::half * K_chunk_f16_ptr = K_chunk_f16.ptr;

    स्थिर भग्न alpha = 1.0f;
    स्थिर भग्न beta  = 0.0f;

    क्रम (पूर्णांक ib = 0; ib < n_batch; ib++) {
        स्थिर भग्न * Q_batch = (स्थिर भग्न *)Q->तथ्य
            + ib * (Q->nb[3] / माप(भग्न));
        भग्न * dst_batch = (भग्न *)KQV->तथ्य
            + ib * (KQV->nb[3] / माप(भग्न));

        स्थिर sycl::half * mask_batch = nullptr;
        अगर (mask) {
            पूर्णांक m_batch = (mask->ne[3] > 1) ? ib : 0;
            mask_batch = (स्थिर sycl::half *)mask->तथ्य
                + m_batch * (mask->nb[3] / 2);  // 2 = actual fp16 device size
        }

        क्रम (पूर्णांक ikvh = 0; ikvh < n_kv_heads; ikvh++) {
            पूर्णांक kvh_base_शीर्ष = ikvh * gqa_ratio;

            // 1. Pack all GQA Q heads into fp16 (full n_query_rows)
            mkl_fa_pack_q_fp16(stream,
                Q_शीर्ष_f16_ptr, Q_batch,
                n_queries, DKQ,
                gqa_ratio, kvh_base_शीर्ष,
                q_scale, q_row_stride, q_शीर्ष_stride, wg_आकार);

            // 2. Initialize softmax state (full n_query_rows)
            mkl_fa_init_softmax_state(stream,
                KQ_max_ptr, KQ_योग_ptr, VKQ_accum_ptr,
                n_query_rows, DV, wg_आकार);

            // Sync before MKL GEMM (MKL may use an internal queue)
            stream->रुको();

            // 3. KV chunk loop (OUTER): dequant each chunk once, then tile queries.
            क्रम (पूर्णांक chunk_start = 0; chunk_start < n_kv; chunk_start += chunk_आकार) {
                पूर्णांक this_chunk = std::min(chunk_आकार, n_kv - chunk_start);

                // 3a. Dequant this KV chunk to dense fp16 (once per chunk)
                {
                    MKL_TAKE_TIME(t0);
                    mkl_fa_dequant_chunk(stream, K_desc, KQV,
                        K_chunk_f16_ptr, ikvh, chunk_start, this_chunk);
                    अगर (!V_is_K_view) {
                        mkl_fa_dequant_chunk(stream, V_desc, KQV,
                            V_chunk_f16_ptr, ikvh, chunk_start, this_chunk);
                    }
                    stream->रुको();  // dequant must be ready before MKL GEMM
                    MKL_ACCUM(dequant_time_us, t0);
                }

                // 3b. Query tile loop (INNER) — bounds KQ_f32/S_f16 footprint.
                क्रम (पूर्णांक q0 = 0; q0 < n_query_rows; q0 += q_tile_rows) {
                    पूर्णांक q_rows = std::min(q_tile_rows, n_query_rows - q0);

                    // GEMM: KQ = Q_tile × K_chunk^T
                    {
                        MKL_TAKE_TIME(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::trans, transpose::nontrans,
                            this_chunk, q_rows, DKQ,
                            alpha,
                            K_chunk_f16_ptr, DKQ,
                            Q_शीर्ष_f16_ptr + (int64_t)q0 * DKQ, DKQ,
                            beta,
                            KQ_f32_ptr, this_chunk);
                        प्रयास { ev.wait_and_throw(); } पकड़ो (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM KQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM KQ failed");
                        }
                        MKL_ACCUM(gemm_kq_time_us, t0);
                    }
                    // Online softmax over this chunk for this query tile
                    {
                        MKL_TAKE_TIME(t0);
                        mkl_fa_online_softmax_chunk(stream,
                            KQ_f32_ptr, S_f16_ptr,
                            KQ_max_ptr, KQ_योग_ptr, VKQ_accum_ptr,
                            q0, q_rows, n_queries, DV,
                            this_chunk, chunk_start,
                            kvh_base_शीर्ष,
                            mask_batch, mask_शीर्ष_stride,
                            mask_row_stride, mask_n_heads,
                            logit_softcap, wg_आकार);
                        stream->रुको();  // S_f16 must be ready for GEMM
                        MKL_ACCUM(softmax_time_us, t0);
                    }

                    // GEMM: VKQ_chunk = S × V_chunk
                    {
                        MKL_TAKE_TIME(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::nontrans, transpose::nontrans,
                            DV, q_rows, this_chunk,
                            alpha,
                            V_chunk_f16_ptr, DV,
                            S_f16_ptr, this_chunk,
                            beta,
                            VKQ_chunk_ptr, DV);
                        प्रयास { ev.wait_and_throw(); } पकड़ो (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM VKQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM VKQ failed");
                        }
                        MKL_ACCUM(gemm_vkq_time_us, t0);
                    }
                    // VKQ_accum[q0..] += VKQ_chunk
                    {
                        स्थिर int64_t n_योग = (int64_t)q_rows * DV;
                        स्थिर int64_t wg = ((n_योग + wg_आकार - 1) / wg_आकार)
                            * wg_आकार;
                        भग्न * accum = VKQ_accum_ptr + (int64_t)q0 * DV;
                        stream->submit([&](sycl::handler & cgh) {
                            cgh.parallel_for(sycl::nd_range<1>(wg, wg_आकार),
                                [=](sycl::nd_item<1> item) {
                                    int64_t i = item.get_global_id(0);
                                    अगर (i < n_योग) {
                                        accum[i] += VKQ_chunk_ptr[i];
                                    }
                                });
                        });
                    }
                }
            }

            // 4. Normalize and scatter each GQA head to dst
            क्रम (पूर्णांक iqg = 0; iqg < gqa_ratio; iqg++) {
                पूर्णांक     iqh        = kvh_base_शीर्ष + iqg;
                int64_t src_offset = (int64_t)iqg * n_queries * DV;
                mkl_fa_normalize_शीर्ष(stream,
                    dst_batch, VKQ_accum_ptr, KQ_योग_ptr,
                    iqh, n_queries, DV, n_q_heads,
                    src_offset, wg_आकार);
            }
        }
    }

#अघोषित MKL_TAKE_TIME
#अघोषित MKL_ACCUM

    अगर (do_print) {
        स्थिर int64_t v_chunk_elems = V_is_K_view ? 0 : (int64_t)chunk_आकार * DV;
        द्विगुन योग_mb = (द्विगुन)(
            (int64_t)q_tile_rows * chunk_आकार * माप(भग्न)      // KQ_f32
          + (int64_t)q_tile_rows * chunk_आकार * माप(sycl::half) // S_f16
          + (int64_t)q_tile_rows * DV * माप(भग्न)              // VKQ_chunk
          + (int64_t)n_query_rows * DV * माप(भग्न)             // VKQ_accum
          + (int64_t)n_query_rows * माप(भग्न)                  // KQ_max
          + (int64_t)n_query_rows * माप(भग्न)                  // KQ_sum
          + (int64_t)n_query_rows * DKQ * माप(sycl::half)       // Q_head_f16
          + (int64_t)chunk_आकार * DKQ * माप(sycl::half)         // K_chunk_f16
          + v_chunk_elems * (int64_t)माप(sycl::half)            // V_chunk_f16
        ) / (1024.0 * 1024.0);
        GGML_LOG_INFO("[MKL-FA] #%d n_kv=%d n_q=%d q_tile=%d time_us: "
                "dequant=%lld GEMM_KQ=%lld softmax=%lld GEMM_VKQ=%lld "
                "buf_mb=%.1f\n",
                mkl_call_गिनती, n_kv, n_queries, q_tile_rows,
                (दीर्घ दीर्घ)dequant_time_us,
                (दीर्घ दीर्घ)gemm_kq_time_us,
                (दीर्घ दीर्घ)softmax_time_us,
                (दीर्घ दीर्घ)gemm_vkq_time_us,
                योग_mb);
    }
}
