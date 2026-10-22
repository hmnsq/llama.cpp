// Flash attention via oneMKL GEMM (XMX-accelerated).
// Uses column_major::gemm for Q*K^T and S*V matmuls
// with an online softmax SYCL kernel.
//
// All GQA query heads sharing a KV head are batched into single
// GEMM calls, amortizing MKL launch overhead across K and V reuse.
//

#ضمن "common.hpp"
#ضمن "fattn-common.hpp"
#ضمن "convert.hpp"
#ضمن "fattn.hpp"

#ضمن <oneapi/mkl.hpp>
#ضمن <cstdio>
#ضمن <chrono>

#تعریف_کرو MKL_FA_CHUNK_ناپ_KV 8192

// Number of query rows processed per tile. The score buffers (KQ_f32, S_f16)
// are sized q_tile_rows * chunk_size, so this bounds their footprint
// regardless of batch size (n_query_rows = n_queries * gqa_ratio). A typical
// single-ubatch prefill (e.g. ubatch 1024 * gqa 8 = 8192 rows) is exactly one
// tile, so it runs with no extra iterations. Larger batches tile and stay
// bounded. Override with GGML_SYCL_MKL_FA_Q_TILE.
#تعریف_کرو MKL_FA_Q_TILE 8192

#تعریف_کرو MKL_FA_WG_ناپ 256

ورتوں oneapi::mkl::transpose;
ورتوں oneapi::mkl::blas::کالم_major::gemm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Pack all GQA Q heads for one KV head into fp16, applying q_scale.
// Launches one kernel per GQA group — each kernel copies exactly
// n_queries * DKQ elements using the per-group dst offset and
// per-head source stride.
ساكن فراغ mkl_fa_pack_q_fp16(
    dpct::قطار_ptr stream,
    sycl::half * __restrict dst,
    ثابت عائم * __restrict q_src,
    صحيح n_queries, صحيح DKQ,
    صحيح gqa_تناسب, صحيح kvh_ادھار_سر,
    عائم q_scale, int64_t q_صف_stride, int64_t q_سر_stride,
    int64_t wg_ناپ) {

    لكل (صحيح iqg = 0; iqg < gqa_تناسب; iqg++) {
        صحيح     iqh       = kvh_ادھار_سر + iqg;
        sycl::half * dst_g = dst + (int64_t)iqg * n_queries * DKQ;

        ثابت int64_t n_elem = (int64_t)n_queries * DKQ;
        ثابت int64_t wg = ((n_elem + wg_ناپ - 1) / wg_ناپ) * wg_ناپ;

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(wg, wg_ناپ),
                [=](sycl::nd_شے<1> شے) {
                    int64_t e = شے.get_global_id(0);
                    إذا (e >= n_elem) أرجع;

                    int64_t q = e / DKQ;
                    int64_t d = e - q * DKQ;

                    // Stride-aware source offset: handles permuted,
                    // sliced, or contiguous Q tensor layouts.
                    int64_t src_off = d
                        + q * q_صف_stride
                        + (int64_t)iqh * q_سر_stride;

                    dst_g[e] = sycl::half(
                        q_src[src_off] * q_scale);
                });
        });
    }
}

// Zero-initialize the online softmax state arrays.
// KQ_max → -inf, KQ_sum → 0, VKQ_accum → 0.
// Merged into one kernel to avoid per-array launch overhead.
ساكن فراغ mkl_fa_init_softmax_state(
    dpct::قطار_ptr stream,
    عائم * kmax, عائم * ksum, عائم * vacc,
    صحيح n_query_صفاں, صحيح DV, int64_t wg_ناپ) {

    ثابت عائم    neg_inf   = -1e30f;
    ثابت int64_t  n_maxsum  = n_query_صفاں;
    ثابت int64_t  n_vacc    = (int64_t)n_query_صفاں * DV;
    ثابت int64_t  کل     = (n_vacc > n_maxsum) ? n_vacc : n_maxsum;
    ثابت int64_t  wg = ((کل + wg_ناپ - 1) / wg_ناپ) * wg_ناپ;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_ناپ),
            [=](sycl::nd_شے<1> شے) {
                int64_t i = شے.get_global_id(0);
                إذا (i < n_maxsum) {
                    kmax[i] = neg_inf;
                    ksum[i] = 0.0f;
                }
                إذا (i < n_vacc) {
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
ساكن فراغ mkl_fa_online_softmax_chunk(
    dpct::قطار_ptr stream,
    عائم * __restrict KQ_f32,
    sycl::half * __restrict S_f16,
    عائم * __restrict KQ_max,
    عائم * __restrict KQ_جوڑ,
    عائم * __restrict VKQ_accum,
    صحيح q0, صحيح q_صفاں, صحيح n_queries, صحيح DV,
    صحيح chunk_ناپ, صحيح chunk_شروع,
    صحيح kvh_سر,
    ثابت sycl::half * نقاب_انکڑے, int64_t نقاب_سر_stride,
    int64_t نقاب_صف_stride, صحيح نقاب_n_heads,
    عائم logit_softcap, int64_t wg_ناپ) {

    // One work-group per query row: exactly q_rows groups of wg_size
    // items. q_rows * wg_size is already a multiple of wg_size, so unlike
    // the one-item-per-row kernels there is no round-up / tail guard.
    ثابت int64_t wg         = q_صفاں * wg_ناپ;
    ثابت صحيح     local_ناپ = (صحيح) wg_ناپ;  // stride in the loops below
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_ناپ),
            [=](sycl::nd_شے<1> شے) {
                ثابت صحيح local_id = (صحيح)شے.get_local_id(0);
                ثابت صحيح صف = (صحيح)شے.get_گروہ(0); // tile-relative
                ثابت صحيح jc_abs    = q0 + صف;
                ثابت صحيح gqa_گروہ = jc_abs / n_queries;
                ثابت صحيح q_صف     = jc_abs % n_queries;
                // Score buffers are tile-local (relative index).
                ثابت عائم * __restrict KQ_صف = KQ_f32
                    + صف * (int64_t)chunk_ناپ;
                sycl::half * __restrict S_صف = S_f16
                    + صف * (int64_t)chunk_ناپ;
                // Persistent accumulator is full-sized (absolute index).
                عائم * __restrict vkq = VKQ_accum
                    + jc_abs * (int64_t)DV;
                ثابت sycl::half * نقاب_h = خالی_سوچک;
                int64_t m_stride = 0;
                إذا (نقاب_انکڑے) {
                    صحيح m_سر = (نقاب_n_heads > 1)
                        ? (kvh_سر + gqa_گروہ) : 0;
                    نقاب_h   = نقاب_انکڑے + (int64_t)m_سر * نقاب_سر_stride;
                    m_stride = نقاب_صف_stride;
                }
                // Score at chunk offset i — original per-element math.
                خودکار score = [&](صحيح i) {
                    عائم s = KQ_صف[i];
                    إذا (logit_softcap != 0.0f) {
                        s = logit_softcap * sycl::tanh(s);
                    }
                    إذا (نقاب_h) {
                        s += (عائم)نقاب_h[q_صف * m_stride
                            + (chunk_شروع + i)];
                    }
                    أرجع s;
                };
                // Pass 1: strided (coalesced) row-wise local maximum.
                عائم local_max = -1e30f;
                لكل (صحيح i = local_id; i < chunk_ناپ; i += local_ناپ) {
                    عائم s = score(i);
                    إذا (s > local_max) local_max = s;
                }
                ثابت عائم final_local_max = sycl::reduce_over_گروہ(
                    شے.get_گروہ(), local_max, sycl::ودھ_توں_ودھ<عائم>());
                // Rescale previous accumulator by exp(old_max - new_max)
                عائم old_max = KQ_max[jc_abs];
                عائم new_max = (old_max > final_local_max) ? old_max : final_local_max;
                عائم rescale = (old_max < -1e29f) ? 1.0f
                    : sycl::native::قوت_نما(old_max - new_max);
                لكل (صحيح v = local_id; v < DV; v += local_ناپ) {
                    vkq[v] *= rescale;
                }
                // Pass 2: softmax numerators, strided; S row written once.
                عائم local_جوڑ = 0.0f;
                لكل (صحيح i = local_id; i < chunk_ناپ; i += local_ناپ) {
                    عائم s = score(i);
                    عائم val = sycl::native::قوت_نما(s - new_max);
                    S_صف[i] = sycl::half(val);
                    local_جوڑ += val;
                }
                ثابت عائم کل_جوڑ = sycl::reduce_over_گروہ(
                    شے.get_گروہ(), local_جوڑ, sycl::plus<عائم>());
                إذا (local_id == 0) {
                    KQ_جوڑ[jc_abs] = KQ_جوڑ[jc_abs] * rescale + کل_جوڑ;
                    KQ_max[jc_abs] = new_max;
                }
            });
    });
}

// Write one GQA group's normalized output to its destination head.
ساكن فراغ mkl_fa_normalize_سر(
    dpct::قطار_ptr stream,
    عائم * __restrict dst_batch,
    ثابت عائم * __restrict VKQ_accum,
    ثابت عائم * __restrict KQ_جوڑ,
    صحيح iqh, صحيح n_queries, صحيح DV, صحيح n_q_heads,
    int64_t src_وتھ, int64_t wg_ناپ) {

    ثابت int64_t wg = ((n_queries + wg_ناپ - 1) / wg_ناپ) * wg_ناپ;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_ناپ),
            [=](sycl::nd_شے<1> شے) {
                صحيح jc = شے.get_global_id(0);
                إذا (jc >= n_queries) أرجع;

                صحيح     ksum_idx = (صحيح)(src_وتھ / DV) + jc;
                عائم   inv_جوڑ  = 1.0f / KQ_جوڑ[ksum_idx];
                ثابت عائم * __restrict src = VKQ_accum
                    + src_وتھ + jc * (int64_t)DV;
                // Interleaved dst layout (matching TILE):
                // rows alternate between heads, then increment query.
                // offset = (query * n_q_heads + head) * DV
                عائم * __restrict dst_صف = dst_batch
                    + ((int64_t)jc * n_q_heads + iqh) * (int64_t)DV;

                لكل (صحيح v = 0; v < DV; v++) {
                    dst_صف[v] = src[v] * inv_جوڑ;
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
تعداد mkl_fa_kv_desc_منوال {
    MKL_FA_KV_منوال_F16_DENSE       = 0,
    MKL_FA_KV_منوال_F16_INTERLEAVED = 1,
    MKL_FA_KV_منوال_QUANT_CONTIG    = 2,
    MKL_FA_KV_منوال_QUANT_NC        = 3,
};

بنية mkl_fa_kv_desc {
    ثابت حرف *         انکڑے = خالی_سوچک;
    ggml_type            type = GGML_TYPE_F16;
    int64_t              D    = 0;      // ne[0]
    int64_t              nb1  = 0;      // byte stride, seq dim
    int64_t              nb2  = 0;      // byte stride, head dim
    mkl_fa_kv_desc_منوال  mode = MKL_FA_KV_منوال_F16_DENSE;
    int64_t              ts   = 0;      // type size (mode 3 base offset)
    int64_t              s01  = 0;      // nc row stride in blocks (mode 3)
    int64_t              s02  = 0;      // nc head stride in blocks (mode 3)
};

ساكن mkl_fa_kv_desc mkl_fa_make_desc(ثابت ggml_tensor * T, منطقي interleaved, صحيح n_kv_heads) {
    mkl_fa_kv_desc d;
    d.انکڑے = (ثابت حرف *)T->انکڑے;
    d.type = T->type;
    d.D    = T->ne[0];
    d.nb1  = (int64_t)T->nb[1];
    d.nb2  = (int64_t)T->nb[2];
    d.ts   = (int64_t)ggml_type_ناپ(T->type);

    إذا (T->type == GGML_TYPE_F16) {
        // MLA's V cache is a 512-wide view of 576-wide K rows. Treat any
        // padded row stride as strided even when there is only one KV head.
        d.mode = interleaved || d.nb1 != d.D * (int64_t)حجم(sycl::half)
            ? MKL_FA_KV_منوال_F16_INTERLEAVED : MKL_FA_KV_منوال_F16_DENSE;
    } وإلا إذا (ggml_is_contiguously_allocated(T) && !interleaved) {
        d.mode = MKL_FA_KV_منوال_QUANT_CONTIG;
    } وإلا {
        d.mode = MKL_FA_KV_منوال_QUANT_NC;
        ثابت int64_t bs          = (int64_t)ggml_blck_ناپ(T->type);
        ثابت int64_t blk_فی_صف = T->ne[0] / bs;
        // True Gemma interleave packs heads within a row (nb[2] < ne[1]*nb[1])
        // → reconstruct physical strides. Padded seq-views (nb[2] > ne[1]*nb[1])
        // already have correct physical strides.
        ثابت منطقي gemma = interleaved &&
            ((int64_t)T->nb[2] < (int64_t)T->ne[1] * (int64_t)T->nb[1]);
        إذا (gemma) {
            d.s01 = (int64_t)n_kv_heads * blk_فی_صف;
            d.s02 = blk_فی_صف;
        } وإلا {
            d.s01 = d.nb1 / d.ts;
            d.s02 = d.nb2 / d.ts;
        }
    }
    أرجع d;
}

// Dequant one KV-head chunk into a dense [this_chunk x D] fp16 buffer.
ساكن فراغ mkl_fa_dequant_chunk(
    dpct::قطار_ptr stream, ثابت mkl_fa_kv_desc & d, ggml_tensor * dst_ctx,
    sycl::half * باہرلا, صحيح ikvh, صحيح chunk_شروع, صحيح this_chunk) {

    ثابت int64_t D = d.D;
    بدل (d.mode) {
        حالة MKL_FA_KV_منوال_F16_DENSE: {
            ثابت حرف * ادھار = d.انکڑے + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_شروع * d.nb1;
            stream->حافظہ_نقل(باہرلا, ادھار, (size_t)this_chunk * D * حجم(sycl::half));
            اكسر;
        }
        حالة MKL_FA_KV_منوال_F16_INTERLEAVED: {
            ثابت حرف * ادھار = d.انکڑے + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_شروع * d.nb1;
            ثابت int64_t صف_halfs = d.nb1 / (int64_t)حجم(sycl::half);
            ثابت sycl::half * src = (ثابت sycl::half *)ادھار;
            stream->parallel_for(
                sycl::range<2>((size_t)this_chunk, (size_t)D),
                [=](sycl::شے<2> it) {
                    int64_t r = it.get_id(0);
                    int64_t c = it.get_id(1);
                    باہرلا[r * D + c] = src[r * صف_halfs + c];
                });
            اكسر;
        }
        حالة MKL_FA_KV_منوال_QUANT_CONTIG: {
            ثابت حرف * ادھار = d.انکڑے + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_شروع * d.nb1;
            منزل_fp16_sycl_t منزل_fp16 = ggml_get_منزل_fp16_sycl(d.type, dst_ctx);
            منزل_fp16(ادھار, باہرلا, (int64_t)this_chunk * D, stream);
            اكسر;
        }
        افتراضي: {  // MKL_FA_KV_MODE_QUANT_NC
            منزل_fp16_nc_sycl_t منزل_fp16 = ggml_get_منزل_fp16_nc_sycl(d.type);
            ثابت int64_t ادھار_بلاکاں = (int64_t)ikvh * d.s02
                + (int64_t)chunk_شروع * d.s01;
            ثابت حرف * ادھار = d.انکڑے + ادھار_بلاکاں * d.ts;
            // ne02 = ne03 = 1 → s02/s03 inert; head+chunk offset carried by base.
            منزل_fp16(ادھار, باہرلا, D, this_chunk, 1, 1, d.s01, d.s02, d.s02, stream);
            اكسر;
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
فراغ ggml_sycl_flash_attn_ext_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ثابت ggml_tensor * Q    = dst->src[0];
    ثابت ggml_tensor * K    = dst->src[1];
    ثابت ggml_tensor * V    = dst->src[2];
    ثابت ggml_tensor * نقاب = dst->src[3];
    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    // --- Op params ---
    عائم scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    حافظہ_نقل(&scale,         (ثابت عائم *)KQV->op_params + 0, حجم(عائم));
    حافظہ_نقل(&max_bias,      (ثابت عائم *)KQV->op_params + 1, حجم(عائم));
    حافظہ_نقل(&logit_softcap, (ثابت عائم *)KQV->op_params + 2, حجم(عائم));

    ثابت عائم q_scale = scale;

    // --- Dimensions ---
    ثابت صحيح DKQ        = (صحيح)K->ne[0];
    ثابت صحيح DV         = (صحيح)V->ne[0];
    ثابت صحيح n_queries  = (صحيح)Q->ne[1];
    ثابت صحيح n_q_heads  = (صحيح)Q->ne[2];
    ثابت صحيح n_kv_heads = (صحيح)K->ne[2];
    ثابت صحيح n_batch    = (صحيح)Q->ne[3];
    ثابت صحيح n_kv       = (صحيح)K->ne[1];
    ثابت صحيح gqa_تناسب  = n_q_heads / n_kv_heads;
    ثابت صحيح n_query_صفاں = n_queries * gqa_تناسب;

    GGML_ASSERT(n_q_heads % n_kv_heads == 0);
    GGML_ASSERT(max_bias == 0.0f);  // ALiBi not supported
    GGML_ASSERT(Q->ne[3] == K->ne[3] || K->ne[3] == 1);

    ثابت صحيح chunk_ناپ = std::min(MKL_FA_CHUNK_ناپ_KV, n_kv);

    // Query rows are processed in tiles of q_tile_rows so the score buffers
    // (KQ_f32/S_f16 = q_tile_rows * chunk_size) stay bounded regardless of
    // batch size. n_query_rows <= Q_TILE is a single tile (no extra work).
    ساكن صحيح q_tile_env = ggml_sycl_get_env("GGML_SYCL_MKL_FA_Q_TILE", MKL_FA_Q_TILE);
    ثابت صحيح q_tile_صفاں = std::max(1, std::min(q_tile_env, n_query_صفاں));

    ثابت int64_t wg_ناپ = MKL_FA_WG_ناپ;

    // --- Debug output (gated by GGML_SYCL_MKL_FA_DEBUG=1) ---
    ساكن صحيح mkl_call_گنتی = 0;
    mkl_call_گنتی++;
    ساكن صحيح mkl_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    ثابت منطقي do_print = (mkl_debug == 1);

    ثابت int64_t q_صف_stride  = Q->nb[1] / حجم(عائم);
    ثابت int64_t q_سر_stride = Q->nb[2] / حجم(عائم);

    // Alias the dequantized buffers only when K and V expose the same values.
    // MLA V is a narrower view of K and needs its own strided dequantization.
    ثابت منطقي V_is_K_view = V->ne[0] == K->ne[0] && V->view_src
        && (V->view_src == K || (V->view_src == K->view_src
            && V->view_offs == K->view_offs));

    // Early interleaved detection for debug output.
    // True interleaved detection happens after dequant (nb12_fp16 == nb11_fp16),
    // but we can pre-detect on the original tensor strides.
    ثابت منطقي k_early_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]);
    ثابت منطقي v_early_interleaved =
        !V_is_K_view && ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]);

    إذا (do_print) {
        GGML_LOG_INFO("[MKL-FA] #%d D=%d DV=%d n_q=%d n_kv=%d "
                "n_qh=%d n_kvh=%d gqa=%d batch=%d K=%s V=%s "
                "chunk=%d buf=%.1fMB%s%s\n",
                mkl_call_گنتی, DKQ, DV, n_queries, n_kv,
                n_q_heads, n_kv_heads, gqa_تناسب, n_batch,
                ggml_type_name(K->type), ggml_type_name(V->type),
                chunk_ناپ,
                (مزدوج)((int64_t)n_query_صفاں * chunk_ناپ * حجم(عائم))
                    / (1024.0 * 1024.0),
                k_early_interleaved ? " K_ILV" : "",
                v_early_interleaved ? " V_ILV" : "");
        GGML_LOG_INFO("[MKL-FA] #%d Q-nb1=%lld Q-nb2=%lld "
                "q_rs=%lld q_hs=%lld dst_rs=%lld dst_hs=%lld\n",
                mkl_call_گنتی,
                (لما لما)Q->nb[1], (لما لما)Q->nb[2],
                (لما لما)q_صف_stride, (لما لما)q_سر_stride,
                (لما لما)(KQV->nb[1] / حجم(عائم)),
                (لما لما)(KQV->nb[2] / حجم(عائم)));
    }

    // --- Stream and allocators ---
    dpct::قطار_ptr stream = ctx.stream();

#تعریف_کرو MKL_TAKE_ویلا(t0)  خودکار t0 = std::chrono::steady_clock::now()
#تعریف_کرو MKL_ACCUM(acc, t0) افعل { إذا (do_print) { \
    acc += (int64_t)std::chrono::duration_cast \
    <std::chrono::microseconds>(std::chrono::steady_clock::now() - (t0)).count(); \
} } طالما(0)

    int64_t gemm_kq_ویلا_us  = 0;
    int64_t gemm_vkq_ویلا_us = 0;
    int64_t softmax_ویلا_us  = 0;
    int64_t dequant_ویلا_us  = 0;

    MKL_TAKE_ویلا(t_deq);

    // --- K/V dequant descriptors ---
    // Dequant is done per-chunk inside the KV loop (footprint independent of
    // context). Output is always dense row-major fp16 [this_chunk x D], lda=D.
    // Interleaved detection: ne[1]*nb[1] != nb[2] means heads are interleaved.
    ثابت منطقي k_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
    ثابت منطقي v_interleaved =
        ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;

    ثابت mkl_fa_kv_desc K_desc = mkl_fa_make_desc(K, k_interleaved, n_kv_heads);
    ثابت mkl_fa_kv_desc V_desc = V_is_K_view
        ? K_desc : mkl_fa_make_desc(V, v_interleaved, n_kv_heads);

    MKL_ACCUM(dequant_ویلا_us, t_deq);

    // --- Resolve mask pointers ---
    int64_t نقاب_سر_stride = 0;
    int64_t نقاب_صف_stride  = 0;
    صحيح     نقاب_n_heads     = 0;

    إذا (نقاب) {
        // Use actual fp16 device size (2 bytes), NOT sizeof(sycl::half)
        // which may be 4 on the host in oneAPI.
        نقاب_سر_stride = نقاب->nb[2] / 2;
        نقاب_صف_stride  = نقاب->nb[1] / 2;
        نقاب_n_heads     = (صحيح)نقاب->ne[2];
    }

    // --- Allocate intermediates from pool ---
    ggml_sycl_pool & pool = ctx.pool();

    ggml_sycl_pool_alloc<عائم>      KQ_f32(pool);      // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<sycl::half> S_f16(pool);       // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<عائم>      VKQ_chunk(pool);   // [q_tile_rows x DV]
    ggml_sycl_pool_alloc<عائم>      VKQ_accum(pool);   // [n_query_rows x DV] (full)
    ggml_sycl_pool_alloc<عائم>      KQ_max(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<عائم>      KQ_جوڑ(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<sycl::half> Q_سر_f16(pool);  // [n_query_rows x DKQ] (full)
    ggml_sycl_pool_alloc<sycl::half> K_chunk_f16(pool); // [chunk x DKQ] (per-chunk dequant)
    ggml_sycl_pool_alloc<sycl::half> V_chunk_f16(pool); // [chunk x DV] (per-chunk dequant)

    KQ_f32.alloc((size_t)q_tile_صفاں * chunk_ناپ);
    S_f16.alloc((size_t)q_tile_صفاں * chunk_ناپ);
    VKQ_chunk.alloc((size_t)q_tile_صفاں * DV);
    VKQ_accum.alloc((size_t)n_query_صفاں * DV);
    KQ_max.alloc(n_query_صفاں);
    KQ_جوڑ.alloc(n_query_صفاں);
    Q_سر_f16.alloc((size_t)n_query_صفاں * DKQ);
    K_chunk_f16.alloc((size_t)chunk_ناپ * DKQ);

    sycl::half * V_chunk_f16_ptr;
    إذا (V_is_K_view) {
        V_chunk_f16_ptr = K_chunk_f16.ptr;   // V aliases K (DV == DKQ)
    } وإلا {
        V_chunk_f16.alloc((size_t)chunk_ناپ * DV);
        V_chunk_f16_ptr = V_chunk_f16.ptr;
    }

    sycl::half * Q_سر_f16_ptr  = Q_سر_f16.ptr;
    عائم      * KQ_f32_ptr      = KQ_f32.ptr;
    sycl::half * S_f16_ptr       = S_f16.ptr;
    عائم      * VKQ_chunk_ptr   = VKQ_chunk.ptr;
    عائم      * VKQ_accum_ptr   = VKQ_accum.ptr;
    عائم      * KQ_max_ptr      = KQ_max.ptr;
    عائم      * KQ_جوڑ_ptr      = KQ_جوڑ.ptr;
    sycl::half * K_chunk_f16_ptr = K_chunk_f16.ptr;

    ثابت عائم alpha = 1.0f;
    ثابت عائم beta  = 0.0f;

    لكل (صحيح ib = 0; ib < n_batch; ib++) {
        ثابت عائم * Q_batch = (ثابت عائم *)Q->انکڑے
            + ib * (Q->nb[3] / حجم(عائم));
        عائم * dst_batch = (عائم *)KQV->انکڑے
            + ib * (KQV->nb[3] / حجم(عائم));

        ثابت sycl::half * نقاب_batch = خالی_سوچک;
        إذا (نقاب) {
            صحيح m_batch = (نقاب->ne[3] > 1) ? ib : 0;
            نقاب_batch = (ثابت sycl::half *)نقاب->انکڑے
                + m_batch * (نقاب->nb[3] / 2);  // 2 = actual fp16 device size
        }

        لكل (صحيح ikvh = 0; ikvh < n_kv_heads; ikvh++) {
            صحيح kvh_ادھار_سر = ikvh * gqa_تناسب;

            // 1. Pack all GQA Q heads into fp16 (full n_query_rows)
            mkl_fa_pack_q_fp16(stream,
                Q_سر_f16_ptr, Q_batch,
                n_queries, DKQ,
                gqa_تناسب, kvh_ادھار_سر,
                q_scale, q_صف_stride, q_سر_stride, wg_ناپ);

            // 2. Initialize softmax state (full n_query_rows)
            mkl_fa_init_softmax_state(stream,
                KQ_max_ptr, KQ_جوڑ_ptr, VKQ_accum_ptr,
                n_query_صفاں, DV, wg_ناپ);

            // Sync before MKL GEMM (MKL may use an internal queue)
            stream->wait();

            // 3. KV chunk loop (OUTER): dequant each chunk once, then tile queries.
            لكل (صحيح chunk_شروع = 0; chunk_شروع < n_kv; chunk_شروع += chunk_ناپ) {
                صحيح this_chunk = std::min(chunk_ناپ, n_kv - chunk_شروع);

                // 3a. Dequant this KV chunk to dense fp16 (once per chunk)
                {
                    MKL_TAKE_ویلا(t0);
                    mkl_fa_dequant_chunk(stream, K_desc, KQV,
                        K_chunk_f16_ptr, ikvh, chunk_شروع, this_chunk);
                    إذا (!V_is_K_view) {
                        mkl_fa_dequant_chunk(stream, V_desc, KQV,
                            V_chunk_f16_ptr, ikvh, chunk_شروع, this_chunk);
                    }
                    stream->wait();  // dequant must be ready before MKL GEMM
                    MKL_ACCUM(dequant_ویلا_us, t0);
                }

                // 3b. Query tile loop (INNER) — bounds KQ_f32/S_f16 footprint.
                لكل (صحيح q0 = 0; q0 < n_query_صفاں; q0 += q_tile_صفاں) {
                    صحيح q_صفاں = std::min(q_tile_صفاں, n_query_صفاں - q0);

                    // GEMM: KQ = Q_tile × K_chunk^T
                    {
                        MKL_TAKE_ویلا(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::trans, transpose::nontrans,
                            this_chunk, q_صفاں, DKQ,
                            alpha,
                            K_chunk_f16_ptr, DKQ,
                            Q_سر_f16_ptr + (int64_t)q0 * DKQ, DKQ,
                            beta,
                            KQ_f32_ptr, this_chunk);
                        حاول { ev.wait_and_throw(); } امسك (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM KQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM KQ failed");
                        }
                        MKL_ACCUM(gemm_kq_ویلا_us, t0);
                    }
                    // Online softmax over this chunk for this query tile
                    {
                        MKL_TAKE_ویلا(t0);
                        mkl_fa_online_softmax_chunk(stream,
                            KQ_f32_ptr, S_f16_ptr,
                            KQ_max_ptr, KQ_جوڑ_ptr, VKQ_accum_ptr,
                            q0, q_صفاں, n_queries, DV,
                            this_chunk, chunk_شروع,
                            kvh_ادھار_سر,
                            نقاب_batch, نقاب_سر_stride,
                            نقاب_صف_stride, نقاب_n_heads,
                            logit_softcap, wg_ناپ);
                        stream->wait();  // S_f16 must be ready for GEMM
                        MKL_ACCUM(softmax_ویلا_us, t0);
                    }

                    // GEMM: VKQ_chunk = S × V_chunk
                    {
                        MKL_TAKE_ویلا(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::nontrans, transpose::nontrans,
                            DV, q_صفاں, this_chunk,
                            alpha,
                            V_chunk_f16_ptr, DV,
                            S_f16_ptr, this_chunk,
                            beta,
                            VKQ_chunk_ptr, DV);
                        حاول { ev.wait_and_throw(); } امسك (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM VKQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM VKQ failed");
                        }
                        MKL_ACCUM(gemm_vkq_ویلا_us, t0);
                    }
                    // VKQ_accum[q0..] += VKQ_chunk
                    {
                        ثابت int64_t n_کل = (int64_t)q_صفاں * DV;
                        ثابت int64_t wg = ((n_کل + wg_ناپ - 1) / wg_ناپ)
                            * wg_ناپ;
                        عائم * accum = VKQ_accum_ptr + (int64_t)q0 * DV;
                        stream->submit([&](sycl::handler & cgh) {
                            cgh.parallel_for(sycl::nd_range<1>(wg, wg_ناپ),
                                [=](sycl::nd_شے<1> شے) {
                                    int64_t i = شے.get_global_id(0);
                                    إذا (i < n_کل) {
                                        accum[i] += VKQ_chunk_ptr[i];
                                    }
                                });
                        });
                    }
                }
            }

            // 4. Normalize and scatter each GQA head to dst
            لكل (صحيح iqg = 0; iqg < gqa_تناسب; iqg++) {
                صحيح     iqh        = kvh_ادھار_سر + iqg;
                int64_t src_وتھ = (int64_t)iqg * n_queries * DV;
                mkl_fa_normalize_سر(stream,
                    dst_batch, VKQ_accum_ptr, KQ_جوڑ_ptr,
                    iqh, n_queries, DV, n_q_heads,
                    src_وتھ, wg_ناپ);
            }
        }
    }

#تعریف_رد MKL_TAKE_ویلا
#تعریف_رد MKL_ACCUM

    إذا (do_print) {
        ثابت int64_t v_chunk_elems = V_is_K_view ? 0 : (int64_t)chunk_ناپ * DV;
        مزدوج کل_mb = (مزدوج)(
            (int64_t)q_tile_صفاں * chunk_ناپ * حجم(عائم)      // KQ_f32
          + (int64_t)q_tile_صفاں * chunk_ناپ * حجم(sycl::half) // S_f16
          + (int64_t)q_tile_صفاں * DV * حجم(عائم)              // VKQ_chunk
          + (int64_t)n_query_صفاں * DV * حجم(عائم)             // VKQ_accum
          + (int64_t)n_query_صفاں * حجم(عائم)                  // KQ_max
          + (int64_t)n_query_صفاں * حجم(عائم)                  // KQ_sum
          + (int64_t)n_query_صفاں * DKQ * حجم(sycl::half)       // Q_head_f16
          + (int64_t)chunk_ناپ * DKQ * حجم(sycl::half)         // K_chunk_f16
          + v_chunk_elems * (int64_t)حجم(sycl::half)            // V_chunk_f16
        ) / (1024.0 * 1024.0);
        GGML_LOG_INFO("[MKL-FA] #%d n_kv=%d n_q=%d q_tile=%d time_us: "
                "dequant=%lld GEMM_KQ=%lld softmax=%lld GEMM_VKQ=%lld "
                "buf_mb=%.1f\n",
                mkl_call_گنتی, n_kv, n_queries, q_tile_صفاں,
                (لما لما)dequant_ویلا_us,
                (لما لما)gemm_kq_ویلا_us,
                (لما لما)softmax_ویلا_us,
                (لما لما)gemm_vkq_ویلا_us,
                کل_mb);
    }
}
