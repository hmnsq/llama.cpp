// Flash attention via oneMKL GEMM (XMX-accelerated).
// Uses column_major::gemm for Q*K^T and S*V matmuls
// with an online softmax SYCL kernel.
//
// All GQA query heads sharing a KV head are batched into single
// GEMM calls, amortizing MKL launch overhead across K and V reuse.
//

#شامل "common.hpp"
#شامل "fattn-common.hpp"
#شامل "convert.hpp"
#شامل "fattn.hpp"

#شامل <oneapi/mkl.hpp>
#شامل <cstdio>
#شامل <chrono>

#تعریف_کرو MKL_FA_CHUNK_سائز_KV 8192

// Number of query rows processed per tile. The score buffers (KQ_f32, S_f16)
// are sized q_tile_rows * chunk_size, so this bounds their footprint
// regardless of batch size (n_query_rows = n_queries * gqa_ratio). A typical
// single-ubatch prefill (e.g. ubatch 1024 * gqa 8 = 8192 rows) is exactly one
// tile, so it runs with no extra iterations. Larger batches tile and stay
// bounded. Override with GGML_SYCL_MKL_FA_Q_TILE.
#تعریف_کرو MKL_FA_Q_TILE 8192

#تعریف_کرو MKL_FA_WG_سائز 256

استعمال oneapi::mkl::transpose;
استعمال oneapi::mkl::blas::عمود_major::gemm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Pack all GQA Q heads for one KV head into fp16, applying q_scale.
// Launches one kernel per GQA group — each kernel copies exactly
// n_queries * DKQ elements using the per-group dst offset and
// per-head source stride.
ساکن خالی mkl_fa_pack_q_fp16(
    dpct::قطار_ptr stream,
    sycl::half * __restrict dst,
    ثابت اعشاری * __restrict q_src,
    صحیح n_queries, صحیح DKQ,
    صحیح gqa_تناسب, صحیح kvh_بنیاد_سر,
    اعشاری q_scale, int64_t q_صف_stride, int64_t q_سر_stride,
    int64_t wg_سائز) {

    برائے (صحیح iqg = 0; iqg < gqa_تناسب; iqg++) {
        صحیح     iqh       = kvh_بنیاد_سر + iqg;
        sycl::half * dst_g = dst + (int64_t)iqg * n_queries * DKQ;

        ثابت int64_t n_elem = (int64_t)n_queries * DKQ;
        ثابت int64_t wg = ((n_elem + wg_سائز - 1) / wg_سائز) * wg_سائز;

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(wg, wg_سائز),
                [=](sycl::nd_شے<1> شے) {
                    int64_t e = شے.get_global_id(0);
                    اگر (e >= n_elem) واپس;

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
ساکن خالی mkl_fa_init_softmax_state(
    dpct::قطار_ptr stream,
    اعشاری * kmax, اعشاری * ksum, اعشاری * vacc,
    صحیح n_query_صفیں, صحیح DV, int64_t wg_سائز) {

    ثابت اعشاری    neg_inf   = -1e30f;
    ثابت int64_t  n_maxsum  = n_query_صفیں;
    ثابت int64_t  n_vacc    = (int64_t)n_query_صفیں * DV;
    ثابت int64_t  کل     = (n_vacc > n_maxsum) ? n_vacc : n_maxsum;
    ثابت int64_t  wg = ((کل + wg_سائز - 1) / wg_سائز) * wg_سائز;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_سائز),
            [=](sycl::nd_شے<1> شے) {
                int64_t i = شے.get_global_id(0);
                اگر (i < n_maxsum) {
                    kmax[i] = neg_inf;
                    ksum[i] = 0.0f;
                }
                اگر (i < n_vacc) {
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
ساکن خالی mkl_fa_online_softmax_chunk(
    dpct::قطار_ptr stream,
    اعشاری * __restrict KQ_f32,
    sycl::half * __restrict S_f16,
    اعشاری * __restrict KQ_max,
    اعشاری * __restrict KQ_مجموع,
    اعشاری * __restrict VKQ_accum,
    صحیح q0, صحیح q_صفیں, صحیح n_queries, صحیح DV,
    صحیح chunk_سائز, صحیح chunk_آغاز,
    صحیح kvh_سر,
    ثابت sycl::half * نقاب_ڈیٹا, int64_t نقاب_سر_stride,
    int64_t نقاب_صف_stride, صحیح نقاب_n_heads,
    اعشاری logit_softcap, int64_t wg_سائز) {

    // One work-group per query row: exactly q_rows groups of wg_size
    // items. q_rows * wg_size is already a multiple of wg_size, so unlike
    // the one-item-per-row kernels there is no round-up / tail guard.
    ثابت int64_t wg         = q_صفیں * wg_سائز;
    ثابت صحیح     local_سائز = (صحیح) wg_سائز;  // stride in the loops below
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_سائز),
            [=](sycl::nd_شے<1> شے) {
                ثابت صحیح local_id = (صحیح)شے.get_local_id(0);
                ثابت صحیح صف = (صحیح)شے.get_گروہ(0); // tile-relative
                ثابت صحیح jc_abs    = q0 + صف;
                ثابت صحیح gqa_گروہ = jc_abs / n_queries;
                ثابت صحیح q_صف     = jc_abs % n_queries;
                // Score buffers are tile-local (relative index).
                ثابت اعشاری * __restrict KQ_صف = KQ_f32
                    + صف * (int64_t)chunk_سائز;
                sycl::half * __restrict S_صف = S_f16
                    + صف * (int64_t)chunk_سائز;
                // Persistent accumulator is full-sized (absolute index).
                اعشاری * __restrict vkq = VKQ_accum
                    + jc_abs * (int64_t)DV;
                ثابت sycl::half * نقاب_h = خالی_اشاریہ;
                int64_t m_stride = 0;
                اگر (نقاب_ڈیٹا) {
                    صحیح m_سر = (نقاب_n_heads > 1)
                        ? (kvh_سر + gqa_گروہ) : 0;
                    نقاب_h   = نقاب_ڈیٹا + (int64_t)m_سر * نقاب_سر_stride;
                    m_stride = نقاب_صف_stride;
                }
                // Score at chunk offset i — original per-element math.
                خودکار score = [&](صحیح i) {
                    اعشاری s = KQ_صف[i];
                    اگر (logit_softcap != 0.0f) {
                        s = logit_softcap * sycl::tanh(s);
                    }
                    اگر (نقاب_h) {
                        s += (اعشاری)نقاب_h[q_صف * m_stride
                            + (chunk_آغاز + i)];
                    }
                    واپس s;
                };
                // Pass 1: strided (coalesced) row-wise local maximum.
                اعشاری local_max = -1e30f;
                برائے (صحیح i = local_id; i < chunk_سائز; i += local_سائز) {
                    اعشاری s = score(i);
                    اگر (s > local_max) local_max = s;
                }
                ثابت اعشاری final_local_max = sycl::reduce_over_گروہ(
                    شے.get_گروہ(), local_max, sycl::زیادہ_سے_زیادہ<اعشاری>());
                // Rescale previous accumulator by exp(old_max - new_max)
                اعشاری old_max = KQ_max[jc_abs];
                اعشاری new_max = (old_max > final_local_max) ? old_max : final_local_max;
                اعشاری rescale = (old_max < -1e29f) ? 1.0f
                    : sycl::native::قوت_نما(old_max - new_max);
                برائے (صحیح v = local_id; v < DV; v += local_سائز) {
                    vkq[v] *= rescale;
                }
                // Pass 2: softmax numerators, strided; S row written once.
                اعشاری local_مجموع = 0.0f;
                برائے (صحیح i = local_id; i < chunk_سائز; i += local_سائز) {
                    اعشاری s = score(i);
                    اعشاری val = sycl::native::قوت_نما(s - new_max);
                    S_صف[i] = sycl::half(val);
                    local_مجموع += val;
                }
                ثابت اعشاری کل_مجموع = sycl::reduce_over_گروہ(
                    شے.get_گروہ(), local_مجموع, sycl::plus<اعشاری>());
                اگر (local_id == 0) {
                    KQ_مجموع[jc_abs] = KQ_مجموع[jc_abs] * rescale + کل_مجموع;
                    KQ_max[jc_abs] = new_max;
                }
            });
    });
}

// Write one GQA group's normalized output to its destination head.
ساکن خالی mkl_fa_normalize_سر(
    dpct::قطار_ptr stream,
    اعشاری * __restrict dst_batch,
    ثابت اعشاری * __restrict VKQ_accum,
    ثابت اعشاری * __restrict KQ_مجموع,
    صحیح iqh, صحیح n_queries, صحیح DV, صحیح n_q_heads,
    int64_t src_انحراف, int64_t wg_سائز) {

    ثابت int64_t wg = ((n_queries + wg_سائز - 1) / wg_سائز) * wg_سائز;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_سائز),
            [=](sycl::nd_شے<1> شے) {
                صحیح jc = شے.get_global_id(0);
                اگر (jc >= n_queries) واپس;

                صحیح     ksum_idx = (صحیح)(src_انحراف / DV) + jc;
                اعشاری   inv_مجموع  = 1.0f / KQ_مجموع[ksum_idx];
                ثابت اعشاری * __restrict src = VKQ_accum
                    + src_انحراف + jc * (int64_t)DV;
                // Interleaved dst layout (matching TILE):
                // rows alternate between heads, then increment query.
                // offset = (query * n_q_heads + head) * DV
                اعشاری * __restrict dst_صف = dst_batch
                    + ((int64_t)jc * n_q_heads + iqh) * (int64_t)DV;

                برائے (صحیح v = 0; v < DV; v++) {
                    dst_صف[v] = src[v] * inv_مجموع;
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
شمار mkl_fa_kv_desc_منوال {
    MKL_FA_KV_منوال_F16_DENSE       = 0,
    MKL_FA_KV_منوال_F16_INTERLEAVED = 1,
    MKL_FA_KV_منوال_QUANT_CONTIG    = 2,
    MKL_FA_KV_منوال_QUANT_NC        = 3,
};

ساخت mkl_fa_kv_desc {
    ثابت حرف *         ڈیٹا = خالی_اشاریہ;
    ggml_type            type = GGML_TYPE_F16;
    int64_t              D    = 0;      // ne[0]
    int64_t              nb1  = 0;      // byte stride, seq dim
    int64_t              nb2  = 0;      // byte stride, head dim
    mkl_fa_kv_desc_منوال  mode = MKL_FA_KV_منوال_F16_DENSE;
    int64_t              ts   = 0;      // type size (mode 3 base offset)
    int64_t              s01  = 0;      // nc row stride in blocks (mode 3)
    int64_t              s02  = 0;      // nc head stride in blocks (mode 3)
};

ساکن mkl_fa_kv_desc mkl_fa_make_desc(ثابت ggml_tensor * T, منطقی interleaved, صحیح n_kv_heads) {
    mkl_fa_kv_desc d;
    d.ڈیٹا = (ثابت حرف *)T->ڈیٹا;
    d.type = T->type;
    d.D    = T->ne[0];
    d.nb1  = (int64_t)T->nb[1];
    d.nb2  = (int64_t)T->nb[2];
    d.ts   = (int64_t)ggml_type_سائز(T->type);

    اگر (T->type == GGML_TYPE_F16) {
        // MLA's V cache is a 512-wide view of 576-wide K rows. Treat any
        // padded row stride as strided even when there is only one KV head.
        d.mode = interleaved || d.nb1 != d.D * (int64_t)حجم(sycl::half)
            ? MKL_FA_KV_منوال_F16_INTERLEAVED : MKL_FA_KV_منوال_F16_DENSE;
    } ورنہ اگر (ggml_is_contiguously_allocated(T) && !interleaved) {
        d.mode = MKL_FA_KV_منوال_QUANT_CONTIG;
    } ورنہ {
        d.mode = MKL_FA_KV_منوال_QUANT_NC;
        ثابت int64_t bs          = (int64_t)ggml_blck_سائز(T->type);
        ثابت int64_t blk_فی_صف = T->ne[0] / bs;
        // True Gemma interleave packs heads within a row (nb[2] < ne[1]*nb[1])
        // → reconstruct physical strides. Padded seq-views (nb[2] > ne[1]*nb[1])
        // already have correct physical strides.
        ثابت منطقی gemma = interleaved &&
            ((int64_t)T->nb[2] < (int64_t)T->ne[1] * (int64_t)T->nb[1]);
        اگر (gemma) {
            d.s01 = (int64_t)n_kv_heads * blk_فی_صف;
            d.s02 = blk_فی_صف;
        } ورنہ {
            d.s01 = d.nb1 / d.ts;
            d.s02 = d.nb2 / d.ts;
        }
    }
    واپس d;
}

// Dequant one KV-head chunk into a dense [this_chunk x D] fp16 buffer.
ساکن خالی mkl_fa_dequant_chunk(
    dpct::قطار_ptr stream, ثابت mkl_fa_kv_desc & d, ggml_tensor * dst_ctx,
    sycl::half * برآمد, صحیح ikvh, صحیح chunk_آغاز, صحیح this_chunk) {

    ثابت int64_t D = d.D;
    انتخاب (d.mode) {
        حالت MKL_FA_KV_منوال_F16_DENSE: {
            ثابت حرف * بنیاد = d.ڈیٹا + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_آغاز * d.nb1;
            stream->حافظہ_نقل(برآمد, بنیاد, (size_t)this_chunk * D * حجم(sycl::half));
            توڑو;
        }
        حالت MKL_FA_KV_منوال_F16_INTERLEAVED: {
            ثابت حرف * بنیاد = d.ڈیٹا + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_آغاز * d.nb1;
            ثابت int64_t صف_halfs = d.nb1 / (int64_t)حجم(sycl::half);
            ثابت sycl::half * src = (ثابت sycl::half *)بنیاد;
            stream->parallel_for(
                sycl::range<2>((size_t)this_chunk, (size_t)D),
                [=](sycl::شے<2> it) {
                    int64_t r = it.get_id(0);
                    int64_t c = it.get_id(1);
                    برآمد[r * D + c] = src[r * صف_halfs + c];
                });
            توڑو;
        }
        حالت MKL_FA_KV_منوال_QUANT_CONTIG: {
            ثابت حرف * بنیاد = d.ڈیٹا + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_آغاز * d.nb1;
            مقصد_fp16_sycl_t مقصد_fp16 = ggml_get_مقصد_fp16_sycl(d.type, dst_ctx);
            مقصد_fp16(بنیاد, برآمد, (int64_t)this_chunk * D, stream);
            توڑو;
        }
        طےشدہ: {  // MKL_FA_KV_MODE_QUANT_NC
            مقصد_fp16_nc_sycl_t مقصد_fp16 = ggml_get_مقصد_fp16_nc_sycl(d.type);
            ثابت int64_t بنیاد_بلاکیں = (int64_t)ikvh * d.s02
                + (int64_t)chunk_آغاز * d.s01;
            ثابت حرف * بنیاد = d.ڈیٹا + بنیاد_بلاکیں * d.ts;
            // ne02 = ne03 = 1 → s02/s03 inert; head+chunk offset carried by base.
            مقصد_fp16(بنیاد, برآمد, D, this_chunk, 1, 1, d.s01, d.s02, d.s02, stream);
            توڑو;
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
خالی ggml_sycl_flash_attn_ext_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    ثابت ggml_tensor * Q    = dst->src[0];
    ثابت ggml_tensor * K    = dst->src[1];
    ثابت ggml_tensor * V    = dst->src[2];
    ثابت ggml_tensor * نقاب = dst->src[3];
    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    // --- Op params ---
    اعشاری scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    حافظہ_نقل(&scale,         (ثابت اعشاری *)KQV->op_params + 0, حجم(اعشاری));
    حافظہ_نقل(&max_bias,      (ثابت اعشاری *)KQV->op_params + 1, حجم(اعشاری));
    حافظہ_نقل(&logit_softcap, (ثابت اعشاری *)KQV->op_params + 2, حجم(اعشاری));

    ثابت اعشاری q_scale = scale;

    // --- Dimensions ---
    ثابت صحیح DKQ        = (صحیح)K->ne[0];
    ثابت صحیح DV         = (صحیح)V->ne[0];
    ثابت صحیح n_queries  = (صحیح)Q->ne[1];
    ثابت صحیح n_q_heads  = (صحیح)Q->ne[2];
    ثابت صحیح n_kv_heads = (صحیح)K->ne[2];
    ثابت صحیح n_batch    = (صحیح)Q->ne[3];
    ثابت صحیح n_kv       = (صحیح)K->ne[1];
    ثابت صحیح gqa_تناسب  = n_q_heads / n_kv_heads;
    ثابت صحیح n_query_صفیں = n_queries * gqa_تناسب;

    GGML_ASSERT(n_q_heads % n_kv_heads == 0);
    GGML_ASSERT(max_bias == 0.0f);  // ALiBi not supported
    GGML_ASSERT(Q->ne[3] == K->ne[3] || K->ne[3] == 1);

    ثابت صحیح chunk_سائز = std::min(MKL_FA_CHUNK_سائز_KV, n_kv);

    // Query rows are processed in tiles of q_tile_rows so the score buffers
    // (KQ_f32/S_f16 = q_tile_rows * chunk_size) stay bounded regardless of
    // batch size. n_query_rows <= Q_TILE is a single tile (no extra work).
    ساکن صحیح q_tile_env = ggml_sycl_get_env("GGML_SYCL_MKL_FA_Q_TILE", MKL_FA_Q_TILE);
    ثابت صحیح q_tile_صفیں = std::max(1, std::min(q_tile_env, n_query_صفیں));

    ثابت int64_t wg_سائز = MKL_FA_WG_سائز;

    // --- Debug output (gated by GGML_SYCL_MKL_FA_DEBUG=1) ---
    ساکن صحیح mkl_call_گنتی = 0;
    mkl_call_گنتی++;
    ساکن صحیح mkl_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    ثابت منطقی do_print = (mkl_debug == 1);

    ثابت int64_t q_صف_stride  = Q->nb[1] / حجم(اعشاری);
    ثابت int64_t q_سر_stride = Q->nb[2] / حجم(اعشاری);

    // Alias the dequantized buffers only when K and V expose the same values.
    // MLA V is a narrower view of K and needs its own strided dequantization.
    ثابت منطقی V_is_K_view = V->ne[0] == K->ne[0] && V->view_src
        && (V->view_src == K || (V->view_src == K->view_src
            && V->view_offs == K->view_offs));

    // Early interleaved detection for debug output.
    // True interleaved detection happens after dequant (nb12_fp16 == nb11_fp16),
    // but we can pre-detect on the original tensor strides.
    ثابت منطقی k_early_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]);
    ثابت منطقی v_early_interleaved =
        !V_is_K_view && ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]);

    اگر (do_print) {
        GGML_LOG_INFO("[MKL-FA] #%d D=%d DV=%d n_q=%d n_kv=%d "
                "n_qh=%d n_kvh=%d gqa=%d batch=%d K=%s V=%s "
                "chunk=%d buf=%.1fMB%s%s\n",
                mkl_call_گنتی, DKQ, DV, n_queries, n_kv,
                n_q_heads, n_kv_heads, gqa_تناسب, n_batch,
                ggml_type_name(K->type), ggml_type_name(V->type),
                chunk_سائز,
                (دوہرا)((int64_t)n_query_صفیں * chunk_سائز * حجم(اعشاری))
                    / (1024.0 * 1024.0),
                k_early_interleaved ? " K_ILV" : "",
                v_early_interleaved ? " V_ILV" : "");
        GGML_LOG_INFO("[MKL-FA] #%d Q-nb1=%lld Q-nb2=%lld "
                "q_rs=%lld q_hs=%lld dst_rs=%lld dst_hs=%lld\n",
                mkl_call_گنتی,
                (طویل طویل)Q->nb[1], (طویل طویل)Q->nb[2],
                (طویل طویل)q_صف_stride, (طویل طویل)q_سر_stride,
                (طویل طویل)(KQV->nb[1] / حجم(اعشاری)),
                (طویل طویل)(KQV->nb[2] / حجم(اعشاری)));
    }

    // --- Stream and allocators ---
    dpct::قطار_ptr stream = ctx.stream();

#تعریف_کرو MKL_TAKE_وقت(t0)  خودکار t0 = std::chrono::steady_clock::now()
#تعریف_کرو MKL_ACCUM(acc, t0) کرو { اگر (do_print) { \
    acc += (int64_t)std::chrono::duration_cast \
    <std::chrono::microseconds>(std::chrono::steady_clock::now() - (t0)).count(); \
} } جب(0)

    int64_t gemm_kq_وقت_us  = 0;
    int64_t gemm_vkq_وقت_us = 0;
    int64_t softmax_وقت_us  = 0;
    int64_t dequant_وقت_us  = 0;

    MKL_TAKE_وقت(t_deq);

    // --- K/V dequant descriptors ---
    // Dequant is done per-chunk inside the KV loop (footprint independent of
    // context). Output is always dense row-major fp16 [this_chunk x D], lda=D.
    // Interleaved detection: ne[1]*nb[1] != nb[2] means heads are interleaved.
    ثابت منطقی k_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
    ثابت منطقی v_interleaved =
        ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;

    ثابت mkl_fa_kv_desc K_desc = mkl_fa_make_desc(K, k_interleaved, n_kv_heads);
    ثابت mkl_fa_kv_desc V_desc = V_is_K_view
        ? K_desc : mkl_fa_make_desc(V, v_interleaved, n_kv_heads);

    MKL_ACCUM(dequant_وقت_us, t_deq);

    // --- Resolve mask pointers ---
    int64_t نقاب_سر_stride = 0;
    int64_t نقاب_صف_stride  = 0;
    صحیح     نقاب_n_heads     = 0;

    اگر (نقاب) {
        // Use actual fp16 device size (2 bytes), NOT sizeof(sycl::half)
        // which may be 4 on the host in oneAPI.
        نقاب_سر_stride = نقاب->nb[2] / 2;
        نقاب_صف_stride  = نقاب->nb[1] / 2;
        نقاب_n_heads     = (صحیح)نقاب->ne[2];
    }

    // --- Allocate intermediates from pool ---
    ggml_sycl_pool & pool = ctx.pool();

    ggml_sycl_pool_alloc<اعشاری>      KQ_f32(pool);      // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<sycl::half> S_f16(pool);       // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<اعشاری>      VKQ_chunk(pool);   // [q_tile_rows x DV]
    ggml_sycl_pool_alloc<اعشاری>      VKQ_accum(pool);   // [n_query_rows x DV] (full)
    ggml_sycl_pool_alloc<اعشاری>      KQ_max(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<اعشاری>      KQ_مجموع(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<sycl::half> Q_سر_f16(pool);  // [n_query_rows x DKQ] (full)
    ggml_sycl_pool_alloc<sycl::half> K_chunk_f16(pool); // [chunk x DKQ] (per-chunk dequant)
    ggml_sycl_pool_alloc<sycl::half> V_chunk_f16(pool); // [chunk x DV] (per-chunk dequant)

    KQ_f32.alloc((size_t)q_tile_صفیں * chunk_سائز);
    S_f16.alloc((size_t)q_tile_صفیں * chunk_سائز);
    VKQ_chunk.alloc((size_t)q_tile_صفیں * DV);
    VKQ_accum.alloc((size_t)n_query_صفیں * DV);
    KQ_max.alloc(n_query_صفیں);
    KQ_مجموع.alloc(n_query_صفیں);
    Q_سر_f16.alloc((size_t)n_query_صفیں * DKQ);
    K_chunk_f16.alloc((size_t)chunk_سائز * DKQ);

    sycl::half * V_chunk_f16_ptr;
    اگر (V_is_K_view) {
        V_chunk_f16_ptr = K_chunk_f16.ptr;   // V aliases K (DV == DKQ)
    } ورنہ {
        V_chunk_f16.alloc((size_t)chunk_سائز * DV);
        V_chunk_f16_ptr = V_chunk_f16.ptr;
    }

    sycl::half * Q_سر_f16_ptr  = Q_سر_f16.ptr;
    اعشاری      * KQ_f32_ptr      = KQ_f32.ptr;
    sycl::half * S_f16_ptr       = S_f16.ptr;
    اعشاری      * VKQ_chunk_ptr   = VKQ_chunk.ptr;
    اعشاری      * VKQ_accum_ptr   = VKQ_accum.ptr;
    اعشاری      * KQ_max_ptr      = KQ_max.ptr;
    اعشاری      * KQ_مجموع_ptr      = KQ_مجموع.ptr;
    sycl::half * K_chunk_f16_ptr = K_chunk_f16.ptr;

    ثابت اعشاری alpha = 1.0f;
    ثابت اعشاری beta  = 0.0f;

    برائے (صحیح ib = 0; ib < n_batch; ib++) {
        ثابت اعشاری * Q_batch = (ثابت اعشاری *)Q->ڈیٹا
            + ib * (Q->nb[3] / حجم(اعشاری));
        اعشاری * dst_batch = (اعشاری *)KQV->ڈیٹا
            + ib * (KQV->nb[3] / حجم(اعشاری));

        ثابت sycl::half * نقاب_batch = خالی_اشاریہ;
        اگر (نقاب) {
            صحیح m_batch = (نقاب->ne[3] > 1) ? ib : 0;
            نقاب_batch = (ثابت sycl::half *)نقاب->ڈیٹا
                + m_batch * (نقاب->nb[3] / 2);  // 2 = actual fp16 device size
        }

        برائے (صحیح ikvh = 0; ikvh < n_kv_heads; ikvh++) {
            صحیح kvh_بنیاد_سر = ikvh * gqa_تناسب;

            // 1. Pack all GQA Q heads into fp16 (full n_query_rows)
            mkl_fa_pack_q_fp16(stream,
                Q_سر_f16_ptr, Q_batch,
                n_queries, DKQ,
                gqa_تناسب, kvh_بنیاد_سر,
                q_scale, q_صف_stride, q_سر_stride, wg_سائز);

            // 2. Initialize softmax state (full n_query_rows)
            mkl_fa_init_softmax_state(stream,
                KQ_max_ptr, KQ_مجموع_ptr, VKQ_accum_ptr,
                n_query_صفیں, DV, wg_سائز);

            // Sync before MKL GEMM (MKL may use an internal queue)
            stream->wait();

            // 3. KV chunk loop (OUTER): dequant each chunk once, then tile queries.
            برائے (صحیح chunk_آغاز = 0; chunk_آغاز < n_kv; chunk_آغاز += chunk_سائز) {
                صحیح this_chunk = std::min(chunk_سائز, n_kv - chunk_آغاز);

                // 3a. Dequant this KV chunk to dense fp16 (once per chunk)
                {
                    MKL_TAKE_وقت(t0);
                    mkl_fa_dequant_chunk(stream, K_desc, KQV,
                        K_chunk_f16_ptr, ikvh, chunk_آغاز, this_chunk);
                    اگر (!V_is_K_view) {
                        mkl_fa_dequant_chunk(stream, V_desc, KQV,
                            V_chunk_f16_ptr, ikvh, chunk_آغاز, this_chunk);
                    }
                    stream->wait();  // dequant must be ready before MKL GEMM
                    MKL_ACCUM(dequant_وقت_us, t0);
                }

                // 3b. Query tile loop (INNER) — bounds KQ_f32/S_f16 footprint.
                برائے (صحیح q0 = 0; q0 < n_query_صفیں; q0 += q_tile_صفیں) {
                    صحیح q_صفیں = std::min(q_tile_صفیں, n_query_صفیں - q0);

                    // GEMM: KQ = Q_tile × K_chunk^T
                    {
                        MKL_TAKE_وقت(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::trans, transpose::nontrans,
                            this_chunk, q_صفیں, DKQ,
                            alpha,
                            K_chunk_f16_ptr, DKQ,
                            Q_سر_f16_ptr + (int64_t)q0 * DKQ, DKQ,
                            beta,
                            KQ_f32_ptr, this_chunk);
                        کوشش { ev.wait_and_throw(); } پکڑو (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM KQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM KQ failed");
                        }
                        MKL_ACCUM(gemm_kq_وقت_us, t0);
                    }
                    // Online softmax over this chunk for this query tile
                    {
                        MKL_TAKE_وقت(t0);
                        mkl_fa_online_softmax_chunk(stream,
                            KQ_f32_ptr, S_f16_ptr,
                            KQ_max_ptr, KQ_مجموع_ptr, VKQ_accum_ptr,
                            q0, q_صفیں, n_queries, DV,
                            this_chunk, chunk_آغاز,
                            kvh_بنیاد_سر,
                            نقاب_batch, نقاب_سر_stride,
                            نقاب_صف_stride, نقاب_n_heads,
                            logit_softcap, wg_سائز);
                        stream->wait();  // S_f16 must be ready for GEMM
                        MKL_ACCUM(softmax_وقت_us, t0);
                    }

                    // GEMM: VKQ_chunk = S × V_chunk
                    {
                        MKL_TAKE_وقت(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::nontrans, transpose::nontrans,
                            DV, q_صفیں, this_chunk,
                            alpha,
                            V_chunk_f16_ptr, DV,
                            S_f16_ptr, this_chunk,
                            beta,
                            VKQ_chunk_ptr, DV);
                        کوشش { ev.wait_and_throw(); } پکڑو (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM VKQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM VKQ failed");
                        }
                        MKL_ACCUM(gemm_vkq_وقت_us, t0);
                    }
                    // VKQ_accum[q0..] += VKQ_chunk
                    {
                        ثابت int64_t n_کل = (int64_t)q_صفیں * DV;
                        ثابت int64_t wg = ((n_کل + wg_سائز - 1) / wg_سائز)
                            * wg_سائز;
                        اعشاری * accum = VKQ_accum_ptr + (int64_t)q0 * DV;
                        stream->submit([&](sycl::handler & cgh) {
                            cgh.parallel_for(sycl::nd_range<1>(wg, wg_سائز),
                                [=](sycl::nd_شے<1> شے) {
                                    int64_t i = شے.get_global_id(0);
                                    اگر (i < n_کل) {
                                        accum[i] += VKQ_chunk_ptr[i];
                                    }
                                });
                        });
                    }
                }
            }

            // 4. Normalize and scatter each GQA head to dst
            برائے (صحیح iqg = 0; iqg < gqa_تناسب; iqg++) {
                صحیح     iqh        = kvh_بنیاد_سر + iqg;
                int64_t src_انحراف = (int64_t)iqg * n_queries * DV;
                mkl_fa_normalize_سر(stream,
                    dst_batch, VKQ_accum_ptr, KQ_مجموع_ptr,
                    iqh, n_queries, DV, n_q_heads,
                    src_انحراف, wg_سائز);
            }
        }
    }

#تعریف_منسوخ MKL_TAKE_وقت
#تعریف_منسوخ MKL_ACCUM

    اگر (do_print) {
        ثابت int64_t v_chunk_elems = V_is_K_view ? 0 : (int64_t)chunk_سائز * DV;
        دوہرا کل_mb = (دوہرا)(
            (int64_t)q_tile_صفیں * chunk_سائز * حجم(اعشاری)      // KQ_f32
          + (int64_t)q_tile_صفیں * chunk_سائز * حجم(sycl::half) // S_f16
          + (int64_t)q_tile_صفیں * DV * حجم(اعشاری)              // VKQ_chunk
          + (int64_t)n_query_صفیں * DV * حجم(اعشاری)             // VKQ_accum
          + (int64_t)n_query_صفیں * حجم(اعشاری)                  // KQ_max
          + (int64_t)n_query_صفیں * حجم(اعشاری)                  // KQ_sum
          + (int64_t)n_query_صفیں * DKQ * حجم(sycl::half)       // Q_head_f16
          + (int64_t)chunk_سائز * DKQ * حجم(sycl::half)         // K_chunk_f16
          + v_chunk_elems * (int64_t)حجم(sycl::half)            // V_chunk_f16
        ) / (1024.0 * 1024.0);
        GGML_LOG_INFO("[MKL-FA] #%d n_kv=%d n_q=%d q_tile=%d time_us: "
                "dequant=%lld GEMM_KQ=%lld softmax=%lld GEMM_VKQ=%lld "
                "buf_mb=%.1f\n",
                mkl_call_گنتی, n_kv, n_queries, q_tile_صفیں,
                (طویل طویل)dequant_وقت_us,
                (طویل طویل)gemm_kq_وقت_us,
                (طویل طویل)softmax_وقت_us,
                (طویل طویل)gemm_vkq_وقت_us,
                کل_mb);
    }
}
