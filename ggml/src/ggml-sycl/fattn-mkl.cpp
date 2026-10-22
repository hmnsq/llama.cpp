// Flash attention via oneMKL GEMM (XMX-accelerated).
// Uses column_major::gemm for Q*K^T and S*V matmuls
// with an online softmax SYCL kernel.
//
// All GQA query heads sharing a KV head are batched into single
// GEMM calls, amortizing MKL launch overhead across K and V reuse.
//

#అన్తర్భావ "common.hpp"
#అన్తర్భావ "fattn-common.hpp"
#అన్తర్భావ "convert.hpp"
#అన్తర్భావ "fattn.hpp"

#అన్తర్భావ <oneapi/mkl.hpp>
#అన్తర్భావ <cstdio>
#అన్తర్భావ <chrono>

#నిర్వచించు MKL_FA_CHUNK_కొలత_KV 8192

// Number of query rows processed per tile. The score buffers (KQ_f32, S_f16)
// are sized q_tile_rows * chunk_size, so this bounds their footprint
// regardless of batch size (n_query_rows = n_queries * gqa_ratio). A typical
// single-ubatch prefill (e.g. ubatch 1024 * gqa 8 = 8192 rows) is exactly one
// tile, so it runs with no extra iterations. Larger batches tile and stay
// bounded. Override with GGML_SYCL_MKL_FA_Q_TILE.
#నిర్వచించు MKL_FA_Q_TILE 8192

#నిర్వచించు MKL_FA_WG_కొలత 256

ఉపయోగించి oneapi::mkl::transpose;
ఉపయోగించి oneapi::mkl::blas::స్తంభం_major::gemm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Pack all GQA Q heads for one KV head into fp16, applying q_scale.
// Launches one kernel per GQA group — each kernel copies exactly
// n_queries * DKQ elements using the per-group dst offset and
// per-head source stride.
స్థిర శూన్య mkl_fa_pack_q_fp16(
    dpct::వరుస_ptr stream,
    sycl::half * __restrict dst,
    స్థిరరాశి దశాంశం * __restrict q_src,
    పూర్ణాంకం n_queries, పూర్ణాంకం DKQ,
    పూర్ణాంకం gqa_నిష్పత్తి, పూర్ణాంకం kvh_పునాది_తల,
    దశాంశం q_scale, int64_t q_అడ్డువరుస_stride, int64_t q_తల_stride,
    int64_t wg_కొలత) {

    కొరకు (పూర్ణాంకం iqg = 0; iqg < gqa_నిష్పత్తి; iqg++) {
        పూర్ణాంకం     iqh       = kvh_పునాది_తల + iqg;
        sycl::half * dst_g = dst + (int64_t)iqg * n_queries * DKQ;

        స్థిరరాశి int64_t n_elem = (int64_t)n_queries * DKQ;
        స్థిరరాశి int64_t wg = ((n_elem + wg_కొలత - 1) / wg_కొలత) * wg_కొలత;

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(wg, wg_కొలత),
                [=](sycl::nd_అంశం<1> అంశం) {
                    int64_t e = అంశం.get_global_id(0);
                    ఒకవేళ (e >= n_elem) తిరిగి;

                    int64_t q = e / DKQ;
                    int64_t d = e - q * DKQ;

                    // Stride-aware source offset: handles permuted,
                    // sliced, or contiguous Q tensor layouts.
                    int64_t src_off = d
                        + q * q_అడ్డువరుస_stride
                        + (int64_t)iqh * q_తల_stride;

                    dst_g[e] = sycl::half(
                        q_src[src_off] * q_scale);
                });
        });
    }
}

// Zero-initialize the online softmax state arrays.
// KQ_max → -inf, KQ_sum → 0, VKQ_accum → 0.
// Merged into one kernel to avoid per-array launch overhead.
స్థిర శూన్య mkl_fa_init_softmax_state(
    dpct::వరుస_ptr stream,
    దశాంశం * kmax, దశాంశం * ksum, దశాంశం * vacc,
    పూర్ణాంకం n_query_అడ్డువరుసలు, పూర్ణాంకం DV, int64_t wg_కొలత) {

    స్థిరరాశి దశాంశం    neg_inf   = -1e30f;
    స్థిరరాశి int64_t  n_maxsum  = n_query_అడ్డువరుసలు;
    స్థిరరాశి int64_t  n_vacc    = (int64_t)n_query_అడ్డువరుసలు * DV;
    స్థిరరాశి int64_t  మొత్తం     = (n_vacc > n_maxsum) ? n_vacc : n_maxsum;
    స్థిరరాశి int64_t  wg = ((మొత్తం + wg_కొలత - 1) / wg_కొలత) * wg_కొలత;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_కొలత),
            [=](sycl::nd_అంశం<1> అంశం) {
                int64_t i = అంశం.get_global_id(0);
                ఒకవేళ (i < n_maxsum) {
                    kmax[i] = neg_inf;
                    ksum[i] = 0.0f;
                }
                ఒకవేళ (i < n_vacc) {
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
స్థిర శూన్య mkl_fa_online_softmax_chunk(
    dpct::వరుస_ptr stream,
    దశాంశం * __restrict KQ_f32,
    sycl::half * __restrict S_f16,
    దశాంశం * __restrict KQ_max,
    దశాంశం * __restrict KQ_కూడిక,
    దశాంశం * __restrict VKQ_accum,
    పూర్ణాంకం q0, పూర్ణాంకం q_అడ్డువరుసలు, పూర్ణాంకం n_queries, పూర్ణాంకం DV,
    పూర్ణాంకం chunk_కొలత, పూర్ణాంకం chunk_ప్రారంభం,
    పూర్ణాంకం kvh_తల,
    స్థిరరాశి sycl::half * ముసుగు_దత్తాంశం, int64_t ముసుగు_తల_stride,
    int64_t ముసుగు_అడ్డువరుస_stride, పూర్ణాంకం ముసుగు_n_heads,
    దశాంశం logit_softcap, int64_t wg_కొలత) {

    // One work-group per query row: exactly q_rows groups of wg_size
    // items. q_rows * wg_size is already a multiple of wg_size, so unlike
    // the one-item-per-row kernels there is no round-up / tail guard.
    స్థిరరాశి int64_t wg         = q_అడ్డువరుసలు * wg_కొలత;
    స్థిరరాశి పూర్ణాంకం     local_కొలత = (పూర్ణాంకం) wg_కొలత;  // stride in the loops below
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_కొలత),
            [=](sycl::nd_అంశం<1> అంశం) {
                స్థిరరాశి పూర్ణాంకం local_id = (పూర్ణాంకం)అంశం.get_local_id(0);
                స్థిరరాశి పూర్ణాంకం అడ్డువరుస = (పూర్ణాంకం)అంశం.get_సమూహం(0); // tile-relative
                స్థిరరాశి పూర్ణాంకం jc_abs    = q0 + అడ్డువరుస;
                స్థిరరాశి పూర్ణాంకం gqa_సమూహం = jc_abs / n_queries;
                స్థిరరాశి పూర్ణాంకం q_అడ్డువరుస     = jc_abs % n_queries;
                // Score buffers are tile-local (relative index).
                స్థిరరాశి దశాంశం * __restrict KQ_అడ్డువరుస = KQ_f32
                    + అడ్డువరుస * (int64_t)chunk_కొలత;
                sycl::half * __restrict S_అడ్డువరుస = S_f16
                    + అడ్డువరుస * (int64_t)chunk_కొలత;
                // Persistent accumulator is full-sized (absolute index).
                దశాంశం * __restrict vkq = VKQ_accum
                    + jc_abs * (int64_t)DV;
                స్థిరరాశి sycl::half * ముసుగు_h = శూన్యసూచిక;
                int64_t m_stride = 0;
                ఒకవేళ (ముసుగు_దత్తాంశం) {
                    పూర్ణాంకం m_తల = (ముసుగు_n_heads > 1)
                        ? (kvh_తల + gqa_సమూహం) : 0;
                    ముసుగు_h   = ముసుగు_దత్తాంశం + (int64_t)m_తల * ముసుగు_తల_stride;
                    m_stride = ముసుగు_అడ్డువరుస_stride;
                }
                // Score at chunk offset i — original per-element math.
                స్వయంచాలక score = [&](పూర్ణాంకం i) {
                    దశాంశం s = KQ_అడ్డువరుస[i];
                    ఒకవేళ (logit_softcap != 0.0f) {
                        s = logit_softcap * sycl::tanh(s);
                    }
                    ఒకవేళ (ముసుగు_h) {
                        s += (దశాంశం)ముసుగు_h[q_అడ్డువరుస * m_stride
                            + (chunk_ప్రారంభం + i)];
                    }
                    తిరిగి s;
                };
                // Pass 1: strided (coalesced) row-wise local maximum.
                దశాంశం local_max = -1e30f;
                కొరకు (పూర్ణాంకం i = local_id; i < chunk_కొలత; i += local_కొలత) {
                    దశాంశం s = score(i);
                    ఒకవేళ (s > local_max) local_max = s;
                }
                స్థిరరాశి దశాంశం final_local_max = sycl::reduce_over_సమూహం(
                    అంశం.get_సమూహం(), local_max, sycl::గరిష్ఠ<దశాంశం>());
                // Rescale previous accumulator by exp(old_max - new_max)
                దశాంశం old_max = KQ_max[jc_abs];
                దశాంశం new_max = (old_max > final_local_max) ? old_max : final_local_max;
                దశాంశం rescale = (old_max < -1e29f) ? 1.0f
                    : sycl::native::ఘాతాంకం(old_max - new_max);
                కొరకు (పూర్ణాంకం v = local_id; v < DV; v += local_కొలత) {
                    vkq[v] *= rescale;
                }
                // Pass 2: softmax numerators, strided; S row written once.
                దశాంశం local_కూడిక = 0.0f;
                కొరకు (పూర్ణాంకం i = local_id; i < chunk_కొలత; i += local_కొలత) {
                    దశాంశం s = score(i);
                    దశాంశం val = sycl::native::ఘాతాంకం(s - new_max);
                    S_అడ్డువరుస[i] = sycl::half(val);
                    local_కూడిక += val;
                }
                స్థిరరాశి దశాంశం మొత్తం_కూడిక = sycl::reduce_over_సమూహం(
                    అంశం.get_సమూహం(), local_కూడిక, sycl::plus<దశాంశం>());
                ఒకవేళ (local_id == 0) {
                    KQ_కూడిక[jc_abs] = KQ_కూడిక[jc_abs] * rescale + మొత్తం_కూడిక;
                    KQ_max[jc_abs] = new_max;
                }
            });
    });
}

// Write one GQA group's normalized output to its destination head.
స్థిర శూన్య mkl_fa_normalize_తల(
    dpct::వరుస_ptr stream,
    దశాంశం * __restrict dst_batch,
    స్థిరరాశి దశాంశం * __restrict VKQ_accum,
    స్థిరరాశి దశాంశం * __restrict KQ_కూడిక,
    పూర్ణాంకం iqh, పూర్ణాంకం n_queries, పూర్ణాంకం DV, పూర్ణాంకం n_q_heads,
    int64_t src_స్థానభ్రంశం, int64_t wg_కొలత) {

    స్థిరరాశి int64_t wg = ((n_queries + wg_కొలత - 1) / wg_కొలత) * wg_కొలత;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_కొలత),
            [=](sycl::nd_అంశం<1> అంశం) {
                పూర్ణాంకం jc = అంశం.get_global_id(0);
                ఒకవేళ (jc >= n_queries) తిరిగి;

                పూర్ణాంకం     ksum_idx = (పూర్ణాంకం)(src_స్థానభ్రంశం / DV) + jc;
                దశాంశం   inv_కూడిక  = 1.0f / KQ_కూడిక[ksum_idx];
                స్థిరరాశి దశాంశం * __restrict src = VKQ_accum
                    + src_స్థానభ్రంశం + jc * (int64_t)DV;
                // Interleaved dst layout (matching TILE):
                // rows alternate between heads, then increment query.
                // offset = (query * n_q_heads + head) * DV
                దశాంశం * __restrict dst_అడ్డువరుస = dst_batch
                    + ((int64_t)jc * n_q_heads + iqh) * (int64_t)DV;

                కొరకు (పూర్ణాంకం v = 0; v < DV; v++) {
                    dst_అడ్డువరుస[v] = src[v] * inv_కూడిక;
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
గణన mkl_fa_kv_desc_బహుళకం {
    MKL_FA_KV_బహుళకం_F16_DENSE       = 0,
    MKL_FA_KV_బహుళకం_F16_INTERLEAVED = 1,
    MKL_FA_KV_బహుళకం_QUANT_CONTIG    = 2,
    MKL_FA_KV_బహుళకం_QUANT_NC        = 3,
};

నిర్మాణం mkl_fa_kv_desc {
    స్థిరరాశి అక్షరం *         దత్తాంశం = శూన్యసూచిక;
    ggml_type            type = GGML_TYPE_F16;
    int64_t              D    = 0;      // ne[0]
    int64_t              nb1  = 0;      // byte stride, seq dim
    int64_t              nb2  = 0;      // byte stride, head dim
    mkl_fa_kv_desc_బహుళకం  mode = MKL_FA_KV_బహుళకం_F16_DENSE;
    int64_t              ts   = 0;      // type size (mode 3 base offset)
    int64_t              s01  = 0;      // nc row stride in blocks (mode 3)
    int64_t              s02  = 0;      // nc head stride in blocks (mode 3)
};

స్థిర mkl_fa_kv_desc mkl_fa_make_desc(స్థిరరాశి ggml_tensor * T, తార్కిక interleaved, పూర్ణాంకం n_kv_heads) {
    mkl_fa_kv_desc d;
    d.దత్తాంశం = (స్థిరరాశి అక్షరం *)T->దత్తాంశం;
    d.type = T->type;
    d.D    = T->ne[0];
    d.nb1  = (int64_t)T->nb[1];
    d.nb2  = (int64_t)T->nb[2];
    d.ts   = (int64_t)ggml_type_కొలత(T->type);

    ఒకవేళ (T->type == GGML_TYPE_F16) {
        // MLA's V cache is a 512-wide view of 576-wide K rows. Treat any
        // padded row stride as strided even when there is only one KV head.
        d.mode = interleaved || d.nb1 != d.D * (int64_t)పరిమాణం(sycl::half)
            ? MKL_FA_KV_బహుళకం_F16_INTERLEAVED : MKL_FA_KV_బహుళకం_F16_DENSE;
    } లేకపోతే ఒకవేళ (ggml_is_contiguously_allocated(T) && !interleaved) {
        d.mode = MKL_FA_KV_బహుళకం_QUANT_CONTIG;
    } లేకపోతే {
        d.mode = MKL_FA_KV_బహుళకం_QUANT_NC;
        స్థిరరాశి int64_t bs          = (int64_t)ggml_blck_కొలత(T->type);
        స్థిరరాశి int64_t blk_చొప్పున_అడ్డువరుస = T->ne[0] / bs;
        // True Gemma interleave packs heads within a row (nb[2] < ne[1]*nb[1])
        // → reconstruct physical strides. Padded seq-views (nb[2] > ne[1]*nb[1])
        // already have correct physical strides.
        స్థిరరాశి తార్కిక gemma = interleaved &&
            ((int64_t)T->nb[2] < (int64_t)T->ne[1] * (int64_t)T->nb[1]);
        ఒకవేళ (gemma) {
            d.s01 = (int64_t)n_kv_heads * blk_చొప్పున_అడ్డువరుస;
            d.s02 = blk_చొప్పున_అడ్డువరుస;
        } లేకపోతే {
            d.s01 = d.nb1 / d.ts;
            d.s02 = d.nb2 / d.ts;
        }
    }
    తిరిగి d;
}

// Dequant one KV-head chunk into a dense [this_chunk x D] fp16 buffer.
స్థిర శూన్య mkl_fa_dequant_chunk(
    dpct::వరుస_ptr stream, స్థిరరాశి mkl_fa_kv_desc & d, ggml_tensor * dst_ctx,
    sycl::half * బహిర్గతం, పూర్ణాంకం ikvh, పూర్ణాంకం chunk_ప్రారంభం, పూర్ణాంకం this_chunk) {

    స్థిరరాశి int64_t D = d.D;
    ఎంపిక (d.mode) {
        స్థితి MKL_FA_KV_బహుళకం_F16_DENSE: {
            స్థిరరాశి అక్షరం * పునాది = d.దత్తాంశం + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_ప్రారంభం * d.nb1;
            stream->జ్ఞాపకం_నకలు(బహిర్గతం, పునాది, (size_t)this_chunk * D * పరిమాణం(sycl::half));
            ఆపు;
        }
        స్థితి MKL_FA_KV_బహుళకం_F16_INTERLEAVED: {
            స్థిరరాశి అక్షరం * పునాది = d.దత్తాంశం + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_ప్రారంభం * d.nb1;
            స్థిరరాశి int64_t అడ్డువరుస_halfs = d.nb1 / (int64_t)పరిమాణం(sycl::half);
            స్థిరరాశి sycl::half * src = (స్థిరరాశి sycl::half *)పునాది;
            stream->parallel_for(
                sycl::range<2>((size_t)this_chunk, (size_t)D),
                [=](sycl::అంశం<2> it) {
                    int64_t r = it.get_id(0);
                    int64_t c = it.get_id(1);
                    బహిర్గతం[r * D + c] = src[r * అడ్డువరుస_halfs + c];
                });
            ఆపు;
        }
        స్థితి MKL_FA_KV_బహుళకం_QUANT_CONTIG: {
            స్థిరరాశి అక్షరం * పునాది = d.దత్తాంశం + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_ప్రారంభం * d.nb1;
            గమ్యం_fp16_sycl_t గమ్యం_fp16 = ggml_get_గమ్యం_fp16_sycl(d.type, dst_ctx);
            గమ్యం_fp16(పునాది, బహిర్గతం, (int64_t)this_chunk * D, stream);
            ఆపు;
        }
        అప్రమేయ: {  // MKL_FA_KV_MODE_QUANT_NC
            గమ్యం_fp16_nc_sycl_t గమ్యం_fp16 = ggml_get_గమ్యం_fp16_nc_sycl(d.type);
            స్థిరరాశి int64_t పునాది_ఖండాలు = (int64_t)ikvh * d.s02
                + (int64_t)chunk_ప్రారంభం * d.s01;
            స్థిరరాశి అక్షరం * పునాది = d.దత్తాంశం + పునాది_ఖండాలు * d.ts;
            // ne02 = ne03 = 1 → s02/s03 inert; head+chunk offset carried by base.
            గమ్యం_fp16(పునాది, బహిర్గతం, D, this_chunk, 1, 1, d.s01, d.s02, d.s02, stream);
            ఆపు;
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
శూన్య ggml_sycl_flash_attn_ext_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * Q    = dst->src[0];
    స్థిరరాశి ggml_tensor * K    = dst->src[1];
    స్థిరరాశి ggml_tensor * V    = dst->src[2];
    స్థిరరాశి ggml_tensor * ముసుగు = dst->src[3];
    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    // --- Op params ---
    దశాంశం scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    జ్ఞాపకం_నకలు(&scale,         (స్థిరరాశి దశాంశం *)KQV->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max_bias,      (స్థిరరాశి దశాంశం *)KQV->op_params + 1, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&logit_softcap, (స్థిరరాశి దశాంశం *)KQV->op_params + 2, పరిమాణం(దశాంశం));

    స్థిరరాశి దశాంశం q_scale = scale;

    // --- Dimensions ---
    స్థిరరాశి పూర్ణాంకం DKQ        = (పూర్ణాంకం)K->ne[0];
    స్థిరరాశి పూర్ణాంకం DV         = (పూర్ణాంకం)V->ne[0];
    స్థిరరాశి పూర్ణాంకం n_queries  = (పూర్ణాంకం)Q->ne[1];
    స్థిరరాశి పూర్ణాంకం n_q_heads  = (పూర్ణాంకం)Q->ne[2];
    స్థిరరాశి పూర్ణాంకం n_kv_heads = (పూర్ణాంకం)K->ne[2];
    స్థిరరాశి పూర్ణాంకం n_batch    = (పూర్ణాంకం)Q->ne[3];
    స్థిరరాశి పూర్ణాంకం n_kv       = (పూర్ణాంకం)K->ne[1];
    స్థిరరాశి పూర్ణాంకం gqa_నిష్పత్తి  = n_q_heads / n_kv_heads;
    స్థిరరాశి పూర్ణాంకం n_query_అడ్డువరుసలు = n_queries * gqa_నిష్పత్తి;

    GGML_ASSERT(n_q_heads % n_kv_heads == 0);
    GGML_ASSERT(max_bias == 0.0f);  // ALiBi not supported
    GGML_ASSERT(Q->ne[3] == K->ne[3] || K->ne[3] == 1);

    స్థిరరాశి పూర్ణాంకం chunk_కొలత = std::min(MKL_FA_CHUNK_కొలత_KV, n_kv);

    // Query rows are processed in tiles of q_tile_rows so the score buffers
    // (KQ_f32/S_f16 = q_tile_rows * chunk_size) stay bounded regardless of
    // batch size. n_query_rows <= Q_TILE is a single tile (no extra work).
    స్థిర పూర్ణాంకం q_tile_env = ggml_sycl_get_env("GGML_SYCL_MKL_FA_Q_TILE", MKL_FA_Q_TILE);
    స్థిరరాశి పూర్ణాంకం q_tile_అడ్డువరుసలు = std::max(1, std::min(q_tile_env, n_query_అడ్డువరుసలు));

    స్థిరరాశి int64_t wg_కొలత = MKL_FA_WG_కొలత;

    // --- Debug output (gated by GGML_SYCL_MKL_FA_DEBUG=1) ---
    స్థిర పూర్ణాంకం mkl_call_లెక్క = 0;
    mkl_call_లెక్క++;
    స్థిర పూర్ణాంకం mkl_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    స్థిరరాశి తార్కిక do_print = (mkl_debug == 1);

    స్థిరరాశి int64_t q_అడ్డువరుస_stride  = Q->nb[1] / పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t q_తల_stride = Q->nb[2] / పరిమాణం(దశాంశం);

    // Alias the dequantized buffers only when K and V expose the same values.
    // MLA V is a narrower view of K and needs its own strided dequantization.
    స్థిరరాశి తార్కిక V_is_K_view = V->ne[0] == K->ne[0] && V->view_src
        && (V->view_src == K || (V->view_src == K->view_src
            && V->view_offs == K->view_offs));

    // Early interleaved detection for debug output.
    // True interleaved detection happens after dequant (nb12_fp16 == nb11_fp16),
    // but we can pre-detect on the original tensor strides.
    స్థిరరాశి తార్కిక k_early_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]);
    స్థిరరాశి తార్కిక v_early_interleaved =
        !V_is_K_view && ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]);

    ఒకవేళ (do_print) {
        GGML_LOG_INFO("[MKL-FA] #%d D=%d DV=%d n_q=%d n_kv=%d "
                "n_qh=%d n_kvh=%d gqa=%d batch=%d K=%s V=%s "
                "chunk=%d buf=%.1fMB%s%s\n",
                mkl_call_లెక్క, DKQ, DV, n_queries, n_kv,
                n_q_heads, n_kv_heads, gqa_నిష్పత్తి, n_batch,
                ggml_type_name(K->type), ggml_type_name(V->type),
                chunk_కొలత,
                (ద్విగుణ)((int64_t)n_query_అడ్డువరుసలు * chunk_కొలత * పరిమాణం(దశాంశం))
                    / (1024.0 * 1024.0),
                k_early_interleaved ? " K_ILV" : "",
                v_early_interleaved ? " V_ILV" : "");
        GGML_LOG_INFO("[MKL-FA] #%d Q-nb1=%lld Q-nb2=%lld "
                "q_rs=%lld q_hs=%lld dst_rs=%lld dst_hs=%lld\n",
                mkl_call_లెక్క,
                (దీర్ఘ దీర్ఘ)Q->nb[1], (దీర్ఘ దీర్ఘ)Q->nb[2],
                (దీర్ఘ దీర్ఘ)q_అడ్డువరుస_stride, (దీర్ఘ దీర్ఘ)q_తల_stride,
                (దీర్ఘ దీర్ఘ)(KQV->nb[1] / పరిమాణం(దశాంశం)),
                (దీర్ఘ దీర్ఘ)(KQV->nb[2] / పరిమాణం(దశాంశం)));
    }

    // --- Stream and allocators ---
    dpct::వరుస_ptr stream = ctx.stream();

#నిర్వచించు MKL_TAKE_సమయం(t0)  స్వయంచాలక t0 = std::chrono::steady_clock::now()
#నిర్వచించు MKL_ACCUM(acc, t0) చేయి { ఒకవేళ (do_print) { \
    acc += (int64_t)std::chrono::duration_cast \
    <std::chrono::microseconds>(std::chrono::steady_clock::now() - (t0)).count(); \
} } అయితే(0)

    int64_t gemm_kq_సమయం_us  = 0;
    int64_t gemm_vkq_సమయం_us = 0;
    int64_t softmax_సమయం_us  = 0;
    int64_t dequant_సమయం_us  = 0;

    MKL_TAKE_సమయం(t_deq);

    // --- K/V dequant descriptors ---
    // Dequant is done per-chunk inside the KV loop (footprint independent of
    // context). Output is always dense row-major fp16 [this_chunk x D], lda=D.
    // Interleaved detection: ne[1]*nb[1] != nb[2] means heads are interleaved.
    స్థిరరాశి తార్కిక k_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
    స్థిరరాశి తార్కిక v_interleaved =
        ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;

    స్థిరరాశి mkl_fa_kv_desc K_desc = mkl_fa_make_desc(K, k_interleaved, n_kv_heads);
    స్థిరరాశి mkl_fa_kv_desc V_desc = V_is_K_view
        ? K_desc : mkl_fa_make_desc(V, v_interleaved, n_kv_heads);

    MKL_ACCUM(dequant_సమయం_us, t_deq);

    // --- Resolve mask pointers ---
    int64_t ముసుగు_తల_stride = 0;
    int64_t ముసుగు_అడ్డువరుస_stride  = 0;
    పూర్ణాంకం     ముసుగు_n_heads     = 0;

    ఒకవేళ (ముసుగు) {
        // Use actual fp16 device size (2 bytes), NOT sizeof(sycl::half)
        // which may be 4 on the host in oneAPI.
        ముసుగు_తల_stride = ముసుగు->nb[2] / 2;
        ముసుగు_అడ్డువరుస_stride  = ముసుగు->nb[1] / 2;
        ముసుగు_n_heads     = (పూర్ణాంకం)ముసుగు->ne[2];
    }

    // --- Allocate intermediates from pool ---
    ggml_sycl_pool & pool = ctx.pool();

    ggml_sycl_pool_alloc<దశాంశం>      KQ_f32(pool);      // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<sycl::half> S_f16(pool);       // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<దశాంశం>      VKQ_chunk(pool);   // [q_tile_rows x DV]
    ggml_sycl_pool_alloc<దశాంశం>      VKQ_accum(pool);   // [n_query_rows x DV] (full)
    ggml_sycl_pool_alloc<దశాంశం>      KQ_max(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<దశాంశం>      KQ_కూడిక(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<sycl::half> Q_తల_f16(pool);  // [n_query_rows x DKQ] (full)
    ggml_sycl_pool_alloc<sycl::half> K_chunk_f16(pool); // [chunk x DKQ] (per-chunk dequant)
    ggml_sycl_pool_alloc<sycl::half> V_chunk_f16(pool); // [chunk x DV] (per-chunk dequant)

    KQ_f32.alloc((size_t)q_tile_అడ్డువరుసలు * chunk_కొలత);
    S_f16.alloc((size_t)q_tile_అడ్డువరుసలు * chunk_కొలత);
    VKQ_chunk.alloc((size_t)q_tile_అడ్డువరుసలు * DV);
    VKQ_accum.alloc((size_t)n_query_అడ్డువరుసలు * DV);
    KQ_max.alloc(n_query_అడ్డువరుసలు);
    KQ_కూడిక.alloc(n_query_అడ్డువరుసలు);
    Q_తల_f16.alloc((size_t)n_query_అడ్డువరుసలు * DKQ);
    K_chunk_f16.alloc((size_t)chunk_కొలత * DKQ);

    sycl::half * V_chunk_f16_ptr;
    ఒకవేళ (V_is_K_view) {
        V_chunk_f16_ptr = K_chunk_f16.ptr;   // V aliases K (DV == DKQ)
    } లేకపోతే {
        V_chunk_f16.alloc((size_t)chunk_కొలత * DV);
        V_chunk_f16_ptr = V_chunk_f16.ptr;
    }

    sycl::half * Q_తల_f16_ptr  = Q_తల_f16.ptr;
    దశాంశం      * KQ_f32_ptr      = KQ_f32.ptr;
    sycl::half * S_f16_ptr       = S_f16.ptr;
    దశాంశం      * VKQ_chunk_ptr   = VKQ_chunk.ptr;
    దశాంశం      * VKQ_accum_ptr   = VKQ_accum.ptr;
    దశాంశం      * KQ_max_ptr      = KQ_max.ptr;
    దశాంశం      * KQ_కూడిక_ptr      = KQ_కూడిక.ptr;
    sycl::half * K_chunk_f16_ptr = K_chunk_f16.ptr;

    స్థిరరాశి దశాంశం alpha = 1.0f;
    స్థిరరాశి దశాంశం beta  = 0.0f;

    కొరకు (పూర్ణాంకం ib = 0; ib < n_batch; ib++) {
        స్థిరరాశి దశాంశం * Q_batch = (స్థిరరాశి దశాంశం *)Q->దత్తాంశం
            + ib * (Q->nb[3] / పరిమాణం(దశాంశం));
        దశాంశం * dst_batch = (దశాంశం *)KQV->దత్తాంశం
            + ib * (KQV->nb[3] / పరిమాణం(దశాంశం));

        స్థిరరాశి sycl::half * ముసుగు_batch = శూన్యసూచిక;
        ఒకవేళ (ముసుగు) {
            పూర్ణాంకం m_batch = (ముసుగు->ne[3] > 1) ? ib : 0;
            ముసుగు_batch = (స్థిరరాశి sycl::half *)ముసుగు->దత్తాంశం
                + m_batch * (ముసుగు->nb[3] / 2);  // 2 = actual fp16 device size
        }

        కొరకు (పూర్ణాంకం ikvh = 0; ikvh < n_kv_heads; ikvh++) {
            పూర్ణాంకం kvh_పునాది_తల = ikvh * gqa_నిష్పత్తి;

            // 1. Pack all GQA Q heads into fp16 (full n_query_rows)
            mkl_fa_pack_q_fp16(stream,
                Q_తల_f16_ptr, Q_batch,
                n_queries, DKQ,
                gqa_నిష్పత్తి, kvh_పునాది_తల,
                q_scale, q_అడ్డువరుస_stride, q_తల_stride, wg_కొలత);

            // 2. Initialize softmax state (full n_query_rows)
            mkl_fa_init_softmax_state(stream,
                KQ_max_ptr, KQ_కూడిక_ptr, VKQ_accum_ptr,
                n_query_అడ్డువరుసలు, DV, wg_కొలత);

            // Sync before MKL GEMM (MKL may use an internal queue)
            stream->wait();

            // 3. KV chunk loop (OUTER): dequant each chunk once, then tile queries.
            కొరకు (పూర్ణాంకం chunk_ప్రారంభం = 0; chunk_ప్రారంభం < n_kv; chunk_ప్రారంభం += chunk_కొలత) {
                పూర్ణాంకం this_chunk = std::min(chunk_కొలత, n_kv - chunk_ప్రారంభం);

                // 3a. Dequant this KV chunk to dense fp16 (once per chunk)
                {
                    MKL_TAKE_సమయం(t0);
                    mkl_fa_dequant_chunk(stream, K_desc, KQV,
                        K_chunk_f16_ptr, ikvh, chunk_ప్రారంభం, this_chunk);
                    ఒకవేళ (!V_is_K_view) {
                        mkl_fa_dequant_chunk(stream, V_desc, KQV,
                            V_chunk_f16_ptr, ikvh, chunk_ప్రారంభం, this_chunk);
                    }
                    stream->wait();  // dequant must be ready before MKL GEMM
                    MKL_ACCUM(dequant_సమయం_us, t0);
                }

                // 3b. Query tile loop (INNER) — bounds KQ_f32/S_f16 footprint.
                కొరకు (పూర్ణాంకం q0 = 0; q0 < n_query_అడ్డువరుసలు; q0 += q_tile_అడ్డువరుసలు) {
                    పూర్ణాంకం q_అడ్డువరుసలు = std::min(q_tile_అడ్డువరుసలు, n_query_అడ్డువరుసలు - q0);

                    // GEMM: KQ = Q_tile × K_chunk^T
                    {
                        MKL_TAKE_సమయం(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::trans, transpose::nontrans,
                            this_chunk, q_అడ్డువరుసలు, DKQ,
                            alpha,
                            K_chunk_f16_ptr, DKQ,
                            Q_తల_f16_ptr + (int64_t)q0 * DKQ, DKQ,
                            beta,
                            KQ_f32_ptr, this_chunk);
                        ప్రయత్నించు { ev.wait_and_throw(); } పట్టుకో (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM KQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM KQ failed");
                        }
                        MKL_ACCUM(gemm_kq_సమయం_us, t0);
                    }
                    // Online softmax over this chunk for this query tile
                    {
                        MKL_TAKE_సమయం(t0);
                        mkl_fa_online_softmax_chunk(stream,
                            KQ_f32_ptr, S_f16_ptr,
                            KQ_max_ptr, KQ_కూడిక_ptr, VKQ_accum_ptr,
                            q0, q_అడ్డువరుసలు, n_queries, DV,
                            this_chunk, chunk_ప్రారంభం,
                            kvh_పునాది_తల,
                            ముసుగు_batch, ముసుగు_తల_stride,
                            ముసుగు_అడ్డువరుస_stride, ముసుగు_n_heads,
                            logit_softcap, wg_కొలత);
                        stream->wait();  // S_f16 must be ready for GEMM
                        MKL_ACCUM(softmax_సమయం_us, t0);
                    }

                    // GEMM: VKQ_chunk = S × V_chunk
                    {
                        MKL_TAKE_సమయం(t0);
                        sycl::event ev = gemm(*stream,
                            transpose::nontrans, transpose::nontrans,
                            DV, q_అడ్డువరుసలు, this_chunk,
                            alpha,
                            V_chunk_f16_ptr, DV,
                            S_f16_ptr, this_chunk,
                            beta,
                            VKQ_chunk_ptr, DV);
                        ప్రయత్నించు { ev.wait_and_throw(); } పట్టుకో (sycl::exception & e) {
                            GGML_LOG_INFO("[MKL-FA] GEMM VKQ: %s\n", e.what());
                            GGML_ABORT("MKL GEMM VKQ failed");
                        }
                        MKL_ACCUM(gemm_vkq_సమయం_us, t0);
                    }
                    // VKQ_accum[q0..] += VKQ_chunk
                    {
                        స్థిరరాశి int64_t n_మొత్తం = (int64_t)q_అడ్డువరుసలు * DV;
                        స్థిరరాశి int64_t wg = ((n_మొత్తం + wg_కొలత - 1) / wg_కొలత)
                            * wg_కొలత;
                        దశాంశం * accum = VKQ_accum_ptr + (int64_t)q0 * DV;
                        stream->submit([&](sycl::handler & cgh) {
                            cgh.parallel_for(sycl::nd_range<1>(wg, wg_కొలత),
                                [=](sycl::nd_అంశం<1> అంశం) {
                                    int64_t i = అంశం.get_global_id(0);
                                    ఒకవేళ (i < n_మొత్తం) {
                                        accum[i] += VKQ_chunk_ptr[i];
                                    }
                                });
                        });
                    }
                }
            }

            // 4. Normalize and scatter each GQA head to dst
            కొరకు (పూర్ణాంకం iqg = 0; iqg < gqa_నిష్పత్తి; iqg++) {
                పూర్ణాంకం     iqh        = kvh_పునాది_తల + iqg;
                int64_t src_స్థానభ్రంశం = (int64_t)iqg * n_queries * DV;
                mkl_fa_normalize_తల(stream,
                    dst_batch, VKQ_accum_ptr, KQ_కూడిక_ptr,
                    iqh, n_queries, DV, n_q_heads,
                    src_స్థానభ్రంశం, wg_కొలత);
            }
        }
    }

#నిర్వచనం_రద్దు MKL_TAKE_సమయం
#నిర్వచనం_రద్దు MKL_ACCUM

    ఒకవేళ (do_print) {
        స్థిరరాశి int64_t v_chunk_elems = V_is_K_view ? 0 : (int64_t)chunk_కొలత * DV;
        ద్విగుణ మొత్తం_mb = (ద్విగుణ)(
            (int64_t)q_tile_అడ్డువరుసలు * chunk_కొలత * పరిమాణం(దశాంశం)      // KQ_f32
          + (int64_t)q_tile_అడ్డువరుసలు * chunk_కొలత * పరిమాణం(sycl::half) // S_f16
          + (int64_t)q_tile_అడ్డువరుసలు * DV * పరిమాణం(దశాంశం)              // VKQ_chunk
          + (int64_t)n_query_అడ్డువరుసలు * DV * పరిమాణం(దశాంశం)             // VKQ_accum
          + (int64_t)n_query_అడ్డువరుసలు * పరిమాణం(దశాంశం)                  // KQ_max
          + (int64_t)n_query_అడ్డువరుసలు * పరిమాణం(దశాంశం)                  // KQ_sum
          + (int64_t)n_query_అడ్డువరుసలు * DKQ * పరిమాణం(sycl::half)       // Q_head_f16
          + (int64_t)chunk_కొలత * DKQ * పరిమాణం(sycl::half)         // K_chunk_f16
          + v_chunk_elems * (int64_t)పరిమాణం(sycl::half)            // V_chunk_f16
        ) / (1024.0 * 1024.0);
        GGML_LOG_INFO("[MKL-FA] #%d n_kv=%d n_q=%d q_tile=%d time_us: "
                "dequant=%lld GEMM_KQ=%lld softmax=%lld GEMM_VKQ=%lld "
                "buf_mb=%.1f\n",
                mkl_call_లెక్క, n_kv, n_queries, q_tile_అడ్డువరుసలు,
                (దీర్ఘ దీర్ఘ)dequant_సమయం_us,
                (దీర్ఘ దీర్ఘ)gemm_kq_సమయం_us,
                (దీర్ఘ దీర్ఘ)softmax_సమయం_us,
                (దీర్ఘ దీర్ఘ)gemm_vkq_సమయం_us,
                మొత్తం_mb);
    }
}
