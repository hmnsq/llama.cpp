#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"
#అన్తర్భావ "tensor.h"

#అన్తర్భావ <etsoc/common/utils.h>
#అన్తర్భావ <stdint.h>

/*
 * F32 Matrix Multiply for ET-SoC-1 — TensorFMA32.
 *
 * K-parallel + interleaved tiles + ring reduce.
 * No batched-K yet (needs investigation on hang).
 * This is the last known working version.
 */

#నిర్వచించు అంకె_COMPUTE_SHIRES 32
#నిర్వచించు MINIONS_చొప్పున_SHIRE  32
#నిర్వచించు TILE_K             16
#నిర్వచించు TILE_M             16

/* ── Tuning knobs ───────────────────────────────────────────────────── */
#నిర్వచించు TILE_N      16
#నిర్వచించు CACHEOP_MAX 0
#నిర్వచించు REP_రేటు    0

/* ─────────────────────────────────────────────────────────────────── */

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_ద్విమాన_params * params, శూన్య * env) {
    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    ఒకవేళ (shire_id >= అంకె_COMPUTE_SHIRES) {
        తిరిగి 0;
    }
    ఒకవేళ (hart_id & 1) {
        తిరిగి 0;
    }

    uint64_t local_minion = (hart_id >> 1) & 0x1F;
    uint64_t my_minion_id = get_minion_id();

    స్థిరరాశి int64_t K = params->src0.ne[0];
    స్థిరరాశి int64_t M = params->src0.ne[1];
    స్థిరరాశి int64_t N = params->src1.ne[1];

    స్థిరరాశి int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    స్థిరరాశి int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    స్థిరరాశి int64_t nb1_0 = params->src0.nb[1];
    స్థిరరాశి int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];
    స్థిరరాశి int64_t nb1_1 = params->src1.nb[1];
    స్థిరరాశి int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];
    స్థిరరాశి int64_t nb1_d = params->dst.nb[1];
    స్థిరరాశి int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    స్థిరరాశి అక్షరం * src0_పునాది = (స్థిరరాశి అక్షరం *) params->src0.దత్తాంశం;
    స్థిరరాశి అక్షరం * src1_పునాది = (స్థిరరాశి అక్షరం *) params->src1.దత్తాంశం;
    అక్షరం *       dst_పునాది  = (అక్షరం *) params->dst.దత్తాంశం;

    అమరిక_cache_scp();
#ఒకవేళ CACHEOP_MAX > 0 || REP_రేటు > 0
    ucache_control(1, REP_రేటు, CACHEOP_MAX);
#అయితే_ముగింపు
    CLEAR_TENSOR_ERROR;

    స్థిరరాశి int64_t m_tiles     = M / TILE_M;
    స్థిరరాశి int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    స్థిరరాశి int64_t batch_లెక్క = ne2_1 * ne3_1;
    స్థిరరాశి int64_t పునాది_tiles  = m_tiles * n_tiles * batch_లెక్క;

    స్థిరరాశి int64_t r2 = ne2_1 / ne2_0;
    స్థిరరాశి int64_t r3 = ne3_1 / ne3_0;

    స్థిరరాశి int64_t మొత్తం_harts = అంకె_COMPUTE_SHIRES * MINIONS_చొప్పున_SHIRE;
    స్థిరరాశి int64_t k_అడుగులు     = K / TILE_K;
    int64_t       k_splits    = 1;
    ఒకవేళ (పునాది_tiles < మొత్తం_harts) {
        k_splits   = (మొత్తం_harts + పునాది_tiles - 1) / పునాది_tiles;
        int64_t ks = 1;
        అయితే (ks * 2 <= k_splits && ks * 2 <= 32 && k_అడుగులు % (ks * 2) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    స్థిరరాశి int64_t tiles_చొప్పున_shire = MINIONS_చొప్పున_SHIRE / k_splits;
    స్థిరరాశి int64_t k_విభజించు         = local_minion % k_splits;
    స్థిరరాశి int64_t local_tile_idx  = local_minion / k_splits;
    స్థిరరాశి int64_t tiles_stride    = (int64_t) అంకె_COMPUTE_SHIRES * tiles_చొప్పున_shire;

    స్థిరరాశి int64_t k_అడుగులు_చొప్పున_విభజించు = k_అడుగులు / k_splits;
    స్థిరరాశి int64_t k_ప్రారంభం           = k_విభజించు * k_అడుగులు_చొప్పున_విభజించు * TILE_K;
    స్థిరరాశి int64_t k_చివర             = k_ప్రారంభం + k_అడుగులు_చొప్పున_విభజించు * TILE_K;

    స్థిరరాశి uint64_t సమూహం_పునాది_global = my_minion_id - k_విభజించు;

    కొరకు (int64_t tile = (int64_t) shire_id + local_tile_idx * అంకె_COMPUTE_SHIRES; tile < పునాది_tiles;
         tile += tiles_stride) {
        స్థిరరాశి int64_t tiles_చొప్పున_batch = m_tiles * n_tiles;
        స్థిరరాశి int64_t batch_idx       = tile / tiles_చొప్పున_batch;
        స్థిరరాశి int64_t tile_లోపల_batch   = tile % tiles_చొప్పున_batch;
        స్థిరరాశి int64_t nb_idx          = tile_లోపల_batch / m_tiles;
        స్థిరరాశి int64_t mb_idx          = tile_లోపల_batch % m_tiles;

        స్థిరరాశి int64_t i3   = batch_idx / ne2_1;
        స్థిరరాశి int64_t i2   = batch_idx % ne2_1;
        స్థిరరాశి int64_t i2_0 = i2 / r2;
        స్థిరరాశి int64_t i3_0 = i3 / r3;

        స్థిరరాశి అక్షరం * src0_batch = src0_పునాది + i3_0 * nb3_0 + i2_0 * nb2_0;
        స్థిరరాశి అక్షరం * src1_batch = src1_పునాది + i3 * nb3_1 + i2 * nb2_1;
        అక్షరం *       dst_batch  = dst_పునాది + i3 * nb3_d + i2 * nb2_d;

        స్థిరరాశి int64_t mb    = mb_idx * TILE_M;
        స్థిరరాశి int64_t nb    = nb_idx * TILE_N;
        స్థిరరాశి int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        కొరకు (int64_t kb = k_ప్రారంభం; kb < k_చివర; kb += TILE_K) {
            tensor_ఎక్కించు(false, false, 0, 0, 0, (uint64_t) (src1_batch + nb * nb1_1 + kb * పరిమాణం(దశాంశం)), 0, n_cur - 1,
                        (uint64_t) nb1_1, 0);

            tensor_ఎక్కించు(false, false, TILE_K, 7, 0, (uint64_t) (src0_batch + mb * nb1_0 + kb * పరిమాణం(దశాంశం)), 0,
                        TILE_K - 1, (uint64_t) nb1_0, 1);

            tensor_wait(TENSOR_ఎక్కించు_WAIT_0);
            tensor_wait(TENSOR_ఎక్కించు_WAIT_1);

            tensor_fma(false, 3, n_cur - 1, TILE_K - 1, 0, false, false, false, false, TILE_K, 0, 0, (kb == k_ప్రారంభం));

            tensor_wait(TENSOR_FMA_WAIT);
        }

        ఒకవేళ (k_splits > 1) {
            స్థిరరాశి uint64_t అంకె_regs = (uint64_t) n_cur * 2;

            ఒకవేళ (k_విభజించు > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, అంకె_regs, సమూహం_పునాది_global + k_విభజించు - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
            ఒకవేళ (k_విభజించు < k_splits - 1) {
                tensor_reduce_send(0, అంకె_regs, సమూహం_పునాది_global + k_విభజించు + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        ఒకవేళ (k_విభజించు == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * పరిమాణం(దశాంశం)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    తిరిగి 0;
}
