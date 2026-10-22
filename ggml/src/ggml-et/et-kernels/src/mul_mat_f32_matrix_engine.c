#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"
#अन्तर्भाव "tensor.h"

#अन्तर्भाव <etsoc/common/utils.h>
#अन्तर्भाव <stdint.h>

/*
 * F32 Matrix Multiply for ET-SoC-1 — TensorFMA32.
 *
 * K-parallel + interleaved tiles + ring reduce.
 * No batched-K yet (needs investigation on hang).
 * This is the last known working version.
 */

#परिभाषय अङ्कः_COMPUTE_SHIRES 32
#परिभाषय MINIONS_प्रत्येकम्_SHIRE  32
#परिभाषय TILE_K             16
#परिभाषय TILE_M             16

/* ── Tuning knobs ───────────────────────────────────────────────────── */
#परिभाषय TILE_N      16
#परिभाषय CACHEOP_MAX 0
#परिभाषय REP_दरः    0

/* ─────────────────────────────────────────────────────────────────── */

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_द्विमान_params * params, शून्य * env) {
    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    यदि (shire_id >= अङ्कः_COMPUTE_SHIRES) {
        प्रत्यागम 0;
    }
    यदि (hart_id & 1) {
        प्रत्यागम 0;
    }

    uint64_t local_minion = (hart_id >> 1) & 0x1F;
    uint64_t my_minion_id = get_minion_id();

    ध्रुव int64_t K = params->src0.ne[0];
    ध्रुव int64_t M = params->src0.ne[1];
    ध्रुव int64_t N = params->src1.ne[1];

    ध्रुव int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    ध्रुव int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    ध्रुव int64_t nb1_0 = params->src0.nb[1];
    ध्रुव int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];
    ध्रुव int64_t nb1_1 = params->src1.nb[1];
    ध्रुव int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];
    ध्रुव int64_t nb1_d = params->dst.nb[1];
    ध्रुव int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    ध्रुव वर्ण * src0_आधारः = (ध्रुव वर्ण *) params->src0.दत्त;
    ध्रुव वर्ण * src1_आधारः = (ध्रुव वर्ण *) params->src1.दत्त;
    वर्ण *       dst_आधारः  = (वर्ण *) params->dst.दत्त;

    सज्जता_cache_scp();
#यदि CACHEOP_MAX > 0 || REP_दरः > 0
    ucache_control(1, REP_दरः, CACHEOP_MAX);
#यदि_अन्तः
    CLEAR_TENSOR_ERROR;

    ध्रुव int64_t m_tiles     = M / TILE_M;
    ध्रुव int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    ध्रुव int64_t batch_गणनम् = ne2_1 * ne3_1;
    ध्रुव int64_t आधारः_tiles  = m_tiles * n_tiles * batch_गणनम्;

    ध्रुव int64_t r2 = ne2_1 / ne2_0;
    ध्रुव int64_t r3 = ne3_1 / ne3_0;

    ध्रुव int64_t योग_harts = अङ्कः_COMPUTE_SHIRES * MINIONS_प्रत्येकम्_SHIRE;
    ध्रुव int64_t k_चरणाः     = K / TILE_K;
    int64_t       k_splits    = 1;
    यदि (आधारः_tiles < योग_harts) {
        k_splits   = (योग_harts + आधारः_tiles - 1) / आधारः_tiles;
        int64_t ks = 1;
        यावत् (ks * 2 <= k_splits && ks * 2 <= 32 && k_चरणाः % (ks * 2) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ध्रुव int64_t tiles_प्रत्येकम्_shire = MINIONS_प्रत्येकम्_SHIRE / k_splits;
    ध्रुव int64_t k_विभज         = local_minion % k_splits;
    ध्रुव int64_t local_tile_idx  = local_minion / k_splits;
    ध्रुव int64_t tiles_stride    = (int64_t) अङ्कः_COMPUTE_SHIRES * tiles_प्रत्येकम्_shire;

    ध्रुव int64_t k_चरणाः_प्रत्येकम्_विभज = k_चरणाः / k_splits;
    ध्रुव int64_t k_start           = k_विभज * k_चरणाः_प्रत्येकम्_विभज * TILE_K;
    ध्रुव int64_t k_अन्तः             = k_start + k_चरणाः_प्रत्येकम्_विभज * TILE_K;

    ध्रुव uint64_t गणः_आधारः_global = my_minion_id - k_विभज;

    प्रति (int64_t tile = (int64_t) shire_id + local_tile_idx * अङ्कः_COMPUTE_SHIRES; tile < आधारः_tiles;
         tile += tiles_stride) {
        ध्रुव int64_t tiles_प्रत्येकम्_batch = m_tiles * n_tiles;
        ध्रुव int64_t batch_idx       = tile / tiles_प्रत्येकम्_batch;
        ध्रुव int64_t tile_अन्तर्_batch   = tile % tiles_प्रत्येकम्_batch;
        ध्रुव int64_t nb_idx          = tile_अन्तर्_batch / m_tiles;
        ध्रुव int64_t mb_idx          = tile_अन्तर्_batch % m_tiles;

        ध्रुव int64_t i3   = batch_idx / ne2_1;
        ध्रुव int64_t i2   = batch_idx % ne2_1;
        ध्रुव int64_t i2_0 = i2 / r2;
        ध्रुव int64_t i3_0 = i3 / r3;

        ध्रुव वर्ण * src0_batch = src0_आधारः + i3_0 * nb3_0 + i2_0 * nb2_0;
        ध्रुव वर्ण * src1_batch = src1_आधारः + i3 * nb3_1 + i2 * nb2_1;
        वर्ण *       dst_batch  = dst_आधारः + i3 * nb3_d + i2 * nb2_d;

        ध्रुव int64_t mb    = mb_idx * TILE_M;
        ध्रुव int64_t nb    = nb_idx * TILE_N;
        ध्रुव int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        प्रति (int64_t kb = k_start; kb < k_अन्तः; kb += TILE_K) {
            tensor_आरोपय(false, false, 0, 0, 0, (uint64_t) (src1_batch + nb * nb1_1 + kb * परिमाण(भिन्न)), 0, n_cur - 1,
                        (uint64_t) nb1_1, 0);

            tensor_आरोपय(false, false, TILE_K, 7, 0, (uint64_t) (src0_batch + mb * nb1_0 + kb * परिमाण(भिन्न)), 0,
                        TILE_K - 1, (uint64_t) nb1_0, 1);

            tensor_wait(TENSOR_आरोपय_WAIT_0);
            tensor_wait(TENSOR_आरोपय_WAIT_1);

            tensor_fma(false, 3, n_cur - 1, TILE_K - 1, 0, false, false, false, false, TILE_K, 0, 0, (kb == k_start));

            tensor_wait(TENSOR_FMA_WAIT);
        }

        यदि (k_splits > 1) {
            ध्रुव uint64_t अङ्कः_regs = (uint64_t) n_cur * 2;

            यदि (k_विभज > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, अङ्कः_regs, गणः_आधारः_global + k_विभज - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
            यदि (k_विभज < k_splits - 1) {
                tensor_reduce_send(0, अङ्कः_regs, गणः_आधारः_global + k_विभज + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        यदि (k_विभज == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * परिमाण(भिन्न)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    प्रत्यागम 0;
}
