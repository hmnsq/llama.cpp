#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"
#ಅನ್ತರ್ಭಾವ "tensor.h"

#ಅನ್ತರ್ಭಾವ <etsoc/common/utils.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

/*
 * F32 Matrix Multiply for ET-SoC-1 — TensorFMA32.
 *
 * K-parallel + interleaved tiles + ring reduce.
 * No batched-K yet (needs investigation on hang).
 * This is the last known working version.
 */

#ವ್ಯಾಖ್ಯಾನಿಸು ಅಂಕೆ_COMPUTE_SHIRES 32
#ವ್ಯಾಖ್ಯಾನಿಸು MINIONS_ಪ್ರತಿ_SHIRE  32
#ವ್ಯಾಖ್ಯಾನಿಸು TILE_K             16
#ವ್ಯಾಖ್ಯಾನಿಸು TILE_M             16

/* ── Tuning knobs ───────────────────────────────────────────────────── */
#ವ್ಯಾಖ್ಯಾನಿಸು TILE_N      16
#ವ್ಯಾಖ್ಯಾನಿಸು CACHEOP_MAX 0
#ವ್ಯಾಖ್ಯಾನಿಸು REP_ದರ    0

/* ─────────────────────────────────────────────────────────────────── */

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params * params, ಶೂನ್ಯ * env) {
    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    ಆದರೆ (shire_id >= ಅಂಕೆ_COMPUTE_SHIRES) {
        ಹಿಂತಿರುಗಿ 0;
    }
    ಆದರೆ (hart_id & 1) {
        ಹಿಂತಿರುಗಿ 0;
    }

    uint64_t local_minion = (hart_id >> 1) & 0x1F;
    uint64_t my_minion_id = get_minion_id();

    ಸ್ಥಿರಾಂಕ int64_t K = params->src0.ne[0];
    ಸ್ಥಿರಾಂಕ int64_t M = params->src0.ne[1];
    ಸ್ಥಿರಾಂಕ int64_t N = params->src1.ne[1];

    ಸ್ಥಿರಾಂಕ int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    ಸ್ಥಿರಾಂಕ int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    ಸ್ಥಿರಾಂಕ int64_t nb1_0 = params->src0.nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];
    ಸ್ಥಿರಾಂಕ int64_t nb1_1 = params->src1.nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];
    ಸ್ಥಿರಾಂಕ int64_t nb1_d = params->dst.nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) params->src0.ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src1_ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) params->src1.ದತ್ತಾಂಶ;
    ಅಕ್ಷರ *       dst_ತಳ  = (ಅಕ್ಷರ *) params->dst.ದತ್ತಾಂಶ;

    ಸಿದ್ಧತೆ_cache_scp();
#ಆದರೆ CACHEOP_MAX > 0 || REP_ದರ > 0
    ucache_control(1, REP_ದರ, CACHEOP_MAX);
#ಆದರೆ_ಅಂತ್ಯ
    CLEAR_TENSOR_ERROR;

    ಸ್ಥಿರಾಂಕ int64_t m_tiles     = M / TILE_M;
    ಸ್ಥಿರಾಂಕ int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    ಸ್ಥಿರಾಂಕ int64_t batch_ಎಣಿಕೆ = ne2_1 * ne3_1;
    ಸ್ಥಿರಾಂಕ int64_t ತಳ_tiles  = m_tiles * n_tiles * batch_ಎಣಿಕೆ;

    ಸ್ಥಿರಾಂಕ int64_t r2 = ne2_1 / ne2_0;
    ಸ್ಥಿರಾಂಕ int64_t r3 = ne3_1 / ne3_0;

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_harts = ಅಂಕೆ_COMPUTE_SHIRES * MINIONS_ಪ್ರತಿ_SHIRE;
    ಸ್ಥಿರಾಂಕ int64_t k_ಹೆಜ್ಜೆಗಳು     = K / TILE_K;
    int64_t       k_splits    = 1;
    ಆದರೆ (ತಳ_tiles < ಒಟ್ಟು_harts) {
        k_splits   = (ಒಟ್ಟು_harts + ತಳ_tiles - 1) / ತಳ_tiles;
        int64_t ks = 1;
        ಆಗ (ks * 2 <= k_splits && ks * 2 <= 32 && k_ಹೆಜ್ಜೆಗಳು % (ks * 2) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ಸ್ಥಿರಾಂಕ int64_t tiles_ಪ್ರತಿ_shire = MINIONS_ಪ್ರತಿ_SHIRE / k_splits;
    ಸ್ಥಿರಾಂಕ int64_t k_ವಿಭಜಿಸು         = local_minion % k_splits;
    ಸ್ಥಿರಾಂಕ int64_t local_tile_idx  = local_minion / k_splits;
    ಸ್ಥಿರಾಂಕ int64_t tiles_stride    = (int64_t) ಅಂಕೆ_COMPUTE_SHIRES * tiles_ಪ್ರತಿ_shire;

    ಸ್ಥಿರಾಂಕ int64_t k_ಹೆಜ್ಜೆಗಳು_ಪ್ರತಿ_ವಿಭಜಿಸು = k_ಹೆಜ್ಜೆಗಳು / k_splits;
    ಸ್ಥಿರಾಂಕ int64_t k_ಆರಂಭ           = k_ವಿಭಜಿಸು * k_ಹೆಜ್ಜೆಗಳು_ಪ್ರತಿ_ವಿಭಜಿಸು * TILE_K;
    ಸ್ಥಿರಾಂಕ int64_t k_ಕೊನೆ             = k_ಆರಂಭ + k_ಹೆಜ್ಜೆಗಳು_ಪ್ರತಿ_ವಿಭಜಿಸು * TILE_K;

    ಸ್ಥಿರಾಂಕ uint64_t ಗುಂಪು_ತಳ_global = my_minion_id - k_ವಿಭಜಿಸು;

    ಗಾಗಿ (int64_t tile = (int64_t) shire_id + local_tile_idx * ಅಂಕೆ_COMPUTE_SHIRES; tile < ತಳ_tiles;
         tile += tiles_stride) {
        ಸ್ಥಿರಾಂಕ int64_t tiles_ಪ್ರತಿ_batch = m_tiles * n_tiles;
        ಸ್ಥಿರಾಂಕ int64_t batch_idx       = tile / tiles_ಪ್ರತಿ_batch;
        ಸ್ಥಿರಾಂಕ int64_t tile_ಒಳ_batch   = tile % tiles_ಪ್ರತಿ_batch;
        ಸ್ಥಿರಾಂಕ int64_t nb_idx          = tile_ಒಳ_batch / m_tiles;
        ಸ್ಥಿರಾಂಕ int64_t mb_idx          = tile_ಒಳ_batch % m_tiles;

        ಸ್ಥಿರಾಂಕ int64_t i3   = batch_idx / ne2_1;
        ಸ್ಥಿರಾಂಕ int64_t i2   = batch_idx % ne2_1;
        ಸ್ಥಿರಾಂಕ int64_t i2_0 = i2 / r2;
        ಸ್ಥಿರಾಂಕ int64_t i3_0 = i3 / r3;

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_batch = src0_ತಳ + i3_0 * nb3_0 + i2_0 * nb2_0;
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src1_batch = src1_ತಳ + i3 * nb3_1 + i2 * nb2_1;
        ಅಕ್ಷರ *       dst_batch  = dst_ತಳ + i3 * nb3_d + i2 * nb2_d;

        ಸ್ಥಿರಾಂಕ int64_t mb    = mb_idx * TILE_M;
        ಸ್ಥಿರಾಂಕ int64_t nb    = nb_idx * TILE_N;
        ಸ್ಥಿರಾಂಕ int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        ಗಾಗಿ (int64_t kb = k_ಆರಂಭ; kb < k_ಕೊನೆ; kb += TILE_K) {
            tensor_ತುಂಬು(false, false, 0, 0, 0, (uint64_t) (src1_batch + nb * nb1_1 + kb * ಗಾತ್ರ(ದಶಮಾಂಶ)), 0, n_cur - 1,
                        (uint64_t) nb1_1, 0);

            tensor_ತುಂಬು(false, false, TILE_K, 7, 0, (uint64_t) (src0_batch + mb * nb1_0 + kb * ಗಾತ್ರ(ದಶಮಾಂಶ)), 0,
                        TILE_K - 1, (uint64_t) nb1_0, 1);

            tensor_wait(TENSOR_ತುಂಬು_WAIT_0);
            tensor_wait(TENSOR_ತುಂಬು_WAIT_1);

            tensor_fma(false, 3, n_cur - 1, TILE_K - 1, 0, false, false, false, false, TILE_K, 0, 0, (kb == k_ಆರಂಭ));

            tensor_wait(TENSOR_FMA_WAIT);
        }

        ಆದರೆ (k_splits > 1) {
            ಸ್ಥಿರಾಂಕ uint64_t ಅಂಕೆ_regs = (uint64_t) n_cur * 2;

            ಆದರೆ (k_ವಿಭಜಿಸು > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, ಅಂಕೆ_regs, ಗುಂಪು_ತಳ_global + k_ವಿಭಜಿಸು - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
            ಆದರೆ (k_ವಿಭಜಿಸು < k_splits - 1) {
                tensor_reduce_send(0, ಅಂಕೆ_regs, ಗುಂಪು_ತಳ_global + k_ವಿಭಜಿಸು + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        ಆದರೆ (k_ವಿಭಜಿಸು == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * ಗಾತ್ರ(ದಶಮಾಂಶ)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    ಹಿಂತಿರುಗಿ 0;
}
