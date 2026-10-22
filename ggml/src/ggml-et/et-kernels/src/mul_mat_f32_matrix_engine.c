#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "tensor.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <etsoc/common/utils.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

/*
 * F32 Matrix Multiply for ET-SoC-1 — TensorFMA32.
 *
 * K-parallel + interleaved tiles + ring reduce.
 * No batched-K yet (needs investigation on hang).
 * This is the last known working version.
 */

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅੰਕ_COMPUTE_SHIRES 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MINIONS_ਪ੍ਰਤੀ_SHIRE  32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE_K             16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE_M             16

/* ── Tuning knobs ───────────────────────────────────────────────────── */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE_N      16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CACHEOP_MAX 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ REP_ਦਰ    0

/* ─────────────────────────────────────────────────────────────────── */

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params * params, ਖਾਲੀ * env) {
    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    ਜੇ (shire_id >= ਅੰਕ_COMPUTE_SHIRES) {
        ਵਾਪਸ 0;
    }
    ਜੇ (hart_id & 1) {
        ਵਾਪਸ 0;
    }

    uint64_t local_minion = (hart_id >> 1) & 0x1F;
    uint64_t my_minion_id = get_minion_id();

    ਅਚਲ int64_t K = params->src0.ne[0];
    ਅਚਲ int64_t M = params->src0.ne[1];
    ਅਚਲ int64_t N = params->src1.ne[1];

    ਅਚਲ int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    ਅਚਲ int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    ਅਚਲ int64_t nb1_0 = params->src0.nb[1];
    ਅਚਲ int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];
    ਅਚਲ int64_t nb1_1 = params->src1.nb[1];
    ਅਚਲ int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];
    ਅਚਲ int64_t nb1_d = params->dst.nb[1];
    ਅਚਲ int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    ਅਚਲ ਅੱਖਰ * src0_ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) params->src0.ਅੰਕੜੇ;
    ਅਚਲ ਅੱਖਰ * src1_ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) params->src1.ਅੰਕੜੇ;
    ਅੱਖਰ *       dst_ਅਧਾਰ  = (ਅੱਖਰ *) params->dst.ਅੰਕੜੇ;

    ਤਿਆਰੀ_cache_scp();
#ਜੇ CACHEOP_MAX > 0 || REP_ਦਰ > 0
    ucache_control(1, REP_ਦਰ, CACHEOP_MAX);
#ਜੇ_ਅੰਤ
    CLEAR_TENSOR_ERROR;

    ਅਚਲ int64_t m_tiles     = M / TILE_M;
    ਅਚਲ int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    ਅਚਲ int64_t batch_ਗਿਣਤੀ = ne2_1 * ne3_1;
    ਅਚਲ int64_t ਅਧਾਰ_tiles  = m_tiles * n_tiles * batch_ਗਿਣਤੀ;

    ਅਚਲ int64_t r2 = ne2_1 / ne2_0;
    ਅਚਲ int64_t r3 = ne3_1 / ne3_0;

    ਅਚਲ int64_t ਕੁੱਲ_harts = ਅੰਕ_COMPUTE_SHIRES * MINIONS_ਪ੍ਰਤੀ_SHIRE;
    ਅਚਲ int64_t k_ਕਦਮਾਂ     = K / TILE_K;
    int64_t       k_splits    = 1;
    ਜੇ (ਅਧਾਰ_tiles < ਕੁੱਲ_harts) {
        k_splits   = (ਕੁੱਲ_harts + ਅਧਾਰ_tiles - 1) / ਅਧਾਰ_tiles;
        int64_t ks = 1;
        ਜਦੋਂ (ks * 2 <= k_splits && ks * 2 <= 32 && k_ਕਦਮਾਂ % (ks * 2) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ਅਚਲ int64_t tiles_ਪ੍ਰਤੀ_shire = MINIONS_ਪ੍ਰਤੀ_SHIRE / k_splits;
    ਅਚਲ int64_t k_ਵੰਡੋ         = local_minion % k_splits;
    ਅਚਲ int64_t local_tile_idx  = local_minion / k_splits;
    ਅਚਲ int64_t tiles_stride    = (int64_t) ਅੰਕ_COMPUTE_SHIRES * tiles_ਪ੍ਰਤੀ_shire;

    ਅਚਲ int64_t k_ਕਦਮਾਂ_ਪ੍ਰਤੀ_ਵੰਡੋ = k_ਕਦਮਾਂ / k_splits;
    ਅਚਲ int64_t k_ਸ਼ੁਰੂ           = k_ਵੰਡੋ * k_ਕਦਮਾਂ_ਪ੍ਰਤੀ_ਵੰਡੋ * TILE_K;
    ਅਚਲ int64_t k_ਸਿਰਾ             = k_ਸ਼ੁਰੂ + k_ਕਦਮਾਂ_ਪ੍ਰਤੀ_ਵੰਡੋ * TILE_K;

    ਅਚਲ uint64_t ਸਮੂਹ_ਅਧਾਰ_global = my_minion_id - k_ਵੰਡੋ;

    ਲਈ (int64_t tile = (int64_t) shire_id + local_tile_idx * ਅੰਕ_COMPUTE_SHIRES; tile < ਅਧਾਰ_tiles;
         tile += tiles_stride) {
        ਅਚਲ int64_t tiles_ਪ੍ਰਤੀ_batch = m_tiles * n_tiles;
        ਅਚਲ int64_t batch_idx       = tile / tiles_ਪ੍ਰਤੀ_batch;
        ਅਚਲ int64_t tile_ਵਿੱਚ_batch   = tile % tiles_ਪ੍ਰਤੀ_batch;
        ਅਚਲ int64_t nb_idx          = tile_ਵਿੱਚ_batch / m_tiles;
        ਅਚਲ int64_t mb_idx          = tile_ਵਿੱਚ_batch % m_tiles;

        ਅਚਲ int64_t i3   = batch_idx / ne2_1;
        ਅਚਲ int64_t i2   = batch_idx % ne2_1;
        ਅਚਲ int64_t i2_0 = i2 / r2;
        ਅਚਲ int64_t i3_0 = i3 / r3;

        ਅਚਲ ਅੱਖਰ * src0_batch = src0_ਅਧਾਰ + i3_0 * nb3_0 + i2_0 * nb2_0;
        ਅਚਲ ਅੱਖਰ * src1_batch = src1_ਅਧਾਰ + i3 * nb3_1 + i2 * nb2_1;
        ਅੱਖਰ *       dst_batch  = dst_ਅਧਾਰ + i3 * nb3_d + i2 * nb2_d;

        ਅਚਲ int64_t mb    = mb_idx * TILE_M;
        ਅਚਲ int64_t nb    = nb_idx * TILE_N;
        ਅਚਲ int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        ਲਈ (int64_t kb = k_ਸ਼ੁਰੂ; kb < k_ਸਿਰਾ; kb += TILE_K) {
            tensor_ਲੱਦੋ(false, false, 0, 0, 0, (uint64_t) (src1_batch + nb * nb1_1 + kb * ਆਕਾਰ(ਦਸ਼ਮਲਵ)), 0, n_cur - 1,
                        (uint64_t) nb1_1, 0);

            tensor_ਲੱਦੋ(false, false, TILE_K, 7, 0, (uint64_t) (src0_batch + mb * nb1_0 + kb * ਆਕਾਰ(ਦਸ਼ਮਲਵ)), 0,
                        TILE_K - 1, (uint64_t) nb1_0, 1);

            tensor_wait(TENSOR_ਲੱਦੋ_WAIT_0);
            tensor_wait(TENSOR_ਲੱਦੋ_WAIT_1);

            tensor_fma(false, 3, n_cur - 1, TILE_K - 1, 0, false, false, false, false, TILE_K, 0, 0, (kb == k_ਸ਼ੁਰੂ));

            tensor_wait(TENSOR_FMA_WAIT);
        }

        ਜੇ (k_splits > 1) {
            ਅਚਲ uint64_t ਅੰਕ_regs = (uint64_t) n_cur * 2;

            ਜੇ (k_ਵੰਡੋ > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, ਅੰਕ_regs, ਸਮੂਹ_ਅਧਾਰ_global + k_ਵੰਡੋ - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
            ਜੇ (k_ਵੰਡੋ < k_splits - 1) {
                tensor_reduce_send(0, ਅੰਕ_regs, ਸਮੂਹ_ਅਧਾਰ_global + k_ਵੰਡੋ + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        ਜੇ (k_ਵੰਡੋ == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * ਆਕਾਰ(ਦਸ਼ਮਲਵ)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    ਵਾਪਸ 0;
}
