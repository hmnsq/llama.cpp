#ضمن "ggml_tensor.h"
#ضمن "platform.h"
#ضمن "tensor.h"

#ضمن <etsoc/common/utils.h>
#ضمن <stdint.h>

/*
 * F32 Matrix Multiply for ET-SoC-1 — TensorFMA32.
 *
 * K-parallel + interleaved tiles + ring reduce.
 * No batched-K yet (needs investigation on hang).
 * This is the last known working version.
 */

#تعریف_کرو عدد_COMPUTE_SHIRES 32
#تعریف_کرو MINIONS_فی_SHIRE  32
#تعریف_کرو TILE_K             16
#تعریف_کرو TILE_M             16

/* ── Tuning knobs ───────────────────────────────────────────────────── */
#تعریف_کرو TILE_N      16
#تعریف_کرو CACHEOP_MAX 0
#تعریف_کرو REP_شرح    0

/* ─────────────────────────────────────────────────────────────────── */

صحيح اندراج_نقطہ(بنية ggml_et_ثنائی_params * params, فراغ * env) {
    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    إذا (shire_id >= عدد_COMPUTE_SHIRES) {
        أرجع 0;
    }
    إذا (hart_id & 1) {
        أرجع 0;
    }

    uint64_t local_minion = (hart_id >> 1) & 0x1F;
    uint64_t my_minion_id = get_minion_id();

    ثابت int64_t K = params->src0.ne[0];
    ثابت int64_t M = params->src0.ne[1];
    ثابت int64_t N = params->src1.ne[1];

    ثابت int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    ثابت int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    ثابت int64_t nb1_0 = params->src0.nb[1];
    ثابت int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];
    ثابت int64_t nb1_1 = params->src1.nb[1];
    ثابت int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];
    ثابت int64_t nb1_d = params->dst.nb[1];
    ثابت int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    ثابت حرف * src0_ادھار = (ثابت حرف *) params->src0.انکڑے;
    ثابت حرف * src1_ادھار = (ثابت حرف *) params->src1.انکڑے;
    حرف *       dst_ادھار  = (حرف *) params->dst.انکڑے;

    تیاری_cache_scp();
#إذا CACHEOP_MAX > 0 || REP_شرح > 0
    ucache_control(1, REP_شرح, CACHEOP_MAX);
#جے_انت
    CLEAR_TENSOR_ERROR;

    ثابت int64_t m_tiles     = M / TILE_M;
    ثابت int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    ثابت int64_t batch_گنتی = ne2_1 * ne3_1;
    ثابت int64_t ادھار_tiles  = m_tiles * n_tiles * batch_گنتی;

    ثابت int64_t r2 = ne2_1 / ne2_0;
    ثابت int64_t r3 = ne3_1 / ne3_0;

    ثابت int64_t کل_harts = عدد_COMPUTE_SHIRES * MINIONS_فی_SHIRE;
    ثابت int64_t k_قدماں     = K / TILE_K;
    int64_t       k_splits    = 1;
    إذا (ادھار_tiles < کل_harts) {
        k_splits   = (کل_harts + ادھار_tiles - 1) / ادھار_tiles;
        int64_t ks = 1;
        طالما (ks * 2 <= k_splits && ks * 2 <= 32 && k_قدماں % (ks * 2) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ثابت int64_t tiles_فی_shire = MINIONS_فی_SHIRE / k_splits;
    ثابت int64_t k_ونڈو         = local_minion % k_splits;
    ثابت int64_t local_tile_idx  = local_minion / k_splits;
    ثابت int64_t tiles_stride    = (int64_t) عدد_COMPUTE_SHIRES * tiles_فی_shire;

    ثابت int64_t k_قدماں_فی_ونڈو = k_قدماں / k_splits;
    ثابت int64_t k_شروع           = k_ونڈو * k_قدماں_فی_ونڈو * TILE_K;
    ثابت int64_t k_سرا             = k_شروع + k_قدماں_فی_ونڈو * TILE_K;

    ثابت uint64_t گروہ_ادھار_global = my_minion_id - k_ونڈو;

    لكل (int64_t tile = (int64_t) shire_id + local_tile_idx * عدد_COMPUTE_SHIRES; tile < ادھار_tiles;
         tile += tiles_stride) {
        ثابت int64_t tiles_فی_batch = m_tiles * n_tiles;
        ثابت int64_t batch_idx       = tile / tiles_فی_batch;
        ثابت int64_t tile_وچ_batch   = tile % tiles_فی_batch;
        ثابت int64_t nb_idx          = tile_وچ_batch / m_tiles;
        ثابت int64_t mb_idx          = tile_وچ_batch % m_tiles;

        ثابت int64_t i3   = batch_idx / ne2_1;
        ثابت int64_t i2   = batch_idx % ne2_1;
        ثابت int64_t i2_0 = i2 / r2;
        ثابت int64_t i3_0 = i3 / r3;

        ثابت حرف * src0_batch = src0_ادھار + i3_0 * nb3_0 + i2_0 * nb2_0;
        ثابت حرف * src1_batch = src1_ادھار + i3 * nb3_1 + i2 * nb2_1;
        حرف *       dst_batch  = dst_ادھار + i3 * nb3_d + i2 * nb2_d;

        ثابت int64_t mb    = mb_idx * TILE_M;
        ثابت int64_t nb    = nb_idx * TILE_N;
        ثابت int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        لكل (int64_t kb = k_شروع; kb < k_سرا; kb += TILE_K) {
            tensor_لدو(false, false, 0, 0, 0, (uint64_t) (src1_batch + nb * nb1_1 + kb * حجم(عائم)), 0, n_cur - 1,
                        (uint64_t) nb1_1, 0);

            tensor_لدو(false, false, TILE_K, 7, 0, (uint64_t) (src0_batch + mb * nb1_0 + kb * حجم(عائم)), 0,
                        TILE_K - 1, (uint64_t) nb1_0, 1);

            tensor_wait(TENSOR_لدو_WAIT_0);
            tensor_wait(TENSOR_لدو_WAIT_1);

            tensor_fma(false, 3, n_cur - 1, TILE_K - 1, 0, false, false, false, false, TILE_K, 0, 0, (kb == k_شروع));

            tensor_wait(TENSOR_FMA_WAIT);
        }

        إذا (k_splits > 1) {
            ثابت uint64_t عدد_regs = (uint64_t) n_cur * 2;

            إذا (k_ونڈو > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, عدد_regs, گروہ_ادھار_global + k_ونڈو - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
            إذا (k_ونڈو < k_splits - 1) {
                tensor_reduce_send(0, عدد_regs, گروہ_ادھار_global + k_ونڈو + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        إذا (k_ونڈو == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * حجم(عائم)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    أرجع 0;
}
