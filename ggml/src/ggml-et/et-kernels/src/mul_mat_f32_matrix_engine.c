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

#تعریف_کن رقم_COMPUTE_SHIRES 32
#تعریف_کن MINIONS_در_هر_SHIRE  32
#تعریف_کن TILE_K             16
#تعریف_کن TILE_M             16

/* ── Tuning knobs ───────────────────────────────────────────────────── */
#تعریف_کن TILE_N      16
#تعریف_کن CACHEOP_MAX 0
#تعریف_کن REP_نرخ    0

/* ─────────────────────────────────────────────────────────────────── */

صحيح مدخل_نقطه(بنية ggml_et_دودویی_params * params, فراغ * env) {
    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    إذا (shire_id >= رقم_COMPUTE_SHIRES) {
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

    ثابت حرف * src0_پایه = (ثابت حرف *) params->src0.داده;
    ثابت حرف * src1_پایه = (ثابت حرف *) params->src1.داده;
    حرف *       dst_پایه  = (حرف *) params->dst.داده;

    راهاندازی_cache_scp();
#إذا CACHEOP_MAX > 0 || REP_نرخ > 0
    ucache_control(1, REP_نرخ, CACHEOP_MAX);
#پایان_اگر
    CLEAR_TENSOR_ERROR;

    ثابت int64_t m_tiles     = M / TILE_M;
    ثابت int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    ثابت int64_t batch_شمار = ne2_1 * ne3_1;
    ثابت int64_t پایه_tiles  = m_tiles * n_tiles * batch_شمار;

    ثابت int64_t r2 = ne2_1 / ne2_0;
    ثابت int64_t r3 = ne3_1 / ne3_0;

    ثابت int64_t مجموع_harts = رقم_COMPUTE_SHIRES * MINIONS_در_هر_SHIRE;
    ثابت int64_t k_گامها     = K / TILE_K;
    int64_t       k_splits    = 1;
    إذا (پایه_tiles < مجموع_harts) {
        k_splits   = (مجموع_harts + پایه_tiles - 1) / پایه_tiles;
        int64_t ks = 1;
        طالما (ks * 2 <= k_splits && ks * 2 <= 32 && k_گامها % (ks * 2) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ثابت int64_t tiles_در_هر_shire = MINIONS_در_هر_SHIRE / k_splits;
    ثابت int64_t k_تقسیم         = local_minion % k_splits;
    ثابت int64_t local_tile_idx  = local_minion / k_splits;
    ثابت int64_t tiles_stride    = (int64_t) رقم_COMPUTE_SHIRES * tiles_در_هر_shire;

    ثابت int64_t k_گامها_در_هر_تقسیم = k_گامها / k_splits;
    ثابت int64_t k_شروع           = k_تقسیم * k_گامها_در_هر_تقسیم * TILE_K;
    ثابت int64_t k_انتها             = k_شروع + k_گامها_در_هر_تقسیم * TILE_K;

    ثابت uint64_t گروه_پایه_global = my_minion_id - k_تقسیم;

    لكل (int64_t tile = (int64_t) shire_id + local_tile_idx * رقم_COMPUTE_SHIRES; tile < پایه_tiles;
         tile += tiles_stride) {
        ثابت int64_t tiles_در_هر_batch = m_tiles * n_tiles;
        ثابت int64_t batch_idx       = tile / tiles_در_هر_batch;
        ثابت int64_t tile_درون_batch   = tile % tiles_در_هر_batch;
        ثابت int64_t nb_idx          = tile_درون_batch / m_tiles;
        ثابت int64_t mb_idx          = tile_درون_batch % m_tiles;

        ثابت int64_t i3   = batch_idx / ne2_1;
        ثابت int64_t i2   = batch_idx % ne2_1;
        ثابت int64_t i2_0 = i2 / r2;
        ثابت int64_t i3_0 = i3 / r3;

        ثابت حرف * src0_batch = src0_پایه + i3_0 * nb3_0 + i2_0 * nb2_0;
        ثابت حرف * src1_batch = src1_پایه + i3 * nb3_1 + i2 * nb2_1;
        حرف *       dst_batch  = dst_پایه + i3 * nb3_d + i2 * nb2_d;

        ثابت int64_t mb    = mb_idx * TILE_M;
        ثابت int64_t nb    = nb_idx * TILE_N;
        ثابت int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        لكل (int64_t kb = k_شروع; kb < k_انتها; kb += TILE_K) {
            tensor_بارگیری(false, false, 0, 0, 0, (uint64_t) (src1_batch + nb * nb1_1 + kb * حجم(عائم)), 0, n_cur - 1,
                        (uint64_t) nb1_1, 0);

            tensor_بارگیری(false, false, TILE_K, 7, 0, (uint64_t) (src0_batch + mb * nb1_0 + kb * حجم(عائم)), 0,
                        TILE_K - 1, (uint64_t) nb1_0, 1);

            tensor_wait(TENSOR_بارگیری_WAIT_0);
            tensor_wait(TENSOR_بارگیری_WAIT_1);

            tensor_fma(false, 3, n_cur - 1, TILE_K - 1, 0, false, false, false, false, TILE_K, 0, 0, (kb == k_شروع));

            tensor_wait(TENSOR_FMA_WAIT);
        }

        إذا (k_splits > 1) {
            ثابت uint64_t رقم_regs = (uint64_t) n_cur * 2;

            إذا (k_تقسیم > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, رقم_regs, گروه_پایه_global + k_تقسیم - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
            إذا (k_تقسیم < k_splits - 1) {
                tensor_reduce_send(0, رقم_regs, گروه_پایه_global + k_تقسیم + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        إذا (k_تقسیم == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * حجم(عائم)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    أرجع 0;
}
