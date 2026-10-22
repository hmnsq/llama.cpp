#অন্তর্ভাৱ "ggml_tensor.h"
#অন্তর্ভাৱ "platform.h"
#অন্তর্ভাৱ "tensor.h"

#অন্তর্ভাৱ <etsoc/common/utils.h>
#অন্তর্ভাৱ <stdint.h>

/*
 * F32 Matrix Multiply for ET-SoC-1 — TensorFMA32.
 *
 * K-parallel + interleaved tiles + ring reduce.
 * No batched-K yet (needs investigation on hang).
 * This is the last known working version.
 */

#define অঙ্ক_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32
#define TILE_K             16
#define TILE_M             16

/* ── Tuning knobs ───────────────────────────────────────────────────── */
#define TILE_N      16
#define CACHEOP_MAX 0
#define REP_RATE    0

/* ─────────────────────────────────────────────────────────────────── */

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_দ্বিমিক_params * params, শূন্য * env) {
    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    যদি (shire_id >= অঙ্ক_COMPUTE_SHIRES) {
        ঘূৰাই 0;
    }
    যদি (hart_id & 1) {
        ঘূৰাই 0;
    }

    uint64_t local_minion = (hart_id >> 1) & 0x1F;
    uint64_t my_minion_id = get_minion_id();

    ধ্রুৱ int64_t K = params->src0.ne[0];
    ধ্রুৱ int64_t M = params->src0.ne[1];
    ধ্রুৱ int64_t N = params->src1.ne[1];

    ধ্রুৱ int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    ধ্রুৱ int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    ধ্রুৱ int64_t nb1_0 = params->src0.nb[1];
    ধ্রুৱ int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];
    ধ্রুৱ int64_t nb1_1 = params->src1.nb[1];
    ধ্রুৱ int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];
    ধ্রুৱ int64_t nb1_d = params->dst.nb[1];
    ধ্রুৱ int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    ধ্রুৱ ৱর্ণ * src0_base = (ধ্রুৱ ৱর্ণ *) params->src0.data;
    ধ্রুৱ ৱর্ণ * src1_base = (ধ্রুৱ ৱর্ণ *) params->src1.data;
    ৱর্ণ *       dst_base  = (ৱর্ণ *) params->dst.data;

    setup_cache_scp();
#যদি CACHEOP_MAX > 0 || REP_RATE > 0
    ucache_control(1, REP_RATE, CACHEOP_MAX);
#endif
    CLEAR_TENSOR_ERROR;

    ধ্রুৱ int64_t m_tiles     = M / TILE_M;
    ধ্রুৱ int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    ধ্রুৱ int64_t batch_গুনতি = ne2_1 * ne3_1;
    ধ্রুৱ int64_t base_tiles  = m_tiles * n_tiles * batch_গুনতি;

    ধ্রুৱ int64_t r2 = ne2_1 / ne2_0;
    ধ্রুৱ int64_t r3 = ne3_1 / ne3_0;

    ধ্রুৱ int64_t total_harts = অঙ্ক_COMPUTE_SHIRES * MINIONS_PER_SHIRE;
    ধ্রুৱ int64_t k_steps     = K / TILE_K;
    int64_t       k_splits    = 1;
    যদি (base_tiles < total_harts) {
        k_splits   = (total_harts + base_tiles - 1) / base_tiles;
        int64_t ks = 1;
        যেতিয়ালৈ (ks * 2 <= k_splits && ks * 2 <= 32 && k_steps % (ks * 2) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ধ্রুৱ int64_t tiles_per_shire = MINIONS_PER_SHIRE / k_splits;
    ধ্রুৱ int64_t k_split         = local_minion % k_splits;
    ধ্রুৱ int64_t local_tile_idx  = local_minion / k_splits;
    ধ্রুৱ int64_t tiles_stride    = (int64_t) অঙ্ক_COMPUTE_SHIRES * tiles_per_shire;

    ধ্রুৱ int64_t k_steps_per_split = k_steps / k_splits;
    ধ্রুৱ int64_t k_start           = k_split * k_steps_per_split * TILE_K;
    ধ্রুৱ int64_t k_end             = k_start + k_steps_per_split * TILE_K;

    ধ্রুৱ uint64_t group_base_global = my_minion_id - k_split;

    বাবে (int64_t tile = (int64_t) shire_id + local_tile_idx * অঙ্ক_COMPUTE_SHIRES; tile < base_tiles;
         tile += tiles_stride) {
        ধ্রুৱ int64_t tiles_per_batch = m_tiles * n_tiles;
        ধ্রুৱ int64_t batch_idx       = tile / tiles_per_batch;
        ধ্রুৱ int64_t tile_in_batch   = tile % tiles_per_batch;
        ধ্রুৱ int64_t nb_idx          = tile_in_batch / m_tiles;
        ধ্রুৱ int64_t mb_idx          = tile_in_batch % m_tiles;

        ধ্রুৱ int64_t i3   = batch_idx / ne2_1;
        ধ্রুৱ int64_t i2   = batch_idx % ne2_1;
        ধ্রুৱ int64_t i2_0 = i2 / r2;
        ধ্রুৱ int64_t i3_0 = i3 / r3;

        ধ্রুৱ ৱর্ণ * src0_batch = src0_base + i3_0 * nb3_0 + i2_0 * nb2_0;
        ধ্রুৱ ৱর্ণ * src1_batch = src1_base + i3 * nb3_1 + i2 * nb2_1;
        ৱর্ণ *       dst_batch  = dst_base + i3 * nb3_d + i2 * nb2_d;

        ধ্রুৱ int64_t mb    = mb_idx * TILE_M;
        ধ্রুৱ int64_t nb    = nb_idx * TILE_N;
        ধ্রুৱ int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        বাবে (int64_t kb = k_start; kb < k_end; kb += TILE_K) {
            tensor_load(false, false, 0, 0, 0, (uint64_t) (src1_batch + nb * nb1_1 + kb * পরিমাণ(ভিন্ন)), 0, n_cur - 1,
                        (uint64_t) nb1_1, 0);

            tensor_load(false, false, TILE_K, 7, 0, (uint64_t) (src0_batch + mb * nb1_0 + kb * পরিমাণ(ভিন্ন)), 0,
                        TILE_K - 1, (uint64_t) nb1_0, 1);

            tensor_wait(TENSOR_LOAD_WAIT_0);
            tensor_wait(TENSOR_LOAD_WAIT_1);

            tensor_fma(false, 3, n_cur - 1, TILE_K - 1, 0, false, false, false, false, TILE_K, 0, 0, (kb == k_start));

            tensor_wait(TENSOR_FMA_WAIT);
        }

        যদি (k_splits > 1) {
            ধ্রুৱ uint64_t অঙ্ক_regs = (uint64_t) n_cur * 2;

            যদি (k_split > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, অঙ্ক_regs, group_base_global + k_split - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
            যদি (k_split < k_splits - 1) {
                tensor_reduce_send(0, অঙ্ক_regs, group_base_global + k_split + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        যদি (k_split == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * পরিমাণ(ভিন্ন)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    ঘূৰাই 0;
}
