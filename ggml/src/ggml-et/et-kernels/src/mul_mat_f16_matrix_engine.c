#समावेश "ggml_tensor.h"
#समावेश "platform.h"
#समावेश "tensor.h"

#समावेश <etsoc/common/utils.h>
#समावेश <stdint.h>

// FP16 x FP16 -> FP32 MUL_MAT with hart 1 B-panel packing
//
// Hart 0: tensor engine (load A, load B from SCP, FMA, reduce, store)
// Hart 1: pack B into double-buffered L2 SCP panels, flush for tensor_load
//
// Sync: monotonic counters in L2 SCP with evict-based coherency.
// Double-buffered bpanel allows pack/FMA overlap.
//
#घोषणा अंक_COMPUTE_SHIRES 32
#घोषणा MINIONS_PER_SHIRE  32

#घोषणा TILE_M 16
#घोषणा TILE_N 16
#घोषणा TILE_K 32

#घोषणा CACHEOP_MAX 0
#घोषणा REP_RATE    0

#घोषणा A_L1_START 0   // SCP lines  0..15 for A
#घोषणा B_L1_START 16  // SCP lines 16..31 for B

प्रकार uint16_t et_fp16_t;

// L2 SCP layout per minion (double-buffered bpanel + sync counters)
//   [0..1023]    bpanel buffer 0 (16 lines x 64 bytes)
//   [1024..2047] bpanel buffer 1
//   [2048..2111] ready counter   (hart1 -> hart0, own cache line)
//   [2112..2175] consumed counter (hart0 -> hart1, own cache line)
#घोषणा SCP_BPANEL_आकार  (16 * 32 * माप(et_fp16_t))  // 1024 bytes
#घोषणा SCP_READY_OFF    (2 * SCP_BPANEL_आकार)          // 2048
#घोषणा SCP_CONSUMED_OFF (SCP_READY_OFF + 64)           // 2112
#घोषणा SCP_PER_MINION   (SCP_CONSUMED_OFF + 64)        // 2176

// Signal a counter value to the other hart via L2 SCP.
अटल अंतरभूत व्योम __attribute__((always_inline)) scp_signal(अस्थिर uint32_t * flag, uint32_t मूल्य) {
    *flag = मूल्य;
    FENCE;
    evict_to_l2((स्थिर व्योम *) flag, 1, 64);
    WAIT_CACHEOPS;
}

// Wait for a counter in L2 SCP to reach the expected value.
अटल अंतरभूत व्योम __attribute__((always_inline)) scp_wait(अस्थिर uint32_t * flag, uint32_t expected) {
    जबतक (1) {
        evict_to_l2((स्थिर व्योम *) flag, 1, 64);
        WAIT_CACHEOPS;
        अगर (*flag >= expected) {
            वापस;
        }
    }
}

/**
 * Build the interleaved B panel that TensorFMA16A32 expects (vectorized).
 *
 * Output: 16 lines x 32 fp16 = 1024 bytes, 64-byte aligned.
 *   out[l][j*2+0] = src0[mb + j][kb + 2*l]
 *   out[l][j*2+1] = src0[mb + j][kb + 2*l + 1]
 *
 * Uses fsch.ps scatter store: load 8 pairs per row, scatter to 8 output lines.
 */
अटल अंतरभूत व्योम __attribute__((always_inline)) pack_b_interleaved(et_fp16_t *  out,
                                                                     स्थिर अक्षर * src0_batch,
                                                                     int64_t      mb,
                                                                     int64_t      kb,
                                                                     int64_t      nb1_0) {
    अटल स्थिर int32_t __attribute__((aligned(32))) scatter_idx[8] = { 0, 64, 128, 192, 256, 320, 384, 448 };

    अचिन्हित दीर्घ old_mask;
    __asm__ अस्थिर(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "flw.ps    f1, 0(%[idx])    \n\t"
        : [ms] "=&r"(old_mask)
        : [idx] "r"(scatter_idx)
        : "f1");

    क्रम (पूर्णांक j = 0; j < TILE_M; ++j) {
        स्थिर et_fp16_t * row = (स्थिर et_fp16_t *) (src0_batch + (mb + j) * nb1_0) + kb;
        अक्षर *            dst = (अक्षर *) out + j * 4;

        __asm__ अस्थिर(
            "flw.ps    f2, 0(%[src])    \n\t"
            "flw.ps    f3, 32(%[src])   \n\t"
            "fscw.ps   f2, f1(%[d0])    \n\t"
            "fscw.ps   f3, f1(%[d1])    \n\t"
            :
            : [src] "r"(row), [d0] "r"(dst), [d1] "r"(dst + 512)
            : "f2", "f3", "memory");
    }

    __asm__ अस्थिर("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_mask));
}

पूर्णांक entry_point(काष्ठा ggml_et_द्विआधारी_params * params, व्योम * env) {
    (व्योम) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    अगर (shire_id >= अंक_COMPUTE_SHIRES) {
        वापस 0;
    }

    स्थिर पूर्णांक is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    // Dimensions (both harts need these for tile assignment)
    स्थिर int64_t K = params->src0.ne[0];
    स्थिर int64_t M = params->src0.ne[1];
    स्थिर int64_t N = params->src1.ne[1];

    स्थिर int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    स्थिर int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    स्थिर int64_t nb1_0 = params->src0.nb[1];
    स्थिर int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];

    स्थिर int64_t nb1_1 = params->src1.nb[1];
    स्थिर int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];

    स्थिर int64_t nb1_d = params->dst.nb[1];
    स्थिर int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    स्थिर अक्षर * src0_base = (स्थिर अक्षर *) params->src0.तथ्य;
    स्थिर अक्षर * src1_base = (स्थिर अक्षर *) params->src1.तथ्य;
    अक्षर *       dst_base  = (अक्षर *) params->dst.तथ्य;

    अगर ((M % TILE_M) != 0) {
        वापस 0;
    }
    अगर ((K % TILE_K) != 0) {
        वापस 0;
    }

    स्थिर int64_t m_tiles     = M / TILE_M;
    स्थिर int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    स्थिर int64_t batch_गिनती = ne2_1 * ne3_1;
    स्थिर int64_t base_tiles  = m_tiles * n_tiles * batch_गिनती;

    स्थिर int64_t r2 = ne2_1 / ne2_0;
    स्थिर int64_t r3 = ne3_1 / ne3_0;

    स्थिर int64_t योग_harts = अंक_COMPUTE_SHIRES * MINIONS_PER_SHIRE;
    स्थिर int64_t k_steps     = K / TILE_K;

    int64_t k_splits = 1;
    अगर (base_tiles < योग_harts) {
        k_splits   = (योग_harts + base_tiles - 1) / base_tiles;
        int64_t ks = 1;
        जबतक (ks * 2 <= k_splits && ks * 2 <= 32 && k_steps % (ks * 2) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    स्थिर int64_t tiles_per_shire = MINIONS_PER_SHIRE / k_splits;
    स्थिर int64_t k_split         = local_minion % k_splits;
    स्थिर int64_t local_tile_idx  = local_minion / k_splits;
    स्थिर int64_t tiles_stride    = (int64_t) अंक_COMPUTE_SHIRES * tiles_per_shire;

    स्थिर int64_t k_steps_per_split = k_steps / k_splits;
    स्थिर int64_t k_start           = k_split * k_steps_per_split * TILE_K;
    स्थिर int64_t k_end             = k_start + k_steps_per_split * TILE_K;

    // L2 SCP pointers for this minion's double-buffered panels + sync
    uint64_t    scp_base  = local_minion * SCP_PER_MINION;
    et_fp16_t * scp_bp[2] = {
        (et_fp16_t *) et_shire_l2scp_local(scp_base),
        (et_fp16_t *) et_shire_l2scp_local(scp_base + SCP_BPANEL_आकार),
    };
    अस्थिर uint32_t * ready_ctr    = (अस्थिर uint32_t *) et_shire_l2scp_local(scp_base + SCP_READY_OFF);
    अस्थिर uint32_t * consumed_ctr = (अस्थिर uint32_t *) et_shire_l2scp_local(scp_base + SCP_CONSUMED_OFF);

    // ================================================================
    // Hart 1: B-panel packer
    // ================================================================
    अगर (is_hart1) {
        // Initialize sync counters
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);

        uint32_t chunk_id = 0;

        क्रम (int64_t tile = (int64_t) shire_id + local_tile_idx * अंक_COMPUTE_SHIRES; tile < base_tiles;
             tile += tiles_stride) {
            स्थिर int64_t tiles_per_batch = m_tiles * n_tiles;
            स्थिर int64_t batch_idx       = tile / tiles_per_batch;
            स्थिर int64_t tile_in_batch   = tile % tiles_per_batch;

            स्थिर int64_t mb_idx = tile_in_batch % m_tiles;

            स्थिर int64_t i3   = batch_idx / ne2_1;
            स्थिर int64_t i2   = batch_idx % ne2_1;
            स्थिर int64_t i2_0 = i2 / r2;
            स्थिर int64_t i3_0 = i3 / r3;

            स्थिर अक्षर *  src0_batch = src0_base + i3_0 * nb3_0 + i2_0 * nb2_0;
            स्थिर int64_t mb         = mb_idx * TILE_M;

            क्रम (int64_t kb = k_start; kb < k_end; kb += TILE_K) {
                पूर्णांक buf = chunk_id & 1;

                // Back-pressure: wait for hart 0 to finish with this buffer
                अगर (chunk_id >= 2) {
                    scp_wait(consumed_ctr, chunk_id - 1);
                }

                pack_b_interleaved(scp_bp[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_to_l2(scp_bp[buf], 16, 64);
                WAIT_CACHEOPS;

                chunk_id++;
                scp_signal(ready_ctr, chunk_id);
            }
        }

        FENCE;
        वापस 0;
    }

    // ================================================================
    // Hart 0: tensor engine compute
    // ================================================================
    uint64_t       my_minion_id      = get_minion_id();
    स्थिर uint64_t group_base_global = my_minion_id - k_split;

    setup_cache_scp();
#अगर CACHEOP_MAX > 0 || REP_RATE > 0
    ucache_control(1, REP_RATE, CACHEOP_MAX);
#पूर्ण_अगर
    CLEAR_TENSOR_ERROR;

    // Evict any stale L1D copies of sync counters
    evict_to_l2((स्थिर व्योम *) ready_ctr, 1, 64);
    WAIT_CACHEOPS;
    evict_to_l2((स्थिर व्योम *) consumed_ctr, 1, 64);
    WAIT_CACHEOPS;

    uint32_t chunk_id = 0;

    क्रम (int64_t tile = (int64_t) shire_id + local_tile_idx * अंक_COMPUTE_SHIRES; tile < base_tiles;
         tile += tiles_stride) {
        स्थिर int64_t tiles_per_batch = m_tiles * n_tiles;
        स्थिर int64_t batch_idx       = tile / tiles_per_batch;
        स्थिर int64_t tile_in_batch   = tile % tiles_per_batch;

        स्थिर int64_t nb_idx = tile_in_batch / m_tiles;
        स्थिर int64_t mb_idx = tile_in_batch % m_tiles;

        स्थिर int64_t i3 = batch_idx / ne2_1;
        स्थिर int64_t i2 = batch_idx % ne2_1;

        स्थिर अक्षर * src1_batch = src1_base + i3 * nb3_1 + i2 * nb2_1;
        अक्षर *       dst_batch  = dst_base + i3 * nb3_d + i2 * nb2_d;

        स्थिर int64_t mb    = mb_idx * TILE_M;
        स्थिर int64_t nb    = nb_idx * TILE_N;
        स्थिर int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        // Set tensor_mask for partial N tiles
        अगर (n_cur < TILE_N) {
            uint64_t mask = (1ULL << n_cur) - 1;
            __asm__ __volatile__("csrw 0x805, %0" : : "r"(mask));
        }

        क्रम (int64_t kb = k_start; kb < k_end; kb += TILE_K) {
            पूर्णांक buf = chunk_id & 1;

            // Start loading A from DRAM (overlaps with waiting for hart 1)
            tensor_load((n_cur < TILE_N), false, A_L1_START, TENSOR_LOAD_PLAIN, 0,
                        (uint64_t) (src1_batch + nb * nb1_1 + kb * (int64_t) माप(et_fp16_t)), 0, n_cur - 1,
                        (uint64_t) nb1_1, 0);

            // Wait for hart 1 to finish packing this chunk
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            // Load B from L2 SCP (hart 1 already flushed it)
            tensor_load(false, false, B_L1_START, TENSOR_LOAD_PLAIN, 0, (uint64_t) scp_bp[buf], 0, 15, 64, 1);

            tensor_wait(TENSOR_LOAD_WAIT_0);
            tensor_wait(TENSOR_LOAD_WAIT_1);

            // TensorFMA16A32
            tensor_fma((n_cur < TILE_N), 3, n_cur - 1, 15, 0, false, false, false, false, B_L1_START, A_L1_START,
                       TENSOR_FMA_OP_FP16, (kb == k_start));

            tensor_wait(TENSOR_FMA_WAIT);

            // Signal that this buffer is free for hart 1 to reuse
            scp_signal(consumed_ctr, chunk_id);
        }

        // K-split ring reduce
        अगर (k_splits > 1) {
            स्थिर uint64_t अंक_regs = (uint64_t) n_cur * 2;

            अगर (k_split > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, अंक_regs, group_base_global + k_split - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }

            अगर (k_split < k_splits - 1) {
                tensor_reduce_send(0, अंक_regs, group_base_global + k_split + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        // Store FP32 result tile
        अगर (k_split == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * (int64_t) माप(भग्न)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    वापस 0;
}
