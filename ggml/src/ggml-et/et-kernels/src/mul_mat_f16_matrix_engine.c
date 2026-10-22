#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "platform.h"
#अन्तर्भाव "tensor.h"

#अन्तर्भाव <etsoc/common/utils.h>
#अन्तर्भाव <stdint.h>

// FP16 x FP16 -> FP32 MUL_MAT with hart 1 B-panel packing
//
// Hart 0: tensor engine (load A, load B from SCP, FMA, reduce, store)
// Hart 1: pack B into double-buffered L2 SCP panels, flush for tensor_load
//
// Sync: monotonic counters in L2 SCP with evict-based coherency.
// Double-buffered bpanel allows pack/FMA overlap.
//
#परिभाषय अङ्कः_COMPUTE_SHIRES 32
#परिभाषय MINIONS_प्रत्येकम्_SHIRE  32

#परिभाषय TILE_M 16
#परिभाषय TILE_N 16
#परिभाषय TILE_K 32

#परिभाषय CACHEOP_MAX 0
#परिभाषय REP_दरः    0

#परिभाषय A_L1_START 0   // SCP lines  0..15 for A
#परिभाषय B_L1_START 16  // SCP lines 16..31 for B

प्रकारनिर्देश uint16_t et_fp16_t;

// L2 SCP layout per minion (double-buffered bpanel + sync counters)
//   [0..1023]    bpanel buffer 0 (16 lines x 64 bytes)
//   [1024..2047] bpanel buffer 1
//   [2048..2111] ready counter   (hart1 -> hart0, own cache line)
//   [2112..2175] consumed counter (hart0 -> hart1, own cache line)
#परिभाषय SCP_BPANEL_आकार  (16 * 32 * परिमाण(et_fp16_t))  // 1024 bytes
#परिभाषय SCP_READY_OFF    (2 * SCP_BPANEL_आकार)          // 2048
#परिभाषय SCP_CONSUMED_OFF (SCP_READY_OFF + 64)           // 2112
#परिभाषय SCP_प्रत्येकम्_MINION   (SCP_CONSUMED_OFF + 64)        // 2176

// Signal a counter value to the other hart via L2 SCP.
स्थिर अन्तःस्थ शून्य __attribute__((always_inline)) scp_signal(चञ्चल uint32_t * flag, uint32_t मानम्) {
    *flag = मानम्;
    FENCE;
    evict_गन्तव्यम्_l2((ध्रुव शून्य *) flag, 1, 64);
    WAIT_CACHEOPS;
}

// Wait for a counter in L2 SCP to reach the expected value.
स्थिर अन्तःस्थ शून्य __attribute__((always_inline)) scp_wait(चञ्चल uint32_t * flag, uint32_t expected) {
    यावत् (1) {
        evict_गन्तव्यम्_l2((ध्रुव शून्य *) flag, 1, 64);
        WAIT_CACHEOPS;
        यदि (*flag >= expected) {
            प्रत्यागम;
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
स्थिर अन्तःस्थ शून्य __attribute__((always_inline)) pack_b_interleaved(et_fp16_t *  बहिर्गमनम्,
                                                                     ध्रुव वर्ण * src0_batch,
                                                                     int64_t      mb,
                                                                     int64_t      kb,
                                                                     int64_t      nb1_0) {
    स्थिर ध्रुव int32_t __attribute__((aligned(32))) scatter_idx[8] = { 0, 64, 128, 192, 256, 320, 384, 448 };

    अचिह्नित दीर्घ old_आवरणम्;
    __asm__ चञ्चल(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "flw.ps    f1, 0(%[idx])    \n\t"
        : [ms] "=&r"(old_आवरणम्)
        : [idx] "r"(scatter_idx)
        : "f1");

    प्रति (पूर्णाङ्क j = 0; j < TILE_M; ++j) {
        ध्रुव et_fp16_t * row = (ध्रुव et_fp16_t *) (src0_batch + (mb + j) * nb1_0) + kb;
        वर्ण *            dst = (वर्ण *) बहिर्गमनम् + j * 4;

        __asm__ चञ्चल(
            "flw.ps    f2, 0(%[src])    \n\t"
            "flw.ps    f3, 32(%[src])   \n\t"
            "fscw.ps   f2, f1(%[d0])    \n\t"
            "fscw.ps   f3, f1(%[d1])    \n\t"
            :
            : [src] "r"(row), [d0] "r"(dst), [d1] "r"(dst + 512)
            : "f2", "f3", "memory");
    }

    __asm__ चञ्चल("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_आवरणम्));
}

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_द्विमान_params * params, शून्य * env) {
    (शून्य) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    यदि (shire_id >= अङ्कः_COMPUTE_SHIRES) {
        प्रत्यागम 0;
    }

    ध्रुव पूर्णाङ्क is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    // Dimensions (both harts need these for tile assignment)
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

    यदि ((M % TILE_M) != 0) {
        प्रत्यागम 0;
    }
    यदि ((K % TILE_K) != 0) {
        प्रत्यागम 0;
    }

    ध्रुव int64_t m_tiles     = M / TILE_M;
    ध्रुव int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    ध्रुव int64_t batch_गणनम् = ne2_1 * ne3_1;
    ध्रुव int64_t आधारः_tiles  = m_tiles * n_tiles * batch_गणनम्;

    ध्रुव int64_t r2 = ne2_1 / ne2_0;
    ध्रुव int64_t r3 = ne3_1 / ne3_0;

    ध्रुव int64_t योग_harts = अङ्कः_COMPUTE_SHIRES * MINIONS_प्रत्येकम्_SHIRE;
    ध्रुव int64_t k_चरणाः     = K / TILE_K;

    int64_t k_splits = 1;
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

    // L2 SCP pointers for this minion's double-buffered panels + sync
    uint64_t    scp_आधारः  = local_minion * SCP_प्रत्येकम्_MINION;
    et_fp16_t * scp_bp[2] = {
        (et_fp16_t *) et_shire_l2scp_local(scp_आधारः),
        (et_fp16_t *) et_shire_l2scp_local(scp_आधारः + SCP_BPANEL_आकार),
    };
    चञ्चल uint32_t * ready_ctr    = (चञ्चल uint32_t *) et_shire_l2scp_local(scp_आधारः + SCP_READY_OFF);
    चञ्चल uint32_t * consumed_ctr = (चञ्चल uint32_t *) et_shire_l2scp_local(scp_आधारः + SCP_CONSUMED_OFF);

    // ================================================================
    // Hart 1: B-panel packer
    // ================================================================
    यदि (is_hart1) {
        // Initialize sync counters
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);

        uint32_t chunk_id = 0;

        प्रति (int64_t tile = (int64_t) shire_id + local_tile_idx * अङ्कः_COMPUTE_SHIRES; tile < आधारः_tiles;
             tile += tiles_stride) {
            ध्रुव int64_t tiles_प्रत्येकम्_batch = m_tiles * n_tiles;
            ध्रुव int64_t batch_idx       = tile / tiles_प्रत्येकम्_batch;
            ध्रुव int64_t tile_अन्तर्_batch   = tile % tiles_प्रत्येकम्_batch;

            ध्रुव int64_t mb_idx = tile_अन्तर्_batch % m_tiles;

            ध्रुव int64_t i3   = batch_idx / ne2_1;
            ध्रुव int64_t i2   = batch_idx % ne2_1;
            ध्रुव int64_t i2_0 = i2 / r2;
            ध्रुव int64_t i3_0 = i3 / r3;

            ध्रुव वर्ण *  src0_batch = src0_आधारः + i3_0 * nb3_0 + i2_0 * nb2_0;
            ध्रुव int64_t mb         = mb_idx * TILE_M;

            प्रति (int64_t kb = k_start; kb < k_अन्तः; kb += TILE_K) {
                पूर्णाङ्क buf = chunk_id & 1;

                // Back-pressure: wait for hart 0 to finish with this buffer
                यदि (chunk_id >= 2) {
                    scp_wait(consumed_ctr, chunk_id - 1);
                }

                pack_b_interleaved(scp_bp[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_गन्तव्यम्_l2(scp_bp[buf], 16, 64);
                WAIT_CACHEOPS;

                chunk_id++;
                scp_signal(ready_ctr, chunk_id);
            }
        }

        FENCE;
        प्रत्यागम 0;
    }

    // ================================================================
    // Hart 0: tensor engine compute
    // ================================================================
    uint64_t       my_minion_id      = get_minion_id();
    ध्रुव uint64_t गणः_आधारः_global = my_minion_id - k_विभज;

    सज्जता_cache_scp();
#यदि CACHEOP_MAX > 0 || REP_दरः > 0
    ucache_control(1, REP_दरः, CACHEOP_MAX);
#यदि_अन्तः
    CLEAR_TENSOR_ERROR;

    // Evict any stale L1D copies of sync counters
    evict_गन्तव्यम्_l2((ध्रुव शून्य *) ready_ctr, 1, 64);
    WAIT_CACHEOPS;
    evict_गन्तव्यम्_l2((ध्रुव शून्य *) consumed_ctr, 1, 64);
    WAIT_CACHEOPS;

    uint32_t chunk_id = 0;

    प्रति (int64_t tile = (int64_t) shire_id + local_tile_idx * अङ्कः_COMPUTE_SHIRES; tile < आधारः_tiles;
         tile += tiles_stride) {
        ध्रुव int64_t tiles_प्रत्येकम्_batch = m_tiles * n_tiles;
        ध्रुव int64_t batch_idx       = tile / tiles_प्रत्येकम्_batch;
        ध्रुव int64_t tile_अन्तर्_batch   = tile % tiles_प्रत्येकम्_batch;

        ध्रुव int64_t nb_idx = tile_अन्तर्_batch / m_tiles;
        ध्रुव int64_t mb_idx = tile_अन्तर्_batch % m_tiles;

        ध्रुव int64_t i3 = batch_idx / ne2_1;
        ध्रुव int64_t i2 = batch_idx % ne2_1;

        ध्रुव वर्ण * src1_batch = src1_आधारः + i3 * nb3_1 + i2 * nb2_1;
        वर्ण *       dst_batch  = dst_आधारः + i3 * nb3_d + i2 * nb2_d;

        ध्रुव int64_t mb    = mb_idx * TILE_M;
        ध्रुव int64_t nb    = nb_idx * TILE_N;
        ध्रुव int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        // Set tensor_mask for partial N tiles
        यदि (n_cur < TILE_N) {
            uint64_t आवरणम् = (1ULL << n_cur) - 1;
            __asm__ __volatile__("csrw 0x805, %0" : : "r"(आवरणम्));
        }

        प्रति (int64_t kb = k_start; kb < k_अन्तः; kb += TILE_K) {
            पूर्णाङ्क buf = chunk_id & 1;

            // Start loading A from DRAM (overlaps with waiting for hart 1)
            tensor_आरोपय((n_cur < TILE_N), false, A_L1_START, TENSOR_आरोपय_PLAIN, 0,
                        (uint64_t) (src1_batch + nb * nb1_1 + kb * (int64_t) परिमाण(et_fp16_t)), 0, n_cur - 1,
                        (uint64_t) nb1_1, 0);

            // Wait for hart 1 to finish packing this chunk
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            // Load B from L2 SCP (hart 1 already flushed it)
            tensor_आरोपय(false, false, B_L1_START, TENSOR_आरोपय_PLAIN, 0, (uint64_t) scp_bp[buf], 0, 15, 64, 1);

            tensor_wait(TENSOR_आरोपय_WAIT_0);
            tensor_wait(TENSOR_आरोपय_WAIT_1);

            // TensorFMA16A32
            tensor_fma((n_cur < TILE_N), 3, n_cur - 1, 15, 0, false, false, false, false, B_L1_START, A_L1_START,
                       TENSOR_FMA_OP_FP16, (kb == k_start));

            tensor_wait(TENSOR_FMA_WAIT);

            // Signal that this buffer is free for hart 1 to reuse
            scp_signal(consumed_ctr, chunk_id);
        }

        // K-split ring reduce
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

        // Store FP32 result tile
        यदि (k_विभज == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * (int64_t) परिमाण(भिन्न)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    प्रत्यागम 0;
}
