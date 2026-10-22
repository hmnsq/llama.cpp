//******************************************************************************
// RWKV WKV6 F32 Kernel
//
// Implements the RWKV-6 linear attention recurrence:
//   dst = r @ (time_faaaa * (k @ v) + state)
//   state = time_decay * state + (k @ v)
//
// For each head h, timestep t, row i:
//   kv[j]       = v[j] * k[i]
//   temp[j]     = kv[j] * tf[i] + state[i][j]
//   dst[j]     += temp[j] * r[i]     (accumulated across all i)
//   state[i][j] = state[i][j] * td[i] + kv[j]
//
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

నిర్మాణం ggml_et_rwkv_wkv6_params {
    దశాంశం * k;         // src[0]: [S, H, T]  key
    దశాంశం * v;         // src[1]: [S, H, T]  value
    దశాంశం * r;         // src[2]: [S, H, T]  receptance
    దశాంశం * tf;        // src[3]: [S, H]     time_faaaa (per-head, not per-token)
    దశాంశం * td;        // src[4]: [S, H, T]  time_decay
    దశాంశం * state_లోపల;  // src[5]: [S*S*H, n_seqs]  initial state
    దశాంశం * dst;       // [C, T + S*n_seqs]  output + state_out
    int32_t C;         // total channels (S * H)
    int32_t H;         // number of heads
    int32_t S;         // head size
    int32_t T;         // number of tokens
    int32_t n_seqs;    // number of sequences
};

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_rwkv_wkv6_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;
    }

    స్థిరరాశి దశాంశం * k        = params->k;
    స్థిరరాశి దశాంశం * v        = params->v;
    స్థిరరాశి దశాంశం * r        = params->r;
    స్థిరరాశి దశాంశం * tf       = params->tf;
    స్థిరరాశి దశాంశం * td       = params->td;
    స్థిరరాశి దశాంశం * state_లోపల = params->state_లోపల;
    దశాంశం *       dst_దత్తాంశం = params->dst;

    స్థిరరాశి int32_t C      = params->C;
    స్థిరరాశి int32_t H      = params->H;
    స్థిరరాశి int32_t S      = params->S;
    స్థిరరాశి int32_t T      = params->T;
    స్థిరరాశి int32_t n_seqs = params->n_seqs;

    ఒకవేళ (!k || !v || !r || !tf || !td || !state_లోపల || !dst_దత్తాంశం) {
        తిరిగి -1;
    }

    స్థిరరాశి int32_t tps       = T / n_seqs;  // tokens per sequence
    దశాంశం *       state_బహిర్గతం = dst_దత్తాంశం + C * T;
    దశాంశం         zero      = 0.0f;

    // Tile j by one cache line so each hart's dst/state writes never share
    // a 64-B line with another hart's writes (the chip is non-coherent).
    // Tiling on j (not i) is required for WKV6 because dst[j] is accumulated
    // across i — splitting i across harts would race on dst writes.
    // For S=64 this gives 4 tiles per head; for S<16 or odd S we fall back
    // to one-hart-per-head (= the original parallelism).
    స్థిరరాశి int32_t j_tile         = (S % 16 == 0) ? 16 : S;
    స్థిరరాశి int32_t tiles_చొప్పున_తల = S / j_tile;
    స్థిరరాశి int32_t మొత్తం_units    = H * tiles_చొప్పున_తల;

    // Parallelize across (head, j-tile) pairs.  The t loop stays inside this
    // unit loop so the same hart owns the same column slice of state across
    // all timesteps — required for the recurrence to read back its own
    // writes without going through L2.
    కొరకు (int32_t u = thread_id; u < మొత్తం_units; u += అంకె_దారాలు) {
        స్థిరరాశి int32_t h       = u / tiles_చొప్పున_తల;
        స్థిరరాశి int32_t tile    = u % tiles_చొప్పున_తల;
        స్థిరరాశి int32_t j_ప్రారంభం = tile * j_tile;
        స్థిరరాశి int32_t j_చివర   = j_ప్రారంభం + j_tile;

        స్థిరరాశి int32_t h_off = h * S;      // offset within C for this head
        స్థిరరాశి int32_t s2d   = h * S * S;  // offset within state for this head

        కొరకు (int32_t t = 0; t < T; t++) {
            స్థిరరాశి int32_t seq       = t / tps;
            స్థిరరాశి int32_t t_లోపల_seq  = t % tps;
            స్థిరరాశి int32_t seq_state = seq * S * C;

            స్థిరరాశి దశాంశం * s_prev;
            దశాంశం *       s_cur = state_బహిర్గతం + seq_state + s2d;

            ఒకవేళ (t_లోపల_seq == 0) {
                s_prev = state_లోపల + seq_state + s2d;
            } లేకపోతే {
                s_prev = s_cur;
            }

            స్థిరరాశి int32_t th = t * C + h_off;

            // Pointers for this timestep/head
            స్థిరరాశి దశాంశం * k_ptr  = k + th;
            స్థిరరాశి దశాంశం * v_ptr  = v + th;
            స్థిరరాశి దశాంశం * r_ptr  = r + th;
            స్థిరరాశి దశాంశం * tf_ptr = tf + h_off;  // tf is per-head, no t offset
            స్థిరరాశి దశాంశం * td_ptr = td + th;

            // Zero this hart's slice of dst: dst[th + j_start..th + j_end-1]
            // WKV6 accumulates dst[j] across all i, so must start from zero
            దశాంశం * dst_అడ్డువరుస = dst_దత్తాంశం + th;
            కొరకు (int32_t j = j_ప్రారంభం; j < j_చివర; j += 8) {
                __asm__ అస్థిర(
                    "fbc.ps f10, %[z]\n"
                    "fsw.ps f10, %[dst_vec]\n"
                    : [dst_vec] "=m"(*(దశాంశం (*)[8]) & dst_అడ్డువరుస[j])
                    : [z] "m"(zero)
                    : "f10");
            }

            కొరకు (int32_t i = 0; i < S; i++) {
                స్థిరరాశి దశాంశం * sp_అడ్డువరుస = s_prev + i * S;  // state_prev row i
                దశాంశం *       sc_అడ్డువరుస = s_cur + i * S;   // state_cur  row i

                దశాంశం k_val  = k_ptr[i];
                దశాంశం r_val  = r_ptr[i];
                దశాంశం tf_val = tf_ptr[i];
                దశాంశం td_val = td_ptr[i];

                // Broadcast k[i], r[i], tf[i], td[i] to vector registers
                __asm__ అస్థిర(
                    "fbc.ps f20, %[kv]\n"   // f20 = k[i] broadcast
                    "fbc.ps f21, %[rv]\n"   // f21 = r[i] broadcast
                    "fbc.ps f22, %[tfv]\n"  // f22 = tf[i] broadcast
                    "fbc.ps f23, %[tdv]\n"  // f23 = td[i] broadcast
                    :
                    : [kv] "m"(k_val), [rv] "m"(r_val), [tfv] "m"(tf_val), [tdv] "m"(td_val)
                    : "f20", "f21", "f22", "f23");

                కొరకు (int32_t j = j_ప్రారంభం; j < j_చివర; j += 8) {
                    __asm__ అస్థిర(
                        // Load v[j], state_prev[i][j], dst[j]
                        "flw.ps f10, %[v_vec]\n"  // v[j..j+7]
                        "flw.ps f11, %[s_vec]\n"  // state_prev[i][j..j+7]
                        "flw.ps f12, %[d_vec]\n"  // dst[j..j+7] (accumulated)

                        // kv = v * k_broadcast
                        "fmul.ps f13, f10, f20\n"  // kv = v * k

                        // temp = kv * tf_broadcast + state_prev
                        "fmadd.ps f14, f13, f22, f11\n"  // temp = kv * tf + state

                        // dst[j] += temp * r_broadcast
                        "fmadd.ps f12, f14, f21, f12\n"  // dst += temp * r
                        "fsw.ps f12, %[d_out]\n"         // store updated dst

                        // state_cur[i][j] = state_prev * td_broadcast + kv
                        "fmadd.ps f11, f11, f23, f13\n"  // state = state * td + kv
                        "fsw.ps f11, %[s_out]\n"         // store new state

                        : [d_బహిర్గతం] "=m"(*(దశాంశం (*)[8]) & dst_అడ్డువరుస[j]), [s_బహిర్గతం] "=m"(*(దశాంశం (*)[8]) & sc_అడ్డువరుస[j])
                        : [v_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & v_ptr[j]), [s_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & sp_అడ్డువరుస[j]),
                          [d_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & dst_అడ్డువరుస[j])
                        : "f10", "f11", "f12", "f13", "f14");
                }
            }
        }
    }

    తిరిగి 0;
}
