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

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

ರಚನೆ ggml_et_rwkv_wkv6_params {
    ದಶಮಾಂಶ * k;         // src[0]: [S, H, T]  key
    ದಶಮಾಂಶ * v;         // src[1]: [S, H, T]  value
    ದಶಮಾಂಶ * r;         // src[2]: [S, H, T]  receptance
    ದಶಮಾಂಶ * tf;        // src[3]: [S, H]     time_faaaa (per-head, not per-token)
    ದಶಮಾಂಶ * td;        // src[4]: [S, H, T]  time_decay
    ದಶಮಾಂಶ * state_ಒಳ;  // src[5]: [S*S*H, n_seqs]  initial state
    ದಶಮಾಂಶ * dst;       // [C, T + S*n_seqs]  output + state_out
    int32_t C;         // total channels (S * H)
    int32_t H;         // number of heads
    int32_t S;         // head size
    int32_t T;         // number of tokens
    int32_t n_seqs;    // number of sequences
};

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_rwkv_wkv6_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k        = params->k;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * v        = params->v;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * r        = params->r;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * tf       = params->tf;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * td       = params->td;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * state_ಒಳ = params->state_ಒಳ;
    ದಶಮಾಂಶ *       dst_ದತ್ತಾಂಶ = params->dst;

    ಸ್ಥಿರಾಂಕ int32_t C      = params->C;
    ಸ್ಥಿರಾಂಕ int32_t H      = params->H;
    ಸ್ಥಿರಾಂಕ int32_t S      = params->S;
    ಸ್ಥಿರಾಂಕ int32_t T      = params->T;
    ಸ್ಥಿರಾಂಕ int32_t n_seqs = params->n_seqs;

    ಆದರೆ (!k || !v || !r || !tf || !td || !state_ಒಳ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int32_t tps       = T / n_seqs;  // tokens per sequence
    ದಶಮಾಂಶ *       state_ಹೊರಹೋಗು = dst_ದತ್ತಾಂಶ + C * T;
    ದಶಮಾಂಶ         zero      = 0.0f;

    // Tile j by one cache line so each hart's dst/state writes never share
    // a 64-B line with another hart's writes (the chip is non-coherent).
    // Tiling on j (not i) is required for WKV6 because dst[j] is accumulated
    // across i — splitting i across harts would race on dst writes.
    // For S=64 this gives 4 tiles per head; for S<16 or odd S we fall back
    // to one-hart-per-head (= the original parallelism).
    ಸ್ಥಿರಾಂಕ int32_t j_tile         = (S % 16 == 0) ? 16 : S;
    ಸ್ಥಿರಾಂಕ int32_t tiles_ಪ್ರತಿ_ತಲೆ = S / j_tile;
    ಸ್ಥಿರಾಂಕ int32_t ಒಟ್ಟು_units    = H * tiles_ಪ್ರತಿ_ತಲೆ;

    // Parallelize across (head, j-tile) pairs.  The t loop stays inside this
    // unit loop so the same hart owns the same column slice of state across
    // all timesteps — required for the recurrence to read back its own
    // writes without going through L2.
    ಗಾಗಿ (int32_t u = thread_id; u < ಒಟ್ಟು_units; u += ಅಂಕೆ_ಎಳೆಗಳು) {
        ಸ್ಥಿರಾಂಕ int32_t h       = u / tiles_ಪ್ರತಿ_ತಲೆ;
        ಸ್ಥಿರಾಂಕ int32_t tile    = u % tiles_ಪ್ರತಿ_ತಲೆ;
        ಸ್ಥಿರಾಂಕ int32_t j_ಆರಂಭ = tile * j_tile;
        ಸ್ಥಿರಾಂಕ int32_t j_ಕೊನೆ   = j_ಆರಂಭ + j_tile;

        ಸ್ಥಿರಾಂಕ int32_t h_off = h * S;      // offset within C for this head
        ಸ್ಥಿರಾಂಕ int32_t s2d   = h * S * S;  // offset within state for this head

        ಗಾಗಿ (int32_t t = 0; t < T; t++) {
            ಸ್ಥಿರಾಂಕ int32_t seq       = t / tps;
            ಸ್ಥಿರಾಂಕ int32_t t_ಒಳ_seq  = t % tps;
            ಸ್ಥಿರಾಂಕ int32_t seq_state = seq * S * C;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s_prev;
            ದಶಮಾಂಶ *       s_cur = state_ಹೊರಹೋಗು + seq_state + s2d;

            ಆದರೆ (t_ಒಳ_seq == 0) {
                s_prev = state_ಒಳ + seq_state + s2d;
            } ಇಲ್ಲದಿದ್ದರೆ {
                s_prev = s_cur;
            }

            ಸ್ಥಿರಾಂಕ int32_t th = t * C + h_off;

            // Pointers for this timestep/head
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k_ptr  = k + th;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * v_ptr  = v + th;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * r_ptr  = r + th;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * tf_ptr = tf + h_off;  // tf is per-head, no t offset
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * td_ptr = td + th;

            // Zero this hart's slice of dst: dst[th + j_start..th + j_end-1]
            // WKV6 accumulates dst[j] across all i, so must start from zero
            ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = dst_ದತ್ತಾಂಶ + th;
            ಗಾಗಿ (int32_t j = j_ಆರಂಭ; j < j_ಕೊನೆ; j += 8) {
                __asm__ ಅಸ್ಥಿರ(
                    "fbc.ps f10, %[z]\n"
                    "fsw.ps f10, %[dst_vec]\n"
                    : [dst_vec] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಅಡ್ಡಸಾಲು[j])
                    : [z] "m"(zero)
                    : "f10");
            }

            ಗಾಗಿ (int32_t i = 0; i < S; i++) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * sp_ಅಡ್ಡಸಾಲು = s_prev + i * S;  // state_prev row i
                ದಶಮಾಂಶ *       sc_ಅಡ್ಡಸಾಲು = s_cur + i * S;   // state_cur  row i

                ದಶಮಾಂಶ k_val  = k_ptr[i];
                ದಶಮಾಂಶ r_val  = r_ptr[i];
                ದಶಮಾಂಶ tf_val = tf_ptr[i];
                ದಶಮಾಂಶ td_val = td_ptr[i];

                // Broadcast k[i], r[i], tf[i], td[i] to vector registers
                __asm__ ಅಸ್ಥಿರ(
                    "fbc.ps f20, %[kv]\n"   // f20 = k[i] broadcast
                    "fbc.ps f21, %[rv]\n"   // f21 = r[i] broadcast
                    "fbc.ps f22, %[tfv]\n"  // f22 = tf[i] broadcast
                    "fbc.ps f23, %[tdv]\n"  // f23 = td[i] broadcast
                    :
                    : [kv] "m"(k_val), [rv] "m"(r_val), [tfv] "m"(tf_val), [tdv] "m"(td_val)
                    : "f20", "f21", "f22", "f23");

                ಗಾಗಿ (int32_t j = j_ಆರಂಭ; j < j_ಕೊನೆ; j += 8) {
                    __asm__ ಅಸ್ಥಿರ(
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

                        : [d_ಹೊರಹೋಗು] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಅಡ್ಡಸಾಲು[j]), [s_ಹೊರಹೋಗು] "=m"(*(ದಶಮಾಂಶ (*)[8]) & sc_ಅಡ್ಡಸಾಲು[j])
                        : [v_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & v_ptr[j]), [s_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & sp_ಅಡ್ಡಸಾಲು[j]),
                          [d_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & dst_ಅಡ್ಡಸಾಲು[j])
                        : "f10", "f11", "f12", "f13", "f14");
                }
            }
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
