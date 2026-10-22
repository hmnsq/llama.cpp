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

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

ਬਣਤਰ ggml_et_rwkv_wkv6_params {
    ਦਸ਼ਮਲਵ * k;         // src[0]: [S, H, T]  key
    ਦਸ਼ਮਲਵ * v;         // src[1]: [S, H, T]  value
    ਦਸ਼ਮਲਵ * r;         // src[2]: [S, H, T]  receptance
    ਦਸ਼ਮਲਵ * tf;        // src[3]: [S, H]     time_faaaa (per-head, not per-token)
    ਦਸ਼ਮਲਵ * td;        // src[4]: [S, H, T]  time_decay
    ਦਸ਼ਮਲਵ * state_ਵਿੱਚ;  // src[5]: [S*S*H, n_seqs]  initial state
    ਦਸ਼ਮਲਵ * dst;       // [C, T + S*n_seqs]  output + state_out
    int32_t C;         // total channels (S * H)
    int32_t H;         // number of heads
    int32_t S;         // head size
    int32_t T;         // number of tokens
    int32_t n_seqs;    // number of sequences
};

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_rwkv_wkv6_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ * k        = params->k;
    ਅਚਲ ਦਸ਼ਮਲਵ * v        = params->v;
    ਅਚਲ ਦਸ਼ਮਲਵ * r        = params->r;
    ਅਚਲ ਦਸ਼ਮਲਵ * tf       = params->tf;
    ਅਚਲ ਦਸ਼ਮਲਵ * td       = params->td;
    ਅਚਲ ਦਸ਼ਮਲਵ * state_ਵਿੱਚ = params->state_ਵਿੱਚ;
    ਦਸ਼ਮਲਵ *       dst_ਅੰਕੜੇ = params->dst;

    ਅਚਲ int32_t C      = params->C;
    ਅਚਲ int32_t H      = params->H;
    ਅਚਲ int32_t S      = params->S;
    ਅਚਲ int32_t T      = params->T;
    ਅਚਲ int32_t n_seqs = params->n_seqs;

    ਜੇ (!k || !v || !r || !tf || !td || !state_ਵਿੱਚ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int32_t tps       = T / n_seqs;  // tokens per sequence
    ਦਸ਼ਮਲਵ *       state_ਬਾਹਰਲਾ = dst_ਅੰਕੜੇ + C * T;
    ਦਸ਼ਮਲਵ         zero      = 0.0f;

    // Tile j by one cache line so each hart's dst/state writes never share
    // a 64-B line with another hart's writes (the chip is non-coherent).
    // Tiling on j (not i) is required for WKV6 because dst[j] is accumulated
    // across i — splitting i across harts would race on dst writes.
    // For S=64 this gives 4 tiles per head; for S<16 or odd S we fall back
    // to one-hart-per-head (= the original parallelism).
    ਅਚਲ int32_t j_tile         = (S % 16 == 0) ? 16 : S;
    ਅਚਲ int32_t tiles_ਪ੍ਰਤੀ_ਸਿਰ = S / j_tile;
    ਅਚਲ int32_t ਕੁੱਲ_units    = H * tiles_ਪ੍ਰਤੀ_ਸਿਰ;

    // Parallelize across (head, j-tile) pairs.  The t loop stays inside this
    // unit loop so the same hart owns the same column slice of state across
    // all timesteps — required for the recurrence to read back its own
    // writes without going through L2.
    ਲਈ (int32_t u = thread_id; u < ਕੁੱਲ_units; u += ਅੰਕ_ਧਾਗੇ) {
        ਅਚਲ int32_t h       = u / tiles_ਪ੍ਰਤੀ_ਸਿਰ;
        ਅਚਲ int32_t tile    = u % tiles_ਪ੍ਰਤੀ_ਸਿਰ;
        ਅਚਲ int32_t j_ਸ਼ੁਰੂ = tile * j_tile;
        ਅਚਲ int32_t j_ਸਿਰਾ   = j_ਸ਼ੁਰੂ + j_tile;

        ਅਚਲ int32_t h_off = h * S;      // offset within C for this head
        ਅਚਲ int32_t s2d   = h * S * S;  // offset within state for this head

        ਲਈ (int32_t t = 0; t < T; t++) {
            ਅਚਲ int32_t seq       = t / tps;
            ਅਚਲ int32_t t_ਵਿੱਚ_seq  = t % tps;
            ਅਚਲ int32_t seq_state = seq * S * C;

            ਅਚਲ ਦਸ਼ਮਲਵ * s_prev;
            ਦਸ਼ਮਲਵ *       s_cur = state_ਬਾਹਰਲਾ + seq_state + s2d;

            ਜੇ (t_ਵਿੱਚ_seq == 0) {
                s_prev = state_ਵਿੱਚ + seq_state + s2d;
            } ਨਹੀਂਤਾਂ {
                s_prev = s_cur;
            }

            ਅਚਲ int32_t th = t * C + h_off;

            // Pointers for this timestep/head
            ਅਚਲ ਦਸ਼ਮਲਵ * k_ptr  = k + th;
            ਅਚਲ ਦਸ਼ਮਲਵ * v_ptr  = v + th;
            ਅਚਲ ਦਸ਼ਮਲਵ * r_ptr  = r + th;
            ਅਚਲ ਦਸ਼ਮਲਵ * tf_ptr = tf + h_off;  // tf is per-head, no t offset
            ਅਚਲ ਦਸ਼ਮਲਵ * td_ptr = td + th;

            // Zero this hart's slice of dst: dst[th + j_start..th + j_end-1]
            // WKV6 accumulates dst[j] across all i, so must start from zero
            ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = dst_ਅੰਕੜੇ + th;
            ਲਈ (int32_t j = j_ਸ਼ੁਰੂ; j < j_ਸਿਰਾ; j += 8) {
                __asm__ ਅਸਥਿਰ(
                    "fbc.ps f10, %[z]\n"
                    "fsw.ps f10, %[dst_vec]\n"
                    : [dst_vec] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਪੰਕਤੀ[j])
                    : [z] "m"(zero)
                    : "f10");
            }

            ਲਈ (int32_t i = 0; i < S; i++) {
                ਅਚਲ ਦਸ਼ਮਲਵ * sp_ਪੰਕਤੀ = s_prev + i * S;  // state_prev row i
                ਦਸ਼ਮਲਵ *       sc_ਪੰਕਤੀ = s_cur + i * S;   // state_cur  row i

                ਦਸ਼ਮਲਵ k_val  = k_ptr[i];
                ਦਸ਼ਮਲਵ r_val  = r_ptr[i];
                ਦਸ਼ਮਲਵ tf_val = tf_ptr[i];
                ਦਸ਼ਮਲਵ td_val = td_ptr[i];

                // Broadcast k[i], r[i], tf[i], td[i] to vector registers
                __asm__ ਅਸਥਿਰ(
                    "fbc.ps f20, %[kv]\n"   // f20 = k[i] broadcast
                    "fbc.ps f21, %[rv]\n"   // f21 = r[i] broadcast
                    "fbc.ps f22, %[tfv]\n"  // f22 = tf[i] broadcast
                    "fbc.ps f23, %[tdv]\n"  // f23 = td[i] broadcast
                    :
                    : [kv] "m"(k_val), [rv] "m"(r_val), [tfv] "m"(tf_val), [tdv] "m"(td_val)
                    : "f20", "f21", "f22", "f23");

                ਲਈ (int32_t j = j_ਸ਼ੁਰੂ; j < j_ਸਿਰਾ; j += 8) {
                    __asm__ ਅਸਥਿਰ(
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

                        : [d_ਬਾਹਰਲਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਪੰਕਤੀ[j]), [s_ਬਾਹਰਲਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & sc_ਪੰਕਤੀ[j])
                        : [v_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & v_ptr[j]), [s_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & sp_ਪੰਕਤੀ[j]),
                          [d_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & dst_ਪੰਕਤੀ[j])
                        : "f10", "f11", "f12", "f13", "f14");
                }
            }
        }
    }

    ਵਾਪਸ 0;
}
