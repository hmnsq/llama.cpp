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

#समावेश "ggml_tensor.h"
#समावेश "platform.h"

#समावेश <stdint.h>

काष्ठा ggml_et_rwkv_wkv6_params {
    भग्न * k;         // src[0]: [S, H, T]  key
    भग्न * v;         // src[1]: [S, H, T]  value
    भग्न * r;         // src[2]: [S, H, T]  receptance
    भग्न * tf;        // src[3]: [S, H]     time_faaaa (per-head, not per-token)
    भग्न * td;        // src[4]: [S, H, T]  time_decay
    भग्न * state_in;  // src[5]: [S*S*H, n_seqs]  initial state
    भग्न * dst;       // [C, T + S*n_seqs]  output + state_out
    int32_t C;         // total channels (S * H)
    int32_t H;         // number of heads
    int32_t S;         // head size
    int32_t T;         // number of tokens
    int32_t n_seqs;    // number of sequences
};

पूर्णांक entry_point(काष्ठा ggml_et_rwkv_wkv6_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस 0;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    स्थिर भग्न * k        = params->k;
    स्थिर भग्न * v        = params->v;
    स्थिर भग्न * r        = params->r;
    स्थिर भग्न * tf       = params->tf;
    स्थिर भग्न * td       = params->td;
    स्थिर भग्न * state_in = params->state_in;
    भग्न *       dst_तथ्य = params->dst;

    स्थिर int32_t C      = params->C;
    स्थिर int32_t H      = params->H;
    स्थिर int32_t S      = params->S;
    स्थिर int32_t T      = params->T;
    स्थिर int32_t n_seqs = params->n_seqs;

    अगर (!k || !v || !r || !tf || !td || !state_in || !dst_तथ्य) {
        वापस -1;
    }

    स्थिर int32_t tps       = T / n_seqs;  // tokens per sequence
    भग्न *       state_out = dst_तथ्य + C * T;
    भग्न         zero      = 0.0f;

    // Tile j by one cache line so each hart's dst/state writes never share
    // a 64-B line with another hart's writes (the chip is non-coherent).
    // Tiling on j (not i) is required for WKV6 because dst[j] is accumulated
    // across i — splitting i across harts would race on dst writes.
    // For S=64 this gives 4 tiles per head; for S<16 or odd S we fall back
    // to one-hart-per-head (= the original parallelism).
    स्थिर int32_t j_tile         = (S % 16 == 0) ? 16 : S;
    स्थिर int32_t tiles_per_शीर्ष = S / j_tile;
    स्थिर int32_t योग_units    = H * tiles_per_शीर्ष;

    // Parallelize across (head, j-tile) pairs.  The t loop stays inside this
    // unit loop so the same hart owns the same column slice of state across
    // all timesteps — required for the recurrence to read back its own
    // writes without going through L2.
    क्रम (int32_t u = thread_id; u < योग_units; u += अंक_threads) {
        स्थिर int32_t h       = u / tiles_per_शीर्ष;
        स्थिर int32_t tile    = u % tiles_per_शीर्ष;
        स्थिर int32_t j_start = tile * j_tile;
        स्थिर int32_t j_end   = j_start + j_tile;

        स्थिर int32_t h_off = h * S;      // offset within C for this head
        स्थिर int32_t s2d   = h * S * S;  // offset within state for this head

        क्रम (int32_t t = 0; t < T; t++) {
            स्थिर int32_t seq       = t / tps;
            स्थिर int32_t t_in_seq  = t % tps;
            स्थिर int32_t seq_state = seq * S * C;

            स्थिर भग्न * s_पूर्व;
            भग्न *       s_cur = state_out + seq_state + s2d;

            अगर (t_in_seq == 0) {
                s_पूर्व = state_in + seq_state + s2d;
            } अन्यथा {
                s_पूर्व = s_cur;
            }

            स्थिर int32_t th = t * C + h_off;

            // Pointers for this timestep/head
            स्थिर भग्न * k_ptr  = k + th;
            स्थिर भग्न * v_ptr  = v + th;
            स्थिर भग्न * r_ptr  = r + th;
            स्थिर भग्न * tf_ptr = tf + h_off;  // tf is per-head, no t offset
            स्थिर भग्न * td_ptr = td + th;

            // Zero this hart's slice of dst: dst[th + j_start..th + j_end-1]
            // WKV6 accumulates dst[j] across all i, so must start from zero
            भग्न * dst_row = dst_तथ्य + th;
            क्रम (int32_t j = j_start; j < j_end; j += 8) {
                __asm__ अस्थिर(
                    "fbc.ps f10, %[z]\n"
                    "fsw.ps f10, %[dst_vec]\n"
                    : [dst_vec] "=m"(*(भग्न (*)[8]) & dst_row[j])
                    : [z] "m"(zero)
                    : "f10");
            }

            क्रम (int32_t i = 0; i < S; i++) {
                स्थिर भग्न * sp_row = s_पूर्व + i * S;  // state_prev row i
                भग्न *       sc_row = s_cur + i * S;   // state_cur  row i

                भग्न k_val  = k_ptr[i];
                भग्न r_val  = r_ptr[i];
                भग्न tf_val = tf_ptr[i];
                भग्न td_val = td_ptr[i];

                // Broadcast k[i], r[i], tf[i], td[i] to vector registers
                __asm__ अस्थिर(
                    "fbc.ps f20, %[kv]\n"   // f20 = k[i] broadcast
                    "fbc.ps f21, %[rv]\n"   // f21 = r[i] broadcast
                    "fbc.ps f22, %[tfv]\n"  // f22 = tf[i] broadcast
                    "fbc.ps f23, %[tdv]\n"  // f23 = td[i] broadcast
                    :
                    : [kv] "m"(k_val), [rv] "m"(r_val), [tfv] "m"(tf_val), [tdv] "m"(td_val)
                    : "f20", "f21", "f22", "f23");

                क्रम (int32_t j = j_start; j < j_end; j += 8) {
                    __asm__ अस्थिर(
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

                        : [d_out] "=m"(*(भग्न (*)[8]) & dst_row[j]), [s_out] "=m"(*(भग्न (*)[8]) & sc_row[j])
                        : [v_vec] "m"(*(स्थिर भग्न (*)[8]) & v_ptr[j]), [s_vec] "m"(*(स्थिर भग्न (*)[8]) & sp_row[j]),
                          [d_vec] "m"(*(स्थिर भग्न (*)[8]) & dst_row[j])
                        : "f10", "f11", "f12", "f13", "f14");
                }
            }
        }
    }

    वापस 0;
}
