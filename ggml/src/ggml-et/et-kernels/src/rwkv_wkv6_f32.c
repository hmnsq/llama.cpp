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

#ضمن "ggml_tensor.h"
#ضمن "platform.h"

#ضمن <stdint.h>

بنية ggml_et_rwkv_wkv6_params {
    عائم * k;         // src[0]: [S, H, T]  key
    عائم * v;         // src[1]: [S, H, T]  value
    عائم * r;         // src[2]: [S, H, T]  receptance
    عائم * tf;        // src[3]: [S, H]     time_faaaa (per-head, not per-token)
    عائم * td;        // src[4]: [S, H, T]  time_decay
    عائم * state_اندريون;  // src[5]: [S*S*H, n_seqs]  initial state
    عائم * dst;       // [C, T + S*n_seqs]  output + state_out
    int32_t C;         // total channels (S * H)
    int32_t H;         // number of heads
    int32_t S;         // head size
    int32_t T;         // number of tokens
    int32_t n_seqs;    // number of sequences
};

صحيح داخلا_نقطو(بنية ggml_et_rwkv_wkv6_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح انگ_ڌاڳا = get_انگ_ڌاڳا(kernel_env->shire_نقاب);

    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    ثابت عائم * k        = params->k;
    ثابت عائم * v        = params->v;
    ثابت عائم * r        = params->r;
    ثابت عائم * tf       = params->tf;
    ثابت عائم * td       = params->td;
    ثابت عائم * state_اندريون = params->state_اندريون;
    عائم *       dst_معلومات = params->dst;

    ثابت int32_t C      = params->C;
    ثابت int32_t H      = params->H;
    ثابت int32_t S      = params->S;
    ثابت int32_t T      = params->T;
    ثابت int32_t n_seqs = params->n_seqs;

    إذا (!k || !v || !r || !tf || !td || !state_اندريون || !dst_معلومات) {
        أرجع -1;
    }

    ثابت int32_t tps       = T / n_seqs;  // tokens per sequence
    عائم *       state_ٻاهريون = dst_معلومات + C * T;
    عائم         zero      = 0.0f;

    // Tile j by one cache line so each hart's dst/state writes never share
    // a 64-B line with another hart's writes (the chip is non-coherent).
    // Tiling on j (not i) is required for WKV6 because dst[j] is accumulated
    // across i — splitting i across harts would race on dst writes.
    // For S=64 this gives 4 tiles per head; for S<16 or odd S we fall back
    // to one-hart-per-head (= the original parallelism).
    ثابت int32_t j_tile         = (S % 16 == 0) ? 16 : S;
    ثابت int32_t tiles_في_سر = S / j_tile;
    ثابت int32_t ڪل_units    = H * tiles_في_سر;

    // Parallelize across (head, j-tile) pairs.  The t loop stays inside this
    // unit loop so the same hart owns the same column slice of state across
    // all timesteps — required for the recurrence to read back its own
    // writes without going through L2.
    لكل (int32_t u = thread_id; u < ڪل_units; u += انگ_ڌاڳا) {
        ثابت int32_t h       = u / tiles_في_سر;
        ثابت int32_t tile    = u % tiles_في_سر;
        ثابت int32_t j_شروعات = tile * j_tile;
        ثابت int32_t j_پڇاڙي   = j_شروعات + j_tile;

        ثابت int32_t h_off = h * S;      // offset within C for this head
        ثابت int32_t s2d   = h * S * S;  // offset within state for this head

        لكل (int32_t t = 0; t < T; t++) {
            ثابت int32_t seq       = t / tps;
            ثابت int32_t t_اندريون_seq  = t % tps;
            ثابت int32_t seq_state = seq * S * C;

            ثابت عائم * s_prev;
            عائم *       s_cur = state_ٻاهريون + seq_state + s2d;

            إذا (t_اندريون_seq == 0) {
                s_prev = state_اندريون + seq_state + s2d;
            } وإلا {
                s_prev = s_cur;
            }

            ثابت int32_t th = t * C + h_off;

            // Pointers for this timestep/head
            ثابت عائم * k_ptr  = k + th;
            ثابت عائم * v_ptr  = v + th;
            ثابت عائم * r_ptr  = r + th;
            ثابت عائم * tf_ptr = tf + h_off;  // tf is per-head, no t offset
            ثابت عائم * td_ptr = td + th;

            // Zero this hart's slice of dst: dst[th + j_start..th + j_end-1]
            // WKV6 accumulates dst[j] across all i, so must start from zero
            عائم * dst_صف = dst_معلومات + th;
            لكل (int32_t j = j_شروعات; j < j_پڇاڙي; j += 8) {
                __asm__ متطاير(
                    "fbc.ps f10, %[z]\n"
                    "fsw.ps f10, %[dst_vec]\n"
                    : [dst_vec] "=m"(*(عائم (*)[8]) & dst_صف[j])
                    : [z] "m"(zero)
                    : "f10");
            }

            لكل (int32_t i = 0; i < S; i++) {
                ثابت عائم * sp_صف = s_prev + i * S;  // state_prev row i
                عائم *       sc_صف = s_cur + i * S;   // state_cur  row i

                عائم k_val  = k_ptr[i];
                عائم r_val  = r_ptr[i];
                عائم tf_val = tf_ptr[i];
                عائم td_val = td_ptr[i];

                // Broadcast k[i], r[i], tf[i], td[i] to vector registers
                __asm__ متطاير(
                    "fbc.ps f20, %[kv]\n"   // f20 = k[i] broadcast
                    "fbc.ps f21, %[rv]\n"   // f21 = r[i] broadcast
                    "fbc.ps f22, %[tfv]\n"  // f22 = tf[i] broadcast
                    "fbc.ps f23, %[tdv]\n"  // f23 = td[i] broadcast
                    :
                    : [kv] "m"(k_val), [rv] "m"(r_val), [tfv] "m"(tf_val), [tdv] "m"(td_val)
                    : "f20", "f21", "f22", "f23");

                لكل (int32_t j = j_شروعات; j < j_پڇاڙي; j += 8) {
                    __asm__ متطاير(
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

                        : [d_ٻاهريون] "=m"(*(عائم (*)[8]) & dst_صف[j]), [s_ٻاهريون] "=m"(*(عائم (*)[8]) & sc_صف[j])
                        : [v_vec] "m"(*(ثابت عائم (*)[8]) & v_ptr[j]), [s_vec] "m"(*(ثابت عائم (*)[8]) & sp_صف[j]),
                          [d_vec] "m"(*(ثابت عائم (*)[8]) & dst_صف[j])
                        : "f10", "f11", "f12", "f13", "f14");
                }
            }
        }
    }

    أرجع 0;
}
