//******************************************************************************
// RWKV WKV7 F32 Kernel
//
// Implements the RWKV-7 linear attention recurrence:
//   For each head h, timestep t, row i:
//     sa       = dot(a, state[i])
//     state[i] = state[i] * w + v[i]*k + sa * b
//     output[i]= dot(state[i], r)
//
//******************************************************************************

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

ସଂରଚନା ggml_et_rwkv_wkv7_params {
    ଦଶମିକ * r;         // [S, H, T]  receptance
    ଦଶମିକ * w;         // [S, H, T]  decay
    ଦଶମିକ * k;         // [S, H, T]  key
    ଦଶମିକ * v;         // [S, H, T]  value
    ଦଶମିକ * a;         // [S, H, T]  bonus gate
    ଦଶମିକ * b;         // [S, H, T]  bonus key
    ଦଶମିକ * state_in;  // [S*S*H, n_seqs]  initial state
    ଦଶମିକ * dst;       // [C, T + S*n_seqs]  output + state_out
    int32_t C;         // total channels (S * H)
    int32_t H;         // number of heads
    int32_t S;         // head size
    int32_t T;         // number of tokens
    int32_t n_seqs;    // number of sequences
};

// Horizontal sum of 8-wide vector register f10 -> scalar float
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ hsum_f10(ଶୂନ୍ୟ) {
    ଦଶମିକ result;
    __asm__ __volatile__(
        "fswizz.ps f1, f10, 0xB1 \n\t"
        "fadd.ps   f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result)::"t0", "f1", "f2", "f3", "f4", "f5");
    ଫେରନ୍ତୁ result;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_rwkv_wkv7_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ 0;
    }

    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ ଦଶମିକ * r        = params->r;
    ଅଚଳ ଦଶମିକ * w        = params->w;
    ଅଚଳ ଦଶମିକ * k        = params->k;
    ଅଚଳ ଦଶମିକ * v        = params->v;
    ଅଚଳ ଦଶମିକ * a        = params->a;
    ଅଚଳ ଦଶମିକ * b        = params->b;
    ଅଚଳ ଦଶମିକ * state_in = params->state_in;
    ଦଶମିକ *       dst_data = params->dst;

    ଅଚଳ int32_t C      = params->C;
    ଅଚଳ int32_t H      = params->H;
    ଅଚଳ int32_t S      = params->S;
    ଅଚଳ int32_t T      = params->T;
    ଅଚଳ int32_t n_seqs = params->n_seqs;

    ଯଦି (!r || !w || !k || !v || !a || !b || !state_in || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int32_t tps       = T / n_seqs;  // tokens per sequence
    ଦଶମିକ *       state_out = dst_data + C * T;

    // Fix #2: hoist w[0..S-1] across the i loop.  In the inner j-loop of pass
    // 2, w/k/b/r are loop-invariant w.r.t. i but were being reloaded for every
    // i value (16 times redundantly after Fix #1).  Pinning all four arrays
    // would need 32 vector regs (won't fit), so we hoist just w — it's used
    // in the critical fmadd chain and lives cleanly in f24-f31, which the
    // existing kernel never touches.  Saves ~20% of pass-2 load issues.
    //
    // GCC local register variables: declared as `float` but the underlying
    // f-reg holds the wide vector loaded by flw.ps.  GCC reserves f24-f31 for
    // these variables for the whole function and never generates code that
    // touches them on its own, so the upper 7 lanes survive between asm
    // blocks.  Only used when S == 64 (the RWKV-7 case); other head sizes
    // fall through to the original unhoisted path.
    ପଞ୍ଜିକରଣ ଦଶମିକ w_h0 __asm__("f24");
    ପଞ୍ଜିକରଣ ଦଶମିକ w_h1 __asm__("f25");
    ପଞ୍ଜିକରଣ ଦଶମିକ w_h2 __asm__("f26");
    ପଞ୍ଜିକରଣ ଦଶମିକ w_h3 __asm__("f27");
    ପଞ୍ଜିକରଣ ଦଶମିକ w_h4 __asm__("f28");
    ପଞ୍ଜିକରଣ ଦଶମିକ w_h5 __asm__("f29");
    ପଞ୍ଜିକରଣ ଦଶମିକ w_h6 __asm__("f30");
    ପଞ୍ଜିକରଣ ଦଶମିକ w_h7 __asm__("f31");
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା      wkv7_fast = (S == 64);

    // Tile i by one cache line so each hart's output writes never share a
    // 64-B line with another hart's writes (the chip is non-coherent).
    // For S=64 this gives 4 tiles per head; for S<16 or odd S we fall back
    // to one-hart-per-head (= the original parallelism).
    ଅଚଳ int32_t i_tile         = (S % 16 == 0) ? 16 : S;
    ଅଚଳ int32_t tiles_per_head = S / i_tile;
    ଅଚଳ int32_t total_units    = H * tiles_per_head;

    // Parallelize across (head, i-tile) pairs.  The t loop stays inside this
    // unit loop so the same hart owns the same state rows across all
    // timesteps — required for the recurrence to read back its own writes
    // without going through L2.
    ପାଇଁ (int32_t u = thread_id; u < total_units; u += ଅଙ୍କ_threads) {
        ଅଚଳ int32_t h       = u / tiles_per_head;
        ଅଚଳ int32_t tile    = u % tiles_per_head;
        ଅଚଳ int32_t i_start = tile * i_tile;
        ଅଚଳ int32_t i_end   = i_start + i_tile;

        ଅଚଳ int32_t h_off = h * S;      // offset within C for this head
        ଅଚଳ int32_t s2d   = h * S * S;  // offset within state for this head

        ପାଇଁ (int32_t t = 0; t < T; t++) {
            ଅଚଳ int32_t seq       = t / tps;
            ଅଚଳ int32_t t_in_seq  = t % tps;
            ଅଚଳ int32_t seq_state = seq * S * C;  // state offset for this sequence

            ଅଚଳ ଦଶମିକ * s_prev;
            ଦଶମିକ *       s_cur = state_out + seq_state + s2d;

            ଯଦି (t_in_seq == 0) {
                s_prev = state_in + seq_state + s2d;
            } ଅନ୍ୟଥା {
                s_prev = s_cur;
            }

            // Pointers for this timestep/head
            ଅଚଳ int32_t th    = t * C + h_off;
            ଅଚଳ ଦଶମିକ * r_ptr = r + th;
            ଅଚଳ ଦଶମିକ * w_ptr = w + th;
            ଅଚଳ ଦଶମିକ * k_ptr = k + th;
            ଅଚଳ ଦଶମିକ * v_ptr = v + th;
            ଅଚଳ ଦଶମିକ * a_ptr = a + th;
            ଅଚଳ ଦଶମିକ * b_ptr = b + th;

            // Hoist w[0..63] into f24-f31 once per (h, t).  These values are
            // invariant across the i loop below, so the inner j-unroll can
            // reference them by register name and skip the per-i reload.
            ଯଦି (wkv7_fast) {
                __asm__ ଅସ୍ଥିର(
                    "flw.ps f24,   0(%[wp])\n"
                    "flw.ps f25,  32(%[wp])\n"
                    "flw.ps f26,  64(%[wp])\n"
                    "flw.ps f27,  96(%[wp])\n"
                    "flw.ps f28, 128(%[wp])\n"
                    "flw.ps f29, 160(%[wp])\n"
                    "flw.ps f30, 192(%[wp])\n"
                    "flw.ps f31, 224(%[wp])\n"
                    : "=f"(w_h0), "=f"(w_h1), "=f"(w_h2), "=f"(w_h3), "=f"(w_h4), "=f"(w_h5), "=f"(w_h6), "=f"(w_h7)
                    : [wp] "r"(w_ptr));
            }

            ପାଇଁ (int32_t i = i_start; i < i_end; i++) {
                ଅଚଳ ଦଶମିକ * sp_row = s_prev + i * S;  // state_prev row i
                ଦଶମିକ *       sc_row = s_cur + i * S;   // state_cur  row i

                // ----------------------------------------------------------
                // Step 1: sa = dot(a, state_prev[i])
                // Accumulate in f10
                // ----------------------------------------------------------
                ଦଶମିକ zero = 0.0f;
                __asm__ ଅସ୍ଥିର("fbc.ps f10, %[z]\n" : : [z] "m"(zero) : "f10");

                ପାଇଁ (int32_t j = 0; j < S; j += 8) {
                    __asm__ ଅସ୍ଥିର(
                        "flw.ps f11, %[a_vec]\n"
                        "flw.ps f12, %[s_vec]\n"
                        "fmadd.ps f10, f11, f12, f10\n"
                        :
                        : [a_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & a_ptr[j]), [s_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & sp_row[j])
                        : "f10", "f11", "f12");
                }

                ଦଶମିକ sa = hsum_f10();

                // ----------------------------------------------------------
                // Step 2: state update + result accumulation
                //   kv       = v[i] * k[j]
                //   state[j] = state[j] * w[j] + kv + sa * b[j]
                //   result  += state[j] * r[j]
                // ----------------------------------------------------------
                ଦଶମିକ v_val = v_ptr[i];

                // Broadcast v_val and sa, zero result accumulator (f10)
                __asm__ ଅସ୍ଥିର(
                    "fbc.ps f20, %[vv]\n"
                    "fbc.ps f21, %[sv]\n"
                    "fbc.ps f10, %[z]\n"
                    :
                    : [vv] "m"(v_val), [sv] "m"(sa), [z] "m"(zero)
                    : "f10", "f20", "f21");

                ଯଦି (wkv7_fast) {
// Fast path: 8 chunks unrolled, w hoisted to f24-f31.
// Saves one flw per chunk vs the original loop.
#define WKV7_PASS2_CHUNK(j_off, w_var)                                                                           \
    __asm__ ଅସ୍ଥିର(                                                                                            \
        "flw.ps f11, %[s_vec]\n"                                                                                 \
        "flw.ps f13, %[k_vec]\n"                                                                                 \
        "flw.ps f14, %[b_vec]\n"                                                                                 \
        "flw.ps f15, %[r_vec]\n"                                                                                 \
        "fmul.ps f16, f20, f13\n"                                                                                \
        "fmadd.ps f11, f11, %[w_h], f16\n"                                                                       \
        "fmadd.ps f11, f21, f14, f11\n"                                                                          \
        "fsw.ps f11, %[sc_vec]\n"                                                                                \
        "fmadd.ps f10, f11, f15, f10\n"                                                                          \
        : [sc_vec] "=m"(*(ଦଶମିକ (*)[8]) & sc_row[j_off])                                                         \
        : [s_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & sp_row[j_off]), [k_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & k_ptr[j_off]), \
          [b_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & b_ptr[j_off]), [r_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & r_ptr[j_off]),  \
          [w_h] "f"(w_var)                                                                                       \
        : "f10", "f11", "f13", "f14", "f15", "f16")

                    WKV7_PASS2_CHUNK(0, w_h0);
                    WKV7_PASS2_CHUNK(8, w_h1);
                    WKV7_PASS2_CHUNK(16, w_h2);
                    WKV7_PASS2_CHUNK(24, w_h3);
                    WKV7_PASS2_CHUNK(32, w_h4);
                    WKV7_PASS2_CHUNK(40, w_h5);
                    WKV7_PASS2_CHUNK(48, w_h6);
                    WKV7_PASS2_CHUNK(56, w_h7);

#undef WKV7_PASS2_CHUNK
                } ଅନ୍ୟଥା {
                    ପାଇଁ (int32_t j = 0; j < S; j += 8) {
                        __asm__ ଅସ୍ଥିର(
                            "flw.ps f11, %[s_vec]\n"         // state_prev[j..j+7]
                            "flw.ps f12, %[w_vec]\n"         // w[j..j+7]
                            "flw.ps f13, %[k_vec]\n"         // k[j..j+7]
                            "flw.ps f14, %[b_vec]\n"         // b[j..j+7]
                            "flw.ps f15, %[r_vec]\n"         // r[j..j+7]
                            "fmul.ps f16, f20, f13\n"        // kv = v_broadcast * k
                            "fmadd.ps f11, f11, f12, f16\n"  // state*w + kv
                            "fmadd.ps f11, f21, f14, f11\n"  // + sa*b
                            "fsw.ps f11, %[sc_vec]\n"        // store new state
                            "fmadd.ps f10, f11, f15, f10\n"  // result += new_state * r

                            : [sc_vec] "=m"(*(ଦଶମିକ (*)[8]) & sc_row[j])
                            : [s_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & sp_row[j]),
                              [w_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & w_ptr[j]),
                              [k_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & k_ptr[j]),
                              [b_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & b_ptr[j]),
                              [r_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & r_ptr[j])
                            : "f10", "f11", "f12", "f13", "f14", "f15", "f16");
                    }
                }

                dst_data[th + i] = hsum_f10();
            }
        }
    }

    ଫେରନ୍ତୁ 0;
}
