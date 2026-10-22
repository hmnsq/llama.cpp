//******************************************************************************
// GLU F32 Kernel (SwiGLU specifically)
// Gated Linear Unit: y[i] = silu(x[i]) * g[i] where silu(x) = x * sigmoid(x)
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdint.h>

// GLU kernel parameters structure (from ET backend ops)
ساخت ggml_et_glu_params {
    ساخت ggml_tensor src0;         // F32 input tensor A (or combined tensor if src1 is null)
    ساخت ggml_tensor src1;         // F32 input tensor B (null for single tensor mode)
    ساخت ggml_tensor dst;          // F32 output tensor (n/2 columns)
    int32_t            glu_op_type;  // GLU operation type (REGLU=0, GEGLU=1, SWIGLU=2, etc.)
    int32_t            swapped;      // Whether gate and value are swapped
    اعشاری              alpha;        // SWIGLU_OAI: sigmoid scaling factor
    اعشاری              حد_بندی;        // GLU clamp limit
};

// SiLU activation function: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
ساکن اندرونی اعشاری silu_f32(اعشاری x) {
    // For numerical stability, use the mathematically equivalent form:
    // silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    // For large negative x, exp(-x) -> inf, so silu(x) -> 0
    // For large positive x, exp(-x) -> 0, so silu(x) -> x

    اگر (x > 20.0f) {
        // For x > 20, exp(-x) is negligible, silu(x) ~ x
        واپس x;
    } ورنہ اگر (x < -20.0f) {
        // For x < -20, silu(x) ~ 0
        واپس 0.0f;
    } ورنہ {
        // Use standard formula: silu(x) = x / (1 + exp(-x))
        // Optimized using ET hardware division
        اعشاری exp_neg_x   = et_expf(-x);
        اعشاری denominator = 1.0f + exp_neg_x;
        واپس et_fdiv(x, denominator);
    }
}

// Vectorized GeGLU block processing (8 elements = 1 cache line, 64B aligned)
// gelu(x) = 0.5*x*(1 + tanh(z)) = x * (1 - 1/(exp(2z)+1))
// where z = sqrt(2/pi) * x * (1 + 0.044715*x^2)
// Reformulated to avoid inf*0 NaN: uses x * sigmoid(2z) identity
ساکن اندرونی خالی بلاک_geglu(اعشاری * dst_بلاک, ثابت اعشاری * x_بلاک, ثابت اعشاری * g_بلاک, صحیح elements) {
    غیرعلامتی طویل عارضی_نقاب;
    __asm__ متلون("mova.x.m %0" : "=r"(عارضی_نقاب));
    __asm__ متلون("mov.m.x m0, x0, 0xFF");

    اعشاری one_const       = 1.0f;
    اعشاری coef_a_const    = 0.044715f;
    اعشاری sqrt2pi_const   = 0.79788456080286535587989211986876f;  // sqrt(2/pi)
    اعشاری two_log2e_const = 2.8853900817779268f;                  // 2 * log2(e)

    برائے (int32_t i = 0; i < elements; i += 8) {
        __asm__ متلون(
            // Load inputs
            "flw.ps f10, %[x_vec]\n"  // f10 = x
            "flw.ps f11, %[g_vec]\n"  // f11 = g

            // Broadcast constants
            "fbc.ps f20, %[one_ptr]\n"        // f20 = 1.0
            "fbc.ps f22, %[coef_ptr]\n"       // f22 = 0.044715
            "fbc.ps f23, %[sqrt2pi_ptr]\n"    // f23 = sqrt(2/pi)
            "fbc.ps f24, %[two_log2e_ptr]\n"  // f24 = 2*log2(e)

            // inner = 1 + 0.044715 * x^2
            "fmul.ps f12, f10, f10\n"        // f12 = x^2
            "fmadd.ps f13, f22, f12, f20\n"  // f13 = 1 + 0.044715*x^2

            // z = sqrt(2/pi) * x * inner
            "fmul.ps f14, f23, f10\n"  // f14 = sqrt(2/pi) * x
            "fmul.ps f14, f14, f13\n"  // f14 = z

            // exp(2z) via fexp.ps: feed z * 2*log2(e) since fexp computes 2^input
            "fmul.ps f15, f14, f24\n"  // f15 = 2z * log2(e)
            "fexp.ps f15, f15\n"       // f15 = exp(2z)

            // gelu(x) = x * (1 - 1/(exp(2z)+1))  [NaN-safe: no inf*0]
            // exp(2z)->inf: rcp(inf)=0, 1-0=1, gelu=x
            // exp(2z)->0:   rcp(1)=1,   1-1=0, gelu=0
            "fadd.ps f16, f15, f20\n"  // f16 = exp(2z) + 1
            "frcp.ps f16, f16\n"       // f16 = 1/(exp(2z) + 1)
            "fsub.ps f16, f20, f16\n"  // f16 = 1 - 1/(exp(2z)+1)
            "fmul.ps f16, f10, f16\n"  // f16 = gelu(x)

            // Final result
            "fmul.ps f18, f16, f11\n"  // f18 = gelu(x) * g

            "fsw.ps f18, %[dst_out]\n"

            : [dst_برآمد] "=m"(*(اعشاری (*)[8]) & dst_بلاک[i])
            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & x_بلاک[i]), [g_vec] "m"(*(ثابت اعشاری (*)[8]) & g_بلاک[i]),
              [one_ptr] "m"(one_const), [coef_ptr] "m"(coef_a_const), [sqrt2pi_ptr] "m"(sqrt2pi_const),
              [two_log2e_ptr] "m"(two_log2e_const)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f18", "f20", "f22", "f23", "f24");
    }

    __asm__ متلون("mova.m.x %0" ::"r"(عارضی_نقاب));
}

// Vectorized SwiGLU block processing (16 elements = 1 cache line)
ساکن اندرونی خالی بلاک_swiglu(اعشاری * dst_بلاک, ثابت اعشاری * x_بلاک, ثابت اعشاری * g_بلاک, صحیح elements) {
    // Process 8 elements at a time using vector instructions
    int32_t vec_انتہا = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    غیرعلامتی طویل عارضی_نقاب;
    __asm__ متلون("mova.x.m %0" : "=r"(عارضی_نقاب));  // Save current mask
    __asm__ متلون("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Constants for broadcasting
    اعشاری zero_const  = 0.0f;
    اعشاری one_const   = 1.0f;
    اعشاری log2e_const = 1.4426950408889634f;  // log2(e)

    برائے (int32_t i = 0; i < vec_انتہا; i += 8) {
        // Vectorized SwiGLU: dst = silu(x) * g = (x / (1 + exp(-x))) * g
        // Using ET hardware: exp, reciprocal, multiply operations
        __asm__ متلون(
            // Load input vectors
            "flw.ps f10, %[x_vec]\n"  // f10 = x[0..7]
            "flw.ps f11, %[g_vec]\n"  // f11 = g[0..7]

            // Broadcast constants to vector registers
            "fbc.ps f20, %[zero_ptr]\n"  // f20 = broadcast(0.0f) to all 8 elements
            "fbc.ps f21, %[one_ptr]\n"   // f21 = broadcast(1.0f) to all 8 elements

            // Compute -x (negate x by subtracting from zero)
            "fsub.ps f12, f20, f10\n"  // f12 = 0 - x = -x

            // Convert to base-2 exponent: -x * log2(e) = -x * 1.44269504
            // Load log2(e) constant
            "fbc.ps f22, %[log2e_ptr]\n"  // f22 = broadcast(1.44269504f)
            "fmul.ps f13, f12, f22\n"     // f13 = -x * log2(e)

            // Compute 2^(-x * log2(e)) = exp(-x)
            "fexp.ps f14, f13\n"  // f14 = 2^(-x * log2(e)) = exp(-x)

            // Compute 1 + exp(-x)
            "fadd.ps f15, f14, f21\n"  // f15 = exp(-x) + 1

            // Compute 1 / (1 + exp(-x)) using reciprocal
            "frcp.ps f16, f15\n"  // f16 = 1 / (1 + exp(-x))

            // Compute silu(x) = x * (1 / (1 + exp(-x)))
            "fmul.ps f17, f10, f16\n"  // f17 = x * (1 / (1 + exp(-x))) = silu(x)

            // Compute final result: silu(x) * g
            "fmul.ps f18, f17, f11\n"  // f18 = silu(x) * g

            // Store result
            "fsw.ps f18, %[dst_out]\n"  // Store 8 results to destination

            : [dst_برآمد] "=m"(*(اعشاری (*)[8]) & dst_بلاک[i])
            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & x_بلاک[i]), [g_vec] "m"(*(ثابت اعشاری (*)[8]) & g_بلاک[i]),
              [zero_ptr] "m"(zero_const),   // Memory reference to 0.0f for broadcasting
              [one_ptr] "m"(one_const),     // Memory reference to 1.0f for broadcasting
              [log2e_ptr] "m"(log2e_const)  // Memory reference to log2(e) for broadcasting
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22");
    }

    // Restore original mask
    __asm__ متلون("mova.m.x %0" ::"r"(عارضی_نقاب));

    // Handle remaining elements (< 8) with scalar operations
    برائے (int32_t i = vec_انتہا; i < elements; i++) {
        dst_بلاک[i] = silu_f32(x_بلاک[i]) * g_بلاک[i];
    }
}

// Vectorized ReGLU block: dst = max(0, x) * g
ساکن اندرونی خالی بلاک_reglu(اعشاری * dst_بلاک, ثابت اعشاری * x_بلاک, ثابت اعشاری * g_بلاک, صحیح elements) {
    int32_t vec_انتہا = (elements / 8) * 8;

    غیرعلامتی طویل عارضی_نقاب;
    __asm__ متلون("mova.x.m %0" : "=r"(عارضی_نقاب));
    __asm__ متلون("mov.m.x m0, x0, 0xFF");

    اعشاری zero_const = 0.0f;

    برائے (int32_t i = 0; i < vec_انتہا; i += 8) {
        __asm__ متلون(
            "flw.ps  f10, %[x_vec]\n"     // f10 = x
            "flw.ps  f11, %[g_vec]\n"     // f11 = g
            "fbc.ps  f20, %[zero_ptr]\n"  // f20 = 0.0
            "fmax.ps f12, f10, f20\n"     // f12 = max(x, 0)
            "fmul.ps f13, f12, f11\n"     // f13 = relu(x) * g
            "fsw.ps  f13, %[dst_out]\n"
            : [dst_برآمد] "=m"(*(اعشاری (*)[8]) & dst_بلاک[i])
            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & x_بلاک[i]), [g_vec] "m"(*(ثابت اعشاری (*)[8]) & g_بلاک[i]),
              [zero_ptr] "m"(zero_const)
            : "f10", "f11", "f12", "f13", "f20");
    }

    __asm__ متلون("mova.m.x %0" ::"r"(عارضی_نقاب));

    برائے (int32_t i = vec_انتہا; i < elements; i++) {
        اعشاری xv     = x_بلاک[i];
        dst_بلاک[i] = (xv > 0.0f) ? xv * g_بلاک[i] : 0.0f;
    }
}

// Vectorized GeGLU-Quick block: dst = x * sigmoid(1.702 * x) * g
// Using gelu_quick(x) = x / (1 + exp(-1.702*x))
ساکن اندرونی خالی بلاک_geglu_تیز(اعشاری * dst_بلاک, ثابت اعشاری * x_بلاک, ثابت اعشاری * g_بلاک, صحیح elements) {
    int32_t vec_انتہا = (elements / 8) * 8;

    غیرعلامتی طویل عارضی_نقاب;
    __asm__ متلون("mova.x.m %0" : "=r"(عارضی_نقاب));
    __asm__ متلون("mov.m.x m0, x0, 0xFF");

    اعشاری zero_const        = 0.0f;
    اعشاری one_const         = 1.0f;
    // -1.702 * log2(e), so that fexp.ps(x * neg_k_log2e) = exp(-1.702*x)
    اعشاری neg_k_log2e_const = -1.702f * 1.4426950408889634f;

    برائے (int32_t i = 0; i < vec_انتہا; i += 8) {
        __asm__ متلون(
            "flw.ps  f10, %[x_vec]\n"     // f10 = x
            "flw.ps  f11, %[g_vec]\n"     // f11 = g
            "fbc.ps  f20, %[zero_ptr]\n"  // f20 = 0
            "fbc.ps  f21, %[one_ptr]\n"   // f21 = 1
            "fbc.ps  f22, %[k_ptr]\n"     // f22 = -1.702*log2(e)
            "fmul.ps f13, f10, f22\n"     // f13 = -1.702*x*log2(e)
            "fexp.ps f14, f13\n"          // f14 = exp(-1.702*x)
            "fadd.ps f15, f14, f21\n"     // f15 = 1 + exp(-1.702*x)
            "frcp.ps f16, f15\n"          // f16 = sigmoid(1.702*x)
            "fmul.ps f17, f10, f16\n"     // f17 = gelu_quick(x)
            "fmul.ps f18, f17, f11\n"     // f18 = gelu_quick(x) * g
            "fsw.ps  f18, %[dst_out]\n"
            : [dst_برآمد] "=m"(*(اعشاری (*)[8]) & dst_بلاک[i])
            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & x_بلاک[i]), [g_vec] "m"(*(ثابت اعشاری (*)[8]) & g_بلاک[i]),
              [zero_ptr] "m"(zero_const), [one_ptr] "m"(one_const), [k_ptr] "m"(neg_k_log2e_const)
            : "f10", "f11", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22");
    }

    __asm__ متلون("mova.m.x %0" ::"r"(عارضی_نقاب));

    برائے (int32_t i = vec_انتہا; i < elements; i++) {
        اعشاری xv     = x_بلاک[i];
        // Reuse silu reciprocal path: sigmoid(1.702*x) = 1/(1+exp(-1.702*x))
        اعشاری e      = et_expf(-1.702f * xv);
        dst_بلاک[i] = et_fdiv(xv, 1.0f + e) * g_بلاک[i];
    }
}

// Vectorized SwiGLU-OAI block (OpenAI gpt-oss variant):
//   x_c = min(x, limit)
//   y_c = clamp(g, -limit, limit)
//   out = (x_c / (1 + exp(-alpha * x_c))) * (y_c + 1)
ساکن اندرونی خالی بلاک_swiglu_oai(اعشاری *       dst_بلاک,
                                    ثابت اعشاری * x_بلاک,
                                    ثابت اعشاری * g_بلاک,
                                    صحیح           elements,
                                    اعشاری         alpha,
                                    اعشاری         حد_بندی) {
    int32_t vec_انتہا = (elements / 8) * 8;

    غیرعلامتی طویل عارضی_نقاب;
    __asm__ متلون("mova.x.m %0" : "=r"(عارضی_نقاب));
    __asm__ متلون("mov.m.x m0, x0, 0xFF");

    اعشاری zero_const    = 0.0f;
    اعشاری one_const     = 1.0f;
    اعشاری حد_بندی_pos     = حد_بندی;
    اعشاری حد_بندی_neg     = -حد_بندی;
    // -alpha * log2(e): feed (x * neg_alpha_log2e) into fexp.ps to get exp(-alpha*x)
    اعشاری neg_alpha_l2e = -alpha * 1.4426950408889634f;

    برائے (int32_t i = 0; i < vec_انتہا; i += 8) {
        __asm__ متلون(
            "flw.ps  f10, %[x_vec]\n"     // f10 = x raw
            "flw.ps  f11, %[g_vec]\n"     // f11 = g raw

            "fbc.ps  f20, %[zero_ptr]\n"  // f20 = 0
            "fbc.ps  f21, %[one_ptr]\n"   // f21 = 1
            "fbc.ps  f23, %[lim_pos]\n"   // f23 = +limit
            "fbc.ps  f24, %[lim_neg]\n"   // f24 = -limit
            "fbc.ps  f25, %[k_ptr]\n"     // f25 = -alpha*log2(e)

            // x_c = min(x, +limit)  (no lower bound on x per OAI spec)
            "fmin.ps f12, f10, f23\n"  // f12 = x_c

            // y_c = clamp(g, -limit, +limit) = min(max(g, -limit), +limit)
            "fmax.ps f13, f11, f24\n"  // f13 = max(g, -limit)
            "fmin.ps f13, f13, f23\n"  // f13 = y_c

            // sigmoid(alpha * x_c) = 1 / (1 + exp(-alpha * x_c))
            "fmul.ps f14, f12, f25\n"  // f14 = -alpha*x_c*log2(e)
            "fexp.ps f15, f14\n"       // f15 = exp(-alpha*x_c)
            "fadd.ps f15, f15, f21\n"  // f15 = 1 + exp(-alpha*x_c)
            "frcp.ps f16, f15\n"       // f16 = sigmoid(alpha*x_c)

            // out_glu = x_c * sigmoid(alpha*x_c)
            "fmul.ps f17, f12, f16\n"  // f17 = swiglu_oai gate output

            // dst = out_glu * (y_c + 1)
            "fadd.ps f18, f13, f21\n"  // f18 = y_c + 1
            "fmul.ps f19, f17, f18\n"  // f19 = final
            "fsw.ps  f19, %[dst_out]\n"

            : [dst_برآمد] "=m"(*(اعشاری (*)[8]) & dst_بلاک[i])
            : [x_vec] "m"(*(ثابت اعشاری (*)[8]) & x_بلاک[i]), [g_vec] "m"(*(ثابت اعشاری (*)[8]) & g_بلاک[i]),
              [zero_ptr] "m"(zero_const), [one_ptr] "m"(one_const), [lim_pos] "m"(حد_بندی_pos), [lim_neg] "m"(حد_بندی_neg),
              [k_ptr] "m"(neg_alpha_l2e)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f23", "f24", "f25");
    }

    __asm__ متلون("mova.m.x %0" ::"r"(عارضی_نقاب));

    // Scalar tail (mirrors CPU reference exactly)
    برائے (int32_t i = vec_انتہا; i < elements; i++) {
        اعشاری xv = x_بلاک[i];
        اعشاری yv = g_بلاک[i];
        اگر (xv > حد_بندی) {
            xv = حد_بندی;
        }
        اگر (yv > حد_بندی) {
            yv = حد_بندی;
        }
        اگر (yv < -حد_بندی) {
            yv = -حد_بندی;
        }
        اعشاری e       = et_expf(-alpha * xv);
        اعشاری برآمد_glu = et_fdiv(xv, 1.0f + e);
        dst_بلاک[i]  = برآمد_glu * (yv + 1.0f);
    }
}

ساکن اندرونی خالی بلاک_swiglu_clamp(اعشاری *       dst_بلاک,
                                      ثابت اعشاری * gate_بلاک,
                                      ثابت اعشاری * اوپر_بلاک,
                                      صحیح           elements,
                                      اعشاری         حد_بندی) {
    int32_t vec_انتہا = (elements / 8) * 8;

    غیرعلامتی طویل عارضی_نقاب;
    __asm__ متلون("mova.x.m %0" : "=r"(عارضی_نقاب));
    __asm__ متلون("mov.m.x m0, x0, 0xFF");

    اعشاری one_const = 1.0f;
    اعشاری حد_بندی_pos = حد_بندی;
    اعشاری حد_بندی_neg = -حد_بندی;
    اعشاری neg_log2e = -1.4426950408889634f;

    برائے (int32_t i = 0; i < vec_انتہا; i += 8) {
        __asm__ متلون(
            "flw.ps  f10, %[gate_vec]\n"
            "flw.ps  f11, %[up_vec]\n"
            "fbc.ps  f21, %[one_ptr]\n"
            "fbc.ps  f23, %[lim_pos]\n"
            "fbc.ps  f24, %[lim_neg]\n"
            "fbc.ps  f25, %[k_ptr]\n"
            "fmin.ps f12, f10, f23\n"
            "fmax.ps f13, f11, f24\n"
            "fmin.ps f13, f13, f23\n"
            "fmul.ps f14, f12, f25\n"
            "fexp.ps f15, f14\n"
            "fadd.ps f15, f15, f21\n"
            "frcp.ps f16, f15\n"
            "fmul.ps f17, f12, f16\n"
            "fmul.ps f18, f17, f13\n"
            "fsw.ps  f18, %[dst_out]\n"
            : [dst_برآمد] "=m"(*(اعشاری (*)[8]) & dst_بلاک[i])
            : [gate_vec] "m"(*(ثابت اعشاری (*)[8]) & gate_بلاک[i]), [اوپر_vec] "m"(*(ثابت اعشاری (*)[8]) & اوپر_بلاک[i]),
              [one_ptr] "m"(one_const), [lim_pos] "m"(حد_بندی_pos), [lim_neg] "m"(حد_بندی_neg), [k_ptr] "m"(neg_log2e)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f21", "f23", "f24", "f25");
    }

    __asm__ متلون("mova.m.x %0" :: "r"(عارضی_نقاب));

    برائے (int32_t i = vec_انتہا; i < elements; i++) {
        اعشاری gate   = gate_بلاک[i] > حد_بندی ? حد_بندی : gate_بلاک[i];
        اعشاری اوپر     = اوپر_بلاک[i];
        اوپر           = اوپر > حد_بندی ? حد_بندی : اوپر;
        اوپر           = اوپر < -حد_بندی ? -حد_بندی : اوپر;
        dst_بلاک[i] = silu_f32(gate) * اوپر;
    }
}

// Scalar erf approximation (Abramowitz & Stegun 7.1.26, max error ~1.5e-7)
ساکن اندرونی اعشاری erf_approx(اعشاری x) {
    ثابت اعشاری a1 = 0.254829592f;
    ثابت اعشاری a2 = -0.284496736f;
    ثابت اعشاری a3 = 1.421413741f;
    ثابت اعشاری a4 = -1.453152027f;
    ثابت اعشاری a5 = 1.061405429f;
    ثابت اعشاری p  = 0.3275911f;

    اعشاری sign = (x < 0.0f) ? -1.0f : 1.0f;
    اعشاری ax   = (x < 0.0f) ? -x : x;
    اعشاری t    = et_fdiv(1.0f, 1.0f + p * ax);
    اعشاری t2   = t * t;
    اعشاری t3   = t2 * t;
    اعشاری t4   = t3 * t;
    اعشاری t5   = t4 * t;
    اعشاری poly = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5;
    اعشاری y    = 1.0f - poly * et_expf(-ax * ax);
    واپس sign * y;
}

// GeGLU-Erf block: dst = 0.5 * x * (1 + erf(x / sqrt(2))) * g
// Scalar implementation — variant is rarely used so we keep complexity low.
ساکن اندرونی خالی بلاک_geglu_erf(اعشاری * dst_بلاک, ثابت اعشاری * x_بلاک, ثابت اعشاری * g_بلاک, صحیح elements) {
    ثابت اعشاری sqrt_2_inv = 0.70710678118654752440f;
    برائے (int32_t i = 0; i < elements; i++) {
        اعشاری xv     = x_بلاک[i];
        dst_بلاک[i] = 0.5f * xv * (1.0f + erf_approx(xv * sqrt_2_inv)) * g_بلاک[i];
    }
}

// Main entry point for GLU kernel
صحیح اندراج_نقطہ(ساخت ggml_et_glu_params * params, خالی * env) {
    // Cast env to proper type
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    // Validate environment pointer
    اگر (!kernel_env) {
        واپس -1;
    }

    // Get thread info using shire mask from environment
    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    // Basic safety check on params
    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;  // Invalid pointer
    }

    // Supported variants: SwiGLU, SwiGLU-OAI, GeGLU, GeGLU-Erf, GeGLU-Quick, ReGLU
    انتخاب (params->glu_op_type) {
        حالت GGML_GLU_OP_SWIGLU:
        حالت GGML_GLU_OP_SWIGLU_OAI:
        حالت GGML_GLU_OP_SWIGLU_CLAMP:
        حالت GGML_GLU_OP_GEGLU:
        حالت GGML_GLU_OP_GEGLU_ERF:
        حالت GGML_GLU_OP_GEGLU_تیز:
        حالت GGML_GLU_OP_REGLU:
            توڑو;
        طےشدہ:
            واپس -1;  // Unsupported GLU operation
    }

    // Extract tensor references
    ساخت ggml_tensor * src0    = &params->src0;
    ساخت ggml_tensor * src1    = params->src1.ڈیٹا ? &params->src1 : 0;
    ساخت ggml_tensor * dst     = &params->dst;
    int32_t              swapped = params->swapped;

    // Validate tensor types (F32 only)
    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;  // Unsupported type combination
    }

    اگر (src1 && src1->type != GGML_TYPE_F32) {
        واپس -1;  // Unsupported src1 type
    }

    // Get data pointers
    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * src1_ڈیٹا = src1 ? (اعشاری *) src1->ڈیٹا : src0_ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    // Validate data pointers
    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;  // Null data pointer
    }

    // Get tensor dimensions
    ثابت int64_t nc = dst->ne[0];                            // Output columns (input columns / 2)
    ثابت int64_t nr = dst->ne[1] * dst->ne[2] * dst->ne[3];  // Total rows

    // Get strides
    ثابت size_t src0_stride = src0->nb[1];                       // Stride between rows in src0
    ثابت size_t src1_stride = src1 ? src1->nb[1] : src0->nb[1];  // Stride between rows in src1
    ثابت size_t dst_stride  = dst->nb[1];                        // Stride between rows in dst

    // Validate dimensions for split SwiGLU
    اگر (src1) {
        // Split tensor mode: src0 and src1 should have same shape as dst
        اگر (src0->ne[0] != nc || src1->ne[0] != nc) {
            واپس -1;  // Dimension mismatch in split mode
        }
    } ورنہ {
        // Single tensor mode: src0 should have 2*nc columns
        اگر (src0->ne[0] != 2 * nc) {
            واپس -1;  // Dimension mismatch in single tensor mode
        }
    }

    // Calculate total elements for cache line distribution
    ثابت int64_t elements_فی_cacheline = 16;  // 64 bytes / 4 bytes per float
    ثابت int64_t کل_elements         = nr * nc;
    ثابت int64_t کل_cachelines       = (کل_elements + elements_فی_cacheline - 1) / elements_فی_cacheline;

    // Distribute cache lines across threads
    int64_t cachelines_فی_thread = (کل_cachelines + عدد_دھاگے - 1) / عدد_دھاگے;
    int64_t آغاز_cacheline       = thread_id * cachelines_فی_thread;
    int64_t انتہا_cacheline         = آغاز_cacheline + cachelines_فی_thread;

    // Clamp end_cacheline to actual number of cache lines
    اگر (انتہا_cacheline > کل_cachelines) {
        انتہا_cacheline = کل_cachelines;
    }

    // Thread should return if no work to do
    اگر (آغاز_cacheline >= کل_cachelines) {
        واپس 0;
    }

    // Process cache lines assigned to this thread
    برائے (int64_t cl = آغاز_cacheline; cl < انتہا_cacheline; cl++) {
        // Map cache line back to element coordinates
        int64_t global_element_آغاز = cl * elements_فی_cacheline;
        int64_t صف                  = global_element_آغاز / nc;
        int64_t ستون                  = global_element_آغاز % nc;

        // Skip if we're past the end of data
        اگر (global_element_آغاز >= کل_elements) {
            توڑو;
        }

        // Calculate how many elements to process in this cache line
        int64_t elements_بقایا = کل_elements - global_element_آغاز;
        صحیح     elements_this_بلاک =
            (صحیح) ((elements_بقایا < elements_فی_cacheline) ? elements_بقایا : elements_فی_cacheline);

        // Process elements that span across rows
        int64_t elements_processed = 0;
        جب (elements_processed < elements_this_بلاک && صف < nr) {
            // Calculate elements to process in current row
            int64_t elements_اندر_صف     = nc - ستون;
            int64_t elements_مقصد_process = elements_this_بلاک - elements_processed;
            اگر (elements_مقصد_process > elements_اندر_صف) {
                elements_مقصد_process = elements_اندر_صف;
            }

            // Get pointers for current row and column range
            اعشاری * dst_ptr = (اعشاری *) ((حرف *) dst_ڈیٹا + صف * dst_stride) + ستون;

            اعشاری * x_ptr;
            اعشاری * g_ptr;

            اگر (src1) {
                // Split tensor mode
                x_ptr = (اعشاری *) ((حرف *) src0_ڈیٹا + صف * src0_stride) + ستون;
                g_ptr = (اعشاری *) ((حرف *) src1_ڈیٹا + صف * src1_stride) + ستون;
            } ورنہ {
                // Single tensor mode - src0 contains both x and g
                اعشاری * src0_صف = (اعشاری *) ((حرف *) src0_ڈیٹا + صف * src0_stride);
                اگر (swapped) {
                    g_ptr = src0_صف + ستون;       // First half is gate
                    x_ptr = src0_صف + nc + ستون;  // Second half is value
                } ورنہ {
                    x_ptr = src0_صف + ستون;       // First half is value
                    g_ptr = src0_صف + nc + ستون;  // Second half is gate
                }
            }

            // Process this segment
            انتخاب (params->glu_op_type) {
                حالت GGML_GLU_OP_GEGLU:
                    بلاک_geglu(dst_ptr, x_ptr, g_ptr, (صحیح) elements_مقصد_process);
                    توڑو;
                حالت GGML_GLU_OP_SWIGLU:
                    بلاک_swiglu(dst_ptr, x_ptr, g_ptr, (صحیح) elements_مقصد_process);
                    توڑو;
                حالت GGML_GLU_OP_REGLU:
                    بلاک_reglu(dst_ptr, x_ptr, g_ptr, (صحیح) elements_مقصد_process);
                    توڑو;
                حالت GGML_GLU_OP_GEGLU_تیز:
                    بلاک_geglu_تیز(dst_ptr, x_ptr, g_ptr, (صحیح) elements_مقصد_process);
                    توڑو;
                حالت GGML_GLU_OP_GEGLU_ERF:
                    بلاک_geglu_erf(dst_ptr, x_ptr, g_ptr, (صحیح) elements_مقصد_process);
                    توڑو;
                حالت GGML_GLU_OP_SWIGLU_OAI:
                    بلاک_swiglu_oai(dst_ptr, x_ptr, g_ptr, (صحیح) elements_مقصد_process, params->alpha, params->حد_بندی);
                    توڑو;
                حالت GGML_GLU_OP_SWIGLU_CLAMP:
                    بلاک_swiglu_clamp(dst_ptr, x_ptr, g_ptr, (صحیح) elements_مقصد_process, params->حد_بندی);
                    توڑو;
                طےشدہ:
                    واپس -1;
            }

            // Update counters
            elements_processed += elements_مقصد_process;
            ستون += elements_مقصد_process;

            // Move to next row if current row is complete
            اگر (ستون >= nc) {
                صف++;
                ستون = 0;
            }
        }
    }

    واپس 0;
}
