//******************************************************************************
// GLU F32 Kernel (SwiGLU specifically)
// Gated Linear Unit: y[i] = silu(x[i]) * g[i] where silu(x) = x * sigmoid(x)
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <stdint.h>

// GLU kernel parameters structure (from ET backend ops)
بنية ggml_et_glu_params {
    بنية ggml_tensor src0;         // F32 input tensor A (or combined tensor if src1 is null)
    بنية ggml_tensor src1;         // F32 input tensor B (null for single tensor mode)
    بنية ggml_tensor dst;          // F32 output tensor (n/2 columns)
    int32_t            glu_op_type;  // GLU operation type (REGLU=0, GEGLU=1, SWIGLU=2, etc.)
    int32_t            swapped;      // Whether gate and value are swapped
    عائم              alpha;        // SWIGLU_OAI: sigmoid scaling factor
    عائم              حد_بندی;        // GLU clamp limit
};

// SiLU activation function: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
ساكن مضمن عائم silu_f32(عائم x) {
    // For numerical stability, use the mathematically equivalent form:
    // silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    // For large negative x, exp(-x) -> inf, so silu(x) -> 0
    // For large positive x, exp(-x) -> 0, so silu(x) -> x

    إذا (x > 20.0f) {
        // For x > 20, exp(-x) is negligible, silu(x) ~ x
        أرجع x;
    } وإلا إذا (x < -20.0f) {
        // For x < -20, silu(x) ~ 0
        أرجع 0.0f;
    } وإلا {
        // Use standard formula: silu(x) = x / (1 + exp(-x))
        // Optimized using ET hardware division
        عائم exp_neg_x   = et_expf(-x);
        عائم denominator = 1.0f + exp_neg_x;
        أرجع et_fdiv(x, denominator);
    }
}

// Vectorized GeGLU block processing (8 elements = 1 cache line, 64B aligned)
// gelu(x) = 0.5*x*(1 + tanh(z)) = x * (1 - 1/(exp(2z)+1))
// where z = sqrt(2/pi) * x * (1 + 0.044715*x^2)
// Reformulated to avoid inf*0 NaN: uses x * sigmoid(2z) identity
ساكن مضمن فراغ بلاک_geglu(عائم * dst_بلاک, ثابت عائم * x_بلاک, ثابت عائم * g_بلاک, صحيح elements) {
    بنانشانی لما عارضی_نقاب;
    __asm__ متطاير("mova.x.m %0" : "=r"(عارضی_نقاب));
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");

    عائم one_const       = 1.0f;
    عائم coef_a_const    = 0.044715f;
    عائم sqrt2pi_const   = 0.79788456080286535587989211986876f;  // sqrt(2/pi)
    عائم two_log2e_const = 2.8853900817779268f;                  // 2 * log2(e)

    لكل (int32_t i = 0; i < elements; i += 8) {
        __asm__ متطاير(
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

            : [dst_باہرلا] "=m"(*(عائم (*)[8]) & dst_بلاک[i])
            : [x_vec] "m"(*(ثابت عائم (*)[8]) & x_بلاک[i]), [g_vec] "m"(*(ثابت عائم (*)[8]) & g_بلاک[i]),
              [one_ptr] "m"(one_const), [coef_ptr] "m"(coef_a_const), [sqrt2pi_ptr] "m"(sqrt2pi_const),
              [two_log2e_ptr] "m"(two_log2e_const)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f18", "f20", "f22", "f23", "f24");
    }

    __asm__ متطاير("mova.m.x %0" ::"r"(عارضی_نقاب));
}

// Vectorized SwiGLU block processing (16 elements = 1 cache line)
ساكن مضمن فراغ بلاک_swiglu(عائم * dst_بلاک, ثابت عائم * x_بلاک, ثابت عائم * g_بلاک, صحيح elements) {
    // Process 8 elements at a time using vector instructions
    int32_t vec_سرا = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    بنانشانی لما عارضی_نقاب;
    __asm__ متطاير("mova.x.m %0" : "=r"(عارضی_نقاب));  // Save current mask
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Constants for broadcasting
    عائم zero_const  = 0.0f;
    عائم one_const   = 1.0f;
    عائم log2e_const = 1.4426950408889634f;  // log2(e)

    لكل (int32_t i = 0; i < vec_سرا; i += 8) {
        // Vectorized SwiGLU: dst = silu(x) * g = (x / (1 + exp(-x))) * g
        // Using ET hardware: exp, reciprocal, multiply operations
        __asm__ متطاير(
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

            : [dst_باہرلا] "=m"(*(عائم (*)[8]) & dst_بلاک[i])
            : [x_vec] "m"(*(ثابت عائم (*)[8]) & x_بلاک[i]), [g_vec] "m"(*(ثابت عائم (*)[8]) & g_بلاک[i]),
              [zero_ptr] "m"(zero_const),   // Memory reference to 0.0f for broadcasting
              [one_ptr] "m"(one_const),     // Memory reference to 1.0f for broadcasting
              [log2e_ptr] "m"(log2e_const)  // Memory reference to log2(e) for broadcasting
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22");
    }

    // Restore original mask
    __asm__ متطاير("mova.m.x %0" ::"r"(عارضی_نقاب));

    // Handle remaining elements (< 8) with scalar operations
    لكل (int32_t i = vec_سرا; i < elements; i++) {
        dst_بلاک[i] = silu_f32(x_بلاک[i]) * g_بلاک[i];
    }
}

// Vectorized ReGLU block: dst = max(0, x) * g
ساكن مضمن فراغ بلاک_reglu(عائم * dst_بلاک, ثابت عائم * x_بلاک, ثابت عائم * g_بلاک, صحيح elements) {
    int32_t vec_سرا = (elements / 8) * 8;

    بنانشانی لما عارضی_نقاب;
    __asm__ متطاير("mova.x.m %0" : "=r"(عارضی_نقاب));
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");

    عائم zero_const = 0.0f;

    لكل (int32_t i = 0; i < vec_سرا; i += 8) {
        __asm__ متطاير(
            "flw.ps  f10, %[x_vec]\n"     // f10 = x
            "flw.ps  f11, %[g_vec]\n"     // f11 = g
            "fbc.ps  f20, %[zero_ptr]\n"  // f20 = 0.0
            "fmax.ps f12, f10, f20\n"     // f12 = max(x, 0)
            "fmul.ps f13, f12, f11\n"     // f13 = relu(x) * g
            "fsw.ps  f13, %[dst_out]\n"
            : [dst_باہرلا] "=m"(*(عائم (*)[8]) & dst_بلاک[i])
            : [x_vec] "m"(*(ثابت عائم (*)[8]) & x_بلاک[i]), [g_vec] "m"(*(ثابت عائم (*)[8]) & g_بلاک[i]),
              [zero_ptr] "m"(zero_const)
            : "f10", "f11", "f12", "f13", "f20");
    }

    __asm__ متطاير("mova.m.x %0" ::"r"(عارضی_نقاب));

    لكل (int32_t i = vec_سرا; i < elements; i++) {
        عائم xv     = x_بلاک[i];
        dst_بلاک[i] = (xv > 0.0f) ? xv * g_بلاک[i] : 0.0f;
    }
}

// Vectorized GeGLU-Quick block: dst = x * sigmoid(1.702 * x) * g
// Using gelu_quick(x) = x / (1 + exp(-1.702*x))
ساكن مضمن فراغ بلاک_geglu_تیز(عائم * dst_بلاک, ثابت عائم * x_بلاک, ثابت عائم * g_بلاک, صحيح elements) {
    int32_t vec_سرا = (elements / 8) * 8;

    بنانشانی لما عارضی_نقاب;
    __asm__ متطاير("mova.x.m %0" : "=r"(عارضی_نقاب));
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");

    عائم zero_const        = 0.0f;
    عائم one_const         = 1.0f;
    // -1.702 * log2(e), so that fexp.ps(x * neg_k_log2e) = exp(-1.702*x)
    عائم neg_k_log2e_const = -1.702f * 1.4426950408889634f;

    لكل (int32_t i = 0; i < vec_سرا; i += 8) {
        __asm__ متطاير(
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
            : [dst_باہرلا] "=m"(*(عائم (*)[8]) & dst_بلاک[i])
            : [x_vec] "m"(*(ثابت عائم (*)[8]) & x_بلاک[i]), [g_vec] "m"(*(ثابت عائم (*)[8]) & g_بلاک[i]),
              [zero_ptr] "m"(zero_const), [one_ptr] "m"(one_const), [k_ptr] "m"(neg_k_log2e_const)
            : "f10", "f11", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22");
    }

    __asm__ متطاير("mova.m.x %0" ::"r"(عارضی_نقاب));

    لكل (int32_t i = vec_سرا; i < elements; i++) {
        عائم xv     = x_بلاک[i];
        // Reuse silu reciprocal path: sigmoid(1.702*x) = 1/(1+exp(-1.702*x))
        عائم e      = et_expf(-1.702f * xv);
        dst_بلاک[i] = et_fdiv(xv, 1.0f + e) * g_بلاک[i];
    }
}

// Vectorized SwiGLU-OAI block (OpenAI gpt-oss variant):
//   x_c = min(x, limit)
//   y_c = clamp(g, -limit, limit)
//   out = (x_c / (1 + exp(-alpha * x_c))) * (y_c + 1)
ساكن مضمن فراغ بلاک_swiglu_oai(عائم *       dst_بلاک,
                                    ثابت عائم * x_بلاک,
                                    ثابت عائم * g_بلاک,
                                    صحيح           elements,
                                    عائم         alpha,
                                    عائم         حد_بندی) {
    int32_t vec_سرا = (elements / 8) * 8;

    بنانشانی لما عارضی_نقاب;
    __asm__ متطاير("mova.x.m %0" : "=r"(عارضی_نقاب));
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");

    عائم zero_const    = 0.0f;
    عائم one_const     = 1.0f;
    عائم حد_بندی_pos     = حد_بندی;
    عائم حد_بندی_neg     = -حد_بندی;
    // -alpha * log2(e): feed (x * neg_alpha_log2e) into fexp.ps to get exp(-alpha*x)
    عائم neg_alpha_l2e = -alpha * 1.4426950408889634f;

    لكل (int32_t i = 0; i < vec_سرا; i += 8) {
        __asm__ متطاير(
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

            : [dst_باہرلا] "=m"(*(عائم (*)[8]) & dst_بلاک[i])
            : [x_vec] "m"(*(ثابت عائم (*)[8]) & x_بلاک[i]), [g_vec] "m"(*(ثابت عائم (*)[8]) & g_بلاک[i]),
              [zero_ptr] "m"(zero_const), [one_ptr] "m"(one_const), [lim_pos] "m"(حد_بندی_pos), [lim_neg] "m"(حد_بندی_neg),
              [k_ptr] "m"(neg_alpha_l2e)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f23", "f24", "f25");
    }

    __asm__ متطاير("mova.m.x %0" ::"r"(عارضی_نقاب));

    // Scalar tail (mirrors CPU reference exactly)
    لكل (int32_t i = vec_سرا; i < elements; i++) {
        عائم xv = x_بلاک[i];
        عائم yv = g_بلاک[i];
        إذا (xv > حد_بندی) {
            xv = حد_بندی;
        }
        إذا (yv > حد_بندی) {
            yv = حد_بندی;
        }
        إذا (yv < -حد_بندی) {
            yv = -حد_بندی;
        }
        عائم e       = et_expf(-alpha * xv);
        عائم باہرلا_glu = et_fdiv(xv, 1.0f + e);
        dst_بلاک[i]  = باہرلا_glu * (yv + 1.0f);
    }
}

ساكن مضمن فراغ بلاک_swiglu_clamp(عائم *       dst_بلاک,
                                      ثابت عائم * gate_بلاک,
                                      ثابت عائم * اتے_بلاک,
                                      صحيح           elements,
                                      عائم         حد_بندی) {
    int32_t vec_سرا = (elements / 8) * 8;

    بنانشانی لما عارضی_نقاب;
    __asm__ متطاير("mova.x.m %0" : "=r"(عارضی_نقاب));
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");

    عائم one_const = 1.0f;
    عائم حد_بندی_pos = حد_بندی;
    عائم حد_بندی_neg = -حد_بندی;
    عائم neg_log2e = -1.4426950408889634f;

    لكل (int32_t i = 0; i < vec_سرا; i += 8) {
        __asm__ متطاير(
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
            : [dst_باہرلا] "=m"(*(عائم (*)[8]) & dst_بلاک[i])
            : [gate_vec] "m"(*(ثابت عائم (*)[8]) & gate_بلاک[i]), [اتے_vec] "m"(*(ثابت عائم (*)[8]) & اتے_بلاک[i]),
              [one_ptr] "m"(one_const), [lim_pos] "m"(حد_بندی_pos), [lim_neg] "m"(حد_بندی_neg), [k_ptr] "m"(neg_log2e)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f21", "f23", "f24", "f25");
    }

    __asm__ متطاير("mova.m.x %0" :: "r"(عارضی_نقاب));

    لكل (int32_t i = vec_سرا; i < elements; i++) {
        عائم gate   = gate_بلاک[i] > حد_بندی ? حد_بندی : gate_بلاک[i];
        عائم اتے     = اتے_بلاک[i];
        اتے           = اتے > حد_بندی ? حد_بندی : اتے;
        اتے           = اتے < -حد_بندی ? -حد_بندی : اتے;
        dst_بلاک[i] = silu_f32(gate) * اتے;
    }
}

// Scalar erf approximation (Abramowitz & Stegun 7.1.26, max error ~1.5e-7)
ساكن مضمن عائم erf_approx(عائم x) {
    ثابت عائم a1 = 0.254829592f;
    ثابت عائم a2 = -0.284496736f;
    ثابت عائم a3 = 1.421413741f;
    ثابت عائم a4 = -1.453152027f;
    ثابت عائم a5 = 1.061405429f;
    ثابت عائم p  = 0.3275911f;

    عائم sign = (x < 0.0f) ? -1.0f : 1.0f;
    عائم ax   = (x < 0.0f) ? -x : x;
    عائم t    = et_fdiv(1.0f, 1.0f + p * ax);
    عائم t2   = t * t;
    عائم t3   = t2 * t;
    عائم t4   = t3 * t;
    عائم t5   = t4 * t;
    عائم poly = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5;
    عائم y    = 1.0f - poly * et_expf(-ax * ax);
    أرجع sign * y;
}

// GeGLU-Erf block: dst = 0.5 * x * (1 + erf(x / sqrt(2))) * g
// Scalar implementation — variant is rarely used so we keep complexity low.
ساكن مضمن فراغ بلاک_geglu_erf(عائم * dst_بلاک, ثابت عائم * x_بلاک, ثابت عائم * g_بلاک, صحيح elements) {
    ثابت عائم sqrt_2_inv = 0.70710678118654752440f;
    لكل (int32_t i = 0; i < elements; i++) {
        عائم xv     = x_بلاک[i];
        dst_بلاک[i] = 0.5f * xv * (1.0f + erf_approx(xv * sqrt_2_inv)) * g_بلاک[i];
    }
}

// Main entry point for GLU kernel
صحيح اندراج_نقطہ(بنية ggml_et_glu_params * params, فراغ * env) {
    // Cast env to proper type
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    // Validate environment pointer
    إذا (!kernel_env) {
        أرجع -1;
    }

    // Get thread info using shire mask from environment
    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    // Basic safety check on params
    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;  // Invalid pointer
    }

    // Supported variants: SwiGLU, SwiGLU-OAI, GeGLU, GeGLU-Erf, GeGLU-Quick, ReGLU
    بدل (params->glu_op_type) {
        حالة GGML_GLU_OP_SWIGLU:
        حالة GGML_GLU_OP_SWIGLU_OAI:
        حالة GGML_GLU_OP_SWIGLU_CLAMP:
        حالة GGML_GLU_OP_GEGLU:
        حالة GGML_GLU_OP_GEGLU_ERF:
        حالة GGML_GLU_OP_GEGLU_تیز:
        حالة GGML_GLU_OP_REGLU:
            اكسر;
        افتراضي:
            أرجع -1;  // Unsupported GLU operation
    }

    // Extract tensor references
    بنية ggml_tensor * src0    = &params->src0;
    بنية ggml_tensor * src1    = params->src1.انکڑے ? &params->src1 : 0;
    بنية ggml_tensor * dst     = &params->dst;
    int32_t              swapped = params->swapped;

    // Validate tensor types (F32 only)
    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;  // Unsupported type combination
    }

    إذا (src1 && src1->type != GGML_TYPE_F32) {
        أرجع -1;  // Unsupported src1 type
    }

    // Get data pointers
    عائم * src0_انکڑے = (عائم *) src0->انکڑے;
    عائم * src1_انکڑے = src1 ? (عائم *) src1->انکڑے : src0_انکڑے;
    عائم * dst_انکڑے  = (عائم *) dst->انکڑے;

    // Validate data pointers
    إذا (!src0_انکڑے || !dst_انکڑے) {
        أرجع -1;  // Null data pointer
    }

    // Get tensor dimensions
    ثابت int64_t nc = dst->ne[0];                            // Output columns (input columns / 2)
    ثابت int64_t nr = dst->ne[1] * dst->ne[2] * dst->ne[3];  // Total rows

    // Get strides
    ثابت size_t src0_stride = src0->nb[1];                       // Stride between rows in src0
    ثابت size_t src1_stride = src1 ? src1->nb[1] : src0->nb[1];  // Stride between rows in src1
    ثابت size_t dst_stride  = dst->nb[1];                        // Stride between rows in dst

    // Validate dimensions for split SwiGLU
    إذا (src1) {
        // Split tensor mode: src0 and src1 should have same shape as dst
        إذا (src0->ne[0] != nc || src1->ne[0] != nc) {
            أرجع -1;  // Dimension mismatch in split mode
        }
    } وإلا {
        // Single tensor mode: src0 should have 2*nc columns
        إذا (src0->ne[0] != 2 * nc) {
            أرجع -1;  // Dimension mismatch in single tensor mode
        }
    }

    // Calculate total elements for cache line distribution
    ثابت int64_t elements_فی_cacheline = 16;  // 64 bytes / 4 bytes per float
    ثابت int64_t کل_elements         = nr * nc;
    ثابت int64_t کل_cachelines       = (کل_elements + elements_فی_cacheline - 1) / elements_فی_cacheline;

    // Distribute cache lines across threads
    int64_t cachelines_فی_thread = (کل_cachelines + عدد_دھاگے - 1) / عدد_دھاگے;
    int64_t شروع_cacheline       = thread_id * cachelines_فی_thread;
    int64_t سرا_cacheline         = شروع_cacheline + cachelines_فی_thread;

    // Clamp end_cacheline to actual number of cache lines
    إذا (سرا_cacheline > کل_cachelines) {
        سرا_cacheline = کل_cachelines;
    }

    // Thread should return if no work to do
    إذا (شروع_cacheline >= کل_cachelines) {
        أرجع 0;
    }

    // Process cache lines assigned to this thread
    لكل (int64_t cl = شروع_cacheline; cl < سرا_cacheline; cl++) {
        // Map cache line back to element coordinates
        int64_t global_element_شروع = cl * elements_فی_cacheline;
        int64_t صف                  = global_element_شروع / nc;
        int64_t خانہ                  = global_element_شروع % nc;

        // Skip if we're past the end of data
        إذا (global_element_شروع >= کل_elements) {
            اكسر;
        }

        // Calculate how many elements to process in this cache line
        int64_t elements_بچیا = کل_elements - global_element_شروع;
        صحيح     elements_this_بلاک =
            (صحيح) ((elements_بچیا < elements_فی_cacheline) ? elements_بچیا : elements_فی_cacheline);

        // Process elements that span across rows
        int64_t elements_processed = 0;
        طالما (elements_processed < elements_this_بلاک && صف < nr) {
            // Calculate elements to process in current row
            int64_t elements_وچ_صف     = nc - خانہ;
            int64_t elements_منزل_process = elements_this_بلاک - elements_processed;
            إذا (elements_منزل_process > elements_وچ_صف) {
                elements_منزل_process = elements_وچ_صف;
            }

            // Get pointers for current row and column range
            عائم * dst_ptr = (عائم *) ((حرف *) dst_انکڑے + صف * dst_stride) + خانہ;

            عائم * x_ptr;
            عائم * g_ptr;

            إذا (src1) {
                // Split tensor mode
                x_ptr = (عائم *) ((حرف *) src0_انکڑے + صف * src0_stride) + خانہ;
                g_ptr = (عائم *) ((حرف *) src1_انکڑے + صف * src1_stride) + خانہ;
            } وإلا {
                // Single tensor mode - src0 contains both x and g
                عائم * src0_صف = (عائم *) ((حرف *) src0_انکڑے + صف * src0_stride);
                إذا (swapped) {
                    g_ptr = src0_صف + خانہ;       // First half is gate
                    x_ptr = src0_صف + nc + خانہ;  // Second half is value
                } وإلا {
                    x_ptr = src0_صف + خانہ;       // First half is value
                    g_ptr = src0_صف + nc + خانہ;  // Second half is gate
                }
            }

            // Process this segment
            بدل (params->glu_op_type) {
                حالة GGML_GLU_OP_GEGLU:
                    بلاک_geglu(dst_ptr, x_ptr, g_ptr, (صحيح) elements_منزل_process);
                    اكسر;
                حالة GGML_GLU_OP_SWIGLU:
                    بلاک_swiglu(dst_ptr, x_ptr, g_ptr, (صحيح) elements_منزل_process);
                    اكسر;
                حالة GGML_GLU_OP_REGLU:
                    بلاک_reglu(dst_ptr, x_ptr, g_ptr, (صحيح) elements_منزل_process);
                    اكسر;
                حالة GGML_GLU_OP_GEGLU_تیز:
                    بلاک_geglu_تیز(dst_ptr, x_ptr, g_ptr, (صحيح) elements_منزل_process);
                    اكسر;
                حالة GGML_GLU_OP_GEGLU_ERF:
                    بلاک_geglu_erf(dst_ptr, x_ptr, g_ptr, (صحيح) elements_منزل_process);
                    اكسر;
                حالة GGML_GLU_OP_SWIGLU_OAI:
                    بلاک_swiglu_oai(dst_ptr, x_ptr, g_ptr, (صحيح) elements_منزل_process, params->alpha, params->حد_بندی);
                    اكسر;
                حالة GGML_GLU_OP_SWIGLU_CLAMP:
                    بلاک_swiglu_clamp(dst_ptr, x_ptr, g_ptr, (صحيح) elements_منزل_process, params->حد_بندی);
                    اكسر;
                افتراضي:
                    أرجع -1;
            }

            // Update counters
            elements_processed += elements_منزل_process;
            خانہ += elements_منزل_process;

            // Move to next row if current row is complete
            إذا (خانہ >= nc) {
                صف++;
                خانہ = 0;
            }
        }
    }

    أرجع 0;
}
