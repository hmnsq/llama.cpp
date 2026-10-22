//******************************************************************************
// GLU F32 Kernel (SwiGLU specifically)
// Gated Linear Unit: y[i] = silu(x[i]) * g[i] where silu(x) = x * sigmoid(x)
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

// GLU kernel parameters structure (from ET backend ops)
ਬਣਤਰ ggml_et_glu_params {
    ਬਣਤਰ ggml_tensor src0;         // F32 input tensor A (or combined tensor if src1 is null)
    ਬਣਤਰ ggml_tensor src1;         // F32 input tensor B (null for single tensor mode)
    ਬਣਤਰ ggml_tensor dst;          // F32 output tensor (n/2 columns)
    int32_t            glu_op_type;  // GLU operation type (REGLU=0, GEGLU=1, SWIGLU=2, etc.)
    int32_t            swapped;      // Whether gate and value are swapped
    ਦਸ਼ਮਲਵ              alpha;        // SWIGLU_OAI: sigmoid scaling factor
    ਦਸ਼ਮਲਵ              ਹੱਦਬੰਦੀ;        // GLU clamp limit
};

// SiLU activation function: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ silu_f32(ਦਸ਼ਮਲਵ x) {
    // For numerical stability, use the mathematically equivalent form:
    // silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    // For large negative x, exp(-x) -> inf, so silu(x) -> 0
    // For large positive x, exp(-x) -> 0, so silu(x) -> x

    ਜੇ (x > 20.0f) {
        // For x > 20, exp(-x) is negligible, silu(x) ~ x
        ਵਾਪਸ x;
    } ਨਹੀਂਤਾਂ ਜੇ (x < -20.0f) {
        // For x < -20, silu(x) ~ 0
        ਵਾਪਸ 0.0f;
    } ਨਹੀਂਤਾਂ {
        // Use standard formula: silu(x) = x / (1 + exp(-x))
        // Optimized using ET hardware division
        ਦਸ਼ਮਲਵ exp_neg_x   = et_expf(-x);
        ਦਸ਼ਮਲਵ denominator = 1.0f + exp_neg_x;
        ਵਾਪਸ et_fdiv(x, denominator);
    }
}

// Vectorized GeGLU block processing (8 elements = 1 cache line, 64B aligned)
// gelu(x) = 0.5*x*(1 + tanh(z)) = x * (1 - 1/(exp(2z)+1))
// where z = sqrt(2/pi) * x * (1 + 0.044715*x^2)
// Reformulated to avoid inf*0 NaN: uses x * sigmoid(2z) identity
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ਖੰਡ_geglu(ਦਸ਼ਮਲਵ * dst_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * x_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * g_ਖੰਡ, ਪੂਰਨ elements) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

    ਦਸ਼ਮਲਵ one_const       = 1.0f;
    ਦਸ਼ਮਲਵ coef_a_const    = 0.044715f;
    ਦਸ਼ਮਲਵ sqrt2pi_const   = 0.79788456080286535587989211986876f;  // sqrt(2/pi)
    ਦਸ਼ਮਲਵ two_log2e_const = 2.8853900817779268f;                  // 2 * log2(e)

    ਲਈ (int32_t i = 0; i < elements; i += 8) {
        __asm__ ਅਸਥਿਰ(
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

            : [dst_ਬਾਹਰਲਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਖੰਡ[i])
            : [x_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & x_ਖੰਡ[i]), [g_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & g_ਖੰਡ[i]),
              [one_ptr] "m"(one_const), [coef_ptr] "m"(coef_a_const), [sqrt2pi_ptr] "m"(sqrt2pi_const),
              [two_log2e_ptr] "m"(two_log2e_const)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f18", "f20", "f22", "f23", "f24");
    }

    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));
}

// Vectorized SwiGLU block processing (16 elements = 1 cache line)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ਖੰਡ_swiglu(ਦਸ਼ਮਲਵ * dst_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * x_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * g_ਖੰਡ, ਪੂਰਨ elements) {
    // Process 8 elements at a time using vector instructions
    int32_t vec_ਸਿਰਾ = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));  // Save current mask
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Constants for broadcasting
    ਦਸ਼ਮਲਵ zero_const  = 0.0f;
    ਦਸ਼ਮਲਵ one_const   = 1.0f;
    ਦਸ਼ਮਲਵ log2e_const = 1.4426950408889634f;  // log2(e)

    ਲਈ (int32_t i = 0; i < vec_ਸਿਰਾ; i += 8) {
        // Vectorized SwiGLU: dst = silu(x) * g = (x / (1 + exp(-x))) * g
        // Using ET hardware: exp, reciprocal, multiply operations
        __asm__ ਅਸਥਿਰ(
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

            : [dst_ਬਾਹਰਲਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਖੰਡ[i])
            : [x_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & x_ਖੰਡ[i]), [g_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & g_ਖੰਡ[i]),
              [zero_ptr] "m"(zero_const),   // Memory reference to 0.0f for broadcasting
              [one_ptr] "m"(one_const),     // Memory reference to 1.0f for broadcasting
              [log2e_ptr] "m"(log2e_const)  // Memory reference to log2(e) for broadcasting
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22");
    }

    // Restore original mask
    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));

    // Handle remaining elements (< 8) with scalar operations
    ਲਈ (int32_t i = vec_ਸਿਰਾ; i < elements; i++) {
        dst_ਖੰਡ[i] = silu_f32(x_ਖੰਡ[i]) * g_ਖੰਡ[i];
    }
}

// Vectorized ReGLU block: dst = max(0, x) * g
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ਖੰਡ_reglu(ਦਸ਼ਮਲਵ * dst_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * x_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * g_ਖੰਡ, ਪੂਰਨ elements) {
    int32_t vec_ਸਿਰਾ = (elements / 8) * 8;

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

    ਦਸ਼ਮਲਵ zero_const = 0.0f;

    ਲਈ (int32_t i = 0; i < vec_ਸਿਰਾ; i += 8) {
        __asm__ ਅਸਥਿਰ(
            "flw.ps  f10, %[x_vec]\n"     // f10 = x
            "flw.ps  f11, %[g_vec]\n"     // f11 = g
            "fbc.ps  f20, %[zero_ptr]\n"  // f20 = 0.0
            "fmax.ps f12, f10, f20\n"     // f12 = max(x, 0)
            "fmul.ps f13, f12, f11\n"     // f13 = relu(x) * g
            "fsw.ps  f13, %[dst_out]\n"
            : [dst_ਬਾਹਰਲਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਖੰਡ[i])
            : [x_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & x_ਖੰਡ[i]), [g_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & g_ਖੰਡ[i]),
              [zero_ptr] "m"(zero_const)
            : "f10", "f11", "f12", "f13", "f20");
    }

    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));

    ਲਈ (int32_t i = vec_ਸਿਰਾ; i < elements; i++) {
        ਦਸ਼ਮਲਵ xv     = x_ਖੰਡ[i];
        dst_ਖੰਡ[i] = (xv > 0.0f) ? xv * g_ਖੰਡ[i] : 0.0f;
    }
}

// Vectorized GeGLU-Quick block: dst = x * sigmoid(1.702 * x) * g
// Using gelu_quick(x) = x / (1 + exp(-1.702*x))
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ਖੰਡ_geglu_ਤੇਜ਼(ਦਸ਼ਮਲਵ * dst_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * x_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * g_ਖੰਡ, ਪੂਰਨ elements) {
    int32_t vec_ਸਿਰਾ = (elements / 8) * 8;

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

    ਦਸ਼ਮਲਵ zero_const        = 0.0f;
    ਦਸ਼ਮਲਵ one_const         = 1.0f;
    // -1.702 * log2(e), so that fexp.ps(x * neg_k_log2e) = exp(-1.702*x)
    ਦਸ਼ਮਲਵ neg_k_log2e_const = -1.702f * 1.4426950408889634f;

    ਲਈ (int32_t i = 0; i < vec_ਸਿਰਾ; i += 8) {
        __asm__ ਅਸਥਿਰ(
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
            : [dst_ਬਾਹਰਲਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਖੰਡ[i])
            : [x_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & x_ਖੰਡ[i]), [g_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & g_ਖੰਡ[i]),
              [zero_ptr] "m"(zero_const), [one_ptr] "m"(one_const), [k_ptr] "m"(neg_k_log2e_const)
            : "f10", "f11", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22");
    }

    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));

    ਲਈ (int32_t i = vec_ਸਿਰਾ; i < elements; i++) {
        ਦਸ਼ਮਲਵ xv     = x_ਖੰਡ[i];
        // Reuse silu reciprocal path: sigmoid(1.702*x) = 1/(1+exp(-1.702*x))
        ਦਸ਼ਮਲਵ e      = et_expf(-1.702f * xv);
        dst_ਖੰਡ[i] = et_fdiv(xv, 1.0f + e) * g_ਖੰਡ[i];
    }
}

// Vectorized SwiGLU-OAI block (OpenAI gpt-oss variant):
//   x_c = min(x, limit)
//   y_c = clamp(g, -limit, limit)
//   out = (x_c / (1 + exp(-alpha * x_c))) * (y_c + 1)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ਖੰਡ_swiglu_oai(ਦਸ਼ਮਲਵ *       dst_ਖੰਡ,
                                    ਅਚਲ ਦਸ਼ਮਲਵ * x_ਖੰਡ,
                                    ਅਚਲ ਦਸ਼ਮਲਵ * g_ਖੰਡ,
                                    ਪੂਰਨ           elements,
                                    ਦਸ਼ਮਲਵ         alpha,
                                    ਦਸ਼ਮਲਵ         ਹੱਦਬੰਦੀ) {
    int32_t vec_ਸਿਰਾ = (elements / 8) * 8;

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

    ਦਸ਼ਮਲਵ zero_const    = 0.0f;
    ਦਸ਼ਮਲਵ one_const     = 1.0f;
    ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ_pos     = ਹੱਦਬੰਦੀ;
    ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ_neg     = -ਹੱਦਬੰਦੀ;
    // -alpha * log2(e): feed (x * neg_alpha_log2e) into fexp.ps to get exp(-alpha*x)
    ਦਸ਼ਮਲਵ neg_alpha_l2e = -alpha * 1.4426950408889634f;

    ਲਈ (int32_t i = 0; i < vec_ਸਿਰਾ; i += 8) {
        __asm__ ਅਸਥਿਰ(
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

            : [dst_ਬਾਹਰਲਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਖੰਡ[i])
            : [x_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & x_ਖੰਡ[i]), [g_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & g_ਖੰਡ[i]),
              [zero_ptr] "m"(zero_const), [one_ptr] "m"(one_const), [lim_pos] "m"(ਹੱਦਬੰਦੀ_pos), [lim_neg] "m"(ਹੱਦਬੰਦੀ_neg),
              [k_ptr] "m"(neg_alpha_l2e)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f23", "f24", "f25");
    }

    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));

    // Scalar tail (mirrors CPU reference exactly)
    ਲਈ (int32_t i = vec_ਸਿਰਾ; i < elements; i++) {
        ਦਸ਼ਮਲਵ xv = x_ਖੰਡ[i];
        ਦਸ਼ਮਲਵ yv = g_ਖੰਡ[i];
        ਜੇ (xv > ਹੱਦਬੰਦੀ) {
            xv = ਹੱਦਬੰਦੀ;
        }
        ਜੇ (yv > ਹੱਦਬੰਦੀ) {
            yv = ਹੱਦਬੰਦੀ;
        }
        ਜੇ (yv < -ਹੱਦਬੰਦੀ) {
            yv = -ਹੱਦਬੰਦੀ;
        }
        ਦਸ਼ਮਲਵ e       = et_expf(-alpha * xv);
        ਦਸ਼ਮਲਵ ਬਾਹਰਲਾ_glu = et_fdiv(xv, 1.0f + e);
        dst_ਖੰਡ[i]  = ਬਾਹਰਲਾ_glu * (yv + 1.0f);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ਖੰਡ_swiglu_clamp(ਦਸ਼ਮਲਵ *       dst_ਖੰਡ,
                                      ਅਚਲ ਦਸ਼ਮਲਵ * gate_ਖੰਡ,
                                      ਅਚਲ ਦਸ਼ਮਲਵ * ਉੱਪਰ_ਖੰਡ,
                                      ਪੂਰਨ           elements,
                                      ਦਸ਼ਮਲਵ         ਹੱਦਬੰਦੀ) {
    int32_t vec_ਸਿਰਾ = (elements / 8) * 8;

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

    ਦਸ਼ਮਲਵ one_const = 1.0f;
    ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ_pos = ਹੱਦਬੰਦੀ;
    ਦਸ਼ਮਲਵ ਹੱਦਬੰਦੀ_neg = -ਹੱਦਬੰਦੀ;
    ਦਸ਼ਮਲਵ neg_log2e = -1.4426950408889634f;

    ਲਈ (int32_t i = 0; i < vec_ਸਿਰਾ; i += 8) {
        __asm__ ਅਸਥਿਰ(
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
            : [dst_ਬਾਹਰਲਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & dst_ਖੰਡ[i])
            : [gate_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & gate_ਖੰਡ[i]), [ਉੱਪਰ_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & ਉੱਪਰ_ਖੰਡ[i]),
              [one_ptr] "m"(one_const), [lim_pos] "m"(ਹੱਦਬੰਦੀ_pos), [lim_neg] "m"(ਹੱਦਬੰਦੀ_neg), [k_ptr] "m"(neg_log2e)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f21", "f23", "f24", "f25");
    }

    __asm__ ਅਸਥਿਰ("mova.m.x %0" :: "r"(ਆਰਜ਼ੀ_ਨਕਾਬ));

    ਲਈ (int32_t i = vec_ਸਿਰਾ; i < elements; i++) {
        ਦਸ਼ਮਲਵ gate   = gate_ਖੰਡ[i] > ਹੱਦਬੰਦੀ ? ਹੱਦਬੰਦੀ : gate_ਖੰਡ[i];
        ਦਸ਼ਮਲਵ ਉੱਪਰ     = ਉੱਪਰ_ਖੰਡ[i];
        ਉੱਪਰ           = ਉੱਪਰ > ਹੱਦਬੰਦੀ ? ਹੱਦਬੰਦੀ : ਉੱਪਰ;
        ਉੱਪਰ           = ਉੱਪਰ < -ਹੱਦਬੰਦੀ ? -ਹੱਦਬੰਦੀ : ਉੱਪਰ;
        dst_ਖੰਡ[i] = silu_f32(gate) * ਉੱਪਰ;
    }
}

// Scalar erf approximation (Abramowitz & Stegun 7.1.26, max error ~1.5e-7)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ erf_approx(ਦਸ਼ਮਲਵ x) {
    ਅਚਲ ਦਸ਼ਮਲਵ a1 = 0.254829592f;
    ਅਚਲ ਦਸ਼ਮਲਵ a2 = -0.284496736f;
    ਅਚਲ ਦਸ਼ਮਲਵ a3 = 1.421413741f;
    ਅਚਲ ਦਸ਼ਮਲਵ a4 = -1.453152027f;
    ਅਚਲ ਦਸ਼ਮਲਵ a5 = 1.061405429f;
    ਅਚਲ ਦਸ਼ਮਲਵ p  = 0.3275911f;

    ਦਸ਼ਮਲਵ sign = (x < 0.0f) ? -1.0f : 1.0f;
    ਦਸ਼ਮਲਵ ax   = (x < 0.0f) ? -x : x;
    ਦਸ਼ਮਲਵ t    = et_fdiv(1.0f, 1.0f + p * ax);
    ਦਸ਼ਮਲਵ t2   = t * t;
    ਦਸ਼ਮਲਵ t3   = t2 * t;
    ਦਸ਼ਮਲਵ t4   = t3 * t;
    ਦਸ਼ਮਲਵ t5   = t4 * t;
    ਦਸ਼ਮਲਵ poly = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5;
    ਦਸ਼ਮਲਵ y    = 1.0f - poly * et_expf(-ax * ax);
    ਵਾਪਸ sign * y;
}

// GeGLU-Erf block: dst = 0.5 * x * (1 + erf(x / sqrt(2))) * g
// Scalar implementation — variant is rarely used so we keep complexity low.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ਖੰਡ_geglu_erf(ਦਸ਼ਮਲਵ * dst_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * x_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * g_ਖੰਡ, ਪੂਰਨ elements) {
    ਅਚਲ ਦਸ਼ਮਲਵ sqrt_2_inv = 0.70710678118654752440f;
    ਲਈ (int32_t i = 0; i < elements; i++) {
        ਦਸ਼ਮਲਵ xv     = x_ਖੰਡ[i];
        dst_ਖੰਡ[i] = 0.5f * xv * (1.0f + erf_approx(xv * sqrt_2_inv)) * g_ਖੰਡ[i];
    }
}

// Main entry point for GLU kernel
ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_glu_params * params, ਖਾਲੀ * env) {
    // Cast env to proper type
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    // Validate environment pointer
    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    // Get thread info using shire mask from environment
    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    // Basic safety check on params
    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;  // Invalid pointer
    }

    // Supported variants: SwiGLU, SwiGLU-OAI, GeGLU, GeGLU-Erf, GeGLU-Quick, ReGLU
    ਚੋਣ (params->glu_op_type) {
        ਹਾਲਤ GGML_GLU_OP_SWIGLU:
        ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:
        ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
        ਹਾਲਤ GGML_GLU_OP_GEGLU:
        ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
        ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
        ਹਾਲਤ GGML_GLU_OP_REGLU:
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ -1;  // Unsupported GLU operation
    }

    // Extract tensor references
    ਬਣਤਰ ggml_tensor * src0    = &params->src0;
    ਬਣਤਰ ggml_tensor * src1    = params->src1.ਅੰਕੜੇ ? &params->src1 : 0;
    ਬਣਤਰ ggml_tensor * dst     = &params->dst;
    int32_t              swapped = params->swapped;

    // Validate tensor types (F32 only)
    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;  // Unsupported type combination
    }

    ਜੇ (src1 && src1->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;  // Unsupported src1 type
    }

    // Get data pointers
    ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ = (ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * src1_ਅੰਕੜੇ = src1 ? (ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ : src0_ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    // Validate data pointers
    ਜੇ (!src0_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;  // Null data pointer
    }

    // Get tensor dimensions
    ਅਚਲ int64_t nc = dst->ne[0];                            // Output columns (input columns / 2)
    ਅਚਲ int64_t nr = dst->ne[1] * dst->ne[2] * dst->ne[3];  // Total rows

    // Get strides
    ਅਚਲ size_t src0_stride = src0->nb[1];                       // Stride between rows in src0
    ਅਚਲ size_t src1_stride = src1 ? src1->nb[1] : src0->nb[1];  // Stride between rows in src1
    ਅਚਲ size_t dst_stride  = dst->nb[1];                        // Stride between rows in dst

    // Validate dimensions for split SwiGLU
    ਜੇ (src1) {
        // Split tensor mode: src0 and src1 should have same shape as dst
        ਜੇ (src0->ne[0] != nc || src1->ne[0] != nc) {
            ਵਾਪਸ -1;  // Dimension mismatch in split mode
        }
    } ਨਹੀਂਤਾਂ {
        // Single tensor mode: src0 should have 2*nc columns
        ਜੇ (src0->ne[0] != 2 * nc) {
            ਵਾਪਸ -1;  // Dimension mismatch in single tensor mode
        }
    }

    // Calculate total elements for cache line distribution
    ਅਚਲ int64_t elements_ਪ੍ਰਤੀ_cacheline = 16;  // 64 bytes / 4 bytes per float
    ਅਚਲ int64_t ਕੁੱਲ_elements         = nr * nc;
    ਅਚਲ int64_t ਕੁੱਲ_cachelines       = (ਕੁੱਲ_elements + elements_ਪ੍ਰਤੀ_cacheline - 1) / elements_ਪ੍ਰਤੀ_cacheline;

    // Distribute cache lines across threads
    int64_t cachelines_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_cachelines + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
    int64_t ਸ਼ੁਰੂ_cacheline       = thread_id * cachelines_ਪ੍ਰਤੀ_thread;
    int64_t ਸਿਰਾ_cacheline         = ਸ਼ੁਰੂ_cacheline + cachelines_ਪ੍ਰਤੀ_thread;

    // Clamp end_cacheline to actual number of cache lines
    ਜੇ (ਸਿਰਾ_cacheline > ਕੁੱਲ_cachelines) {
        ਸਿਰਾ_cacheline = ਕੁੱਲ_cachelines;
    }

    // Thread should return if no work to do
    ਜੇ (ਸ਼ੁਰੂ_cacheline >= ਕੁੱਲ_cachelines) {
        ਵਾਪਸ 0;
    }

    // Process cache lines assigned to this thread
    ਲਈ (int64_t cl = ਸ਼ੁਰੂ_cacheline; cl < ਸਿਰਾ_cacheline; cl++) {
        // Map cache line back to element coordinates
        int64_t global_element_ਸ਼ੁਰੂ = cl * elements_ਪ੍ਰਤੀ_cacheline;
        int64_t ਪੰਕਤੀ                  = global_element_ਸ਼ੁਰੂ / nc;
        int64_t ਖਾਨਾ                  = global_element_ਸ਼ੁਰੂ % nc;

        // Skip if we're past the end of data
        ਜੇ (global_element_ਸ਼ੁਰੂ >= ਕੁੱਲ_elements) {
            ਰੋਕੋ;
        }

        // Calculate how many elements to process in this cache line
        int64_t elements_ਬਚਿਆ = ਕੁੱਲ_elements - global_element_ਸ਼ੁਰੂ;
        ਪੂਰਨ     elements_this_ਖੰਡ =
            (ਪੂਰਨ) ((elements_ਬਚਿਆ < elements_ਪ੍ਰਤੀ_cacheline) ? elements_ਬਚਿਆ : elements_ਪ੍ਰਤੀ_cacheline);

        // Process elements that span across rows
        int64_t elements_processed = 0;
        ਜਦੋਂ (elements_processed < elements_this_ਖੰਡ && ਪੰਕਤੀ < nr) {
            // Calculate elements to process in current row
            int64_t elements_ਵਿੱਚ_ਪੰਕਤੀ     = nc - ਖਾਨਾ;
            int64_t elements_ਮੰਜ਼ਿਲ_process = elements_this_ਖੰਡ - elements_processed;
            ਜੇ (elements_ਮੰਜ਼ਿਲ_process > elements_ਵਿੱਚ_ਪੰਕਤੀ) {
                elements_ਮੰਜ਼ਿਲ_process = elements_ਵਿੱਚ_ਪੰਕਤੀ;
            }

            // Get pointers for current row and column range
            ਦਸ਼ਮਲਵ * dst_ptr = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + ਪੰਕਤੀ * dst_stride) + ਖਾਨਾ;

            ਦਸ਼ਮਲਵ * x_ptr;
            ਦਸ਼ਮਲਵ * g_ptr;

            ਜੇ (src1) {
                // Split tensor mode
                x_ptr = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0_ਅੰਕੜੇ + ਪੰਕਤੀ * src0_stride) + ਖਾਨਾ;
                g_ptr = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src1_ਅੰਕੜੇ + ਪੰਕਤੀ * src1_stride) + ਖਾਨਾ;
            } ਨਹੀਂਤਾਂ {
                // Single tensor mode - src0 contains both x and g
                ਦਸ਼ਮਲਵ * src0_ਪੰਕਤੀ = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) src0_ਅੰਕੜੇ + ਪੰਕਤੀ * src0_stride);
                ਜੇ (swapped) {
                    g_ptr = src0_ਪੰਕਤੀ + ਖਾਨਾ;       // First half is gate
                    x_ptr = src0_ਪੰਕਤੀ + nc + ਖਾਨਾ;  // Second half is value
                } ਨਹੀਂਤਾਂ {
                    x_ptr = src0_ਪੰਕਤੀ + ਖਾਨਾ;       // First half is value
                    g_ptr = src0_ਪੰਕਤੀ + nc + ਖਾਨਾ;  // Second half is gate
                }
            }

            // Process this segment
            ਚੋਣ (params->glu_op_type) {
                ਹਾਲਤ GGML_GLU_OP_GEGLU:
                    ਖੰਡ_geglu(dst_ptr, x_ptr, g_ptr, (ਪੂਰਨ) elements_ਮੰਜ਼ਿਲ_process);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_SWIGLU:
                    ਖੰਡ_swiglu(dst_ptr, x_ptr, g_ptr, (ਪੂਰਨ) elements_ਮੰਜ਼ਿਲ_process);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_REGLU:
                    ਖੰਡ_reglu(dst_ptr, x_ptr, g_ptr, (ਪੂਰਨ) elements_ਮੰਜ਼ਿਲ_process);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
                    ਖੰਡ_geglu_ਤੇਜ਼(dst_ptr, x_ptr, g_ptr, (ਪੂਰਨ) elements_ਮੰਜ਼ਿਲ_process);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
                    ਖੰਡ_geglu_erf(dst_ptr, x_ptr, g_ptr, (ਪੂਰਨ) elements_ਮੰਜ਼ਿਲ_process);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:
                    ਖੰਡ_swiglu_oai(dst_ptr, x_ptr, g_ptr, (ਪੂਰਨ) elements_ਮੰਜ਼ਿਲ_process, params->alpha, params->ਹੱਦਬੰਦੀ);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
                    ਖੰਡ_swiglu_clamp(dst_ptr, x_ptr, g_ptr, (ਪੂਰਨ) elements_ਮੰਜ਼ਿਲ_process, params->ਹੱਦਬੰਦੀ);
                    ਰੋਕੋ;
                ਮੂਲ:
                    ਵਾਪਸ -1;
            }

            // Update counters
            elements_processed += elements_ਮੰਜ਼ਿਲ_process;
            ਖਾਨਾ += elements_ਮੰਜ਼ਿਲ_process;

            // Move to next row if current row is complete
            ਜੇ (ਖਾਨਾ >= nc) {
                ਪੰਕਤੀ++;
                ਖਾਨਾ = 0;
            }
        }
    }

    ਵਾਪਸ 0;
}
