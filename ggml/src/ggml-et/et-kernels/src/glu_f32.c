//******************************************************************************
// GLU F32 Kernel (SwiGLU specifically)
// Gated Linear Unit: y[i] = silu(x[i]) * g[i] where silu(x) = x * sigmoid(x)
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

// GLU kernel parameters structure (from ET backend ops)
ರಚನೆ ggml_et_glu_params {
    ರಚನೆ ggml_tensor src0;         // F32 input tensor A (or combined tensor if src1 is null)
    ರಚನೆ ggml_tensor src1;         // F32 input tensor B (null for single tensor mode)
    ರಚನೆ ggml_tensor dst;          // F32 output tensor (n/2 columns)
    int32_t            glu_op_type;  // GLU operation type (REGLU=0, GEGLU=1, SWIGLU=2, etc.)
    int32_t            swapped;      // Whether gate and value are swapped
    ದಶಮಾಂಶ              alpha;        // SWIGLU_OAI: sigmoid scaling factor
    ದಶಮಾಂಶ              ಎಲ್ಲೆ;        // GLU clamp limit
};

// SiLU activation function: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ silu_f32(ದಶಮಾಂಶ x) {
    // For numerical stability, use the mathematically equivalent form:
    // silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    // For large negative x, exp(-x) -> inf, so silu(x) -> 0
    // For large positive x, exp(-x) -> 0, so silu(x) -> x

    ಆದರೆ (x > 20.0f) {
        // For x > 20, exp(-x) is negligible, silu(x) ~ x
        ಹಿಂತಿರುಗಿ x;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (x < -20.0f) {
        // For x < -20, silu(x) ~ 0
        ಹಿಂತಿರುಗಿ 0.0f;
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Use standard formula: silu(x) = x / (1 + exp(-x))
        // Optimized using ET hardware division
        ದಶಮಾಂಶ exp_neg_x   = et_expf(-x);
        ದಶಮಾಂಶ denominator = 1.0f + exp_neg_x;
        ಹಿಂತಿರುಗಿ et_fdiv(x, denominator);
    }
}

// Vectorized GeGLU block processing (8 elements = 1 cache line, 64B aligned)
// gelu(x) = 0.5*x*(1 + tanh(z)) = x * (1 - 1/(exp(2z)+1))
// where z = sqrt(2/pi) * x * (1 + 0.044715*x^2)
// Reformulated to avoid inf*0 NaN: uses x * sigmoid(2z) identity
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಖಂಡ_geglu(ದಶಮಾಂಶ * dst_ಖಂಡ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_ಖಂಡ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g_ಖಂಡ, ಪೂರ್ಣಾಂಕ elements) {
    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));
    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");

    ದಶಮಾಂಶ one_const       = 1.0f;
    ದಶಮಾಂಶ coef_a_const    = 0.044715f;
    ದಶಮಾಂಶ sqrt2pi_const   = 0.79788456080286535587989211986876f;  // sqrt(2/pi)
    ದಶಮಾಂಶ two_log2e_const = 2.8853900817779268f;                  // 2 * log2(e)

    ಗಾಗಿ (int32_t i = 0; i < elements; i += 8) {
        __asm__ ಅಸ್ಥಿರ(
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

            : [dst_ಹೊರಹೋಗು] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಖಂಡ[i])
            : [x_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & x_ಖಂಡ[i]), [g_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & g_ಖಂಡ[i]),
              [one_ptr] "m"(one_const), [coef_ptr] "m"(coef_a_const), [sqrt2pi_ptr] "m"(sqrt2pi_const),
              [two_log2e_ptr] "m"(two_log2e_const)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f18", "f20", "f22", "f23", "f24");
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));
}

// Vectorized SwiGLU block processing (16 elements = 1 cache line)
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಖಂಡ_swiglu(ದಶಮಾಂಶ * dst_ಖಂಡ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_ಖಂಡ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g_ಖಂಡ, ಪೂರ್ಣಾಂಕ elements) {
    // Process 8 elements at a time using vector instructions
    int32_t vec_ಕೊನೆ = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));  // Save current mask
    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Constants for broadcasting
    ದಶಮಾಂಶ zero_const  = 0.0f;
    ದಶಮಾಂಶ one_const   = 1.0f;
    ದಶಮಾಂಶ log2e_const = 1.4426950408889634f;  // log2(e)

    ಗಾಗಿ (int32_t i = 0; i < vec_ಕೊನೆ; i += 8) {
        // Vectorized SwiGLU: dst = silu(x) * g = (x / (1 + exp(-x))) * g
        // Using ET hardware: exp, reciprocal, multiply operations
        __asm__ ಅಸ್ಥಿರ(
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

            : [dst_ಹೊರಹೋಗು] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಖಂಡ[i])
            : [x_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & x_ಖಂಡ[i]), [g_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & g_ಖಂಡ[i]),
              [zero_ptr] "m"(zero_const),   // Memory reference to 0.0f for broadcasting
              [one_ptr] "m"(one_const),     // Memory reference to 1.0f for broadcasting
              [log2e_ptr] "m"(log2e_const)  // Memory reference to log2(e) for broadcasting
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22");
    }

    // Restore original mask
    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));

    // Handle remaining elements (< 8) with scalar operations
    ಗಾಗಿ (int32_t i = vec_ಕೊನೆ; i < elements; i++) {
        dst_ಖಂಡ[i] = silu_f32(x_ಖಂಡ[i]) * g_ಖಂಡ[i];
    }
}

// Vectorized ReGLU block: dst = max(0, x) * g
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಖಂಡ_reglu(ದಶಮಾಂಶ * dst_ಖಂಡ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_ಖಂಡ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g_ಖಂಡ, ಪೂರ್ಣಾಂಕ elements) {
    int32_t vec_ಕೊನೆ = (elements / 8) * 8;

    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));
    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");

    ದಶಮಾಂಶ zero_const = 0.0f;

    ಗಾಗಿ (int32_t i = 0; i < vec_ಕೊನೆ; i += 8) {
        __asm__ ಅಸ್ಥಿರ(
            "flw.ps  f10, %[x_vec]\n"     // f10 = x
            "flw.ps  f11, %[g_vec]\n"     // f11 = g
            "fbc.ps  f20, %[zero_ptr]\n"  // f20 = 0.0
            "fmax.ps f12, f10, f20\n"     // f12 = max(x, 0)
            "fmul.ps f13, f12, f11\n"     // f13 = relu(x) * g
            "fsw.ps  f13, %[dst_out]\n"
            : [dst_ಹೊರಹೋಗು] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಖಂಡ[i])
            : [x_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & x_ಖಂಡ[i]), [g_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & g_ಖಂಡ[i]),
              [zero_ptr] "m"(zero_const)
            : "f10", "f11", "f12", "f13", "f20");
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));

    ಗಾಗಿ (int32_t i = vec_ಕೊನೆ; i < elements; i++) {
        ದಶಮಾಂಶ xv     = x_ಖಂಡ[i];
        dst_ಖಂಡ[i] = (xv > 0.0f) ? xv * g_ಖಂಡ[i] : 0.0f;
    }
}

// Vectorized GeGLU-Quick block: dst = x * sigmoid(1.702 * x) * g
// Using gelu_quick(x) = x / (1 + exp(-1.702*x))
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಖಂಡ_geglu_ಶೀಘ್ರ(ದಶಮಾಂಶ * dst_ಖಂಡ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_ಖಂಡ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g_ಖಂಡ, ಪೂರ್ಣಾಂಕ elements) {
    int32_t vec_ಕೊನೆ = (elements / 8) * 8;

    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));
    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");

    ದಶಮಾಂಶ zero_const        = 0.0f;
    ದಶಮಾಂಶ one_const         = 1.0f;
    // -1.702 * log2(e), so that fexp.ps(x * neg_k_log2e) = exp(-1.702*x)
    ದಶಮಾಂಶ neg_k_log2e_const = -1.702f * 1.4426950408889634f;

    ಗಾಗಿ (int32_t i = 0; i < vec_ಕೊನೆ; i += 8) {
        __asm__ ಅಸ್ಥಿರ(
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
            : [dst_ಹೊರಹೋಗು] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಖಂಡ[i])
            : [x_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & x_ಖಂಡ[i]), [g_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & g_ಖಂಡ[i]),
              [zero_ptr] "m"(zero_const), [one_ptr] "m"(one_const), [k_ptr] "m"(neg_k_log2e_const)
            : "f10", "f11", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22");
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));

    ಗಾಗಿ (int32_t i = vec_ಕೊನೆ; i < elements; i++) {
        ದಶಮಾಂಶ xv     = x_ಖಂಡ[i];
        // Reuse silu reciprocal path: sigmoid(1.702*x) = 1/(1+exp(-1.702*x))
        ದಶಮಾಂಶ e      = et_expf(-1.702f * xv);
        dst_ಖಂಡ[i] = et_fdiv(xv, 1.0f + e) * g_ಖಂಡ[i];
    }
}

// Vectorized SwiGLU-OAI block (OpenAI gpt-oss variant):
//   x_c = min(x, limit)
//   y_c = clamp(g, -limit, limit)
//   out = (x_c / (1 + exp(-alpha * x_c))) * (y_c + 1)
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಖಂಡ_swiglu_oai(ದಶಮಾಂಶ *       dst_ಖಂಡ,
                                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_ಖಂಡ,
                                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g_ಖಂಡ,
                                    ಪೂರ್ಣಾಂಕ           elements,
                                    ದಶಮಾಂಶ         alpha,
                                    ದಶಮಾಂಶ         ಎಲ್ಲೆ) {
    int32_t vec_ಕೊನೆ = (elements / 8) * 8;

    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));
    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");

    ದಶಮಾಂಶ zero_const    = 0.0f;
    ದಶಮಾಂಶ one_const     = 1.0f;
    ದಶಮಾಂಶ ಎಲ್ಲೆ_pos     = ಎಲ್ಲೆ;
    ದಶಮಾಂಶ ಎಲ್ಲೆ_neg     = -ಎಲ್ಲೆ;
    // -alpha * log2(e): feed (x * neg_alpha_log2e) into fexp.ps to get exp(-alpha*x)
    ದಶಮಾಂಶ neg_alpha_l2e = -alpha * 1.4426950408889634f;

    ಗಾಗಿ (int32_t i = 0; i < vec_ಕೊನೆ; i += 8) {
        __asm__ ಅಸ್ಥಿರ(
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

            : [dst_ಹೊರಹೋಗು] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಖಂಡ[i])
            : [x_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & x_ಖಂಡ[i]), [g_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & g_ಖಂಡ[i]),
              [zero_ptr] "m"(zero_const), [one_ptr] "m"(one_const), [lim_pos] "m"(ಎಲ್ಲೆ_pos), [lim_neg] "m"(ಎಲ್ಲೆ_neg),
              [k_ptr] "m"(neg_alpha_l2e)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f23", "f24", "f25");
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));

    // Scalar tail (mirrors CPU reference exactly)
    ಗಾಗಿ (int32_t i = vec_ಕೊನೆ; i < elements; i++) {
        ದಶಮಾಂಶ xv = x_ಖಂಡ[i];
        ದಶಮಾಂಶ yv = g_ಖಂಡ[i];
        ಆದರೆ (xv > ಎಲ್ಲೆ) {
            xv = ಎಲ್ಲೆ;
        }
        ಆದರೆ (yv > ಎಲ್ಲೆ) {
            yv = ಎಲ್ಲೆ;
        }
        ಆದರೆ (yv < -ಎಲ್ಲೆ) {
            yv = -ಎಲ್ಲೆ;
        }
        ದಶಮಾಂಶ e       = et_expf(-alpha * xv);
        ದಶಮಾಂಶ ಹೊರಹೋಗು_glu = et_fdiv(xv, 1.0f + e);
        dst_ಖಂಡ[i]  = ಹೊರಹೋಗು_glu * (yv + 1.0f);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಖಂಡ_swiglu_clamp(ದಶಮಾಂಶ *       dst_ಖಂಡ,
                                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * gate_ಖಂಡ,
                                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಮೇಲೆ_ಖಂಡ,
                                      ಪೂರ್ಣಾಂಕ           elements,
                                      ದಶಮಾಂಶ         ಎಲ್ಲೆ) {
    int32_t vec_ಕೊನೆ = (elements / 8) * 8;

    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));
    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");

    ದಶಮಾಂಶ one_const = 1.0f;
    ದಶಮಾಂಶ ಎಲ್ಲೆ_pos = ಎಲ್ಲೆ;
    ದಶಮಾಂಶ ಎಲ್ಲೆ_neg = -ಎಲ್ಲೆ;
    ದಶಮಾಂಶ neg_log2e = -1.4426950408889634f;

    ಗಾಗಿ (int32_t i = 0; i < vec_ಕೊನೆ; i += 8) {
        __asm__ ಅಸ್ಥಿರ(
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
            : [dst_ಹೊರಹೋಗು] "=m"(*(ದಶಮಾಂಶ (*)[8]) & dst_ಖಂಡ[i])
            : [gate_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & gate_ಖಂಡ[i]), [ಮೇಲೆ_vec] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & ಮೇಲೆ_ಖಂಡ[i]),
              [one_ptr] "m"(one_const), [lim_pos] "m"(ಎಲ್ಲೆ_pos), [lim_neg] "m"(ಎಲ್ಲೆ_neg), [k_ptr] "m"(neg_log2e)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f21", "f23", "f24", "f25");
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" :: "r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));

    ಗಾಗಿ (int32_t i = vec_ಕೊನೆ; i < elements; i++) {
        ದಶಮಾಂಶ gate   = gate_ಖಂಡ[i] > ಎಲ್ಲೆ ? ಎಲ್ಲೆ : gate_ಖಂಡ[i];
        ದಶಮಾಂಶ ಮೇಲೆ     = ಮೇಲೆ_ಖಂಡ[i];
        ಮೇಲೆ           = ಮೇಲೆ > ಎಲ್ಲೆ ? ಎಲ್ಲೆ : ಮೇಲೆ;
        ಮೇಲೆ           = ಮೇಲೆ < -ಎಲ್ಲೆ ? -ಎಲ್ಲೆ : ಮೇಲೆ;
        dst_ಖಂಡ[i] = silu_f32(gate) * ಮೇಲೆ;
    }
}

// Scalar erf approximation (Abramowitz & Stegun 7.1.26, max error ~1.5e-7)
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ erf_approx(ದಶಮಾಂಶ x) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a1 = 0.254829592f;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a2 = -0.284496736f;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a3 = 1.421413741f;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a4 = -1.453152027f;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a5 = 1.061405429f;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ p  = 0.3275911f;

    ದಶಮಾಂಶ sign = (x < 0.0f) ? -1.0f : 1.0f;
    ದಶಮಾಂಶ ax   = (x < 0.0f) ? -x : x;
    ದಶಮಾಂಶ t    = et_fdiv(1.0f, 1.0f + p * ax);
    ದಶಮಾಂಶ t2   = t * t;
    ದಶಮಾಂಶ t3   = t2 * t;
    ದಶಮಾಂಶ t4   = t3 * t;
    ದಶಮಾಂಶ t5   = t4 * t;
    ದಶಮಾಂಶ poly = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5;
    ದಶಮಾಂಶ y    = 1.0f - poly * et_expf(-ax * ax);
    ಹಿಂತಿರುಗಿ sign * y;
}

// GeGLU-Erf block: dst = 0.5 * x * (1 + erf(x / sqrt(2))) * g
// Scalar implementation — variant is rarely used so we keep complexity low.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಖಂಡ_geglu_erf(ದಶಮಾಂಶ * dst_ಖಂಡ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_ಖಂಡ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * g_ಖಂಡ, ಪೂರ್ಣಾಂಕ elements) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sqrt_2_inv = 0.70710678118654752440f;
    ಗಾಗಿ (int32_t i = 0; i < elements; i++) {
        ದಶಮಾಂಶ xv     = x_ಖಂಡ[i];
        dst_ಖಂಡ[i] = 0.5f * xv * (1.0f + erf_approx(xv * sqrt_2_inv)) * g_ಖಂಡ[i];
    }
}

// Main entry point for GLU kernel
ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_glu_params * params, ಶೂನ್ಯ * env) {
    // Cast env to proper type
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    // Validate environment pointer
    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    // Get thread info using shire mask from environment
    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    // Basic safety check on params
    ಆದರೆ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;  // Invalid pointer
    }

    // Supported variants: SwiGLU, SwiGLU-OAI, GeGLU, GeGLU-Erf, GeGLU-Quick, ReGLU
    ಆಯ್ಕೆ (params->glu_op_type) {
        ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
        ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:
        ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
        ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
        ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
        ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
        ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ -1;  // Unsupported GLU operation
    }

    // Extract tensor references
    ರಚನೆ ggml_tensor * src0    = &params->src0;
    ರಚನೆ ggml_tensor * src1    = params->src1.ದತ್ತಾಂಶ ? &params->src1 : 0;
    ರಚನೆ ggml_tensor * dst     = &params->dst;
    int32_t              swapped = params->swapped;

    // Validate tensor types (F32 only)
    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;  // Unsupported type combination
    }

    ಆದರೆ (src1 && src1->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;  // Unsupported src1 type
    }

    // Get data pointers
    ದಶಮಾಂಶ * src0_ದತ್ತಾಂಶ = (ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * src1_ದತ್ತಾಂಶ = src1 ? (ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ : src0_ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    // Validate data pointers
    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;  // Null data pointer
    }

    // Get tensor dimensions
    ಸ್ಥಿರಾಂಕ int64_t nc = dst->ne[0];                            // Output columns (input columns / 2)
    ಸ್ಥಿರಾಂಕ int64_t nr = dst->ne[1] * dst->ne[2] * dst->ne[3];  // Total rows

    // Get strides
    ಸ್ಥಿರಾಂಕ size_t src0_stride = src0->nb[1];                       // Stride between rows in src0
    ಸ್ಥಿರಾಂಕ size_t src1_stride = src1 ? src1->nb[1] : src0->nb[1];  // Stride between rows in src1
    ಸ್ಥಿರಾಂಕ size_t dst_stride  = dst->nb[1];                        // Stride between rows in dst

    // Validate dimensions for split SwiGLU
    ಆದರೆ (src1) {
        // Split tensor mode: src0 and src1 should have same shape as dst
        ಆದರೆ (src0->ne[0] != nc || src1->ne[0] != nc) {
            ಹಿಂತಿರುಗಿ -1;  // Dimension mismatch in split mode
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Single tensor mode: src0 should have 2*nc columns
        ಆದರೆ (src0->ne[0] != 2 * nc) {
            ಹಿಂತಿರುಗಿ -1;  // Dimension mismatch in single tensor mode
        }
    }

    // Calculate total elements for cache line distribution
    ಸ್ಥಿರಾಂಕ int64_t elements_ಪ್ರತಿ_cacheline = 16;  // 64 bytes / 4 bytes per float
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elements         = nr * nc;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cachelines       = (ಒಟ್ಟು_elements + elements_ಪ್ರತಿ_cacheline - 1) / elements_ಪ್ರತಿ_cacheline;

    // Distribute cache lines across threads
    int64_t cachelines_ಪ್ರತಿ_thread = (ಒಟ್ಟು_cachelines + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
    int64_t ಆರಂಭ_cacheline       = thread_id * cachelines_ಪ್ರತಿ_thread;
    int64_t ಕೊನೆ_cacheline         = ಆರಂಭ_cacheline + cachelines_ಪ್ರತಿ_thread;

    // Clamp end_cacheline to actual number of cache lines
    ಆದರೆ (ಕೊನೆ_cacheline > ಒಟ್ಟು_cachelines) {
        ಕೊನೆ_cacheline = ಒಟ್ಟು_cachelines;
    }

    // Thread should return if no work to do
    ಆದರೆ (ಆರಂಭ_cacheline >= ಒಟ್ಟು_cachelines) {
        ಹಿಂತಿರುಗಿ 0;
    }

    // Process cache lines assigned to this thread
    ಗಾಗಿ (int64_t cl = ಆರಂಭ_cacheline; cl < ಕೊನೆ_cacheline; cl++) {
        // Map cache line back to element coordinates
        int64_t global_element_ಆರಂಭ = cl * elements_ಪ್ರತಿ_cacheline;
        int64_t ಅಡ್ಡಸಾಲು                  = global_element_ಆರಂಭ / nc;
        int64_t ಲಂಬಸಾಲು                  = global_element_ಆರಂಭ % nc;

        // Skip if we're past the end of data
        ಆದರೆ (global_element_ಆರಂಭ >= ಒಟ್ಟು_elements) {
            ನಿಲ್ಲಿಸು;
        }

        // Calculate how many elements to process in this cache line
        int64_t elements_ಉಳಿದ = ಒಟ್ಟು_elements - global_element_ಆರಂಭ;
        ಪೂರ್ಣಾಂಕ     elements_this_ಖಂಡ =
            (ಪೂರ್ಣಾಂಕ) ((elements_ಉಳಿದ < elements_ಪ್ರತಿ_cacheline) ? elements_ಉಳಿದ : elements_ಪ್ರತಿ_cacheline);

        // Process elements that span across rows
        int64_t elements_processed = 0;
        ಆಗ (elements_processed < elements_this_ಖಂಡ && ಅಡ್ಡಸಾಲು < nr) {
            // Calculate elements to process in current row
            int64_t elements_ಒಳ_ಅಡ್ಡಸಾಲು     = nc - ಲಂಬಸಾಲು;
            int64_t elements_ಗಮ್ಯ_process = elements_this_ಖಂಡ - elements_processed;
            ಆದರೆ (elements_ಗಮ್ಯ_process > elements_ಒಳ_ಅಡ್ಡಸಾಲು) {
                elements_ಗಮ್ಯ_process = elements_ಒಳ_ಅಡ್ಡಸಾಲು;
            }

            // Get pointers for current row and column range
            ದಶಮಾಂಶ * dst_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + ಅಡ್ಡಸಾಲು * dst_stride) + ಲಂಬಸಾಲು;

            ದಶಮಾಂಶ * x_ptr;
            ದಶಮಾಂಶ * g_ptr;

            ಆದರೆ (src1) {
                // Split tensor mode
                x_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + ಅಡ್ಡಸಾಲು * src0_stride) + ಲಂಬಸಾಲು;
                g_ptr = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src1_ದತ್ತಾಂಶ + ಅಡ್ಡಸಾಲು * src1_stride) + ಲಂಬಸಾಲು;
            } ಇಲ್ಲದಿದ್ದರೆ {
                // Single tensor mode - src0 contains both x and g
                ದಶಮಾಂಶ * src0_ಅಡ್ಡಸಾಲು = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + ಅಡ್ಡಸಾಲು * src0_stride);
                ಆದರೆ (swapped) {
                    g_ptr = src0_ಅಡ್ಡಸಾಲು + ಲಂಬಸಾಲು;       // First half is gate
                    x_ptr = src0_ಅಡ್ಡಸಾಲು + nc + ಲಂಬಸಾಲು;  // Second half is value
                } ಇಲ್ಲದಿದ್ದರೆ {
                    x_ptr = src0_ಅಡ್ಡಸಾಲು + ಲಂಬಸಾಲು;       // First half is value
                    g_ptr = src0_ಅಡ್ಡಸಾಲು + nc + ಲಂಬಸಾಲು;  // Second half is gate
                }
            }

            // Process this segment
            ಆಯ್ಕೆ (params->glu_op_type) {
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
                    ಖಂಡ_geglu(dst_ptr, x_ptr, g_ptr, (ಪೂರ್ಣಾಂಕ) elements_ಗಮ್ಯ_process);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
                    ಖಂಡ_swiglu(dst_ptr, x_ptr, g_ptr, (ಪೂರ್ಣಾಂಕ) elements_ಗಮ್ಯ_process);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:
                    ಖಂಡ_reglu(dst_ptr, x_ptr, g_ptr, (ಪೂರ್ಣಾಂಕ) elements_ಗಮ್ಯ_process);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
                    ಖಂಡ_geglu_ಶೀಘ್ರ(dst_ptr, x_ptr, g_ptr, (ಪೂರ್ಣಾಂಕ) elements_ಗಮ್ಯ_process);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
                    ಖಂಡ_geglu_erf(dst_ptr, x_ptr, g_ptr, (ಪೂರ್ಣಾಂಕ) elements_ಗಮ್ಯ_process);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:
                    ಖಂಡ_swiglu_oai(dst_ptr, x_ptr, g_ptr, (ಪೂರ್ಣಾಂಕ) elements_ಗಮ್ಯ_process, params->alpha, params->ಎಲ್ಲೆ);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
                    ಖಂಡ_swiglu_clamp(dst_ptr, x_ptr, g_ptr, (ಪೂರ್ಣಾಂಕ) elements_ಗಮ್ಯ_process, params->ಎಲ್ಲೆ);
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ -1;
            }

            // Update counters
            elements_processed += elements_ಗಮ್ಯ_process;
            ಲಂಬಸಾಲು += elements_ಗಮ್ಯ_process;

            // Move to next row if current row is complete
            ಆದರೆ (ಲಂಬಸಾಲು >= nc) {
                ಅಡ್ಡಸಾಲು++;
                ಲಂಬಸಾಲು = 0;
            }
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
