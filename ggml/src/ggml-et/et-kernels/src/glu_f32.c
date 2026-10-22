//******************************************************************************
// GLU F32 Kernel (SwiGLU specifically)
// Gated Linear Unit: y[i] = silu(x[i]) * g[i] where silu(x) = x * sigmoid(x)
//******************************************************************************

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "math_fp.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <stdint.h>

// GLU kernel parameters structure (from ET backend ops)
સંરચના ggml_et_glu_params {
    સંરચના ggml_tensor src0;         // F32 input tensor A (or combined tensor if src1 is null)
    સંરચના ggml_tensor src1;         // F32 input tensor B (null for single tensor mode)
    સંરચના ggml_tensor dst;          // F32 output tensor (n/2 columns)
    int32_t            glu_op_type;  // GLU operation type (REGLU=0, GEGLU=1, SWIGLU=2, etc.)
    int32_t            swapped;      // Whether gate and value are swapped
    દશાંશ              alpha;        // SWIGLU_OAI: sigmoid scaling factor
    દશાંશ              limit;        // GLU clamp limit
};

// SiLU activation function: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
સ્થિર અંતઃરેખ દશાંશ silu_f32(દશાંશ x) {
    // For numerical stability, use the mathematically equivalent form:
    // silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    // For large negative x, exp(-x) -> inf, so silu(x) -> 0
    // For large positive x, exp(-x) -> 0, so silu(x) -> x

    જો (x > 20.0f) {
        // For x > 20, exp(-x) is negligible, silu(x) ~ x
        પાછા x;
    } નહીંતર જો (x < -20.0f) {
        // For x < -20, silu(x) ~ 0
        પાછા 0.0f;
    } નહીંતર {
        // Use standard formula: silu(x) = x / (1 + exp(-x))
        // Optimized using ET hardware division
        દશાંશ exp_neg_x   = et_expf(-x);
        દશાંશ denominator = 1.0f + exp_neg_x;
        પાછા et_fdiv(x, denominator);
    }
}

// Vectorized GeGLU block processing (8 elements = 1 cache line, 64B aligned)
// gelu(x) = 0.5*x*(1 + tanh(z)) = x * (1 - 1/(exp(2z)+1))
// where z = sqrt(2/pi) * x * (1 + 0.044715*x^2)
// Reformulated to avoid inf*0 NaN: uses x * sigmoid(2z) identity
સ્થિર અંતઃરેખ શૂન્ય block_geglu(દશાંશ * dst_block, અચલ દશાંશ * x_block, અચલ દશાંશ * g_block, પૂર્ણાંક elements) {
    unsigned long temp_mask;
    __asm__ અસ્થિર("mova.x.m %0" : "=r"(temp_mask));
    __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");

    દશાંશ one_const       = 1.0f;
    દશાંશ coef_a_const    = 0.044715f;
    દશાંશ sqrt2pi_const   = 0.79788456080286535587989211986876f;  // sqrt(2/pi)
    દશાંશ two_log2e_const = 2.8853900817779268f;                  // 2 * log2(e)

    માટે (int32_t i = 0; i < elements; i += 8) {
        __asm__ અસ્થિર(
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

            : [dst_out] "=m"(*(દશાંશ (*)[8]) & dst_block[i])
            : [x_vec] "m"(*(અચલ દશાંશ (*)[8]) & x_block[i]), [g_vec] "m"(*(અચલ દશાંશ (*)[8]) & g_block[i]),
              [one_ptr] "m"(one_const), [coef_ptr] "m"(coef_a_const), [sqrt2pi_ptr] "m"(sqrt2pi_const),
              [two_log2e_ptr] "m"(two_log2e_const)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f18", "f20", "f22", "f23", "f24");
    }

    __asm__ અસ્થિર("mova.m.x %0" ::"r"(temp_mask));
}

// Vectorized SwiGLU block processing (16 elements = 1 cache line)
સ્થિર અંતઃરેખ શૂન્ય block_swiglu(દશાંશ * dst_block, અચલ દશાંશ * x_block, અચલ દશાંશ * g_block, પૂર્ણાંક elements) {
    // Process 8 elements at a time using vector instructions
    int32_t vec_end = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ અસ્થિર("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Constants for broadcasting
    દશાંશ zero_const  = 0.0f;
    દશાંશ one_const   = 1.0f;
    દશાંશ log2e_const = 1.4426950408889634f;  // log2(e)

    માટે (int32_t i = 0; i < vec_end; i += 8) {
        // Vectorized SwiGLU: dst = silu(x) * g = (x / (1 + exp(-x))) * g
        // Using ET hardware: exp, reciprocal, multiply operations
        __asm__ અસ્થિર(
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

            : [dst_out] "=m"(*(દશાંશ (*)[8]) & dst_block[i])
            : [x_vec] "m"(*(અચલ દશાંશ (*)[8]) & x_block[i]), [g_vec] "m"(*(અચલ દશાંશ (*)[8]) & g_block[i]),
              [zero_ptr] "m"(zero_const),   // Memory reference to 0.0f for broadcasting
              [one_ptr] "m"(one_const),     // Memory reference to 1.0f for broadcasting
              [log2e_ptr] "m"(log2e_const)  // Memory reference to log2(e) for broadcasting
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22");
    }

    // Restore original mask
    __asm__ અસ્થિર("mova.m.x %0" ::"r"(temp_mask));

    // Handle remaining elements (< 8) with scalar operations
    માટે (int32_t i = vec_end; i < elements; i++) {
        dst_block[i] = silu_f32(x_block[i]) * g_block[i];
    }
}

// Vectorized ReGLU block: dst = max(0, x) * g
સ્થિર અંતઃરેખ શૂન્ય block_reglu(દશાંશ * dst_block, અચલ દશાંશ * x_block, અચલ દશાંશ * g_block, પૂર્ણાંક elements) {
    int32_t vec_end = (elements / 8) * 8;

    unsigned long temp_mask;
    __asm__ અસ્થિર("mova.x.m %0" : "=r"(temp_mask));
    __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");

    દશાંશ zero_const = 0.0f;

    માટે (int32_t i = 0; i < vec_end; i += 8) {
        __asm__ અસ્થિર(
            "flw.ps  f10, %[x_vec]\n"     // f10 = x
            "flw.ps  f11, %[g_vec]\n"     // f11 = g
            "fbc.ps  f20, %[zero_ptr]\n"  // f20 = 0.0
            "fmax.ps f12, f10, f20\n"     // f12 = max(x, 0)
            "fmul.ps f13, f12, f11\n"     // f13 = relu(x) * g
            "fsw.ps  f13, %[dst_out]\n"
            : [dst_out] "=m"(*(દશાંશ (*)[8]) & dst_block[i])
            : [x_vec] "m"(*(અચલ દશાંશ (*)[8]) & x_block[i]), [g_vec] "m"(*(અચલ દશાંશ (*)[8]) & g_block[i]),
              [zero_ptr] "m"(zero_const)
            : "f10", "f11", "f12", "f13", "f20");
    }

    __asm__ અસ્થિર("mova.m.x %0" ::"r"(temp_mask));

    માટે (int32_t i = vec_end; i < elements; i++) {
        દશાંશ xv     = x_block[i];
        dst_block[i] = (xv > 0.0f) ? xv * g_block[i] : 0.0f;
    }
}

// Vectorized GeGLU-Quick block: dst = x * sigmoid(1.702 * x) * g
// Using gelu_quick(x) = x / (1 + exp(-1.702*x))
સ્થિર અંતઃરેખ શૂન્ય block_geglu_quick(દશાંશ * dst_block, અચલ દશાંશ * x_block, અચલ દશાંશ * g_block, પૂર્ણાંક elements) {
    int32_t vec_end = (elements / 8) * 8;

    unsigned long temp_mask;
    __asm__ અસ્થિર("mova.x.m %0" : "=r"(temp_mask));
    __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");

    દશાંશ zero_const        = 0.0f;
    દશાંશ one_const         = 1.0f;
    // -1.702 * log2(e), so that fexp.ps(x * neg_k_log2e) = exp(-1.702*x)
    દશાંશ neg_k_log2e_const = -1.702f * 1.4426950408889634f;

    માટે (int32_t i = 0; i < vec_end; i += 8) {
        __asm__ અસ્થિર(
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
            : [dst_out] "=m"(*(દશાંશ (*)[8]) & dst_block[i])
            : [x_vec] "m"(*(અચલ દશાંશ (*)[8]) & x_block[i]), [g_vec] "m"(*(અચલ દશાંશ (*)[8]) & g_block[i]),
              [zero_ptr] "m"(zero_const), [one_ptr] "m"(one_const), [k_ptr] "m"(neg_k_log2e_const)
            : "f10", "f11", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22");
    }

    __asm__ અસ્થિર("mova.m.x %0" ::"r"(temp_mask));

    માટે (int32_t i = vec_end; i < elements; i++) {
        દશાંશ xv     = x_block[i];
        // Reuse silu reciprocal path: sigmoid(1.702*x) = 1/(1+exp(-1.702*x))
        દશાંશ e      = et_expf(-1.702f * xv);
        dst_block[i] = et_fdiv(xv, 1.0f + e) * g_block[i];
    }
}

// Vectorized SwiGLU-OAI block (OpenAI gpt-oss variant):
//   x_c = min(x, limit)
//   y_c = clamp(g, -limit, limit)
//   out = (x_c / (1 + exp(-alpha * x_c))) * (y_c + 1)
સ્થિર અંતઃરેખ શૂન્ય block_swiglu_oai(દશાંશ *       dst_block,
                                    અચલ દશાંશ * x_block,
                                    અચલ દશાંશ * g_block,
                                    પૂર્ણાંક           elements,
                                    દશાંશ         alpha,
                                    દશાંશ         limit) {
    int32_t vec_end = (elements / 8) * 8;

    unsigned long temp_mask;
    __asm__ અસ્થિર("mova.x.m %0" : "=r"(temp_mask));
    __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");

    દશાંશ zero_const    = 0.0f;
    દશાંશ one_const     = 1.0f;
    દશાંશ limit_pos     = limit;
    દશાંશ limit_neg     = -limit;
    // -alpha * log2(e): feed (x * neg_alpha_log2e) into fexp.ps to get exp(-alpha*x)
    દશાંશ neg_alpha_l2e = -alpha * 1.4426950408889634f;

    માટે (int32_t i = 0; i < vec_end; i += 8) {
        __asm__ અસ્થિર(
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

            : [dst_out] "=m"(*(દશાંશ (*)[8]) & dst_block[i])
            : [x_vec] "m"(*(અચલ દશાંશ (*)[8]) & x_block[i]), [g_vec] "m"(*(અચલ દશાંશ (*)[8]) & g_block[i]),
              [zero_ptr] "m"(zero_const), [one_ptr] "m"(one_const), [lim_pos] "m"(limit_pos), [lim_neg] "m"(limit_neg),
              [k_ptr] "m"(neg_alpha_l2e)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f23", "f24", "f25");
    }

    __asm__ અસ્થિર("mova.m.x %0" ::"r"(temp_mask));

    // Scalar tail (mirrors CPU reference exactly)
    માટે (int32_t i = vec_end; i < elements; i++) {
        દશાંશ xv = x_block[i];
        દશાંશ yv = g_block[i];
        જો (xv > limit) {
            xv = limit;
        }
        જો (yv > limit) {
            yv = limit;
        }
        જો (yv < -limit) {
            yv = -limit;
        }
        દશાંશ e       = et_expf(-alpha * xv);
        દશાંશ out_glu = et_fdiv(xv, 1.0f + e);
        dst_block[i]  = out_glu * (yv + 1.0f);
    }
}

સ્થિર અંતઃરેખ શૂન્ય block_swiglu_clamp(દશાંશ *       dst_block,
                                      અચલ દશાંશ * gate_block,
                                      અચલ દશાંશ * up_block,
                                      પૂર્ણાંક           elements,
                                      દશાંશ         limit) {
    int32_t vec_end = (elements / 8) * 8;

    unsigned long temp_mask;
    __asm__ અસ્થિર("mova.x.m %0" : "=r"(temp_mask));
    __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");

    દશાંશ one_const = 1.0f;
    દશાંશ limit_pos = limit;
    દશાંશ limit_neg = -limit;
    દશાંશ neg_log2e = -1.4426950408889634f;

    માટે (int32_t i = 0; i < vec_end; i += 8) {
        __asm__ અસ્થિર(
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
            : [dst_out] "=m"(*(દશાંશ (*)[8]) & dst_block[i])
            : [gate_vec] "m"(*(અચલ દશાંશ (*)[8]) & gate_block[i]), [up_vec] "m"(*(અચલ દશાંશ (*)[8]) & up_block[i]),
              [one_ptr] "m"(one_const), [lim_pos] "m"(limit_pos), [lim_neg] "m"(limit_neg), [k_ptr] "m"(neg_log2e)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f21", "f23", "f24", "f25");
    }

    __asm__ અસ્થિર("mova.m.x %0" :: "r"(temp_mask));

    માટે (int32_t i = vec_end; i < elements; i++) {
        દશાંશ gate   = gate_block[i] > limit ? limit : gate_block[i];
        દશાંશ up     = up_block[i];
        up           = up > limit ? limit : up;
        up           = up < -limit ? -limit : up;
        dst_block[i] = silu_f32(gate) * up;
    }
}

// Scalar erf approximation (Abramowitz & Stegun 7.1.26, max error ~1.5e-7)
સ્થિર અંતઃરેખ દશાંશ erf_approx(દશાંશ x) {
    અચલ દશાંશ a1 = 0.254829592f;
    અચલ દશાંશ a2 = -0.284496736f;
    અચલ દશાંશ a3 = 1.421413741f;
    અચલ દશાંશ a4 = -1.453152027f;
    અચલ દશાંશ a5 = 1.061405429f;
    અચલ દશાંશ p  = 0.3275911f;

    દશાંશ sign = (x < 0.0f) ? -1.0f : 1.0f;
    દશાંશ ax   = (x < 0.0f) ? -x : x;
    દશાંશ t    = et_fdiv(1.0f, 1.0f + p * ax);
    દશાંશ t2   = t * t;
    દશાંશ t3   = t2 * t;
    દશાંશ t4   = t3 * t;
    દશાંશ t5   = t4 * t;
    દશાંશ poly = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5;
    દશાંશ y    = 1.0f - poly * et_expf(-ax * ax);
    પાછા sign * y;
}

// GeGLU-Erf block: dst = 0.5 * x * (1 + erf(x / sqrt(2))) * g
// Scalar implementation — variant is rarely used so we keep complexity low.
સ્થિર અંતઃરેખ શૂન્ય block_geglu_erf(દશાંશ * dst_block, અચલ દશાંશ * x_block, અચલ દશાંશ * g_block, પૂર્ણાંક elements) {
    અચલ દશાંશ sqrt_2_inv = 0.70710678118654752440f;
    માટે (int32_t i = 0; i < elements; i++) {
        દશાંશ xv     = x_block[i];
        dst_block[i] = 0.5f * xv * (1.0f + erf_approx(xv * sqrt_2_inv)) * g_block[i];
    }
}

// Main entry point for GLU kernel
પૂર્ણાંક entry_point(સંરચના ggml_et_glu_params * params, શૂન્ય * env) {
    // Cast env to proper type
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    // Validate environment pointer
    જો (!kernel_env) {
        પાછા -1;
    }

    // Get thread info using shire mask from environment
    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    // Basic safety check on params
    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;  // Invalid pointer
    }

    // Supported variants: SwiGLU, SwiGLU-OAI, GeGLU, GeGLU-Erf, GeGLU-Quick, ReGLU
    પસંદગી (params->glu_op_type) {
        સ્થિતિ GGML_GLU_OP_SWIGLU:
        સ્થિતિ GGML_GLU_OP_SWIGLU_OAI:
        સ્થિતિ GGML_GLU_OP_SWIGLU_CLAMP:
        સ્થિતિ GGML_GLU_OP_GEGLU:
        સ્થિતિ GGML_GLU_OP_GEGLU_ERF:
        સ્થિતિ GGML_GLU_OP_GEGLU_QUICK:
        સ્થિતિ GGML_GLU_OP_REGLU:
            તોડો;
        મૂળભૂત:
            પાછા -1;  // Unsupported GLU operation
    }

    // Extract tensor references
    સંરચના ggml_tensor * src0    = &params->src0;
    સંરચના ggml_tensor * src1    = params->src1.data ? &params->src1 : 0;
    સંરચના ggml_tensor * dst     = &params->dst;
    int32_t              swapped = params->swapped;

    // Validate tensor types (F32 only)
    જો (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;  // Unsupported type combination
    }

    જો (src1 && src1->type != GGML_TYPE_F32) {
        પાછા -1;  // Unsupported src1 type
    }

    // Get data pointers
    દશાંશ * src0_data = (દશાંશ *) src0->data;
    દશાંશ * src1_data = src1 ? (દશાંશ *) src1->data : src0_data;
    દશાંશ * dst_data  = (દશાંશ *) dst->data;

    // Validate data pointers
    જો (!src0_data || !dst_data) {
        પાછા -1;  // Null data pointer
    }

    // Get tensor dimensions
    અચલ int64_t nc = dst->ne[0];                            // Output columns (input columns / 2)
    અચલ int64_t nr = dst->ne[1] * dst->ne[2] * dst->ne[3];  // Total rows

    // Get strides
    અચલ size_t src0_stride = src0->nb[1];                       // Stride between rows in src0
    અચલ size_t src1_stride = src1 ? src1->nb[1] : src0->nb[1];  // Stride between rows in src1
    અચલ size_t dst_stride  = dst->nb[1];                        // Stride between rows in dst

    // Validate dimensions for split SwiGLU
    જો (src1) {
        // Split tensor mode: src0 and src1 should have same shape as dst
        જો (src0->ne[0] != nc || src1->ne[0] != nc) {
            પાછા -1;  // Dimension mismatch in split mode
        }
    } નહીંતર {
        // Single tensor mode: src0 should have 2*nc columns
        જો (src0->ne[0] != 2 * nc) {
            પાછા -1;  // Dimension mismatch in single tensor mode
        }
    }

    // Calculate total elements for cache line distribution
    અચલ int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    અચલ int64_t total_elements         = nr * nc;
    અચલ int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    // Distribute cache lines across threads
    int64_t cachelines_per_thread = (total_cachelines + અંક_threads - 1) / અંક_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    // Clamp end_cacheline to actual number of cache lines
    જો (end_cacheline > total_cachelines) {
        end_cacheline = total_cachelines;
    }

    // Thread should return if no work to do
    જો (start_cacheline >= total_cachelines) {
        પાછા 0;
    }

    // Process cache lines assigned to this thread
    માટે (int64_t cl = start_cacheline; cl < end_cacheline; cl++) {
        // Map cache line back to element coordinates
        int64_t global_element_start = cl * elements_per_cacheline;
        int64_t row                  = global_element_start / nc;
        int64_t col                  = global_element_start % nc;

        // Skip if we're past the end of data
        જો (global_element_start >= total_elements) {
            તોડો;
        }

        // Calculate how many elements to process in this cache line
        int64_t elements_remaining = total_elements - global_element_start;
        પૂર્ણાંક     elements_this_block =
            (પૂર્ણાંક) ((elements_remaining < elements_per_cacheline) ? elements_remaining : elements_per_cacheline);

        // Process elements that span across rows
        int64_t elements_processed = 0;
        જ્યારે (elements_processed < elements_this_block && row < nr) {
            // Calculate elements to process in current row
            int64_t elements_in_row     = nc - col;
            int64_t elements_to_process = elements_this_block - elements_processed;
            જો (elements_to_process > elements_in_row) {
                elements_to_process = elements_in_row;
            }

            // Get pointers for current row and column range
            દશાંશ * dst_ptr = (દશાંશ *) ((અક્ષર *) dst_data + row * dst_stride) + col;

            દશાંશ * x_ptr;
            દશાંશ * g_ptr;

            જો (src1) {
                // Split tensor mode
                x_ptr = (દશાંશ *) ((અક્ષર *) src0_data + row * src0_stride) + col;
                g_ptr = (દશાંશ *) ((અક્ષર *) src1_data + row * src1_stride) + col;
            } નહીંતર {
                // Single tensor mode - src0 contains both x and g
                દશાંશ * src0_row = (દશાંશ *) ((અક્ષર *) src0_data + row * src0_stride);
                જો (swapped) {
                    g_ptr = src0_row + col;       // First half is gate
                    x_ptr = src0_row + nc + col;  // Second half is value
                } નહીંતર {
                    x_ptr = src0_row + col;       // First half is value
                    g_ptr = src0_row + nc + col;  // Second half is gate
                }
            }

            // Process this segment
            પસંદગી (params->glu_op_type) {
                સ્થિતિ GGML_GLU_OP_GEGLU:
                    block_geglu(dst_ptr, x_ptr, g_ptr, (પૂર્ણાંક) elements_to_process);
                    તોડો;
                સ્થિતિ GGML_GLU_OP_SWIGLU:
                    block_swiglu(dst_ptr, x_ptr, g_ptr, (પૂર્ણાંક) elements_to_process);
                    તોડો;
                સ્થિતિ GGML_GLU_OP_REGLU:
                    block_reglu(dst_ptr, x_ptr, g_ptr, (પૂર્ણાંક) elements_to_process);
                    તોડો;
                સ્થિતિ GGML_GLU_OP_GEGLU_QUICK:
                    block_geglu_quick(dst_ptr, x_ptr, g_ptr, (પૂર્ણાંક) elements_to_process);
                    તોડો;
                સ્થિતિ GGML_GLU_OP_GEGLU_ERF:
                    block_geglu_erf(dst_ptr, x_ptr, g_ptr, (પૂર્ણાંક) elements_to_process);
                    તોડો;
                સ્થિતિ GGML_GLU_OP_SWIGLU_OAI:
                    block_swiglu_oai(dst_ptr, x_ptr, g_ptr, (પૂર્ણાંક) elements_to_process, params->alpha, params->limit);
                    તોડો;
                સ્થિતિ GGML_GLU_OP_SWIGLU_CLAMP:
                    block_swiglu_clamp(dst_ptr, x_ptr, g_ptr, (પૂર્ણાંક) elements_to_process, params->limit);
                    તોડો;
                મૂળભૂત:
                    પાછા -1;
            }

            // Update counters
            elements_processed += elements_to_process;
            col += elements_to_process;

            // Move to next row if current row is complete
            જો (col >= nc) {
                row++;
                col = 0;
            }
        }
    }

    પાછા 0;
}
