//******************************************************************************
// GLU F32 Kernel (SwiGLU specifically)
// Gated Linear Unit: y[i] = silu(x[i]) * g[i] where silu(x) = x * sigmoid(x)
//******************************************************************************

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "math_fp.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <stdint.h>

// GLU kernel parameters structure (from ET backend ops)
ସଂରଚନା ggml_et_glu_params {
    ସଂରଚନା ggml_tensor src0;         // F32 input tensor A (or combined tensor if src1 is null)
    ସଂରଚନା ggml_tensor src1;         // F32 input tensor B (null for single tensor mode)
    ସଂରଚନା ggml_tensor dst;          // F32 output tensor (n/2 columns)
    int32_t            glu_op_type;  // GLU operation type (REGLU=0, GEGLU=1, SWIGLU=2, etc.)
    int32_t            swapped;      // Whether gate and value are swapped
    ଦଶମିକ              alpha;        // SWIGLU_OAI: sigmoid scaling factor
    ଦଶମିକ              limit;        // GLU clamp limit
};

// SiLU activation function: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ silu_f32(ଦଶମିକ x) {
    // For numerical stability, use the mathematically equivalent form:
    // silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    // For large negative x, exp(-x) -> inf, so silu(x) -> 0
    // For large positive x, exp(-x) -> 0, so silu(x) -> x

    ଯଦି (x > 20.0f) {
        // For x > 20, exp(-x) is negligible, silu(x) ~ x
        ଫେରନ୍ତୁ x;
    } ଅନ୍ୟଥା ଯଦି (x < -20.0f) {
        // For x < -20, silu(x) ~ 0
        ଫେରନ୍ତୁ 0.0f;
    } ଅନ୍ୟଥା {
        // Use standard formula: silu(x) = x / (1 + exp(-x))
        // Optimized using ET hardware division
        ଦଶମିକ exp_neg_x   = et_expf(-x);
        ଦଶମିକ denominator = 1.0f + exp_neg_x;
        ଫେରନ୍ତୁ et_fdiv(x, denominator);
    }
}

// Vectorized GeGLU block processing (8 elements = 1 cache line, 64B aligned)
// gelu(x) = 0.5*x*(1 + tanh(z)) = x * (1 - 1/(exp(2z)+1))
// where z = sqrt(2/pi) * x * (1 + 0.044715*x^2)
// Reformulated to avoid inf*0 NaN: uses x * sigmoid(2z) identity
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ block_geglu(ଦଶମିକ * dst_block, ଅଚଳ ଦଶମିକ * x_block, ଅଚଳ ଦଶମିକ * g_block, ପୂର୍ଣ୍ଣସଂଖ୍ୟା elements) {
    unsigned long temp_mask;
    __asm__ ଅସ୍ଥିର("mova.x.m %0" : "=r"(temp_mask));
    __asm__ ଅସ୍ଥିର("mov.m.x m0, x0, 0xFF");

    ଦଶମିକ one_const       = 1.0f;
    ଦଶମିକ coef_a_const    = 0.044715f;
    ଦଶମିକ sqrt2pi_const   = 0.79788456080286535587989211986876f;  // sqrt(2/pi)
    ଦଶମିକ two_log2e_const = 2.8853900817779268f;                  // 2 * log2(e)

    ପାଇଁ (int32_t i = 0; i < elements; i += 8) {
        __asm__ ଅସ୍ଥିର(
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

            : [dst_out] "=m"(*(ଦଶମିକ (*)[8]) & dst_block[i])
            : [x_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & x_block[i]), [g_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & g_block[i]),
              [one_ptr] "m"(one_const), [coef_ptr] "m"(coef_a_const), [sqrt2pi_ptr] "m"(sqrt2pi_const),
              [two_log2e_ptr] "m"(two_log2e_const)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f18", "f20", "f22", "f23", "f24");
    }

    __asm__ ଅସ୍ଥିର("mova.m.x %0" ::"r"(temp_mask));
}

// Vectorized SwiGLU block processing (16 elements = 1 cache line)
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ block_swiglu(ଦଶମିକ * dst_block, ଅଚଳ ଦଶମିକ * x_block, ଅଚଳ ଦଶମିକ * g_block, ପୂର୍ଣ୍ଣସଂଖ୍ୟା elements) {
    // Process 8 elements at a time using vector instructions
    int32_t vec_end = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ ଅସ୍ଥିର("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ ଅସ୍ଥିର("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Constants for broadcasting
    ଦଶମିକ zero_const  = 0.0f;
    ଦଶମିକ one_const   = 1.0f;
    ଦଶମିକ log2e_const = 1.4426950408889634f;  // log2(e)

    ପାଇଁ (int32_t i = 0; i < vec_end; i += 8) {
        // Vectorized SwiGLU: dst = silu(x) * g = (x / (1 + exp(-x))) * g
        // Using ET hardware: exp, reciprocal, multiply operations
        __asm__ ଅସ୍ଥିର(
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

            : [dst_out] "=m"(*(ଦଶମିକ (*)[8]) & dst_block[i])
            : [x_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & x_block[i]), [g_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & g_block[i]),
              [zero_ptr] "m"(zero_const),   // Memory reference to 0.0f for broadcasting
              [one_ptr] "m"(one_const),     // Memory reference to 1.0f for broadcasting
              [log2e_ptr] "m"(log2e_const)  // Memory reference to log2(e) for broadcasting
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22");
    }

    // Restore original mask
    __asm__ ଅସ୍ଥିର("mova.m.x %0" ::"r"(temp_mask));

    // Handle remaining elements (< 8) with scalar operations
    ପାଇଁ (int32_t i = vec_end; i < elements; i++) {
        dst_block[i] = silu_f32(x_block[i]) * g_block[i];
    }
}

// Vectorized ReGLU block: dst = max(0, x) * g
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ block_reglu(ଦଶମିକ * dst_block, ଅଚଳ ଦଶମିକ * x_block, ଅଚଳ ଦଶମିକ * g_block, ପୂର୍ଣ୍ଣସଂଖ୍ୟା elements) {
    int32_t vec_end = (elements / 8) * 8;

    unsigned long temp_mask;
    __asm__ ଅସ୍ଥିର("mova.x.m %0" : "=r"(temp_mask));
    __asm__ ଅସ୍ଥିର("mov.m.x m0, x0, 0xFF");

    ଦଶମିକ zero_const = 0.0f;

    ପାଇଁ (int32_t i = 0; i < vec_end; i += 8) {
        __asm__ ଅସ୍ଥିର(
            "flw.ps  f10, %[x_vec]\n"     // f10 = x
            "flw.ps  f11, %[g_vec]\n"     // f11 = g
            "fbc.ps  f20, %[zero_ptr]\n"  // f20 = 0.0
            "fmax.ps f12, f10, f20\n"     // f12 = max(x, 0)
            "fmul.ps f13, f12, f11\n"     // f13 = relu(x) * g
            "fsw.ps  f13, %[dst_out]\n"
            : [dst_out] "=m"(*(ଦଶମିକ (*)[8]) & dst_block[i])
            : [x_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & x_block[i]), [g_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & g_block[i]),
              [zero_ptr] "m"(zero_const)
            : "f10", "f11", "f12", "f13", "f20");
    }

    __asm__ ଅସ୍ଥିର("mova.m.x %0" ::"r"(temp_mask));

    ପାଇଁ (int32_t i = vec_end; i < elements; i++) {
        ଦଶମିକ xv     = x_block[i];
        dst_block[i] = (xv > 0.0f) ? xv * g_block[i] : 0.0f;
    }
}

// Vectorized GeGLU-Quick block: dst = x * sigmoid(1.702 * x) * g
// Using gelu_quick(x) = x / (1 + exp(-1.702*x))
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ block_geglu_quick(ଦଶମିକ * dst_block, ଅଚଳ ଦଶମିକ * x_block, ଅଚଳ ଦଶମିକ * g_block, ପୂର୍ଣ୍ଣସଂଖ୍ୟା elements) {
    int32_t vec_end = (elements / 8) * 8;

    unsigned long temp_mask;
    __asm__ ଅସ୍ଥିର("mova.x.m %0" : "=r"(temp_mask));
    __asm__ ଅସ୍ଥିର("mov.m.x m0, x0, 0xFF");

    ଦଶମିକ zero_const        = 0.0f;
    ଦଶମିକ one_const         = 1.0f;
    // -1.702 * log2(e), so that fexp.ps(x * neg_k_log2e) = exp(-1.702*x)
    ଦଶମିକ neg_k_log2e_const = -1.702f * 1.4426950408889634f;

    ପାଇଁ (int32_t i = 0; i < vec_end; i += 8) {
        __asm__ ଅସ୍ଥିର(
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
            : [dst_out] "=m"(*(ଦଶମିକ (*)[8]) & dst_block[i])
            : [x_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & x_block[i]), [g_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & g_block[i]),
              [zero_ptr] "m"(zero_const), [one_ptr] "m"(one_const), [k_ptr] "m"(neg_k_log2e_const)
            : "f10", "f11", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22");
    }

    __asm__ ଅସ୍ଥିର("mova.m.x %0" ::"r"(temp_mask));

    ପାଇଁ (int32_t i = vec_end; i < elements; i++) {
        ଦଶମିକ xv     = x_block[i];
        // Reuse silu reciprocal path: sigmoid(1.702*x) = 1/(1+exp(-1.702*x))
        ଦଶମିକ e      = et_expf(-1.702f * xv);
        dst_block[i] = et_fdiv(xv, 1.0f + e) * g_block[i];
    }
}

// Vectorized SwiGLU-OAI block (OpenAI gpt-oss variant):
//   x_c = min(x, limit)
//   y_c = clamp(g, -limit, limit)
//   out = (x_c / (1 + exp(-alpha * x_c))) * (y_c + 1)
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ block_swiglu_oai(ଦଶମିକ *       dst_block,
                                    ଅଚଳ ଦଶମିକ * x_block,
                                    ଅଚଳ ଦଶମିକ * g_block,
                                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           elements,
                                    ଦଶମିକ         alpha,
                                    ଦଶମିକ         limit) {
    int32_t vec_end = (elements / 8) * 8;

    unsigned long temp_mask;
    __asm__ ଅସ୍ଥିର("mova.x.m %0" : "=r"(temp_mask));
    __asm__ ଅସ୍ଥିର("mov.m.x m0, x0, 0xFF");

    ଦଶମିକ zero_const    = 0.0f;
    ଦଶମିକ one_const     = 1.0f;
    ଦଶମିକ limit_pos     = limit;
    ଦଶମିକ limit_neg     = -limit;
    // -alpha * log2(e): feed (x * neg_alpha_log2e) into fexp.ps to get exp(-alpha*x)
    ଦଶମିକ neg_alpha_l2e = -alpha * 1.4426950408889634f;

    ପାଇଁ (int32_t i = 0; i < vec_end; i += 8) {
        __asm__ ଅସ୍ଥିର(
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

            : [dst_out] "=m"(*(ଦଶମିକ (*)[8]) & dst_block[i])
            : [x_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & x_block[i]), [g_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & g_block[i]),
              [zero_ptr] "m"(zero_const), [one_ptr] "m"(one_const), [lim_pos] "m"(limit_pos), [lim_neg] "m"(limit_neg),
              [k_ptr] "m"(neg_alpha_l2e)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f23", "f24", "f25");
    }

    __asm__ ଅସ୍ଥିର("mova.m.x %0" ::"r"(temp_mask));

    // Scalar tail (mirrors CPU reference exactly)
    ପାଇଁ (int32_t i = vec_end; i < elements; i++) {
        ଦଶମିକ xv = x_block[i];
        ଦଶମିକ yv = g_block[i];
        ଯଦି (xv > limit) {
            xv = limit;
        }
        ଯଦି (yv > limit) {
            yv = limit;
        }
        ଯଦି (yv < -limit) {
            yv = -limit;
        }
        ଦଶମିକ e       = et_expf(-alpha * xv);
        ଦଶମିକ out_glu = et_fdiv(xv, 1.0f + e);
        dst_block[i]  = out_glu * (yv + 1.0f);
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ block_swiglu_clamp(ଦଶମିକ *       dst_block,
                                      ଅଚଳ ଦଶମିକ * gate_block,
                                      ଅଚଳ ଦଶମିକ * up_block,
                                      ପୂର୍ଣ୍ଣସଂଖ୍ୟା           elements,
                                      ଦଶମିକ         limit) {
    int32_t vec_end = (elements / 8) * 8;

    unsigned long temp_mask;
    __asm__ ଅସ୍ଥିର("mova.x.m %0" : "=r"(temp_mask));
    __asm__ ଅସ୍ଥିର("mov.m.x m0, x0, 0xFF");

    ଦଶମିକ one_const = 1.0f;
    ଦଶମିକ limit_pos = limit;
    ଦଶମିକ limit_neg = -limit;
    ଦଶମିକ neg_log2e = -1.4426950408889634f;

    ପାଇଁ (int32_t i = 0; i < vec_end; i += 8) {
        __asm__ ଅସ୍ଥିର(
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
            : [dst_out] "=m"(*(ଦଶମିକ (*)[8]) & dst_block[i])
            : [gate_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & gate_block[i]), [up_vec] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & up_block[i]),
              [one_ptr] "m"(one_const), [lim_pos] "m"(limit_pos), [lim_neg] "m"(limit_neg), [k_ptr] "m"(neg_log2e)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f21", "f23", "f24", "f25");
    }

    __asm__ ଅସ୍ଥିର("mova.m.x %0" :: "r"(temp_mask));

    ପାଇଁ (int32_t i = vec_end; i < elements; i++) {
        ଦଶମିକ gate   = gate_block[i] > limit ? limit : gate_block[i];
        ଦଶମିକ up     = up_block[i];
        up           = up > limit ? limit : up;
        up           = up < -limit ? -limit : up;
        dst_block[i] = silu_f32(gate) * up;
    }
}

// Scalar erf approximation (Abramowitz & Stegun 7.1.26, max error ~1.5e-7)
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ erf_approx(ଦଶମିକ x) {
    ଅଚଳ ଦଶମିକ a1 = 0.254829592f;
    ଅଚଳ ଦଶମିକ a2 = -0.284496736f;
    ଅଚଳ ଦଶମିକ a3 = 1.421413741f;
    ଅଚଳ ଦଶମିକ a4 = -1.453152027f;
    ଅଚଳ ଦଶମିକ a5 = 1.061405429f;
    ଅଚଳ ଦଶମିକ p  = 0.3275911f;

    ଦଶମିକ sign = (x < 0.0f) ? -1.0f : 1.0f;
    ଦଶମିକ ax   = (x < 0.0f) ? -x : x;
    ଦଶମିକ t    = et_fdiv(1.0f, 1.0f + p * ax);
    ଦଶମିକ t2   = t * t;
    ଦଶମିକ t3   = t2 * t;
    ଦଶମିକ t4   = t3 * t;
    ଦଶମିକ t5   = t4 * t;
    ଦଶମିକ poly = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5;
    ଦଶମିକ y    = 1.0f - poly * et_expf(-ax * ax);
    ଫେରନ୍ତୁ sign * y;
}

// GeGLU-Erf block: dst = 0.5 * x * (1 + erf(x / sqrt(2))) * g
// Scalar implementation — variant is rarely used so we keep complexity low.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ block_geglu_erf(ଦଶମିକ * dst_block, ଅଚଳ ଦଶମିକ * x_block, ଅଚଳ ଦଶମିକ * g_block, ପୂର୍ଣ୍ଣସଂଖ୍ୟା elements) {
    ଅଚଳ ଦଶମିକ sqrt_2_inv = 0.70710678118654752440f;
    ପାଇଁ (int32_t i = 0; i < elements; i++) {
        ଦଶମିକ xv     = x_block[i];
        dst_block[i] = 0.5f * xv * (1.0f + erf_approx(xv * sqrt_2_inv)) * g_block[i];
    }
}

// Main entry point for GLU kernel
ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_glu_params * params, ଶୂନ୍ୟ * env) {
    // Cast env to proper type
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    // Validate environment pointer
    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    // Get thread info using shire mask from environment
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    // Basic safety check on params
    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;  // Invalid pointer
    }

    // Supported variants: SwiGLU, SwiGLU-OAI, GeGLU, GeGLU-Erf, GeGLU-Quick, ReGLU
    ବାଛ (params->glu_op_type) {
        ସ୍ଥିତି GGML_GLU_OP_SWIGLU:
        ସ୍ଥିତି GGML_GLU_OP_SWIGLU_OAI:
        ସ୍ଥିତି GGML_GLU_OP_SWIGLU_CLAMP:
        ସ୍ଥିତି GGML_GLU_OP_GEGLU:
        ସ୍ଥିତି GGML_GLU_OP_GEGLU_ERF:
        ସ୍ଥିତି GGML_GLU_OP_GEGLU_QUICK:
        ସ୍ଥିତି GGML_GLU_OP_REGLU:
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ -1;  // Unsupported GLU operation
    }

    // Extract tensor references
    ସଂରଚନା ggml_tensor * src0    = &params->src0;
    ସଂରଚନା ggml_tensor * src1    = params->src1.data ? &params->src1 : 0;
    ସଂରଚନା ggml_tensor * dst     = &params->dst;
    int32_t              swapped = params->swapped;

    // Validate tensor types (F32 only)
    ଯଦି (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;  // Unsupported type combination
    }

    ଯଦି (src1 && src1->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;  // Unsupported src1 type
    }

    // Get data pointers
    ଦଶମିକ * src0_data = (ଦଶମିକ *) src0->data;
    ଦଶମିକ * src1_data = src1 ? (ଦଶମିକ *) src1->data : src0_data;
    ଦଶମିକ * dst_data  = (ଦଶମିକ *) dst->data;

    // Validate data pointers
    ଯଦି (!src0_data || !dst_data) {
        ଫେରନ୍ତୁ -1;  // Null data pointer
    }

    // Get tensor dimensions
    ଅଚଳ int64_t nc = dst->ne[0];                            // Output columns (input columns / 2)
    ଅଚଳ int64_t nr = dst->ne[1] * dst->ne[2] * dst->ne[3];  // Total rows

    // Get strides
    ଅଚଳ size_t src0_stride = src0->nb[1];                       // Stride between rows in src0
    ଅଚଳ size_t src1_stride = src1 ? src1->nb[1] : src0->nb[1];  // Stride between rows in src1
    ଅଚଳ size_t dst_stride  = dst->nb[1];                        // Stride between rows in dst

    // Validate dimensions for split SwiGLU
    ଯଦି (src1) {
        // Split tensor mode: src0 and src1 should have same shape as dst
        ଯଦି (src0->ne[0] != nc || src1->ne[0] != nc) {
            ଫେରନ୍ତୁ -1;  // Dimension mismatch in split mode
        }
    } ଅନ୍ୟଥା {
        // Single tensor mode: src0 should have 2*nc columns
        ଯଦି (src0->ne[0] != 2 * nc) {
            ଫେରନ୍ତୁ -1;  // Dimension mismatch in single tensor mode
        }
    }

    // Calculate total elements for cache line distribution
    ଅଚଳ int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    ଅଚଳ int64_t total_elements         = nr * nc;
    ଅଚଳ int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    // Distribute cache lines across threads
    int64_t cachelines_per_thread = (total_cachelines + ଅଙ୍କ_threads - 1) / ଅଙ୍କ_threads;
    int64_t start_cacheline       = thread_id * cachelines_per_thread;
    int64_t end_cacheline         = start_cacheline + cachelines_per_thread;

    // Clamp end_cacheline to actual number of cache lines
    ଯଦି (end_cacheline > total_cachelines) {
        end_cacheline = total_cachelines;
    }

    // Thread should return if no work to do
    ଯଦି (start_cacheline >= total_cachelines) {
        ଫେରନ୍ତୁ 0;
    }

    // Process cache lines assigned to this thread
    ପାଇଁ (int64_t cl = start_cacheline; cl < end_cacheline; cl++) {
        // Map cache line back to element coordinates
        int64_t global_element_start = cl * elements_per_cacheline;
        int64_t row                  = global_element_start / nc;
        int64_t col                  = global_element_start % nc;

        // Skip if we're past the end of data
        ଯଦି (global_element_start >= total_elements) {
            ଭାଙ୍ଗ;
        }

        // Calculate how many elements to process in this cache line
        int64_t elements_remaining = total_elements - global_element_start;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା     elements_this_block =
            (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ((elements_remaining < elements_per_cacheline) ? elements_remaining : elements_per_cacheline);

        // Process elements that span across rows
        int64_t elements_processed = 0;
        ଯେତେବେଳେ (elements_processed < elements_this_block && row < nr) {
            // Calculate elements to process in current row
            int64_t elements_in_row     = nc - col;
            int64_t elements_to_process = elements_this_block - elements_processed;
            ଯଦି (elements_to_process > elements_in_row) {
                elements_to_process = elements_in_row;
            }

            // Get pointers for current row and column range
            ଦଶମିକ * dst_ptr = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + row * dst_stride) + col;

            ଦଶମିକ * x_ptr;
            ଦଶମିକ * g_ptr;

            ଯଦି (src1) {
                // Split tensor mode
                x_ptr = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0_data + row * src0_stride) + col;
                g_ptr = (ଦଶମିକ *) ((ଅକ୍ଷର *) src1_data + row * src1_stride) + col;
            } ଅନ୍ୟଥା {
                // Single tensor mode - src0 contains both x and g
                ଦଶମିକ * src0_row = (ଦଶମିକ *) ((ଅକ୍ଷର *) src0_data + row * src0_stride);
                ଯଦି (swapped) {
                    g_ptr = src0_row + col;       // First half is gate
                    x_ptr = src0_row + nc + col;  // Second half is value
                } ଅନ୍ୟଥା {
                    x_ptr = src0_row + col;       // First half is value
                    g_ptr = src0_row + nc + col;  // Second half is gate
                }
            }

            // Process this segment
            ବାଛ (params->glu_op_type) {
                ସ୍ଥିତି GGML_GLU_OP_GEGLU:
                    block_geglu(dst_ptr, x_ptr, g_ptr, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) elements_to_process);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU:
                    block_swiglu(dst_ptr, x_ptr, g_ptr, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) elements_to_process);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_GLU_OP_REGLU:
                    block_reglu(dst_ptr, x_ptr, g_ptr, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) elements_to_process);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_GLU_OP_GEGLU_QUICK:
                    block_geglu_quick(dst_ptr, x_ptr, g_ptr, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) elements_to_process);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_GLU_OP_GEGLU_ERF:
                    block_geglu_erf(dst_ptr, x_ptr, g_ptr, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) elements_to_process);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU_OAI:
                    block_swiglu_oai(dst_ptr, x_ptr, g_ptr, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) elements_to_process, params->alpha, params->limit);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU_CLAMP:
                    block_swiglu_clamp(dst_ptr, x_ptr, g_ptr, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) elements_to_process, params->limit);
                    ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    ଫେରନ୍ତୁ -1;
            }

            // Update counters
            elements_processed += elements_to_process;
            col += elements_to_process;

            // Move to next row if current row is complete
            ଯଦି (col >= nc) {
                row++;
                col = 0;
            }
        }
    }

    ଫେରନ୍ତୁ 0;
}
