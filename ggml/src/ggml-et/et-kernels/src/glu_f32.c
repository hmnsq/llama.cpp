//******************************************************************************
// GLU F32 Kernel (SwiGLU specifically)
// Gated Linear Unit: y[i] = silu(x[i]) * g[i] where silu(x) = x * sigmoid(x)
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <stdint.h>

// GLU kernel parameters structure (from ET backend ops)
నిర్మాణం ggml_et_glu_params {
    నిర్మాణం ggml_tensor src0;         // F32 input tensor A (or combined tensor if src1 is null)
    నిర్మాణం ggml_tensor src1;         // F32 input tensor B (null for single tensor mode)
    నిర్మాణం ggml_tensor dst;          // F32 output tensor (n/2 columns)
    int32_t            glu_op_type;  // GLU operation type (REGLU=0, GEGLU=1, SWIGLU=2, etc.)
    int32_t            swapped;      // Whether gate and value are swapped
    దశాంశం              alpha;        // SWIGLU_OAI: sigmoid scaling factor
    దశాంశం              హద్దు;        // GLU clamp limit
};

// SiLU activation function: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
స్థిర అంతఃవరుస దశాంశం silu_f32(దశాంశం x) {
    // For numerical stability, use the mathematically equivalent form:
    // silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    // For large negative x, exp(-x) -> inf, so silu(x) -> 0
    // For large positive x, exp(-x) -> 0, so silu(x) -> x

    ఒకవేళ (x > 20.0f) {
        // For x > 20, exp(-x) is negligible, silu(x) ~ x
        తిరిగి x;
    } లేకపోతే ఒకవేళ (x < -20.0f) {
        // For x < -20, silu(x) ~ 0
        తిరిగి 0.0f;
    } లేకపోతే {
        // Use standard formula: silu(x) = x / (1 + exp(-x))
        // Optimized using ET hardware division
        దశాంశం exp_neg_x   = et_expf(-x);
        దశాంశం denominator = 1.0f + exp_neg_x;
        తిరిగి et_fdiv(x, denominator);
    }
}

// Vectorized GeGLU block processing (8 elements = 1 cache line, 64B aligned)
// gelu(x) = 0.5*x*(1 + tanh(z)) = x * (1 - 1/(exp(2z)+1))
// where z = sqrt(2/pi) * x * (1 + 0.044715*x^2)
// Reformulated to avoid inf*0 NaN: uses x * sigmoid(2z) identity
స్థిర అంతఃవరుస శూన్య ఖండం_geglu(దశాంశం * dst_ఖండం, స్థిరరాశి దశాంశం * x_ఖండం, స్థిరరాశి దశాంశం * g_ఖండం, పూర్ణాంకం elements) {
    గుర్తులేని దీర్ఘ తాత్కాలిక_ముసుగు;
    __asm__ అస్థిర("mova.x.m %0" : "=r"(తాత్కాలిక_ముసుగు));
    __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");

    దశాంశం one_const       = 1.0f;
    దశాంశం coef_a_const    = 0.044715f;
    దశాంశం sqrt2pi_const   = 0.79788456080286535587989211986876f;  // sqrt(2/pi)
    దశాంశం two_log2e_const = 2.8853900817779268f;                  // 2 * log2(e)

    కొరకు (int32_t i = 0; i < elements; i += 8) {
        __asm__ అస్థిర(
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

            : [dst_బహిర్గతం] "=m"(*(దశాంశం (*)[8]) & dst_ఖండం[i])
            : [x_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & x_ఖండం[i]), [g_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & g_ఖండం[i]),
              [one_ptr] "m"(one_const), [coef_ptr] "m"(coef_a_const), [sqrt2pi_ptr] "m"(sqrt2pi_const),
              [two_log2e_ptr] "m"(two_log2e_const)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f18", "f20", "f22", "f23", "f24");
    }

    __asm__ అస్థిర("mova.m.x %0" ::"r"(తాత్కాలిక_ముసుగు));
}

// Vectorized SwiGLU block processing (16 elements = 1 cache line)
స్థిర అంతఃవరుస శూన్య ఖండం_swiglu(దశాంశం * dst_ఖండం, స్థిరరాశి దశాంశం * x_ఖండం, స్థిరరాశి దశాంశం * g_ఖండం, పూర్ణాంకం elements) {
    // Process 8 elements at a time using vector instructions
    int32_t vec_చివర = (elements / 8) * 8;

    // Set mask register to enable all 8 vector elements
    గుర్తులేని దీర్ఘ తాత్కాలిక_ముసుగు;
    __asm__ అస్థిర("mova.x.m %0" : "=r"(తాత్కాలిక_ముసుగు));  // Save current mask
    __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Constants for broadcasting
    దశాంశం zero_const  = 0.0f;
    దశాంశం one_const   = 1.0f;
    దశాంశం log2e_const = 1.4426950408889634f;  // log2(e)

    కొరకు (int32_t i = 0; i < vec_చివర; i += 8) {
        // Vectorized SwiGLU: dst = silu(x) * g = (x / (1 + exp(-x))) * g
        // Using ET hardware: exp, reciprocal, multiply operations
        __asm__ అస్థిర(
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

            : [dst_బహిర్గతం] "=m"(*(దశాంశం (*)[8]) & dst_ఖండం[i])
            : [x_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & x_ఖండం[i]), [g_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & g_ఖండం[i]),
              [zero_ptr] "m"(zero_const),   // Memory reference to 0.0f for broadcasting
              [one_ptr] "m"(one_const),     // Memory reference to 1.0f for broadcasting
              [log2e_ptr] "m"(log2e_const)  // Memory reference to log2(e) for broadcasting
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22");
    }

    // Restore original mask
    __asm__ అస్థిర("mova.m.x %0" ::"r"(తాత్కాలిక_ముసుగు));

    // Handle remaining elements (< 8) with scalar operations
    కొరకు (int32_t i = vec_చివర; i < elements; i++) {
        dst_ఖండం[i] = silu_f32(x_ఖండం[i]) * g_ఖండం[i];
    }
}

// Vectorized ReGLU block: dst = max(0, x) * g
స్థిర అంతఃవరుస శూన్య ఖండం_reglu(దశాంశం * dst_ఖండం, స్థిరరాశి దశాంశం * x_ఖండం, స్థిరరాశి దశాంశం * g_ఖండం, పూర్ణాంకం elements) {
    int32_t vec_చివర = (elements / 8) * 8;

    గుర్తులేని దీర్ఘ తాత్కాలిక_ముసుగు;
    __asm__ అస్థిర("mova.x.m %0" : "=r"(తాత్కాలిక_ముసుగు));
    __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");

    దశాంశం zero_const = 0.0f;

    కొరకు (int32_t i = 0; i < vec_చివర; i += 8) {
        __asm__ అస్థిర(
            "flw.ps  f10, %[x_vec]\n"     // f10 = x
            "flw.ps  f11, %[g_vec]\n"     // f11 = g
            "fbc.ps  f20, %[zero_ptr]\n"  // f20 = 0.0
            "fmax.ps f12, f10, f20\n"     // f12 = max(x, 0)
            "fmul.ps f13, f12, f11\n"     // f13 = relu(x) * g
            "fsw.ps  f13, %[dst_out]\n"
            : [dst_బహిర్గతం] "=m"(*(దశాంశం (*)[8]) & dst_ఖండం[i])
            : [x_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & x_ఖండం[i]), [g_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & g_ఖండం[i]),
              [zero_ptr] "m"(zero_const)
            : "f10", "f11", "f12", "f13", "f20");
    }

    __asm__ అస్థిర("mova.m.x %0" ::"r"(తాత్కాలిక_ముసుగు));

    కొరకు (int32_t i = vec_చివర; i < elements; i++) {
        దశాంశం xv     = x_ఖండం[i];
        dst_ఖండం[i] = (xv > 0.0f) ? xv * g_ఖండం[i] : 0.0f;
    }
}

// Vectorized GeGLU-Quick block: dst = x * sigmoid(1.702 * x) * g
// Using gelu_quick(x) = x / (1 + exp(-1.702*x))
స్థిర అంతఃవరుస శూన్య ఖండం_geglu_శీఘ్ర(దశాంశం * dst_ఖండం, స్థిరరాశి దశాంశం * x_ఖండం, స్థిరరాశి దశాంశం * g_ఖండం, పూర్ణాంకం elements) {
    int32_t vec_చివర = (elements / 8) * 8;

    గుర్తులేని దీర్ఘ తాత్కాలిక_ముసుగు;
    __asm__ అస్థిర("mova.x.m %0" : "=r"(తాత్కాలిక_ముసుగు));
    __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");

    దశాంశం zero_const        = 0.0f;
    దశాంశం one_const         = 1.0f;
    // -1.702 * log2(e), so that fexp.ps(x * neg_k_log2e) = exp(-1.702*x)
    దశాంశం neg_k_log2e_const = -1.702f * 1.4426950408889634f;

    కొరకు (int32_t i = 0; i < vec_చివర; i += 8) {
        __asm__ అస్థిర(
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
            : [dst_బహిర్గతం] "=m"(*(దశాంశం (*)[8]) & dst_ఖండం[i])
            : [x_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & x_ఖండం[i]), [g_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & g_ఖండం[i]),
              [zero_ptr] "m"(zero_const), [one_ptr] "m"(one_const), [k_ptr] "m"(neg_k_log2e_const)
            : "f10", "f11", "f13", "f14", "f15", "f16", "f17", "f18", "f20", "f21", "f22");
    }

    __asm__ అస్థిర("mova.m.x %0" ::"r"(తాత్కాలిక_ముసుగు));

    కొరకు (int32_t i = vec_చివర; i < elements; i++) {
        దశాంశం xv     = x_ఖండం[i];
        // Reuse silu reciprocal path: sigmoid(1.702*x) = 1/(1+exp(-1.702*x))
        దశాంశం e      = et_expf(-1.702f * xv);
        dst_ఖండం[i] = et_fdiv(xv, 1.0f + e) * g_ఖండం[i];
    }
}

// Vectorized SwiGLU-OAI block (OpenAI gpt-oss variant):
//   x_c = min(x, limit)
//   y_c = clamp(g, -limit, limit)
//   out = (x_c / (1 + exp(-alpha * x_c))) * (y_c + 1)
స్థిర అంతఃవరుస శూన్య ఖండం_swiglu_oai(దశాంశం *       dst_ఖండం,
                                    స్థిరరాశి దశాంశం * x_ఖండం,
                                    స్థిరరాశి దశాంశం * g_ఖండం,
                                    పూర్ణాంకం           elements,
                                    దశాంశం         alpha,
                                    దశాంశం         హద్దు) {
    int32_t vec_చివర = (elements / 8) * 8;

    గుర్తులేని దీర్ఘ తాత్కాలిక_ముసుగు;
    __asm__ అస్థిర("mova.x.m %0" : "=r"(తాత్కాలిక_ముసుగు));
    __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");

    దశాంశం zero_const    = 0.0f;
    దశాంశం one_const     = 1.0f;
    దశాంశం హద్దు_pos     = హద్దు;
    దశాంశం హద్దు_neg     = -హద్దు;
    // -alpha * log2(e): feed (x * neg_alpha_log2e) into fexp.ps to get exp(-alpha*x)
    దశాంశం neg_alpha_l2e = -alpha * 1.4426950408889634f;

    కొరకు (int32_t i = 0; i < vec_చివర; i += 8) {
        __asm__ అస్థిర(
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

            : [dst_బహిర్గతం] "=m"(*(దశాంశం (*)[8]) & dst_ఖండం[i])
            : [x_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & x_ఖండం[i]), [g_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & g_ఖండం[i]),
              [zero_ptr] "m"(zero_const), [one_ptr] "m"(one_const), [lim_pos] "m"(హద్దు_pos), [lim_neg] "m"(హద్దు_neg),
              [k_ptr] "m"(neg_alpha_l2e)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f23", "f24", "f25");
    }

    __asm__ అస్థిర("mova.m.x %0" ::"r"(తాత్కాలిక_ముసుగు));

    // Scalar tail (mirrors CPU reference exactly)
    కొరకు (int32_t i = vec_చివర; i < elements; i++) {
        దశాంశం xv = x_ఖండం[i];
        దశాంశం yv = g_ఖండం[i];
        ఒకవేళ (xv > హద్దు) {
            xv = హద్దు;
        }
        ఒకవేళ (yv > హద్దు) {
            yv = హద్దు;
        }
        ఒకవేళ (yv < -హద్దు) {
            yv = -హద్దు;
        }
        దశాంశం e       = et_expf(-alpha * xv);
        దశాంశం బహిర్గతం_glu = et_fdiv(xv, 1.0f + e);
        dst_ఖండం[i]  = బహిర్గతం_glu * (yv + 1.0f);
    }
}

స్థిర అంతఃవరుస శూన్య ఖండం_swiglu_clamp(దశాంశం *       dst_ఖండం,
                                      స్థిరరాశి దశాంశం * gate_ఖండం,
                                      స్థిరరాశి దశాంశం * పైకి_ఖండం,
                                      పూర్ణాంకం           elements,
                                      దశాంశం         హద్దు) {
    int32_t vec_చివర = (elements / 8) * 8;

    గుర్తులేని దీర్ఘ తాత్కాలిక_ముసుగు;
    __asm__ అస్థిర("mova.x.m %0" : "=r"(తాత్కాలిక_ముసుగు));
    __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");

    దశాంశం one_const = 1.0f;
    దశాంశం హద్దు_pos = హద్దు;
    దశాంశం హద్దు_neg = -హద్దు;
    దశాంశం neg_log2e = -1.4426950408889634f;

    కొరకు (int32_t i = 0; i < vec_చివర; i += 8) {
        __asm__ అస్థిర(
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
            : [dst_బహిర్గతం] "=m"(*(దశాంశం (*)[8]) & dst_ఖండం[i])
            : [gate_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & gate_ఖండం[i]), [పైకి_vec] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & పైకి_ఖండం[i]),
              [one_ptr] "m"(one_const), [lim_pos] "m"(హద్దు_pos), [lim_neg] "m"(హద్దు_neg), [k_ptr] "m"(neg_log2e)
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f21", "f23", "f24", "f25");
    }

    __asm__ అస్థిర("mova.m.x %0" :: "r"(తాత్కాలిక_ముసుగు));

    కొరకు (int32_t i = vec_చివర; i < elements; i++) {
        దశాంశం gate   = gate_ఖండం[i] > హద్దు ? హద్దు : gate_ఖండం[i];
        దశాంశం పైకి     = పైకి_ఖండం[i];
        పైకి           = పైకి > హద్దు ? హద్దు : పైకి;
        పైకి           = పైకి < -హద్దు ? -హద్దు : పైకి;
        dst_ఖండం[i] = silu_f32(gate) * పైకి;
    }
}

// Scalar erf approximation (Abramowitz & Stegun 7.1.26, max error ~1.5e-7)
స్థిర అంతఃవరుస దశాంశం erf_approx(దశాంశం x) {
    స్థిరరాశి దశాంశం a1 = 0.254829592f;
    స్థిరరాశి దశాంశం a2 = -0.284496736f;
    స్థిరరాశి దశాంశం a3 = 1.421413741f;
    స్థిరరాశి దశాంశం a4 = -1.453152027f;
    స్థిరరాశి దశాంశం a5 = 1.061405429f;
    స్థిరరాశి దశాంశం p  = 0.3275911f;

    దశాంశం sign = (x < 0.0f) ? -1.0f : 1.0f;
    దశాంశం ax   = (x < 0.0f) ? -x : x;
    దశాంశం t    = et_fdiv(1.0f, 1.0f + p * ax);
    దశాంశం t2   = t * t;
    దశాంశం t3   = t2 * t;
    దశాంశం t4   = t3 * t;
    దశాంశం t5   = t4 * t;
    దశాంశం poly = a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5;
    దశాంశం y    = 1.0f - poly * et_expf(-ax * ax);
    తిరిగి sign * y;
}

// GeGLU-Erf block: dst = 0.5 * x * (1 + erf(x / sqrt(2))) * g
// Scalar implementation — variant is rarely used so we keep complexity low.
స్థిర అంతఃవరుస శూన్య ఖండం_geglu_erf(దశాంశం * dst_ఖండం, స్థిరరాశి దశాంశం * x_ఖండం, స్థిరరాశి దశాంశం * g_ఖండం, పూర్ణాంకం elements) {
    స్థిరరాశి దశాంశం sqrt_2_inv = 0.70710678118654752440f;
    కొరకు (int32_t i = 0; i < elements; i++) {
        దశాంశం xv     = x_ఖండం[i];
        dst_ఖండం[i] = 0.5f * xv * (1.0f + erf_approx(xv * sqrt_2_inv)) * g_ఖండం[i];
    }
}

// Main entry point for GLU kernel
పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_glu_params * params, శూన్య * env) {
    // Cast env to proper type
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    // Validate environment pointer
    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    // Get thread info using shire mask from environment
    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    // Basic safety check on params
    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;  // Invalid pointer
    }

    // Supported variants: SwiGLU, SwiGLU-OAI, GeGLU, GeGLU-Erf, GeGLU-Quick, ReGLU
    ఎంపిక (params->glu_op_type) {
        స్థితి GGML_GLU_OP_SWIGLU:
        స్థితి GGML_GLU_OP_SWIGLU_OAI:
        స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
        స్థితి GGML_GLU_OP_GEGLU:
        స్థితి GGML_GLU_OP_GEGLU_ERF:
        స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
        స్థితి GGML_GLU_OP_REGLU:
            ఆపు;
        అప్రమేయ:
            తిరిగి -1;  // Unsupported GLU operation
    }

    // Extract tensor references
    నిర్మాణం ggml_tensor * src0    = &params->src0;
    నిర్మాణం ggml_tensor * src1    = params->src1.దత్తాంశం ? &params->src1 : 0;
    నిర్మాణం ggml_tensor * dst     = &params->dst;
    int32_t              swapped = params->swapped;

    // Validate tensor types (F32 only)
    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;  // Unsupported type combination
    }

    ఒకవేళ (src1 && src1->type != GGML_TYPE_F32) {
        తిరిగి -1;  // Unsupported src1 type
    }

    // Get data pointers
    దశాంశం * src0_దత్తాంశం = (దశాంశం *) src0->దత్తాంశం;
    దశాంశం * src1_దత్తాంశం = src1 ? (దశాంశం *) src1->దత్తాంశం : src0_దత్తాంశం;
    దశాంశం * dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;

    // Validate data pointers
    ఒకవేళ (!src0_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;  // Null data pointer
    }

    // Get tensor dimensions
    స్థిరరాశి int64_t nc = dst->ne[0];                            // Output columns (input columns / 2)
    స్థిరరాశి int64_t nr = dst->ne[1] * dst->ne[2] * dst->ne[3];  // Total rows

    // Get strides
    స్థిరరాశి size_t src0_stride = src0->nb[1];                       // Stride between rows in src0
    స్థిరరాశి size_t src1_stride = src1 ? src1->nb[1] : src0->nb[1];  // Stride between rows in src1
    స్థిరరాశి size_t dst_stride  = dst->nb[1];                        // Stride between rows in dst

    // Validate dimensions for split SwiGLU
    ఒకవేళ (src1) {
        // Split tensor mode: src0 and src1 should have same shape as dst
        ఒకవేళ (src0->ne[0] != nc || src1->ne[0] != nc) {
            తిరిగి -1;  // Dimension mismatch in split mode
        }
    } లేకపోతే {
        // Single tensor mode: src0 should have 2*nc columns
        ఒకవేళ (src0->ne[0] != 2 * nc) {
            తిరిగి -1;  // Dimension mismatch in single tensor mode
        }
    }

    // Calculate total elements for cache line distribution
    స్థిరరాశి int64_t elements_చొప్పున_cacheline = 16;  // 64 bytes / 4 bytes per float
    స్థిరరాశి int64_t మొత్తం_elements         = nr * nc;
    స్థిరరాశి int64_t మొత్తం_cachelines       = (మొత్తం_elements + elements_చొప్పున_cacheline - 1) / elements_చొప్పున_cacheline;

    // Distribute cache lines across threads
    int64_t cachelines_చొప్పున_thread = (మొత్తం_cachelines + అంకె_దారాలు - 1) / అంకె_దారాలు;
    int64_t ప్రారంభం_cacheline       = thread_id * cachelines_చొప్పున_thread;
    int64_t చివర_cacheline         = ప్రారంభం_cacheline + cachelines_చొప్పున_thread;

    // Clamp end_cacheline to actual number of cache lines
    ఒకవేళ (చివర_cacheline > మొత్తం_cachelines) {
        చివర_cacheline = మొత్తం_cachelines;
    }

    // Thread should return if no work to do
    ఒకవేళ (ప్రారంభం_cacheline >= మొత్తం_cachelines) {
        తిరిగి 0;
    }

    // Process cache lines assigned to this thread
    కొరకు (int64_t cl = ప్రారంభం_cacheline; cl < చివర_cacheline; cl++) {
        // Map cache line back to element coordinates
        int64_t global_element_ప్రారంభం = cl * elements_చొప్పున_cacheline;
        int64_t అడ్డువరుస                  = global_element_ప్రారంభం / nc;
        int64_t నిలువు                  = global_element_ప్రారంభం % nc;

        // Skip if we're past the end of data
        ఒకవేళ (global_element_ప్రారంభం >= మొత్తం_elements) {
            ఆపు;
        }

        // Calculate how many elements to process in this cache line
        int64_t elements_మిగిలిన = మొత్తం_elements - global_element_ప్రారంభం;
        పూర్ణాంకం     elements_this_ఖండం =
            (పూర్ణాంకం) ((elements_మిగిలిన < elements_చొప్పున_cacheline) ? elements_మిగిలిన : elements_చొప్పున_cacheline);

        // Process elements that span across rows
        int64_t elements_processed = 0;
        అయితే (elements_processed < elements_this_ఖండం && అడ్డువరుస < nr) {
            // Calculate elements to process in current row
            int64_t elements_లోపల_అడ్డువరుస     = nc - నిలువు;
            int64_t elements_గమ్యం_process = elements_this_ఖండం - elements_processed;
            ఒకవేళ (elements_గమ్యం_process > elements_లోపల_అడ్డువరుస) {
                elements_గమ్యం_process = elements_లోపల_అడ్డువరుస;
            }

            // Get pointers for current row and column range
            దశాంశం * dst_ptr = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + అడ్డువరుస * dst_stride) + నిలువు;

            దశాంశం * x_ptr;
            దశాంశం * g_ptr;

            ఒకవేళ (src1) {
                // Split tensor mode
                x_ptr = (దశాంశం *) ((అక్షరం *) src0_దత్తాంశం + అడ్డువరుస * src0_stride) + నిలువు;
                g_ptr = (దశాంశం *) ((అక్షరం *) src1_దత్తాంశం + అడ్డువరుస * src1_stride) + నిలువు;
            } లేకపోతే {
                // Single tensor mode - src0 contains both x and g
                దశాంశం * src0_అడ్డువరుస = (దశాంశం *) ((అక్షరం *) src0_దత్తాంశం + అడ్డువరుస * src0_stride);
                ఒకవేళ (swapped) {
                    g_ptr = src0_అడ్డువరుస + నిలువు;       // First half is gate
                    x_ptr = src0_అడ్డువరుస + nc + నిలువు;  // Second half is value
                } లేకపోతే {
                    x_ptr = src0_అడ్డువరుస + నిలువు;       // First half is value
                    g_ptr = src0_అడ్డువరుస + nc + నిలువు;  // Second half is gate
                }
            }

            // Process this segment
            ఎంపిక (params->glu_op_type) {
                స్థితి GGML_GLU_OP_GEGLU:
                    ఖండం_geglu(dst_ptr, x_ptr, g_ptr, (పూర్ణాంకం) elements_గమ్యం_process);
                    ఆపు;
                స్థితి GGML_GLU_OP_SWIGLU:
                    ఖండం_swiglu(dst_ptr, x_ptr, g_ptr, (పూర్ణాంకం) elements_గమ్యం_process);
                    ఆపు;
                స్థితి GGML_GLU_OP_REGLU:
                    ఖండం_reglu(dst_ptr, x_ptr, g_ptr, (పూర్ణాంకం) elements_గమ్యం_process);
                    ఆపు;
                స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
                    ఖండం_geglu_శీఘ్ర(dst_ptr, x_ptr, g_ptr, (పూర్ణాంకం) elements_గమ్యం_process);
                    ఆపు;
                స్థితి GGML_GLU_OP_GEGLU_ERF:
                    ఖండం_geglu_erf(dst_ptr, x_ptr, g_ptr, (పూర్ణాంకం) elements_గమ్యం_process);
                    ఆపు;
                స్థితి GGML_GLU_OP_SWIGLU_OAI:
                    ఖండం_swiglu_oai(dst_ptr, x_ptr, g_ptr, (పూర్ణాంకం) elements_గమ్యం_process, params->alpha, params->హద్దు);
                    ఆపు;
                స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
                    ఖండం_swiglu_clamp(dst_ptr, x_ptr, g_ptr, (పూర్ణాంకం) elements_గమ్యం_process, params->హద్దు);
                    ఆపు;
                అప్రమేయ:
                    తిరిగి -1;
            }

            // Update counters
            elements_processed += elements_గమ్యం_process;
            నిలువు += elements_గమ్యం_process;

            // Move to next row if current row is complete
            ఒకవేళ (నిలువు >= nc) {
                అడ్డువరుస++;
                నిలువు = 0;
            }
        }
    }

    తిరిగి 0;
}
