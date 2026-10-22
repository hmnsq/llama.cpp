#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ "wkv.hpp"

constexpr ಪೂರ್ಣಾಂಕ WKV_ಖಂಡ_ಗಾತ್ರ = 64;

// Helper function for the main kernel
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3>
ಸ್ಥಿರ ಶೂನ್ಯ rwkv_wkv6_f32_kernel(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ B, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ T, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ C, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ H,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* k, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* v, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* r,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* tf, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* td, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* s,
    ದಶಮಾಂಶ* dst, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3>& ಅಂಶ_ct1, ದಶಮಾಂಶ* shared_mem) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bid = ಅಂಶ_ct1.get_ಗುಂಪು(2);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ತಲೆ_ಗಾತ್ರ = ಖಂಡ_ಗಾತ್ರ_3;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ batch_i = bid / H;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ತಲೆ_i = bid % H;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ state_ಗಾತ್ರ = C * ತಲೆ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_seq_ಟೋಕನುಗಳು = T / B;

    // Set up shared memory pointers
    ದಶಮಾಂಶ* _k = shared_mem;
    ದಶಮಾಂಶ* _r = _k + ತಲೆ_ಗಾತ್ರ;
    ದಶಮಾಂಶ* _tf = _r + ತಲೆ_ಗಾತ್ರ;
    ದಶಮಾಂಶ* _td = _tf + ತಲೆ_ಗಾತ್ರ;

    // Local state array
    ದಶಮಾಂಶ state[ಖಂಡ_ಗಾತ್ರ_3];

    // Load initial state
    #ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ತಲೆ_ಗಾತ್ರ; i++) {
        state[i] = s[batch_i * state_ಗಾತ್ರ + ತಲೆ_i * ತಲೆ_ಗಾತ್ರ * ತಲೆ_ಗಾತ್ರ + i * ತಲೆ_ಗಾತ್ರ + tid];
    }

    // Sync threads before shared memory operations
    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    // Load time-mixing parameters
    _tf[tid] = tf[ತಲೆ_i * ತಲೆ_ಗಾತ್ರ + tid];
    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    // Main sequence processing loop
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = batch_i * n_seq_ಟೋಕನುಗಳು * C + ತಲೆ_i * ತಲೆ_ಗಾತ್ರ + tid;
         t < (batch_i + 1) * n_seq_ಟೋಕನುಗಳು * C + ತಲೆ_i * ತಲೆ_ಗಾತ್ರ + tid;
         t += C) {

        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

        // Load current timestep data to shared memory
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];

        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ _v = v[t];
        ದಶಮಾಂಶ y = 0;

        // Process in chunks of 4 for better vectorization
        sycl::float4 k4, r4, tf4, td4, s4;
        #ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ತಲೆ_ಗಾತ್ರ; j += 4) {
            // Load data in vec4 chunks
            k4 = sycl::float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            r4 = sycl::float4(_r[j], _r[j+1], _r[j+2], _r[j+3]);
            tf4 = sycl::float4(_tf[j], _tf[j+1], _tf[j+2], _tf[j+3]);
            td4 = sycl::float4(_td[j], _td[j+1], _td[j+2], _td[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);

            // Compute key-value product
            sycl::float4 kv4 = k4 * _v;

            // Accumulate weighted sum
            y += sycl::dot(r4, tf4 * kv4 + s4);

            // Update state
            s4 = s4 * td4 + kv4;

            // Store updated state
            state[j] = s4.x();
            state[j+1] = s4.y();
            state[j+2] = s4.z();
            state[j+3] = s4.w();
        }

        dst[t] = y;
    }

    // Save final state
    #ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ತಲೆ_ಗಾತ್ರ; i++) {
        dst[T * C + batch_i * state_ಗಾತ್ರ + ತಲೆ_i * ತಲೆ_ಗಾತ್ರ * ತಲೆ_ಗಾತ್ರ + i * ತಲೆ_ಗಾತ್ರ + tid] = state[i];
    }
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3>
ಸ್ಥಿರ ಶೂನ್ಯ rwkv_wkv7_f32_kernel(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ B, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ T, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ C, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ H,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* r, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* w, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* k, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* v,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* a, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* b, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* s,
    ದಶಮಾಂಶ* dst, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3>& ಅಂಶ_ct1, ದಶಮಾಂಶ* shared_mem) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bid = ಅಂಶ_ct1.get_ಗುಂಪು(2);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ತಲೆ_ಗಾತ್ರ = ಖಂಡ_ಗಾತ್ರ_3;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ batch_i = bid / H;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ತಲೆ_i = bid % H;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ state_ಗಾತ್ರ = C * ತಲೆ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_seq_ಟೋಕನುಗಳು = T / B;

    ದಶಮಾಂಶ* _r = shared_mem;
    ದಶಮಾಂಶ* _w = _r + ತಲೆ_ಗಾತ್ರ;
    ದಶಮಾಂಶ* _k = _w + ತಲೆ_ಗಾತ್ರ;
    ದಶಮಾಂಶ* _a = _k + ತಲೆ_ಗಾತ್ರ;
    ದಶಮಾಂಶ* _b = _a + ತಲೆ_ಗಾತ್ರ;

    ದಶಮಾಂಶ state[ಖಂಡ_ಗಾತ್ರ_3];

    #ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ತಲೆ_ಗಾತ್ರ; i++) {
        state[i] = s[batch_i * state_ಗಾತ್ರ + ತಲೆ_i * ತಲೆ_ಗಾತ್ರ * ತಲೆ_ಗಾತ್ರ + tid * ತಲೆ_ಗಾತ್ರ + i];
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = batch_i * n_seq_ಟೋಕನುಗಳು * C + ತಲೆ_i * ತಲೆ_ಗಾತ್ರ + tid;
         t < (batch_i + 1) * n_seq_ಟೋಕನುಗಳು * C + ತಲೆ_i * ತಲೆ_ಗಾತ್ರ + tid;
         t += C) {

        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];

        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ _v = v[t];
        ದಶಮಾಂಶ y = 0, sa = 0;
        sycl::float4 a4, s4;

        #ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ತಲೆ_ಗಾತ್ರ; j += 4) {
            a4 = sycl::float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa += sycl::dot(a4, s4);
        }

        sycl::float4 r4, w4, k4, b4;
        #ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < ತಲೆ_ಗಾತ್ರ; j += 4) {
            r4 = sycl::float4(_r[j], _r[j+1], _r[j+2], _r[j+3]);
            w4 = sycl::float4(_w[j], _w[j+1], _w[j+2], _w[j+3]);
            k4 = sycl::float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            b4 = sycl::float4(_b[j], _b[j+1], _b[j+2], _b[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);

            sycl::float4 kv4 = k4 * _v;

            s4 = s4 * w4 + kv4 + sa * b4;
            y += sycl::dot(r4, s4);

            state[j] = s4.x();
            state[j+1] = s4.y();
            state[j+2] = s4.z();
            state[j+3] = s4.w();
        }

        dst[t] = y;
    }

    #ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ತಲೆ_ಗಾತ್ರ; i++) {
        dst[T * C + batch_i * state_ಗಾತ್ರ + ತಲೆ_i * ತಲೆ_ಗಾತ್ರ * ತಲೆ_ಗಾತ್ರ + tid * ತಲೆ_ಗಾತ್ರ + i] = state[i];
    }
}

ಶೂನ್ಯ ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* k_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)dst->src[0]->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* v_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)dst->src[1]->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* r_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)dst->src[2]->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* tf_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)dst->src[3]->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* td_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)dst->src[4]->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* s_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)dst->src[5]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ* dst_d = (ದಶಮಾಂಶ*)dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ int64_t B = dst->src[5]->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t T = dst->src[0]->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t C = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_ಖಂಡ_ಗಾತ್ರ || C / H == WKV_ಖಂಡ_ಗಾತ್ರ * 2); // The current sycl kernel is designed for RWKV6, HEAD_SIZE == 64

    dpct::ಸರತಿ_ptr stream = ctx.stream();

    // Calculate execution configuration
    ಸ್ಥಿರಾಂಕ size_t shared_mem_ಗಾತ್ರ = C / H * 4 * ಗಾತ್ರ(ದಶಮಾಂಶ); // For k, r, tf, td
    sycl::range<3> ಖಂಡ_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    ಆದರೆ (C / H == WKV_ಖಂಡ_ಗಾತ್ರ) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> shared_mem_acc(shared_mem_ಗಾತ್ರ, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_ಖಂಡ_ಗಾತ್ರ>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        ಅಂಶ_ct1, (ದಶಮಾಂಶ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } ಇಲ್ಲದಿದ್ದರೆ {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> shared_mem_acc(shared_mem_ಗಾತ್ರ, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_ಖಂಡ_ಗಾತ್ರ * 2>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        ಅಂಶ_ct1, (ದಶಮಾಂಶ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}

ಶೂನ್ಯ ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* r_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)dst->src[0]->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* w_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)dst->src[1]->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* k_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)dst->src[2]->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* v_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)dst->src[3]->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* a_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)dst->src[4]->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* b_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)dst->src[5]->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ* s_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ*)dst->src[6]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ* dst_d = (ದಶಮಾಂಶ*)dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ int64_t B = dst->src[6]->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t T = dst->src[0]->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t C = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_ಖಂಡ_ಗಾತ್ರ || C / H == WKV_ಖಂಡ_ಗಾತ್ರ * 2);

    dpct::ಸರತಿ_ptr stream = ctx.stream();

    // Calculate execution configuration
    ಸ್ಥಿರಾಂಕ size_t shared_mem_ಗಾತ್ರ = C / H * 5 * ಗಾತ್ರ(ದಶಮಾಂಶ); // For r, w, k, a, b
    sycl::range<3> ಖಂಡ_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    ಆದರೆ (C / H == WKV_ಖಂಡ_ಗಾತ್ರ) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> shared_mem_acc(shared_mem_ಗಾತ್ರ, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_ಖಂಡ_ಗಾತ್ರ>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        ಅಂಶ_ct1, (ದಶಮಾಂಶ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } ಇಲ್ಲದಿದ್ದರೆ {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> shared_mem_acc(shared_mem_ಗಾತ್ರ, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_ಖಂಡ_ಗಾತ್ರ * 2>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        ಅಂಶ_ct1, (ದಶಮಾಂಶ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}
