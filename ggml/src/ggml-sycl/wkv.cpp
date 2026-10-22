#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ "wkv.hpp"

constexpr ਪੂਰਨ WKV_ਖੰਡ_ਮਾਪ = 64;

// Helper function for the main kernel
ਸਾਂਚਾ <ਪੂਰਨ ਖੰਡ_ਮਾਪ_3>
ਸਥਿਰ ਖਾਲੀ rwkv_wkv6_f32_kernel(
    ਅਚਲ ਪੂਰਨ B, ਅਚਲ ਪੂਰਨ T, ਅਚਲ ਪੂਰਨ C, ਅਚਲ ਪੂਰਨ H,
    ਅਚਲ ਦਸ਼ਮਲਵ* k, ਅਚਲ ਦਸ਼ਮਲਵ* v, ਅਚਲ ਦਸ਼ਮਲਵ* r,
    ਅਚਲ ਦਸ਼ਮਲਵ* tf, ਅਚਲ ਦਸ਼ਮਲਵ* td, ਅਚਲ ਦਸ਼ਮਲਵ* s,
    ਦਸ਼ਮਲਵ* dst, ਅਚਲ sycl::nd_ਵਸਤੂ<3>& ਵਸਤੂ_ct1, ਦਸ਼ਮਲਵ* shared_mem) {

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ ਪੂਰਨ bid = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);

    ਅਚਲ ਪੂਰਨ ਸਿਰ_ਮਾਪ = ਖੰਡ_ਮਾਪ_3;
    ਅਚਲ ਪੂਰਨ batch_i = bid / H;
    ਅਚਲ ਪੂਰਨ ਸਿਰ_i = bid % H;
    ਅਚਲ ਪੂਰਨ state_ਮਾਪ = C * ਸਿਰ_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ n_seq_ਟੋਕਨ = T / B;

    // Set up shared memory pointers
    ਦਸ਼ਮਲਵ* _k = shared_mem;
    ਦਸ਼ਮਲਵ* _r = _k + ਸਿਰ_ਮਾਪ;
    ਦਸ਼ਮਲਵ* _tf = _r + ਸਿਰ_ਮਾਪ;
    ਦਸ਼ਮਲਵ* _td = _tf + ਸਿਰ_ਮਾਪ;

    // Local state array
    ਦਸ਼ਮਲਵ state[ਖੰਡ_ਮਾਪ_3];

    // Load initial state
    #ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < ਸਿਰ_ਮਾਪ; i++) {
        state[i] = s[batch_i * state_ਮਾਪ + ਸਿਰ_i * ਸਿਰ_ਮਾਪ * ਸਿਰ_ਮਾਪ + i * ਸਿਰ_ਮਾਪ + tid];
    }

    // Sync threads before shared memory operations
    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    // Load time-mixing parameters
    _tf[tid] = tf[ਸਿਰ_i * ਸਿਰ_ਮਾਪ + tid];
    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    // Main sequence processing loop
    ਲਈ (ਪੂਰਨ t = batch_i * n_seq_ਟੋਕਨ * C + ਸਿਰ_i * ਸਿਰ_ਮਾਪ + tid;
         t < (batch_i + 1) * n_seq_ਟੋਕਨ * C + ਸਿਰ_i * ਸਿਰ_ਮਾਪ + tid;
         t += C) {

        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

        // Load current timestep data to shared memory
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];

        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

        ਅਚਲ ਦਸ਼ਮਲਵ _v = v[t];
        ਦਸ਼ਮਲਵ y = 0;

        // Process in chunks of 4 for better vectorization
        sycl::float4 k4, r4, tf4, td4, s4;
        #ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < ਸਿਰ_ਮਾਪ; j += 4) {
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
    #ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < ਸਿਰ_ਮਾਪ; i++) {
        dst[T * C + batch_i * state_ਮਾਪ + ਸਿਰ_i * ਸਿਰ_ਮਾਪ * ਸਿਰ_ਮਾਪ + i * ਸਿਰ_ਮਾਪ + tid] = state[i];
    }
}

ਸਾਂਚਾ <ਪੂਰਨ ਖੰਡ_ਮਾਪ_3>
ਸਥਿਰ ਖਾਲੀ rwkv_wkv7_f32_kernel(
    ਅਚਲ ਪੂਰਨ B, ਅਚਲ ਪੂਰਨ T, ਅਚਲ ਪੂਰਨ C, ਅਚਲ ਪੂਰਨ H,
    ਅਚਲ ਦਸ਼ਮਲਵ* r, ਅਚਲ ਦਸ਼ਮਲਵ* w, ਅਚਲ ਦਸ਼ਮਲਵ* k, ਅਚਲ ਦਸ਼ਮਲਵ* v,
    ਅਚਲ ਦਸ਼ਮਲਵ* a, ਅਚਲ ਦਸ਼ਮਲਵ* b, ਅਚਲ ਦਸ਼ਮਲਵ* s,
    ਦਸ਼ਮਲਵ* dst, ਅਚਲ sycl::nd_ਵਸਤੂ<3>& ਵਸਤੂ_ct1, ਦਸ਼ਮਲਵ* shared_mem) {

    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ ਪੂਰਨ bid = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);

    ਅਚਲ ਪੂਰਨ ਸਿਰ_ਮਾਪ = ਖੰਡ_ਮਾਪ_3;
    ਅਚਲ ਪੂਰਨ batch_i = bid / H;
    ਅਚਲ ਪੂਰਨ ਸਿਰ_i = bid % H;
    ਅਚਲ ਪੂਰਨ state_ਮਾਪ = C * ਸਿਰ_ਮਾਪ;
    ਅਚਲ ਪੂਰਨ n_seq_ਟੋਕਨ = T / B;

    ਦਸ਼ਮਲਵ* _r = shared_mem;
    ਦਸ਼ਮਲਵ* _w = _r + ਸਿਰ_ਮਾਪ;
    ਦਸ਼ਮਲਵ* _k = _w + ਸਿਰ_ਮਾਪ;
    ਦਸ਼ਮਲਵ* _a = _k + ਸਿਰ_ਮਾਪ;
    ਦਸ਼ਮਲਵ* _b = _a + ਸਿਰ_ਮਾਪ;

    ਦਸ਼ਮਲਵ state[ਖੰਡ_ਮਾਪ_3];

    #ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < ਸਿਰ_ਮਾਪ; i++) {
        state[i] = s[batch_i * state_ਮਾਪ + ਸਿਰ_i * ਸਿਰ_ਮਾਪ * ਸਿਰ_ਮਾਪ + tid * ਸਿਰ_ਮਾਪ + i];
    }

    ਲਈ (ਪੂਰਨ t = batch_i * n_seq_ਟੋਕਨ * C + ਸਿਰ_i * ਸਿਰ_ਮਾਪ + tid;
         t < (batch_i + 1) * n_seq_ਟੋਕਨ * C + ਸਿਰ_i * ਸਿਰ_ਮਾਪ + tid;
         t += C) {

        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];

        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

        ਅਚਲ ਦਸ਼ਮਲਵ _v = v[t];
        ਦਸ਼ਮਲਵ y = 0, sa = 0;
        sycl::float4 a4, s4;

        #ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < ਸਿਰ_ਮਾਪ; j += 4) {
            a4 = sycl::float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa += sycl::dot(a4, s4);
        }

        sycl::float4 r4, w4, k4, b4;
        #ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < ਸਿਰ_ਮਾਪ; j += 4) {
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

    #ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < ਸਿਰ_ਮਾਪ; i++) {
        dst[T * C + batch_i * state_ਮਾਪ + ਸਿਰ_i * ਸਿਰ_ਮਾਪ * ਸਿਰ_ਮਾਪ + tid * ਸਿਰ_ਮਾਪ + i] = state[i];
    }
}

ਖਾਲੀ ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ਅਚਲ ਦਸ਼ਮਲਵ* k_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)dst->src[0]->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ* v_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)dst->src[1]->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ* r_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)dst->src[2]->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ* tf_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)dst->src[3]->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ* td_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)dst->src[4]->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ* s_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)dst->src[5]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ* dst_d = (ਦਸ਼ਮਲਵ*)dst->ਅੰਕੜੇ;

    ਅਚਲ int64_t B = dst->src[5]->ne[1];
    ਅਚਲ int64_t T = dst->src[0]->ne[2];
    ਅਚਲ int64_t C = dst->ne[0];
    ਅਚਲ int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_ਖੰਡ_ਮਾਪ || C / H == WKV_ਖੰਡ_ਮਾਪ * 2); // The current sycl kernel is designed for RWKV6, HEAD_SIZE == 64

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    // Calculate execution configuration
    ਅਚਲ size_t shared_mem_ਮਾਪ = C / H * 4 * ਆਕਾਰ(ਦਸ਼ਮਲਵ); // For k, r, tf, td
    sycl::range<3> ਖੰਡ_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    ਜੇ (C / H == WKV_ਖੰਡ_ਮਾਪ) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> shared_mem_acc(shared_mem_ਮਾਪ, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_ਖੰਡ_ਮਾਪ>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        ਵਸਤੂ_ct1, (ਦਸ਼ਮਲਵ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } ਨਹੀਂਤਾਂ {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> shared_mem_acc(shared_mem_ਮਾਪ, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_ਖੰਡ_ਮਾਪ * 2>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        ਵਸਤੂ_ct1, (ਦਸ਼ਮਲਵ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}

ਖਾਲੀ ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ਅਚਲ ਦਸ਼ਮਲਵ* r_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)dst->src[0]->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ* w_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)dst->src[1]->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ* k_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)dst->src[2]->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ* v_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)dst->src[3]->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ* a_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)dst->src[4]->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ* b_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)dst->src[5]->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ* s_d = (ਅਚਲ ਦਸ਼ਮਲਵ*)dst->src[6]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ* dst_d = (ਦਸ਼ਮਲਵ*)dst->ਅੰਕੜੇ;

    ਅਚਲ int64_t B = dst->src[6]->ne[1];
    ਅਚਲ int64_t T = dst->src[0]->ne[2];
    ਅਚਲ int64_t C = dst->ne[0];
    ਅਚਲ int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_ਖੰਡ_ਮਾਪ || C / H == WKV_ਖੰਡ_ਮਾਪ * 2);

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    // Calculate execution configuration
    ਅਚਲ size_t shared_mem_ਮਾਪ = C / H * 5 * ਆਕਾਰ(ਦਸ਼ਮਲਵ); // For r, w, k, a, b
    sycl::range<3> ਖੰਡ_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    ਜੇ (C / H == WKV_ਖੰਡ_ਮਾਪ) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> shared_mem_acc(shared_mem_ਮਾਪ, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_ਖੰਡ_ਮਾਪ>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        ਵਸਤੂ_ct1, (ਦਸ਼ਮਲਵ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } ਨਹੀਂਤਾਂ {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> shared_mem_acc(shared_mem_ਮਾਪ, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_ਖੰਡ_ਮਾਪ * 2>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        ਵਸਤੂ_ct1, (ਦਸ਼ਮਲਵ*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}
