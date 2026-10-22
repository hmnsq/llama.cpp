#అన్తర్భావ <sycl/sycl.hpp>
#అన్తర్భావ "wkv.hpp"

constexpr పూర్ణాంకం WKV_ఖండం_కొలత = 64;

// Helper function for the main kernel
మూస <పూర్ణాంకం ఖండం_కొలత_3>
స్థిర శూన్య rwkv_wkv6_f32_kernel(
    స్థిరరాశి పూర్ణాంకం B, స్థిరరాశి పూర్ణాంకం T, స్థిరరాశి పూర్ణాంకం C, స్థిరరాశి పూర్ణాంకం H,
    స్థిరరాశి దశాంశం* k, స్థిరరాశి దశాంశం* v, స్థిరరాశి దశాంశం* r,
    స్థిరరాశి దశాంశం* tf, స్థిరరాశి దశాంశం* td, స్థిరరాశి దశాంశం* s,
    దశాంశం* dst, స్థిరరాశి sycl::nd_అంశం<3>& అంశం_ct1, దశాంశం* shared_mem) {

    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి పూర్ణాంకం bid = అంశం_ct1.get_సమూహం(2);

    స్థిరరాశి పూర్ణాంకం తల_కొలత = ఖండం_కొలత_3;
    స్థిరరాశి పూర్ణాంకం batch_i = bid / H;
    స్థిరరాశి పూర్ణాంకం తల_i = bid % H;
    స్థిరరాశి పూర్ణాంకం state_కొలత = C * తల_కొలత;
    స్థిరరాశి పూర్ణాంకం n_seq_టోకెన్లు = T / B;

    // Set up shared memory pointers
    దశాంశం* _k = shared_mem;
    దశాంశం* _r = _k + తల_కొలత;
    దశాంశం* _tf = _r + తల_కొలత;
    దశాంశం* _td = _tf + తల_కొలత;

    // Local state array
    దశాంశం state[ఖండం_కొలత_3];

    // Load initial state
    #సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < తల_కొలత; i++) {
        state[i] = s[batch_i * state_కొలత + తల_i * తల_కొలత * తల_కొలత + i * తల_కొలత + tid];
    }

    // Sync threads before shared memory operations
    అంశం_ct1.barrier(sycl::access::fence_space::local_space);

    // Load time-mixing parameters
    _tf[tid] = tf[తల_i * తల_కొలత + tid];
    అంశం_ct1.barrier(sycl::access::fence_space::local_space);

    // Main sequence processing loop
    కొరకు (పూర్ణాంకం t = batch_i * n_seq_టోకెన్లు * C + తల_i * తల_కొలత + tid;
         t < (batch_i + 1) * n_seq_టోకెన్లు * C + తల_i * తల_కొలత + tid;
         t += C) {

        అంశం_ct1.barrier(sycl::access::fence_space::local_space);

        // Load current timestep data to shared memory
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];

        అంశం_ct1.barrier(sycl::access::fence_space::local_space);

        స్థిరరాశి దశాంశం _v = v[t];
        దశాంశం y = 0;

        // Process in chunks of 4 for better vectorization
        sycl::float4 k4, r4, tf4, td4, s4;
        #సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < తల_కొలత; j += 4) {
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
    #సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < తల_కొలత; i++) {
        dst[T * C + batch_i * state_కొలత + తల_i * తల_కొలత * తల_కొలత + i * తల_కొలత + tid] = state[i];
    }
}

మూస <పూర్ణాంకం ఖండం_కొలత_3>
స్థిర శూన్య rwkv_wkv7_f32_kernel(
    స్థిరరాశి పూర్ణాంకం B, స్థిరరాశి పూర్ణాంకం T, స్థిరరాశి పూర్ణాంకం C, స్థిరరాశి పూర్ణాంకం H,
    స్థిరరాశి దశాంశం* r, స్థిరరాశి దశాంశం* w, స్థిరరాశి దశాంశం* k, స్థిరరాశి దశాంశం* v,
    స్థిరరాశి దశాంశం* a, స్థిరరాశి దశాంశం* b, స్థిరరాశి దశాంశం* s,
    దశాంశం* dst, స్థిరరాశి sycl::nd_అంశం<3>& అంశం_ct1, దశాంశం* shared_mem) {

    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి పూర్ణాంకం bid = అంశం_ct1.get_సమూహం(2);

    స్థిరరాశి పూర్ణాంకం తల_కొలత = ఖండం_కొలత_3;
    స్థిరరాశి పూర్ణాంకం batch_i = bid / H;
    స్థిరరాశి పూర్ణాంకం తల_i = bid % H;
    స్థిరరాశి పూర్ణాంకం state_కొలత = C * తల_కొలత;
    స్థిరరాశి పూర్ణాంకం n_seq_టోకెన్లు = T / B;

    దశాంశం* _r = shared_mem;
    దశాంశం* _w = _r + తల_కొలత;
    దశాంశం* _k = _w + తల_కొలత;
    దశాంశం* _a = _k + తల_కొలత;
    దశాంశం* _b = _a + తల_కొలత;

    దశాంశం state[ఖండం_కొలత_3];

    #సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < తల_కొలత; i++) {
        state[i] = s[batch_i * state_కొలత + తల_i * తల_కొలత * తల_కొలత + tid * తల_కొలత + i];
    }

    కొరకు (పూర్ణాంకం t = batch_i * n_seq_టోకెన్లు * C + తల_i * తల_కొలత + tid;
         t < (batch_i + 1) * n_seq_టోకెన్లు * C + తల_i * తల_కొలత + tid;
         t += C) {

        అంశం_ct1.barrier(sycl::access::fence_space::local_space);

        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];

        అంశం_ct1.barrier(sycl::access::fence_space::local_space);

        స్థిరరాశి దశాంశం _v = v[t];
        దశాంశం y = 0, sa = 0;
        sycl::float4 a4, s4;

        #సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < తల_కొలత; j += 4) {
            a4 = sycl::float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa += sycl::dot(a4, s4);
        }

        sycl::float4 r4, w4, k4, b4;
        #సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < తల_కొలత; j += 4) {
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

    #సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < తల_కొలత; i++) {
        dst[T * C + batch_i * state_కొలత + తల_i * తల_కొలత * తల_కొలత + tid * తల_కొలత + i] = state[i];
    }
}

శూన్య ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    స్థిరరాశి దశాంశం* k_d = (స్థిరరాశి దశాంశం*)dst->src[0]->దత్తాంశం;
    స్థిరరాశి దశాంశం* v_d = (స్థిరరాశి దశాంశం*)dst->src[1]->దత్తాంశం;
    స్థిరరాశి దశాంశం* r_d = (స్థిరరాశి దశాంశం*)dst->src[2]->దత్తాంశం;
    స్థిరరాశి దశాంశం* tf_d = (స్థిరరాశి దశాంశం*)dst->src[3]->దత్తాంశం;
    స్థిరరాశి దశాంశం* td_d = (స్థిరరాశి దశాంశం*)dst->src[4]->దత్తాంశం;
    స్థిరరాశి దశాంశం* s_d = (స్థిరరాశి దశాంశం*)dst->src[5]->దత్తాంశం;
    దశాంశం* dst_d = (దశాంశం*)dst->దత్తాంశం;

    స్థిరరాశి int64_t B = dst->src[5]->ne[1];
    స్థిరరాశి int64_t T = dst->src[0]->ne[2];
    స్థిరరాశి int64_t C = dst->ne[0];
    స్థిరరాశి int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_ఖండం_కొలత || C / H == WKV_ఖండం_కొలత * 2); // The current sycl kernel is designed for RWKV6, HEAD_SIZE == 64

    dpct::వరుస_ptr stream = ctx.stream();

    // Calculate execution configuration
    స్థిరరాశి size_t shared_mem_కొలత = C / H * 4 * పరిమాణం(దశాంశం); // For k, r, tf, td
    sycl::range<3> ఖండం_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    ఒకవేళ (C / H == WKV_ఖండం_కొలత) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<దశాంశం, 1> shared_mem_acc(shared_mem_కొలత, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_ఖండం_కొలత>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        అంశం_ct1, (దశాంశం*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } లేకపోతే {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<దశాంశం, 1> shared_mem_acc(shared_mem_కొలత, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_ఖండం_కొలత * 2>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        అంశం_ct1, (దశాంశం*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}

శూన్య ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    స్థిరరాశి దశాంశం* r_d = (స్థిరరాశి దశాంశం*)dst->src[0]->దత్తాంశం;
    స్థిరరాశి దశాంశం* w_d = (స్థిరరాశి దశాంశం*)dst->src[1]->దత్తాంశం;
    స్థిరరాశి దశాంశం* k_d = (స్థిరరాశి దశాంశం*)dst->src[2]->దత్తాంశం;
    స్థిరరాశి దశాంశం* v_d = (స్థిరరాశి దశాంశం*)dst->src[3]->దత్తాంశం;
    స్థిరరాశి దశాంశం* a_d = (స్థిరరాశి దశాంశం*)dst->src[4]->దత్తాంశం;
    స్థిరరాశి దశాంశం* b_d = (స్థిరరాశి దశాంశం*)dst->src[5]->దత్తాంశం;
    స్థిరరాశి దశాంశం* s_d = (స్థిరరాశి దశాంశం*)dst->src[6]->దత్తాంశం;
    దశాంశం* dst_d = (దశాంశం*)dst->దత్తాంశం;

    స్థిరరాశి int64_t B = dst->src[6]->ne[1];
    స్థిరరాశి int64_t T = dst->src[0]->ne[2];
    స్థిరరాశి int64_t C = dst->ne[0];
    స్థిరరాశి int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_ఖండం_కొలత || C / H == WKV_ఖండం_కొలత * 2);

    dpct::వరుస_ptr stream = ctx.stream();

    // Calculate execution configuration
    స్థిరరాశి size_t shared_mem_కొలత = C / H * 5 * పరిమాణం(దశాంశం); // For r, w, k, a, b
    sycl::range<3> ఖండం_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    ఒకవేళ (C / H == WKV_ఖండం_కొలత) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<దశాంశం, 1> shared_mem_acc(shared_mem_కొలత, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_ఖండం_కొలత>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        అంశం_ct1, (దశాంశం*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } లేకపోతే {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<దశాంశం, 1> shared_mem_acc(shared_mem_కొలత, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_ఖండం_కొలత * 2>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        అంశం_ct1, (దశాంశం*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}
