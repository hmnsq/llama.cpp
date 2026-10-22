#अन्तर्भाव <sycl/sycl.hpp>
#अन्तर्भाव "wkv.hpp"

constexpr पूर्णाङ्क WKV_खण्डम्_आकार = 64;

// Helper function for the main kernel
प्रतिरूपम् <पूर्णाङ्क खण्डम्_आकार_3>
स्थिर शून्य rwkv_wkv6_f32_kernel(
    ध्रुव पूर्णाङ्क B, ध्रुव पूर्णाङ्क T, ध्रुव पूर्णाङ्क C, ध्रुव पूर्णाङ्क H,
    ध्रुव भिन्न* k, ध्रुव भिन्न* v, ध्रुव भिन्न* r,
    ध्रुव भिन्न* tf, ध्रुव भिन्न* td, ध्रुव भिन्न* s,
    भिन्न* dst, ध्रुव sycl::nd_item<3>& item_ct1, भिन्न* shared_mem) {

    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2);
    ध्रुव पूर्णाङ्क bid = item_ct1.get_गणः(2);

    ध्रुव पूर्णाङ्क शीर्ष_आकार = खण्डम्_आकार_3;
    ध्रुव पूर्णाङ्क batch_i = bid / H;
    ध्रुव पूर्णाङ्क शीर्ष_i = bid % H;
    ध्रुव पूर्णाङ्क state_आकार = C * शीर्ष_आकार;
    ध्रुव पूर्णाङ्क n_seq_tokens = T / B;

    // Set up shared memory pointers
    भिन्न* _k = shared_mem;
    भिन्न* _r = _k + शीर्ष_आकार;
    भिन्न* _tf = _r + शीर्ष_आकार;
    भिन्न* _td = _tf + शीर्ष_आकार;

    // Local state array
    भिन्न state[खण्डम्_आकार_3];

    // Load initial state
    #निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < शीर्ष_आकार; i++) {
        state[i] = s[batch_i * state_आकार + शीर्ष_i * शीर्ष_आकार * शीर्ष_आकार + i * शीर्ष_आकार + tid];
    }

    // Sync threads before shared memory operations
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Load time-mixing parameters
    _tf[tid] = tf[शीर्ष_i * शीर्ष_आकार + tid];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Main sequence processing loop
    प्रति (पूर्णाङ्क t = batch_i * n_seq_tokens * C + शीर्ष_i * शीर्ष_आकार + tid;
         t < (batch_i + 1) * n_seq_tokens * C + शीर्ष_i * शीर्ष_आकार + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Load current timestep data to shared memory
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        ध्रुव भिन्न _v = v[t];
        भिन्न y = 0;

        // Process in chunks of 4 for better vectorization
        sycl::float4 k4, r4, tf4, td4, s4;
        #निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < शीर्ष_आकार; j += 4) {
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
    #निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < शीर्ष_आकार; i++) {
        dst[T * C + batch_i * state_आकार + शीर्ष_i * शीर्ष_आकार * शीर्ष_आकार + i * शीर्ष_आकार + tid] = state[i];
    }
}

प्रतिरूपम् <पूर्णाङ्क खण्डम्_आकार_3>
स्थिर शून्य rwkv_wkv7_f32_kernel(
    ध्रुव पूर्णाङ्क B, ध्रुव पूर्णाङ्क T, ध्रुव पूर्णाङ्क C, ध्रुव पूर्णाङ्क H,
    ध्रुव भिन्न* r, ध्रुव भिन्न* w, ध्रुव भिन्न* k, ध्रुव भिन्न* v,
    ध्रुव भिन्न* a, ध्रुव भिन्न* b, ध्रुव भिन्न* s,
    भिन्न* dst, ध्रुव sycl::nd_item<3>& item_ct1, भिन्न* shared_mem) {

    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2);
    ध्रुव पूर्णाङ्क bid = item_ct1.get_गणः(2);

    ध्रुव पूर्णाङ्क शीर्ष_आकार = खण्डम्_आकार_3;
    ध्रुव पूर्णाङ्क batch_i = bid / H;
    ध्रुव पूर्णाङ्क शीर्ष_i = bid % H;
    ध्रुव पूर्णाङ्क state_आकार = C * शीर्ष_आकार;
    ध्रुव पूर्णाङ्क n_seq_tokens = T / B;

    भिन्न* _r = shared_mem;
    भिन्न* _w = _r + शीर्ष_आकार;
    भिन्न* _k = _w + शीर्ष_आकार;
    भिन्न* _a = _k + शीर्ष_आकार;
    भिन्न* _b = _a + शीर्ष_आकार;

    भिन्न state[खण्डम्_आकार_3];

    #निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < शीर्ष_आकार; i++) {
        state[i] = s[batch_i * state_आकार + शीर्ष_i * शीर्ष_आकार * शीर्ष_आकार + tid * शीर्ष_आकार + i];
    }

    प्रति (पूर्णाङ्क t = batch_i * n_seq_tokens * C + शीर्ष_i * शीर्ष_आकार + tid;
         t < (batch_i + 1) * n_seq_tokens * C + शीर्ष_i * शीर्ष_आकार + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        ध्रुव भिन्न _v = v[t];
        भिन्न y = 0, sa = 0;
        sycl::float4 a4, s4;

        #निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < शीर्ष_आकार; j += 4) {
            a4 = sycl::float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa += sycl::dot(a4, s4);
        }

        sycl::float4 r4, w4, k4, b4;
        #निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < शीर्ष_आकार; j += 4) {
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

    #निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < शीर्ष_आकार; i++) {
        dst[T * C + batch_i * state_आकार + शीर्ष_i * शीर्ष_आकार * शीर्ष_आकार + tid * शीर्ष_आकार + i] = state[i];
    }
}

शून्य ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    ध्रुव भिन्न* k_d = (ध्रुव भिन्न*)dst->src[0]->दत्त;
    ध्रुव भिन्न* v_d = (ध्रुव भिन्न*)dst->src[1]->दत्त;
    ध्रुव भिन्न* r_d = (ध्रुव भिन्न*)dst->src[2]->दत्त;
    ध्रुव भिन्न* tf_d = (ध्रुव भिन्न*)dst->src[3]->दत्त;
    ध्रुव भिन्न* td_d = (ध्रुव भिन्न*)dst->src[4]->दत्त;
    ध्रुव भिन्न* s_d = (ध्रुव भिन्न*)dst->src[5]->दत्त;
    भिन्न* dst_d = (भिन्न*)dst->दत्त;

    ध्रुव int64_t B = dst->src[5]->ne[1];
    ध्रुव int64_t T = dst->src[0]->ne[2];
    ध्रुव int64_t C = dst->ne[0];
    ध्रुव int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_खण्डम्_आकार || C / H == WKV_खण्डम्_आकार * 2); // The current sycl kernel is designed for RWKV6, HEAD_SIZE == 64

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    ध्रुव size_t shared_mem_आकार = C / H * 4 * परिमाण(भिन्न); // For k, r, tf, td
    sycl::range<3> खण्डम्_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    यदि (C / H == WKV_खण्डम्_आकार) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<भिन्न, 1> shared_mem_acc(shared_mem_आकार, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_खण्डम्_आकार>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (भिन्न*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } अन्यथा {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<भिन्न, 1> shared_mem_acc(shared_mem_आकार, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_खण्डम्_आकार * 2>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (भिन्न*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}

शून्य ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    ध्रुव भिन्न* r_d = (ध्रुव भिन्न*)dst->src[0]->दत्त;
    ध्रुव भिन्न* w_d = (ध्रुव भिन्न*)dst->src[1]->दत्त;
    ध्रुव भिन्न* k_d = (ध्रुव भिन्न*)dst->src[2]->दत्त;
    ध्रुव भिन्न* v_d = (ध्रुव भिन्न*)dst->src[3]->दत्त;
    ध्रुव भिन्न* a_d = (ध्रुव भिन्न*)dst->src[4]->दत्त;
    ध्रुव भिन्न* b_d = (ध्रुव भिन्न*)dst->src[5]->दत्त;
    ध्रुव भिन्न* s_d = (ध्रुव भिन्न*)dst->src[6]->दत्त;
    भिन्न* dst_d = (भिन्न*)dst->दत्त;

    ध्रुव int64_t B = dst->src[6]->ne[1];
    ध्रुव int64_t T = dst->src[0]->ne[2];
    ध्रुव int64_t C = dst->ne[0];
    ध्रुव int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_खण्डम्_आकार || C / H == WKV_खण्डम्_आकार * 2);

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    ध्रुव size_t shared_mem_आकार = C / H * 5 * परिमाण(भिन्न); // For r, w, k, a, b
    sycl::range<3> खण्डम्_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    यदि (C / H == WKV_खण्डम्_आकार) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<भिन्न, 1> shared_mem_acc(shared_mem_आकार, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_खण्डम्_आकार>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (भिन्न*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    } अन्यथा {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<भिन्न, 1> shared_mem_acc(shared_mem_आकार, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_खण्डम्_आकार * 2>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (भिन्न*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().get()
                    );
                });
        });
    }
}
