#समावेश <sycl/sycl.hpp>
#समावेश "wkv.hpp"

constexpr पूर्णांक WKV_BLOCK_आकार = 64;

// Helper function for the main kernel
ढाँचा <पूर्णांक block_आकार_2>
अटल व्योम rwkv_wkv6_f32_kernel(
    स्थिर पूर्णांक B, स्थिर पूर्णांक T, स्थिर पूर्णांक C, स्थिर पूर्णांक H,
    स्थिर भग्न* k, स्थिर भग्न* v, स्थिर भग्न* r,
    स्थिर भग्न* tf, स्थिर भग्न* td, स्थिर भग्न* s,
    भग्न* dst, स्थिर sycl::nd_item<3>& item_ct1, भग्न* shared_mem) {

    स्थिर पूर्णांक tid = item_ct1.get_local_id(2);
    स्थिर पूर्णांक bid = item_ct1.get_group(2);

    स्थिर पूर्णांक शीर्ष_आकार = block_आकार_2;
    स्थिर पूर्णांक batch_i = bid / H;
    स्थिर पूर्णांक शीर्ष_i = bid % H;
    स्थिर पूर्णांक state_आकार = C * शीर्ष_आकार;
    स्थिर पूर्णांक n_seq_tokens = T / B;

    // Set up shared memory pointers
    भग्न* _k = shared_mem;
    भग्न* _r = _k + शीर्ष_आकार;
    भग्न* _tf = _r + शीर्ष_आकार;
    भग्न* _td = _tf + शीर्ष_आकार;

    // Local state array
    भग्न state[block_आकार_2];

    // Load initial state
    #आशय unroll
    क्रम (पूर्णांक i = 0; i < शीर्ष_आकार; i++) {
        state[i] = s[batch_i * state_आकार + शीर्ष_i * शीर्ष_आकार * शीर्ष_आकार + i * शीर्ष_आकार + tid];
    }

    // Sync threads before shared memory operations
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Load time-mixing parameters
    _tf[tid] = tf[शीर्ष_i * शीर्ष_आकार + tid];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // Main sequence processing loop
    क्रम (पूर्णांक t = batch_i * n_seq_tokens * C + शीर्ष_i * शीर्ष_आकार + tid;
         t < (batch_i + 1) * n_seq_tokens * C + शीर्ष_i * शीर्ष_आकार + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        // Load current timestep data to shared memory
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        स्थिर भग्न _v = v[t];
        भग्न y = 0;

        // Process in chunks of 4 for better vectorization
        sycl::float4 k4, r4, tf4, td4, s4;
        #आशय unroll
        क्रम (पूर्णांक j = 0; j < शीर्ष_आकार; j += 4) {
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
    #आशय unroll
    क्रम (पूर्णांक i = 0; i < शीर्ष_आकार; i++) {
        dst[T * C + batch_i * state_आकार + शीर्ष_i * शीर्ष_आकार * शीर्ष_आकार + i * शीर्ष_आकार + tid] = state[i];
    }
}

ढाँचा <पूर्णांक block_आकार_2>
अटल व्योम rwkv_wkv7_f32_kernel(
    स्थिर पूर्णांक B, स्थिर पूर्णांक T, स्थिर पूर्णांक C, स्थिर पूर्णांक H,
    स्थिर भग्न* r, स्थिर भग्न* w, स्थिर भग्न* k, स्थिर भग्न* v,
    स्थिर भग्न* a, स्थिर भग्न* b, स्थिर भग्न* s,
    भग्न* dst, स्थिर sycl::nd_item<3>& item_ct1, भग्न* shared_mem) {

    स्थिर पूर्णांक tid = item_ct1.get_local_id(2);
    स्थिर पूर्णांक bid = item_ct1.get_group(2);

    स्थिर पूर्णांक शीर्ष_आकार = block_आकार_2;
    स्थिर पूर्णांक batch_i = bid / H;
    स्थिर पूर्णांक शीर्ष_i = bid % H;
    स्थिर पूर्णांक state_आकार = C * शीर्ष_आकार;
    स्थिर पूर्णांक n_seq_tokens = T / B;

    भग्न* _r = shared_mem;
    भग्न* _w = _r + शीर्ष_आकार;
    भग्न* _k = _w + शीर्ष_आकार;
    भग्न* _a = _k + शीर्ष_आकार;
    भग्न* _b = _a + शीर्ष_आकार;

    भग्न state[block_आकार_2];

    #आशय unroll
    क्रम (पूर्णांक i = 0; i < शीर्ष_आकार; i++) {
        state[i] = s[batch_i * state_आकार + शीर्ष_i * शीर्ष_आकार * शीर्ष_आकार + tid * शीर्ष_आकार + i];
    }

    क्रम (पूर्णांक t = batch_i * n_seq_tokens * C + शीर्ष_i * शीर्ष_आकार + tid;
         t < (batch_i + 1) * n_seq_tokens * C + शीर्ष_i * शीर्ष_आकार + tid;
         t += C) {

        item_ct1.barrier(sycl::access::fence_space::local_space);

        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];

        item_ct1.barrier(sycl::access::fence_space::local_space);

        स्थिर भग्न _v = v[t];
        भग्न y = 0, sa = 0;
        sycl::float4 a4, s4;

        #आशय unroll
        क्रम (पूर्णांक j = 0; j < शीर्ष_आकार; j += 4) {
            a4 = sycl::float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            s4 = sycl::float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa += sycl::dot(a4, s4);
        }

        sycl::float4 r4, w4, k4, b4;
        #आशय unroll
        क्रम (पूर्णांक j = 0; j < शीर्ष_आकार; j += 4) {
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

    #आशय unroll
    क्रम (पूर्णांक i = 0; i < शीर्ष_आकार; i++) {
        dst[T * C + batch_i * state_आकार + शीर्ष_i * शीर्ष_आकार * शीर्ष_आकार + tid * शीर्ष_आकार + i] = state[i];
    }
}

व्योम ggml_sycl_op_rwkv_wkv6(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/6);
    स्थिर भग्न* k_d = (स्थिर भग्न*)dst->src[0]->तथ्य;
    स्थिर भग्न* v_d = (स्थिर भग्न*)dst->src[1]->तथ्य;
    स्थिर भग्न* r_d = (स्थिर भग्न*)dst->src[2]->तथ्य;
    स्थिर भग्न* tf_d = (स्थिर भग्न*)dst->src[3]->तथ्य;
    स्थिर भग्न* td_d = (स्थिर भग्न*)dst->src[4]->तथ्य;
    स्थिर भग्न* s_d = (स्थिर भग्न*)dst->src[5]->तथ्य;
    भग्न* dst_d = (भग्न*)dst->तथ्य;

    स्थिर int64_t B = dst->src[5]->ne[1];
    स्थिर int64_t T = dst->src[0]->ne[2];
    स्थिर int64_t C = dst->ne[0];
    स्थिर int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_आकार || C / H == WKV_BLOCK_आकार * 2); // The current sycl kernel is designed for RWKV6, HEAD_SIZE == 64

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    स्थिर माप_प्रकार shared_mem_आकार = C / H * 4 * माप(भग्न); // For k, r, tf, td
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    अगर (C / H == WKV_BLOCK_आकार) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<भग्न, 1> shared_mem_acc(shared_mem_आकार, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_BLOCK_आकार>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (भग्न*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().लो()
                    );
                });
        });
    } अन्यथा {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<भग्न, 1> shared_mem_acc(shared_mem_आकार, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv6_f32_kernel<WKV_BLOCK_आकार * 2>(
                        B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d,
                        item_ct1, (भग्न*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().लो()
                    );
                });
        });
    }
}

व्योम ggml_sycl_op_rwkv_wkv7(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/7);
    स्थिर भग्न* r_d = (स्थिर भग्न*)dst->src[0]->तथ्य;
    स्थिर भग्न* w_d = (स्थिर भग्न*)dst->src[1]->तथ्य;
    स्थिर भग्न* k_d = (स्थिर भग्न*)dst->src[2]->तथ्य;
    स्थिर भग्न* v_d = (स्थिर भग्न*)dst->src[3]->तथ्य;
    स्थिर भग्न* a_d = (स्थिर भग्न*)dst->src[4]->तथ्य;
    स्थिर भग्न* b_d = (स्थिर भग्न*)dst->src[5]->तथ्य;
    स्थिर भग्न* s_d = (स्थिर भग्न*)dst->src[6]->तथ्य;
    भग्न* dst_d = (भग्न*)dst->तथ्य;

    स्थिर int64_t B = dst->src[6]->ne[1];
    स्थिर int64_t T = dst->src[0]->ne[2];
    स्थिर int64_t C = dst->ne[0];
    स्थिर int64_t H = dst->src[0]->ne[1];

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == WKV_BLOCK_आकार || C / H == WKV_BLOCK_आकार * 2);

    dpct::queue_ptr stream = ctx.stream();

    // Calculate execution configuration
    स्थिर माप_प्रकार shared_mem_आकार = C / H * 5 * माप(भग्न); // For r, w, k, a, b
    sycl::range<3> block_dims(1, 1, C / H);
    sycl::range<3> grid_dims(1, 1, B * H);

    // Submit kernel
    अगर (C / H == WKV_BLOCK_आकार) {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<भग्न, 1> shared_mem_acc(shared_mem_आकार, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_BLOCK_आकार>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (भग्न*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().लो()
                    );
                });
        });
    } अन्यथा {
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<भग्न, 1> shared_mem_acc(shared_mem_आकार, cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    rwkv_wkv7_f32_kernel<WKV_BLOCK_आकार * 2>(
                        B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d,
                        item_ct1, (भग्न*)shared_mem_acc.get_multi_ptr<sycl::access::decorated::no>().लो()
                    );
                });
        });
    }
}
