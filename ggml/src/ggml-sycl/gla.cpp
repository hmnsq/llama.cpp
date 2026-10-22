#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "gla.hpp"

ਸਾਂਚਾ <u_int ਸਿਰ_ਮਾਪ_2>
ਸਥਿਰ ਖਾਲੀ gated_ਰੇਖਿਕ_attn_f32_kernel(ਅਚਲ dpct::ਕਤਾਰ_ptr stream, u_int B, u_int T, u_int C, u_int H, ਦਸ਼ਮਲਵ scale,
                                         ਅਚਲ ਦਸ਼ਮਲਵ * k, ਅਚਲ ਦਸ਼ਮਲਵ * v, ਅਚਲ ਦਸ਼ਮਲਵ * r, ਅਚਲ ਦਸ਼ਮਲਵ * td,
                                         ਅਚਲ ਦਸ਼ਮਲਵ * s, ਦਸ਼ਮਲਵ * dst) {
    ਅਚਲ u_int ਸਿਰ_ਮਾਪ    = ਸਿਰ_ਮਾਪ_2;
    ਅਚਲ u_int state_ਮਾਪ   = C * ਸਿਰ_ਮਾਪ;
    ਅਚਲ u_int n_seq_ਟੋਕਨ = T / B;
    sycl::range<1> ਖੰਡ_dims((C / H));
    sycl::range<1> grid_dims((B * H));
    stream->submit([&](sycl::handler & cgh) {
        /* local memory accessors*/
        ਸਵੈਚਾਲਿਤ _k  = sycl::local_accessor<ਦਸ਼ਮਲਵ, 1>(sycl::range<1>(ਸਿਰ_ਮਾਪ), cgh);
        ਸਵੈਚਾਲਿਤ _r  = sycl::local_accessor<ਦਸ਼ਮਲਵ, 1>(sycl::range<1>(ਸਿਰ_ਮਾਪ), cgh);
        ਸਵੈਚਾਲਿਤ _td = sycl::local_accessor<ਦਸ਼ਮਲਵ, 1>(sycl::range<1>(ਸਿਰ_ਮਾਪ), cgh);

        cgh.parallel_for(sycl::nd_range<1>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims), [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) {
            u_int tid = ਵਸਤੂ.get_local_id(0);
            u_int bid = ਵਸਤੂ.get_ਸਮੂਹ(0);

            u_int batch_i = bid / H;
            u_int ਸਿਰ_i  = bid % H;

            ਦਸ਼ਮਲਵ state[ਸਿਰ_ਮਾਪ];

#ਹਦਾਇਤ unroll
            ਲਈ (u_int i = 0; i < ਸਿਰ_ਮਾਪ; i++) {
                state[i] = s[batch_i * state_ਮਾਪ + ਸਿਰ_i * ਸਿਰ_ਮਾਪ * ਸਿਰ_ਮਾਪ + i * ਸਿਰ_ਮਾਪ + tid];
            }

            ਲਈ (u_int t = batch_i * n_seq_ਟੋਕਨ * C + ਸਿਰ_i * ਸਿਰ_ਮਾਪ + tid;
                 t < (batch_i + 1) * n_seq_ਟੋਕਨ * C + ਸਿਰ_i * ਸਿਰ_ਮਾਪ + tid; t += C) {

                ਵਸਤੂ.barrier(sycl::access::fence_space::local_space);  //sync threads
                _k[tid]  = k[t];
                _r[tid]  = r[t];
                _td[tid] = td[t];
                ਵਸਤੂ.barrier(sycl::access::fence_space::local_space);  //sync threads

                ਅਚਲ ਦਸ਼ਮਲਵ _v = v[t];
                ਦਸ਼ਮਲਵ       y  = 0;

                ਲਈ (u_int j = 0; j < ਸਿਰ_ਮਾਪ; j += 4) {
                    ਅਚਲ sycl::float4 & k  = (sycl::float4 &) (_k[j]);
                    ਅਚਲ sycl::float4 & r  = (sycl::float4 &) (_r[j]);
                    ਅਚਲ sycl::float4 & td = (sycl::float4 &) (_td[j]);
                    sycl::float4 &       s  = (sycl::float4 &) (state[j]);
                    sycl::float4         kv;

                    kv.x() = k.x() * _v;
                    kv.y() = k.y() * _v;
                    kv.z() = k.z() * _v;
                    kv.w() = k.w() * _v;

                    s.x() = s.x() * td.x() + kv.x();
                    s.y() = s.y() * td.y() + kv.y();
                    s.z() = s.z() * td.z() + kv.z();
                    s.w() = s.w() * td.w() + kv.w();

                    y += r.x() * s.x();
                    y += r.y() * s.y();
                    y += r.z() * s.z();
                    y += r.w() * s.w();
                }
                dst[t] = y * scale;
            }
#ਹਦਾਇਤ unroll
            ਲਈ (u_int i = 0; i < ਸਿਰ_ਮਾਪ; i++) {
                dst[T * C + batch_i * state_ਮਾਪ + ਸਿਰ_i * ਸਿਰ_ਮਾਪ * ਸਿਰ_ਮਾਪ + i * ਸਿਰ_ਮਾਪ + tid] = state[i];
            }
        });
    });
}

ਖਾਲੀ ggml_sycl_op_gated_ਰੇਖਿਕ_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);
    ਅਚਲ ਦਸ਼ਮਲਵ * k_d  = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[0]->ਅੰਕੜੇ);
    ਅਚਲ ਦਸ਼ਮਲਵ * v_d  = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[1]->ਅੰਕੜੇ);
    ਅਚਲ ਦਸ਼ਮਲਵ * r_d  = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[2]->ਅੰਕੜੇ);
    ਅਚਲ ਦਸ਼ਮਲਵ * td_d = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[3]->ਅੰਕੜੇ);
    ਅਚਲ ਦਸ਼ਮਲਵ * s_d  = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[4]->ਅੰਕੜੇ);

    ਅਚਲ int64_t B = dst->src[4]->ne[1];
    ਅਚਲ int64_t T = dst->src[0]->ne[2];
    ਅਚਲ int64_t C = dst->ne[0];
    ਅਚਲ int64_t H = dst->src[0]->ne[1];

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();
    GGML_ASSERT(dst->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == 64 || C / H == 128);

    ਦਸ਼ਮਲਵ scale;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, dst->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਦਸ਼ਮਲਵ * dst_d = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਜੇ (C / H == 64) {
        gated_ਰੇਖਿਕ_attn_f32_kernel<64>(stream, B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    } ਨਹੀਂਤਾਂ {
        gated_ਰੇਖਿਕ_attn_f32_kernel<128>(stream, B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    }
}
