#ضمن <sycl/sycl.hpp>

#ضمن "common.hpp"
#ضمن "gla.hpp"

قالب <u_int سر_اندازه_2>
ساكن فراغ gated_خطی_attn_f32_kernel(ثابت dpct::صف_ptr stream, u_int B, u_int T, u_int C, u_int H, عائم scale,
                                         ثابت عائم * k, ثابت عائم * v, ثابت عائم * r, ثابت عائم * td,
                                         ثابت عائم * s, عائم * dst) {
    ثابت u_int سر_اندازه    = سر_اندازه_2;
    ثابت u_int state_اندازه   = C * سر_اندازه;
    ثابت u_int n_seq_نشانهها = T / B;
    sycl::range<1> بلوک_dims((C / H));
    sycl::range<1> grid_dims((B * H));
    stream->submit([&](sycl::handler & cgh) {
        /* local memory accessors*/
        خودکار _k  = sycl::local_accessor<عائم, 1>(sycl::range<1>(سر_اندازه), cgh);
        خودکار _r  = sycl::local_accessor<عائم, 1>(sycl::range<1>(سر_اندازه), cgh);
        خودکار _td = sycl::local_accessor<عائم, 1>(sycl::range<1>(سر_اندازه), cgh);

        cgh.parallel_for(sycl::nd_range<1>(grid_dims * بلوک_dims, بلوک_dims), [=](sycl::nd_قلم<1> قلم) {
            u_int tid = قلم.get_local_id(0);
            u_int bid = قلم.get_گروه(0);

            u_int batch_i = bid / H;
            u_int سر_i  = bid % H;

            عائم state[سر_اندازه];

#رهنمود unroll
            لكل (u_int i = 0; i < سر_اندازه; i++) {
                state[i] = s[batch_i * state_اندازه + سر_i * سر_اندازه * سر_اندازه + i * سر_اندازه + tid];
            }

            لكل (u_int t = batch_i * n_seq_نشانهها * C + سر_i * سر_اندازه + tid;
                 t < (batch_i + 1) * n_seq_نشانهها * C + سر_i * سر_اندازه + tid; t += C) {

                قلم.barrier(sycl::access::fence_space::local_space);  //sync threads
                _k[tid]  = k[t];
                _r[tid]  = r[t];
                _td[tid] = td[t];
                قلم.barrier(sycl::access::fence_space::local_space);  //sync threads

                ثابت عائم _v = v[t];
                عائم       y  = 0;

                لكل (u_int j = 0; j < سر_اندازه; j += 4) {
                    ثابت sycl::float4 & k  = (sycl::float4 &) (_k[j]);
                    ثابت sycl::float4 & r  = (sycl::float4 &) (_r[j]);
                    ثابت sycl::float4 & td = (sycl::float4 &) (_td[j]);
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
#رهنمود unroll
            لكل (u_int i = 0; i < سر_اندازه; i++) {
                dst[T * C + batch_i * state_اندازه + سر_i * سر_اندازه * سر_اندازه + i * سر_اندازه + tid] = state[i];
            }
        });
    });
}

فراغ ggml_sycl_op_gated_خطی_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);
    ثابت عائم * k_d  = static_cast<ثابت عائم *>(dst->src[0]->داده);
    ثابت عائم * v_d  = static_cast<ثابت عائم *>(dst->src[1]->داده);
    ثابت عائم * r_d  = static_cast<ثابت عائم *>(dst->src[2]->داده);
    ثابت عائم * td_d = static_cast<ثابت عائم *>(dst->src[3]->داده);
    ثابت عائم * s_d  = static_cast<ثابت عائم *>(dst->src[4]->داده);

    ثابت int64_t B = dst->src[4]->ne[1];
    ثابت int64_t T = dst->src[0]->ne[2];
    ثابت int64_t C = dst->ne[0];
    ثابت int64_t H = dst->src[0]->ne[1];

    dpct::صف_ptr stream = ctx.stream();
    GGML_ASSERT(dst->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == 64 || C / H == 128);

    عائم scale;
    کپی_حافظه(&scale, dst->op_params, حجم(عائم));

    عائم * dst_d = (عائم *) dst->داده;

    إذا (C / H == 64) {
        gated_خطی_attn_f32_kernel<64>(stream, B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    } وإلا {
        gated_خطی_attn_f32_kernel<128>(stream, B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    }
}
