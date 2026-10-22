#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>

#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "gla.hpp"

ಮಾದರಿ <u_int ತಲೆ_ಗಾತ್ರ_2>
ಸ್ಥಿರ ಶೂನ್ಯ gated_ರೇಖೀಯ_attn_f32_kernel(ಸ್ಥಿರಾಂಕ dpct::ಸರತಿ_ptr stream, u_int B, u_int T, u_int C, u_int H, ದಶಮಾಂಶ scale,
                                         ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * v, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * r, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * td,
                                         ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s, ದಶಮಾಂಶ * dst) {
    ಸ್ಥಿರಾಂಕ u_int ತಲೆ_ಗಾತ್ರ    = ತಲೆ_ಗಾತ್ರ_2;
    ಸ್ಥಿರಾಂಕ u_int state_ಗಾತ್ರ   = C * ತಲೆ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ u_int n_seq_ಟೋಕನುಗಳು = T / B;
    sycl::range<1> ಖಂಡ_dims((C / H));
    sycl::range<1> grid_dims((B * H));
    stream->submit([&](sycl::handler & cgh) {
        /* local memory accessors*/
        ಸ್ವಯಂಚಾಲಿತ _k  = sycl::local_accessor<ದಶಮಾಂಶ, 1>(sycl::range<1>(ತಲೆ_ಗಾತ್ರ), cgh);
        ಸ್ವಯಂಚಾಲಿತ _r  = sycl::local_accessor<ದಶಮಾಂಶ, 1>(sycl::range<1>(ತಲೆ_ಗಾತ್ರ), cgh);
        ಸ್ವಯಂಚಾಲಿತ _td = sycl::local_accessor<ದಶಮಾಂಶ, 1>(sycl::range<1>(ತಲೆ_ಗಾತ್ರ), cgh);

        cgh.parallel_for(sycl::nd_range<1>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims), [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) {
            u_int tid = ಅಂಶ.get_local_id(0);
            u_int bid = ಅಂಶ.get_ಗುಂಪು(0);

            u_int batch_i = bid / H;
            u_int ತಲೆ_i  = bid % H;

            ದಶಮಾಂಶ state[ತಲೆ_ಗಾತ್ರ];

#ಸೂಚನೆ unroll
            ಗಾಗಿ (u_int i = 0; i < ತಲೆ_ಗಾತ್ರ; i++) {
                state[i] = s[batch_i * state_ಗಾತ್ರ + ತಲೆ_i * ತಲೆ_ಗಾತ್ರ * ತಲೆ_ಗಾತ್ರ + i * ತಲೆ_ಗಾತ್ರ + tid];
            }

            ಗಾಗಿ (u_int t = batch_i * n_seq_ಟೋಕನುಗಳು * C + ತಲೆ_i * ತಲೆ_ಗಾತ್ರ + tid;
                 t < (batch_i + 1) * n_seq_ಟೋಕನುಗಳು * C + ತಲೆ_i * ತಲೆ_ಗಾತ್ರ + tid; t += C) {

                ಅಂಶ.barrier(sycl::access::fence_space::local_space);  //sync threads
                _k[tid]  = k[t];
                _r[tid]  = r[t];
                _td[tid] = td[t];
                ಅಂಶ.barrier(sycl::access::fence_space::local_space);  //sync threads

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ _v = v[t];
                ದಶಮಾಂಶ       y  = 0;

                ಗಾಗಿ (u_int j = 0; j < ತಲೆ_ಗಾತ್ರ; j += 4) {
                    ಸ್ಥಿರಾಂಕ sycl::float4 & k  = (sycl::float4 &) (_k[j]);
                    ಸ್ಥಿರಾಂಕ sycl::float4 & r  = (sycl::float4 &) (_r[j]);
                    ಸ್ಥಿರಾಂಕ sycl::float4 & td = (sycl::float4 &) (_td[j]);
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
#ಸೂಚನೆ unroll
            ಗಾಗಿ (u_int i = 0; i < ತಲೆ_ಗಾತ್ರ; i++) {
                dst[T * C + batch_i * state_ಗಾತ್ರ + ತಲೆ_i * ತಲೆ_ಗಾತ್ರ * ತಲೆ_ಗಾತ್ರ + i * ತಲೆ_ಗಾತ್ರ + tid] = state[i];
            }
        });
    });
}

ಶೂನ್ಯ ggml_sycl_op_gated_ರೇಖೀಯ_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * k_d  = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[0]->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * v_d  = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[1]->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * r_d  = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[2]->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * td_d = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[3]->ದತ್ತಾಂಶ);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * s_d  = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[4]->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ int64_t B = dst->src[4]->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t T = dst->src[0]->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t C = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t H = dst->src[0]->ne[1];

    dpct::ಸರತಿ_ptr stream = ctx.stream();
    GGML_ASSERT(dst->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == 64 || C / H == 128);

    ದಶಮಾಂಶ scale;
    ಸ್ಮರಣೆ_ನಕಲು(&scale, dst->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    ದಶಮಾಂಶ * dst_d = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (C / H == 64) {
        gated_ರೇಖೀಯ_attn_f32_kernel<64>(stream, B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    } ಇಲ್ಲದಿದ್ದರೆ {
        gated_ರೇಖೀಯ_attn_f32_kernel<128>(stream, B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    }
}
