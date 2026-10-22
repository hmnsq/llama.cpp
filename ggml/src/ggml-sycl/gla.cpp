#అన్తర్భావ <sycl/sycl.hpp>

#అన్తర్భావ "common.hpp"
#అన్తర్భావ "gla.hpp"

మూస <u_int తల_కొలత_2>
స్థిర శూన్య gated_రేఖీయ_attn_f32_kernel(స్థిరరాశి dpct::వరుస_ptr stream, u_int B, u_int T, u_int C, u_int H, దశాంశం scale,
                                         స్థిరరాశి దశాంశం * k, స్థిరరాశి దశాంశం * v, స్థిరరాశి దశాంశం * r, స్థిరరాశి దశాంశం * td,
                                         స్థిరరాశి దశాంశం * s, దశాంశం * dst) {
    స్థిరరాశి u_int తల_కొలత    = తల_కొలత_2;
    స్థిరరాశి u_int state_కొలత   = C * తల_కొలత;
    స్థిరరాశి u_int n_seq_టోకెన్లు = T / B;
    sycl::range<1> ఖండం_dims((C / H));
    sycl::range<1> grid_dims((B * H));
    stream->submit([&](sycl::handler & cgh) {
        /* local memory accessors*/
        స్వయంచాలక _k  = sycl::local_accessor<దశాంశం, 1>(sycl::range<1>(తల_కొలత), cgh);
        స్వయంచాలక _r  = sycl::local_accessor<దశాంశం, 1>(sycl::range<1>(తల_కొలత), cgh);
        స్వయంచాలక _td = sycl::local_accessor<దశాంశం, 1>(sycl::range<1>(తల_కొలత), cgh);

        cgh.parallel_for(sycl::nd_range<1>(grid_dims * ఖండం_dims, ఖండం_dims), [=](sycl::nd_అంశం<1> అంశం) {
            u_int tid = అంశం.get_local_id(0);
            u_int bid = అంశం.get_సమూహం(0);

            u_int batch_i = bid / H;
            u_int తల_i  = bid % H;

            దశాంశం state[తల_కొలత];

#సూచన unroll
            కొరకు (u_int i = 0; i < తల_కొలత; i++) {
                state[i] = s[batch_i * state_కొలత + తల_i * తల_కొలత * తల_కొలత + i * తల_కొలత + tid];
            }

            కొరకు (u_int t = batch_i * n_seq_టోకెన్లు * C + తల_i * తల_కొలత + tid;
                 t < (batch_i + 1) * n_seq_టోకెన్లు * C + తల_i * తల_కొలత + tid; t += C) {

                అంశం.barrier(sycl::access::fence_space::local_space);  //sync threads
                _k[tid]  = k[t];
                _r[tid]  = r[t];
                _td[tid] = td[t];
                అంశం.barrier(sycl::access::fence_space::local_space);  //sync threads

                స్థిరరాశి దశాంశం _v = v[t];
                దశాంశం       y  = 0;

                కొరకు (u_int j = 0; j < తల_కొలత; j += 4) {
                    స్థిరరాశి sycl::float4 & k  = (sycl::float4 &) (_k[j]);
                    స్థిరరాశి sycl::float4 & r  = (sycl::float4 &) (_r[j]);
                    స్థిరరాశి sycl::float4 & td = (sycl::float4 &) (_td[j]);
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
#సూచన unroll
            కొరకు (u_int i = 0; i < తల_కొలత; i++) {
                dst[T * C + batch_i * state_కొలత + తల_i * తల_కొలత * తల_కొలత + i * తల_కొలత + tid] = state[i];
            }
        });
    });
}

శూన్య ggml_sycl_op_gated_రేఖీయ_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);
    స్థిరరాశి దశాంశం * k_d  = static_cast<స్థిరరాశి దశాంశం *>(dst->src[0]->దత్తాంశం);
    స్థిరరాశి దశాంశం * v_d  = static_cast<స్థిరరాశి దశాంశం *>(dst->src[1]->దత్తాంశం);
    స్థిరరాశి దశాంశం * r_d  = static_cast<స్థిరరాశి దశాంశం *>(dst->src[2]->దత్తాంశం);
    స్థిరరాశి దశాంశం * td_d = static_cast<స్థిరరాశి దశాంశం *>(dst->src[3]->దత్తాంశం);
    స్థిరరాశి దశాంశం * s_d  = static_cast<స్థిరరాశి దశాంశం *>(dst->src[4]->దత్తాంశం);

    స్థిరరాశి int64_t B = dst->src[4]->ne[1];
    స్థిరరాశి int64_t T = dst->src[0]->ne[2];
    స్థిరరాశి int64_t C = dst->ne[0];
    స్థిరరాశి int64_t H = dst->src[0]->ne[1];

    dpct::వరుస_ptr stream = ctx.stream();
    GGML_ASSERT(dst->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == 64 || C / H == 128);

    దశాంశం scale;
    జ్ఞాపకం_నకలు(&scale, dst->op_params, పరిమాణం(దశాంశం));

    దశాంశం * dst_d = (దశాంశం *) dst->దత్తాంశం;

    ఒకవేళ (C / H == 64) {
        gated_రేఖీయ_attn_f32_kernel<64>(stream, B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    } లేకపోతే {
        gated_రేఖీయ_attn_f32_kernel<128>(stream, B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    }
}
