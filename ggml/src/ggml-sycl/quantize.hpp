/***************************************************************************
 *
 *  Copyright (C) 2025 Codeplay Software Ltd.
 *  Copyright (C) 2025 Intel Corporation
 *
 *  MIT License
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  quantize.hpp
 *
 *  Description:
 *     Sycl backend specific quantization functions
 **************************************************************************/

#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/nd_item.hpp>

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/dpct/helper.hpp"

ਸਾਂਚਾ <ਪੂਰਨ Elements_ਪ੍ਰਤੀ_WI>
__dpct_inline__ ਸਥਿਰ ਖਾਲੀ quantize_q8_1_impl(ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ x,
                                               sycl::vec<int8_t, Elements_ਪ੍ਰਤੀ_WI> & quantized_ਮੁੱਲਾਂ, ਦਸ਼ਮਲਵ & d,
                                               ਦਸ਼ਮਲਵ & ਜੋੜ, ਅਚਲ sycl::nd_ਵਸਤੂ<1> & it) {
    ਸਵੈਚਾਲਿਤ subgroup_id = it.get_ਸਮੂਹ(0);
    ਸਵੈਚਾਲਿਤ wi_id       = it.get_local_id(0);

    sycl::vec<ਦਸ਼ਮਲਵ, Elements_ਪ੍ਰਤੀ_WI> wi_f32_vals;

    ਸਵੈਚਾਲਿਤ float_ptr_ਵਿੱਥ = subgroup_id * QK8_1 + Elements_ਪ੍ਰਤੀ_WI * wi_id;
    wi_f32_vals           = *reinterpret_cast<ਅਚਲ sycl::vec<ਦਸ਼ਮਲਵ, Elements_ਪ੍ਰਤੀ_WI> *>(x + float_ptr_ਵਿੱਥ);

    ਦਸ਼ਮਲਵ amax = 0.0f;

#ਹਦਾਇਤ unroll(Elements_ਪ੍ਰਤੀ_WI)
    ਲਈ (ਪੂਰਨ i = 0; i < Elements_ਪ੍ਰਤੀ_WI; i++) {
        ਜੋੜ += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_ਮੁੱਲਾਂ[i] = 0;
    }
    ਜੋੜ  = sycl::reduce_over_ਸਮੂਹ(it.get_sub_ਸਮੂਹ(), ਜੋੜ, sycl::plus<ਦਸ਼ਮਲਵ>());
    amax = sycl::reduce_over_ਸਮੂਹ(it.get_sub_ਸਮੂਹ(), amax, sycl::ਵੱਧ_ਤੋਂ_ਵੱਧ<ਦਸ਼ਮਲਵ>());
    d    = amax == 0 ? 1 : amax / 127;

#ਹਦਾਇਤ unroll(Elements_ਪ੍ਰਤੀ_WI)
    ਲਈ (ਪੂਰਨ i = 0; i < Elements_ਪ੍ਰਤੀ_WI; i++) {
        quantized_ਮੁੱਲਾਂ[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
ਸਾਂਚਾ <ਪੂਰਨ Elements_ਪ੍ਰਤੀ_WI> ਬਣਤਰ no_quantize_q8_1 {
    ਖਾਲੀ ਸੰਚਾਲਕ()(ਅਚਲ ਦਸ਼ਮਲਵ *, ਖਾਲੀ *, ਪੂਰਨ, ਪੂਰਨ, ਅਚਲ sycl::nd_ਵਸਤੂ<1> &) ਅਚਲ {}
};

ਸਾਂਚਾ <ਪੂਰਨ Elements_ਪ੍ਰਤੀ_WI> ਬਣਤਰ quantize_and_reorder_q8_1_soa {
    __dpct_inline__ ਖਾਲੀ ਸੰਚਾਲਕ()(ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ x, ਖਾਲੀ * reordered_q8_tensor, ਅਚਲ ਪੂਰਨ kx,
                                    ਅਚਲ ਪੂਰਨ kx_padded, ਅਚਲ sycl::nd_ਵਸਤੂ<1> & it) ਅਚਲ {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        ਸਵੈਚਾਲਿਤ subgroup_id = it.get_ਸਮੂਹ(0);
        ਸਵੈਚਾਲਿਤ wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, Elements_ਪ੍ਰਤੀ_WI> quantized_ਮੁੱਲਾਂ;
        ਦਸ਼ਮਲਵ                            d   = 0.0f;
        ਦਸ਼ਮਲਵ                            ਜੋੜ = 0.0f;
        quantize_q8_1_impl<Elements_ਪ੍ਰਤੀ_WI>(x, quantized_ਮੁੱਲਾਂ, d, ਜੋੜ, it);

        ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = kx / QK8_1;
        ਸਵੈਚਾਲਿਤ      ਪੰਕਤੀ                = subgroup_id / ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        ਸਵੈਚਾਲਿਤ      ਖਾਨਾ                = subgroup_id % ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        ਸਵੈਚਾਲਿਤ      ਪੰਕਤੀ_ਵਿੱਥ         = ਪੰਕਤੀ * (kx_padded / QK8_1) * ਆਕਾਰ(ਖੰਡ_q8_1);
        ਸਵੈਚਾਲਿਤ      ਖਾਨਾ_ਵਿੱਥ         = QK8_1 * ਖਾਨਾ + wi_id * Elements_ਪ੍ਰਤੀ_WI;

        ਸਵੈਚਾਲਿਤ quant_ptr = (int8_t *) ((ਅੱਖਰ *) reordered_q8_tensor + ਪੰਕਤੀ_ਵਿੱਥ + ਖਾਨਾ_ਵਿੱਥ);
        *reinterpret_cast<sycl::vec<int8_t, Elements_ਪ੍ਰਤੀ_WI> *>(quant_ptr) = quantized_ਮੁੱਲਾਂ;

        ਸਵੈਚਾਲਿਤ ds_ptr = (sycl::half2 *) ((ਅੱਖਰ *) reordered_q8_tensor + ਪੰਕਤੀ_ਵਿੱਥ + kx + ਖਾਨਾ * ਆਕਾਰ(sycl::half2));
        ਜੇ (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(ਜੋੜ));
        }
    }
};

ਸਾਂਚਾ <ਪੂਰਨ Elements_ਪ੍ਰਤੀ_WI> ਬਣਤਰ quantize_q8_1 {
    __dpct_inline__ ਖਾਲੀ ਸੰਚਾਲਕ()(ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ x, ਖਾਲੀ * q8_tensor, ਅਚਲ ਪੂਰਨ kx, ਅਚਲ ਪੂਰਨ kx_padded,
                                    ਅਚਲ sycl::nd_ਵਸਤੂ<1> & it) ਅਚਲ {
        ਸਵੈਚਾਲਿਤ subgroup_id = it.get_ਸਮੂਹ(0);
        ਸਵੈਚਾਲਿਤ wi_id       = it.get_local_id(0);

        ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ = kx / QK8_1;
        ਸਵੈਚਾਲਿਤ      ਪੰਕਤੀ                = subgroup_id / ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        ਅਚਲ ਪੂਰਨ pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, Elements_ਪ੍ਰਤੀ_WI> quantized_ਮੁੱਲਾਂ;
        ਦਸ਼ਮਲਵ                            d   = 0.0f;
        ਦਸ਼ਮਲਵ                            ਜੋੜ = 0.0f;
        quantize_q8_1_impl<Elements_ਪ੍ਰਤੀ_WI>(x, quantized_ਮੁੱਲਾਂ, d, ਜੋੜ, it);

        ਖੰਡ_q8_1 * quant_ptr = (ਖੰਡ_q8_1 *) q8_tensor;
        ਸਵੈਚਾਲਿਤ         ਖੰਡ_id  = subgroup_id % ਅੰਕ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ + ਪੰਕਤੀ * pitch;

        int8_t * qs                                               = &(quant_ptr[ਖੰਡ_id].qs[wi_id * Elements_ਪ੍ਰਤੀ_WI]);
        *reinterpret_cast<sycl::vec<int8_t, Elements_ਪ੍ਰਤੀ_WI> *>(qs) = quantized_ਮੁੱਲਾਂ;
        ਜੇ (wi_id == 0) {
            quant_ptr[ਖੰਡ_id].ds = sycl::half2(sycl::half(d), sycl::half(ਜੋੜ));
        }
    }
};

ਸਾਂਚਾ <ਸਾਂਚਾ <ਪੂਰਨ> ਕਿਸਮ_ਨਾਂ quantize_f>
ਖਾਲੀ quantize_ਪੰਕਤੀ_q8_1_sycl(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਖਾਲੀ * vy, ਅਚਲ ਪੂਰਨ kx, ਅਚਲ ਪੂਰਨ ky, ਅਚਲ ਪੂਰਨ kx_padded,
                            dpct::ਕਤਾਰ_ptr stream) {
    static_assert(QK8_1 % WARP_ਮਾਪ == 0);
    ਸਵੈਚਾਲਿਤ local_range      = std::size_t(WARP_ਮਾਪ);
    ਸਵੈਚਾਲਿਤ ਅੰਕ_quant_ਖੰਡਾਂ = ky * (kx / QK8_1);
    ਸਵੈਚਾਲਿਤ global_range     = ਅੰਕ_quant_ਖੰਡਾਂ * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_ਵਸਤੂ<1> it) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                             quantize_f<QK8_1 / WARP_ਮਾਪ>()(x, vy, kx, kx_padded, it);
                         });
}
