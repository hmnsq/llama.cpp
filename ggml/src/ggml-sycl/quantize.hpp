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

#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ <sycl/nd_item.hpp>

#ಅನ್ತರ್ಭಾವ "ggml-sycl/dpct/helper.hpp"

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ Elements_ಪ್ರತಿ_WI>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ quantize_q8_1_impl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ x,
                                               sycl::vec<int8_t, Elements_ಪ್ರತಿ_WI> & quantized_ಮೌಲ್ಯಗಳು, ದಶಮಾಂಶ & d,
                                               ದಶಮಾಂಶ & ಮೊತ್ತ, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & it) {
    ಸ್ವಯಂಚಾಲಿತ subgroup_id = it.get_ಗುಂಪು(0);
    ಸ್ವಯಂಚಾಲಿತ wi_id       = it.get_local_id(0);

    sycl::vec<ದಶಮಾಂಶ, Elements_ಪ್ರತಿ_WI> wi_f32_vals;

    ಸ್ವಯಂಚಾಲಿತ float_ptr_ಸ್ಥಾನಾಂತರ = subgroup_id * QK8_1 + Elements_ಪ್ರತಿ_WI * wi_id;
    wi_f32_vals           = *reinterpret_cast<ಸ್ಥಿರಾಂಕ sycl::vec<ದಶಮಾಂಶ, Elements_ಪ್ರತಿ_WI> *>(x + float_ptr_ಸ್ಥಾನಾಂತರ);

    ದಶಮಾಂಶ amax = 0.0f;

#ಸೂಚನೆ unroll(Elements_ಪ್ರತಿ_WI)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < Elements_ಪ್ರತಿ_WI; i++) {
        ಮೊತ್ತ += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_ಮೌಲ್ಯಗಳು[i] = 0;
    }
    ಮೊತ್ತ  = sycl::reduce_over_ಗುಂಪು(it.get_sub_ಗುಂಪು(), ಮೊತ್ತ, sycl::plus<ದಶಮಾಂಶ>());
    amax = sycl::reduce_over_ಗುಂಪು(it.get_sub_ಗುಂಪು(), amax, sycl::ಗರಿಷ್ಠ<ದಶಮಾಂಶ>());
    d    = amax == 0 ? 1 : amax / 127;

#ಸೂಚನೆ unroll(Elements_ಪ್ರತಿ_WI)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < Elements_ಪ್ರತಿ_WI; i++) {
        quantized_ಮೌಲ್ಯಗಳು[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ Elements_ಪ್ರತಿ_WI> ರಚನೆ no_quantize_q8_1 {
    ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *, ಶೂನ್ಯ *, ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> &) ಸ್ಥಿರಾಂಕ {}
};

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ Elements_ಪ್ರತಿ_WI> ರಚನೆ quantize_and_reorder_q8_1_soa {
    __dpct_inline__ ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ x, ಶೂನ್ಯ * reordered_q8_tensor, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kx,
                                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kx_padded, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & it) ಸ್ಥಿರಾಂಕ {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        ಸ್ವಯಂಚಾಲಿತ subgroup_id = it.get_ಗುಂಪು(0);
        ಸ್ವಯಂಚಾಲಿತ wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, Elements_ಪ್ರತಿ_WI> quantized_ಮೌಲ್ಯಗಳು;
        ದಶಮಾಂಶ                            d   = 0.0f;
        ದಶಮಾಂಶ                            ಮೊತ್ತ = 0.0f;
        quantize_q8_1_impl<Elements_ಪ್ರತಿ_WI>(x, quantized_ಮೌಲ್ಯಗಳು, d, ಮೊತ್ತ, it);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = kx / QK8_1;
        ಸ್ವಯಂಚಾಲಿತ      ಅಡ್ಡಸಾಲು                = subgroup_id / ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಸ್ವಯಂಚಾಲಿತ      ಲಂಬಸಾಲು                = subgroup_id % ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಸ್ವಯಂಚಾಲಿತ      ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ         = ಅಡ್ಡಸಾಲು * (kx_padded / QK8_1) * ಗಾತ್ರ(ಖಂಡ_q8_1);
        ಸ್ವಯಂಚಾಲಿತ      ಲಂಬಸಾಲು_ಸ್ಥಾನಾಂತರ         = QK8_1 * ಲಂಬಸಾಲು + wi_id * Elements_ಪ್ರತಿ_WI;

        ಸ್ವಯಂಚಾಲಿತ quant_ptr = (int8_t *) ((ಅಕ್ಷರ *) reordered_q8_tensor + ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ + ಲಂಬಸಾಲು_ಸ್ಥಾನಾಂತರ);
        *reinterpret_cast<sycl::vec<int8_t, Elements_ಪ್ರತಿ_WI> *>(quant_ptr) = quantized_ಮೌಲ್ಯಗಳು;

        ಸ್ವಯಂಚಾಲಿತ ds_ptr = (sycl::half2 *) ((ಅಕ್ಷರ *) reordered_q8_tensor + ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ + kx + ಲಂಬಸಾಲು * ಗಾತ್ರ(sycl::half2));
        ಆದರೆ (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(ಮೊತ್ತ));
        }
    }
};

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ Elements_ಪ್ರತಿ_WI> ರಚನೆ quantize_q8_1 {
    __dpct_inline__ ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ x, ಶೂನ್ಯ * q8_tensor, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kx, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kx_padded,
                                    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & it) ಸ್ಥಿರಾಂಕ {
        ಸ್ವಯಂಚಾಲಿತ subgroup_id = it.get_ಗುಂಪು(0);
        ಸ್ವಯಂಚಾಲಿತ wi_id       = it.get_local_id(0);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು = kx / QK8_1;
        ಸ್ವಯಂಚಾಲಿತ      ಅಡ್ಡಸಾಲು                = subgroup_id / ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, Elements_ಪ್ರತಿ_WI> quantized_ಮೌಲ್ಯಗಳು;
        ದಶಮಾಂಶ                            d   = 0.0f;
        ದಶಮಾಂಶ                            ಮೊತ್ತ = 0.0f;
        quantize_q8_1_impl<Elements_ಪ್ರತಿ_WI>(x, quantized_ಮೌಲ್ಯಗಳು, d, ಮೊತ್ತ, it);

        ಖಂಡ_q8_1 * quant_ptr = (ಖಂಡ_q8_1 *) q8_tensor;
        ಸ್ವಯಂಚಾಲಿತ         ಖಂಡ_id  = subgroup_id % ಅಂಕೆ_ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು + ಅಡ್ಡಸಾಲು * pitch;

        int8_t * qs                                               = &(quant_ptr[ಖಂಡ_id].qs[wi_id * Elements_ಪ್ರತಿ_WI]);
        *reinterpret_cast<sycl::vec<int8_t, Elements_ಪ್ರತಿ_WI> *>(qs) = quantized_ಮೌಲ್ಯಗಳು;
        ಆದರೆ (wi_id == 0) {
            quant_ptr[ಖಂಡ_id].ds = sycl::half2(sycl::half(d), sycl::half(ಮೊತ್ತ));
        }
    }
};

ಮಾದರಿ <ಮಾದರಿ <ಪೂರ್ಣಾಂಕ> ಬಗೆ_ಹೆಸರು quantize_f>
ಶೂನ್ಯ quantize_ಅಡ್ಡಸಾಲು_q8_1_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ಶೂನ್ಯ * vy, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kx, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ky, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kx_padded,
                            dpct::ಸರತಿ_ptr stream) {
    static_assert(QK8_1 % WARP_ಗಾತ್ರ == 0);
    ಸ್ವಯಂಚಾಲಿತ local_range      = std::size_t(WARP_ಗಾತ್ರ);
    ಸ್ವಯಂಚಾಲಿತ ಅಂಕೆ_quant_ಖಂಡಗಳು = ky * (kx / QK8_1);
    ಸ್ವಯಂಚಾಲಿತ global_range     = ಅಂಕೆ_quant_ಖಂಡಗಳು * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_ಅಂಶ<1> it) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                             quantize_f<QK8_1 / WARP_ಗಾತ್ರ>()(x, vy, kx, kx_padded, it);
                         });
}
