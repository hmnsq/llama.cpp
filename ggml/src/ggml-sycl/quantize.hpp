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

#निर्देशः once

#अन्तर्भाव <sycl/nd_item.hpp>

#अन्तर्भाव "ggml-sycl/dpct/helper.hpp"

प्रतिरूपम् <पूर्णाङ्क Elements_प्रत्येकम्_WI>
__dpct_inline__ स्थिर शून्य quantize_q8_1_impl(ध्रुव भिन्न * __restrict__ x,
                                               sycl::vec<int8_t, Elements_प्रत्येकम्_WI> & quantized_मानानि, भिन्न & d,
                                               भिन्न & योग_2, ध्रुव sycl::nd_item<1> & it) {
    स्वचालित subgroup_id = it.get_गणः(0);
    स्वचालित wi_id       = it.get_local_id(0);

    sycl::vec<भिन्न, Elements_प्रत्येकम्_WI> wi_f32_vals;

    स्वचालित float_ptr_विस्थापनम् = subgroup_id * QK8_1 + Elements_प्रत्येकम्_WI * wi_id;
    wi_f32_vals           = *reinterpret_cast<ध्रुव sycl::vec<भिन्न, Elements_प्रत्येकम्_WI> *>(x + float_ptr_विस्थापनम्);

    भिन्न amax = 0.0f;

#निर्देशः unroll(Elements_प्रत्येकम्_WI)
    प्रति (पूर्णाङ्क i = 0; i < Elements_प्रत्येकम्_WI; i++) {
        योग_2 += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_मानानि[i] = 0;
    }
    योग_2  = sycl::reduce_over_गणः(it.get_sub_गणः(), योग_2, sycl::plus<भिन्न>());
    amax = sycl::reduce_over_गणः(it.get_sub_गणः(), amax, sycl::अधिकतमम्<भिन्न>());
    d    = amax == 0 ? 1 : amax / 127;

#निर्देशः unroll(Elements_प्रत्येकम्_WI)
    प्रति (पूर्णाङ्क i = 0; i < Elements_प्रत्येकम्_WI; i++) {
        quantized_मानानि[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
प्रतिरूपम् <पूर्णाङ्क Elements_प्रत्येकम्_WI> संरचना no_quantize_q8_1 {
    शून्य संक्रियकः()(ध्रुव भिन्न *, शून्य *, पूर्णाङ्क, पूर्णाङ्क, ध्रुव sycl::nd_item<1> &) ध्रुव {}
};

प्रतिरूपम् <पूर्णाङ्क Elements_प्रत्येकम्_WI> संरचना quantize_and_reorder_q8_1_soa {
    __dpct_inline__ शून्य संक्रियकः()(ध्रुव भिन्न * __restrict__ x, शून्य * reordered_q8_tensor, ध्रुव पूर्णाङ्क kx,
                                    ध्रुव पूर्णाङ्क kx_padded, ध्रुव sycl::nd_item<1> & it) ध्रुव {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        स्वचालित subgroup_id = it.get_गणः(0);
        स्वचालित wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, Elements_प्रत्येकम्_WI> quantized_मानानि;
        भिन्न                            d   = 0.0f;
        भिन्न                            योग_2 = 0.0f;
        quantize_q8_1_impl<Elements_प्रत्येकम्_WI>(x, quantized_मानानि, d, योग_2, it);

        ध्रुव पूर्णाङ्क अङ्कः_खण्डानि_प्रत्येकम्_row = kx / QK8_1;
        स्वचालित      row                = subgroup_id / अङ्कः_खण्डानि_प्रत्येकम्_row;
        स्वचालित      col                = subgroup_id % अङ्कः_खण्डानि_प्रत्येकम्_row;
        स्वचालित      row_विस्थापनम्         = row * (kx_padded / QK8_1) * परिमाण(खण्डम्_q8_1);
        स्वचालित      col_विस्थापनम्         = QK8_1 * col + wi_id * Elements_प्रत्येकम्_WI;

        स्वचालित quant_ptr = (int8_t *) ((वर्ण *) reordered_q8_tensor + row_विस्थापनम् + col_विस्थापनम्);
        *reinterpret_cast<sycl::vec<int8_t, Elements_प्रत्येकम्_WI> *>(quant_ptr) = quantized_मानानि;

        स्वचालित ds_ptr = (sycl::half2 *) ((वर्ण *) reordered_q8_tensor + row_विस्थापनम् + kx + col * परिमाण(sycl::half2));
        यदि (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(योग_2));
        }
    }
};

प्रतिरूपम् <पूर्णाङ्क Elements_प्रत्येकम्_WI> संरचना quantize_q8_1 {
    __dpct_inline__ शून्य संक्रियकः()(ध्रुव भिन्न * __restrict__ x, शून्य * q8_tensor, ध्रुव पूर्णाङ्क kx, ध्रुव पूर्णाङ्क kx_padded,
                                    ध्रुव sycl::nd_item<1> & it) ध्रुव {
        स्वचालित subgroup_id = it.get_गणः(0);
        स्वचालित wi_id       = it.get_local_id(0);

        ध्रुव पूर्णाङ्क अङ्कः_खण्डानि_प्रत्येकम्_row = kx / QK8_1;
        स्वचालित      row                = subgroup_id / अङ्कः_खण्डानि_प्रत्येकम्_row;
        ध्रुव पूर्णाङ्क pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, Elements_प्रत्येकम्_WI> quantized_मानानि;
        भिन्न                            d   = 0.0f;
        भिन्न                            योग_2 = 0.0f;
        quantize_q8_1_impl<Elements_प्रत्येकम्_WI>(x, quantized_मानानि, d, योग_2, it);

        खण्डम्_q8_1 * quant_ptr = (खण्डम्_q8_1 *) q8_tensor;
        स्वचालित         खण्डम्_id  = subgroup_id % अङ्कः_खण्डानि_प्रत्येकम्_row + row * pitch;

        int8_t * qs                                               = &(quant_ptr[खण्डम्_id].qs[wi_id * Elements_प्रत्येकम्_WI]);
        *reinterpret_cast<sycl::vec<int8_t, Elements_प्रत्येकम्_WI> *>(qs) = quantized_मानानि;
        यदि (wi_id == 0) {
            quant_ptr[खण्डम्_id].ds = sycl::half2(sycl::half(d), sycl::half(योग_2));
        }
    }
};

प्रतिरूपम् <प्रतिरूपम् <पूर्णाङ्क> प्रकारनाम quantize_f>
शून्य quantize_row_q8_1_sycl(ध्रुव भिन्न * x, शून्य * vy, ध्रुव पूर्णाङ्क kx, ध्रुव पूर्णाङ्क ky, ध्रुव पूर्णाङ्क kx_padded,
                            dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_आकार == 0);
    स्वचालित local_range      = std::size_t(WARP_आकार);
    स्वचालित अङ्कः_quant_खण्डानि = ky * (kx / QK8_1);
    स्वचालित global_range     = अङ्कः_quant_खण्डानि * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                             quantize_f<QK8_1 / WARP_आकार>()(x, vy, kx, kx_padded, it);
                         });
}
