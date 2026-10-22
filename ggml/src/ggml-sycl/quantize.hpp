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

#pragma once

#अन्तर्भाव <sycl/nd_item.hpp>

#अन्तर्भाव "ggml-sycl/dpct/helper.hpp"

template <पूर्णाङ्क ElementsPerWI>
__dpct_inline__ स्थिर शून्य quantize_q8_1_impl(ध्रुव भिन्न * __restrict__ x,
                                               sycl::vec<int8_t, ElementsPerWI> & quantized_मानानि, भिन्न & d,
                                               भिन्न & sum, ध्रुव sycl::nd_item<1> & it) {
    auto subgroup_id = it.get_group(0);
    auto wi_id       = it.get_local_id(0);

    sycl::vec<भिन्न, ElementsPerWI> wi_f32_vals;

    auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
    wi_f32_vals           = *reinterpret_cast<ध्रुव sycl::vec<भिन्न, ElementsPerWI> *>(x + float_ptr_offset);

    भिन्न amax = 0.0f;

#pragma unroll(ElementsPerWI)
    अत्थाय (पूर्णाङ्क i = 0; i < ElementsPerWI; i++) {
        sum += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_मानानि[i] = 0;
    }
    sum  = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<भिन्न>());
    amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<भिन्न>());
    d    = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
    अत्थाय (पूर्णाङ्क i = 0; i < ElementsPerWI; i++) {
        quantized_मानानि[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
template <पूर्णाङ्क ElementsPerWI> संरचना no_quantize_q8_1 {
    शून्य operator()(ध्रुव भिन्न *, शून्य *, पूर्णाङ्क, पूर्णाङ्क, ध्रुव sycl::nd_item<1> &) ध्रुव {}
};

template <पूर्णाङ्क ElementsPerWI> संरचना quantize_and_reorder_q8_1_soa {
    __dpct_inline__ शून्य operator()(ध्रुव भिन्न * __restrict__ x, शून्य * reordered_q8_tensor, ध्रुव पूर्णाङ्क kx,
                                    ध्रुव पूर्णाङ्क kx_padded, ध्रुव sycl::nd_item<1> & it) ध्रुव {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_मानानि;
        भिन्न                            d   = 0.0f;
        भिन्न                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_मानानि, d, sum, it);

        ध्रुव पूर्णाङ्क अङ्कः_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / अङ्कः_blocks_per_row;
        auto      col                = subgroup_id % अङ्कः_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * परिमाण(block_q8_1);
        auto      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((वर्ण *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_मानानि;

        auto ds_ptr = (sycl::half2 *) ((वर्ण *) reordered_q8_tensor + row_offset + kx + col * परिमाण(sycl::half2));
        सचे (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <पूर्णाङ्क ElementsPerWI> संरचना quantize_q8_1 {
    __dpct_inline__ शून्य operator()(ध्रुव भिन्न * __restrict__ x, शून्य * q8_tensor, ध्रुव पूर्णाङ्क kx, ध्रुव पूर्णाङ्क kx_padded,
                                    ध्रुव sycl::nd_item<1> & it) ध्रुव {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        ध्रुव पूर्णाङ्क अङ्कः_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / अङ्कः_blocks_per_row;
        ध्रुव पूर्णाङ्क pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, ElementsPerWI> quantized_मानानि;
        भिन्न                            d   = 0.0f;
        भिन्न                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_मानानि, d, sum, it);

        block_q8_1 * quant_ptr = (block_q8_1 *) q8_tensor;
        auto         block_id  = subgroup_id % अङ्कः_blocks_per_row + row * pitch;

        int8_t * qs                                               = &(quant_ptr[block_id].qs[wi_id * ElementsPerWI]);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(qs) = quantized_मानानि;
        सचे (wi_id == 0) {
            quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <template <पूर्णाङ्क> typename quantize_f>
शून्य quantize_row_q8_1_sycl(ध्रुव भिन्न * x, शून्य * vy, ध्रुव पूर्णाङ्क kx, ध्रुव पूर्णाङ्क ky, ध्रुव पूर्णाङ्क kx_padded,
                            dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range      = std::size_t(WARP_SIZE);
    auto अङ्कः_quant_blocks = ky * (kx / QK8_1);
    auto global_range     = अङ्कः_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             quantize_f<QK8_1 / WARP_SIZE>()(x, vy, kx, kx_padded, it);
                         });
}
