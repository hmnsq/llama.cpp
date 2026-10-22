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

#ಅನ್ತರ್ಭಾವ <sycl/nd_item.hpp>

#ಅನ್ತರ್ಭಾವ "ggml-sycl/dpct/helper.hpp"

template <ಪೂರ್ಣಾಙ್ಕ ElementsPerWI>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ quantize_q8_1_impl(ಧ್ರುವ ಭಿನ್ನ * __restrict__ x,
                                               sycl::vec<int8_t, ElementsPerWI> & quantized_ಮೌಲ್ಯಗಳು, ಭಿನ್ನ & d,
                                               ಭಿನ್ನ & sum, ಧ್ರುವ sycl::nd_item<1> & it) {
    auto subgroup_id = it.get_group(0);
    auto wi_id       = it.get_local_id(0);

    sycl::vec<ಭಿನ್ನ, ElementsPerWI> wi_f32_vals;

    auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
    wi_f32_vals           = *reinterpret_cast<ಧ್ರುವ sycl::vec<ಭಿನ್ನ, ElementsPerWI> *>(x + float_ptr_offset);

    ಭಿನ್ನ amax = 0.0f;

#pragma unroll(ElementsPerWI)
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ElementsPerWI; i++) {
        sum += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_ಮೌಲ್ಯಗಳು[i] = 0;
    }
    sum  = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<ಭಿನ್ನ>());
    amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<ಭಿನ್ನ>());
    d    = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < ElementsPerWI; i++) {
        quantized_ಮೌಲ್ಯಗಳು[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
template <ಪೂರ್ಣಾಙ್ಕ ElementsPerWI> ಸಂರಚನಾ no_quantize_q8_1 {
    ಶೂನ್ಯ operator()(ಧ್ರುವ ಭಿನ್ನ *, ಶೂನ್ಯ *, ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ, ಧ್ರುವ sycl::nd_item<1> &) ಧ್ರುವ {}
};

template <ಪೂರ್ಣಾಙ್ಕ ElementsPerWI> ಸಂರಚನಾ quantize_and_reorder_q8_1_soa {
    __dpct_inline__ ಶೂನ್ಯ operator()(ಧ್ರುವ ಭಿನ್ನ * __restrict__ x, ಶೂನ್ಯ * reordered_q8_tensor, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kx,
                                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kx_padded, ಧ್ರುವ sycl::nd_item<1> & it) ಧ್ರುವ {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_ಮೌಲ್ಯಗಳು;
        ಭಿನ್ನ                            d   = 0.0f;
        ಭಿನ್ನ                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_ಮೌಲ್ಯಗಳು, d, sum, it);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / ಅಂಕೆ_blocks_per_row;
        auto      col                = subgroup_id % ಅಂಕೆ_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * ಪರಿಮಾಣ(block_q8_1);
        auto      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((ವರ್ಣ *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_ಮೌಲ್ಯಗಳು;

        auto ds_ptr = (sycl::half2 *) ((ವರ್ಣ *) reordered_q8_tensor + row_offset + kx + col * ಪರಿಮಾಣ(sycl::half2));
        ಯದಿ (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <ಪೂರ್ಣಾಙ್ಕ ElementsPerWI> ಸಂರಚನಾ quantize_q8_1 {
    __dpct_inline__ ಶೂನ್ಯ operator()(ಧ್ರುವ ಭಿನ್ನ * __restrict__ x, ಶೂನ್ಯ * q8_tensor, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kx, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kx_padded,
                                    ಧ್ರುವ sycl::nd_item<1> & it) ಧ್ರುವ {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / ಅಂಕೆ_blocks_per_row;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, ElementsPerWI> quantized_ಮೌಲ್ಯಗಳು;
        ಭಿನ್ನ                            d   = 0.0f;
        ಭಿನ್ನ                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_ಮೌಲ್ಯಗಳು, d, sum, it);

        block_q8_1 * quant_ptr = (block_q8_1 *) q8_tensor;
        auto         block_id  = subgroup_id % ಅಂಕೆ_blocks_per_row + row * pitch;

        int8_t * qs                                               = &(quant_ptr[block_id].qs[wi_id * ElementsPerWI]);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(qs) = quantized_ಮೌಲ್ಯಗಳು;
        ಯದಿ (wi_id == 0) {
            quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <template <ಪೂರ್ಣಾಙ್ಕ> typename quantize_f>
ಶೂನ್ಯ quantize_row_q8_1_sycl(ಧ್ರುವ ಭಿನ್ನ * x, ಶೂನ್ಯ * vy, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kx, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ky, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kx_padded,
                            dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range      = std::size_t(WARP_SIZE);
    auto ಅಂಕೆ_quant_blocks = ky * (kx / QK8_1);
    auto global_range     = ಅಂಕೆ_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             quantize_f<QK8_1 / WARP_SIZE>()(x, vy, kx, kx_padded, it);
                         });
}
