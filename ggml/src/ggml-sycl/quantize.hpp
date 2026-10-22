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

#அந்தர்பாவ <sycl/nd_item.hpp>

#அந்தர்பாவ "ggml-sycl/dpct/helper.hpp"

template <முழு ElementsPerWI>
__dpct_inline__ நிலையான வெறுமை quantize_q8_1_impl(மாறிலி தசம * __restrict__ x,
                                               sycl::vec<int8_t, ElementsPerWI> & quantized_மதிப்புகள், தசம & d,
                                               தசம & sum, மாறிலி sycl::nd_item<1> & it) {
    auto subgroup_id = it.get_group(0);
    auto wi_id       = it.get_local_id(0);

    sycl::vec<தசம, ElementsPerWI> wi_f32_vals;

    auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
    wi_f32_vals           = *reinterpret_cast<மாறிலி sycl::vec<தசம, ElementsPerWI> *>(x + float_ptr_offset);

    தசம amax = 0.0f;

#pragma unroll(ElementsPerWI)
    க்கு (முழு i = 0; i < ElementsPerWI; i++) {
        sum += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_மதிப்புகள்[i] = 0;
    }
    sum  = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<தசம>());
    amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<தசம>());
    d    = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
    க்கு (முழு i = 0; i < ElementsPerWI; i++) {
        quantized_மதிப்புகள்[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
template <முழு ElementsPerWI> கட்டமைப்பு no_quantize_q8_1 {
    வெறுமை operator()(மாறிலி தசம *, வெறுமை *, முழு, முழு, மாறிலி sycl::nd_item<1> &) மாறிலி {}
};

template <முழு ElementsPerWI> கட்டமைப்பு quantize_and_reorder_q8_1_soa {
    __dpct_inline__ வெறுமை operator()(மாறிலி தசம * __restrict__ x, வெறுமை * reordered_q8_tensor, மாறிலி முழு kx,
                                    மாறிலி முழு kx_padded, மாறிலி sycl::nd_item<1> & it) மாறிலி {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_மதிப்புகள்;
        தசம                            d   = 0.0f;
        தசம                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_மதிப்புகள், d, sum, it);

        மாறிலி முழு இலக்கம்_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / இலக்கம்_blocks_per_row;
        auto      col                = subgroup_id % இலக்கம்_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * அளவு(block_q8_1);
        auto      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((எழுத்து *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_மதிப்புகள்;

        auto ds_ptr = (sycl::half2 *) ((எழுத்து *) reordered_q8_tensor + row_offset + kx + col * அளவு(sycl::half2));
        எனில் (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <முழு ElementsPerWI> கட்டமைப்பு quantize_q8_1 {
    __dpct_inline__ வெறுமை operator()(மாறிலி தசம * __restrict__ x, வெறுமை * q8_tensor, மாறிலி முழு kx, மாறிலி முழு kx_padded,
                                    மாறிலி sycl::nd_item<1> & it) மாறிலி {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        மாறிலி முழு இலக்கம்_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / இலக்கம்_blocks_per_row;
        மாறிலி முழு pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, ElementsPerWI> quantized_மதிப்புகள்;
        தசம                            d   = 0.0f;
        தசம                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_மதிப்புகள், d, sum, it);

        block_q8_1 * quant_ptr = (block_q8_1 *) q8_tensor;
        auto         block_id  = subgroup_id % இலக்கம்_blocks_per_row + row * pitch;

        int8_t * qs                                               = &(quant_ptr[block_id].qs[wi_id * ElementsPerWI]);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(qs) = quantized_மதிப்புகள்;
        எனில் (wi_id == 0) {
            quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <template <முழு> typename quantize_f>
வெறுமை quantize_row_q8_1_sycl(மாறிலி தசம * x, வெறுமை * vy, மாறிலி முழு kx, மாறிலி முழு ky, மாறிலி முழு kx_padded,
                            dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range      = std::size_t(WARP_SIZE);
    auto இலக்கம்_quant_blocks = ky * (kx / QK8_1);
    auto global_range     = இலக்கம்_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             quantize_f<QK8_1 / WARP_SIZE>()(x, vy, kx, kx_padded, it);
                         });
}
