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

#شامل <sycl/nd_item.hpp>

#شامل "ggml-sycl/dpct/helper.hpp"

template <صحیح ElementsPerWI>
__dpct_inline__ ایستا تهی quantize_q8_1_impl(ثابت اعشاری * __restrict__ x,
                                               sycl::vec<int8_t, ElementsPerWI> & quantized_مقادیر, اعشاری & d,
                                               اعشاری & sum, ثابت sycl::nd_item<1> & it) {
    auto subgroup_id = it.get_group(0);
    auto wi_id       = it.get_local_id(0);

    sycl::vec<اعشاری, ElementsPerWI> wi_f32_vals;

    auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
    wi_f32_vals           = *reinterpret_cast<ثابت sycl::vec<اعشاری, ElementsPerWI> *>(x + float_ptr_offset);

    اعشاری amax = 0.0f;

#pragma unroll(ElementsPerWI)
    برای (صحیح i = 0; i < ElementsPerWI; i++) {
        sum += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_مقادیر[i] = 0;
    }
    sum  = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<اعشاری>());
    amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<اعشاری>());
    d    = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
    برای (صحیح i = 0; i < ElementsPerWI; i++) {
        quantized_مقادیر[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
template <صحیح ElementsPerWI> ساختار no_quantize_q8_1 {
    تهی operator()(ثابت اعشاری *, تهی *, صحیح, صحیح, ثابت sycl::nd_item<1> &) ثابت {}
};

template <صحیح ElementsPerWI> ساختار quantize_and_reorder_q8_1_soa {
    __dpct_inline__ تهی operator()(ثابت اعشاری * __restrict__ x, تهی * reordered_q8_tensor, ثابت صحیح kx,
                                    ثابت صحیح kx_padded, ثابت sycl::nd_item<1> & it) ثابت {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_مقادیر;
        اعشاری                            d   = 0.0f;
        اعشاری                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_مقادیر, d, sum, it);

        ثابت صحیح رقم_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / رقم_blocks_per_row;
        auto      col                = subgroup_id % رقم_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * اندازه(block_q8_1);
        auto      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((نویسه *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_مقادیر;

        auto ds_ptr = (sycl::half2 *) ((نویسه *) reordered_q8_tensor + row_offset + kx + col * اندازه(sycl::half2));
        اگر (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <صحیح ElementsPerWI> ساختار quantize_q8_1 {
    __dpct_inline__ تهی operator()(ثابت اعشاری * __restrict__ x, تهی * q8_tensor, ثابت صحیح kx, ثابت صحیح kx_padded,
                                    ثابت sycl::nd_item<1> & it) ثابت {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        ثابت صحیح رقم_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / رقم_blocks_per_row;
        ثابت صحیح pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, ElementsPerWI> quantized_مقادیر;
        اعشاری                            d   = 0.0f;
        اعشاری                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_مقادیر, d, sum, it);

        block_q8_1 * quant_ptr = (block_q8_1 *) q8_tensor;
        auto         block_id  = subgroup_id % رقم_blocks_per_row + row * pitch;

        int8_t * qs                                               = &(quant_ptr[block_id].qs[wi_id * ElementsPerWI]);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(qs) = quantized_مقادیر;
        اگر (wi_id == 0) {
            quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <template <صحیح> typename quantize_f>
تهی quantize_row_q8_1_sycl(ثابت اعشاری * x, تهی * vy, ثابت صحیح kx, ثابت صحیح ky, ثابت صحیح kx_padded,
                            dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range      = std::size_t(WARP_SIZE);
    auto رقم_quant_blocks = ky * (kx / QK8_1);
    auto global_range     = رقم_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             quantize_f<QK8_1 / WARP_SIZE>()(x, vy, kx, kx_padded, it);
                         });
}
