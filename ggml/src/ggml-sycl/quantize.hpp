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

#অন্তর্ভাৱ <sycl/nd_item.hpp>

#অন্তর্ভাৱ "ggml-sycl/dpct/helper.hpp"

template <পূর্ণাঙ্ক ElementsPerWI>
__dpct_inline__ স্থির শূন্য quantize_q8_1_impl(ধ্রুৱ ভিন্ন * __restrict__ x,
                                               sycl::vec<int8_t, ElementsPerWI> & quantized_মানসমূহ, ভিন্ন & d,
                                               ভিন্ন & sum, ধ্রুৱ sycl::nd_item<1> & it) {
    auto subgroup_id = it.get_group(0);
    auto wi_id       = it.get_local_id(0);

    sycl::vec<ভিন্ন, ElementsPerWI> wi_f32_vals;

    auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
    wi_f32_vals           = *reinterpret_cast<ধ্রুৱ sycl::vec<ভিন্ন, ElementsPerWI> *>(x + float_ptr_offset);

    ভিন্ন amax = 0.0f;

#pragma unroll(ElementsPerWI)
    বাবে (পূর্ণাঙ্ক i = 0; i < ElementsPerWI; i++) {
        sum += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_মানসমূহ[i] = 0;
    }
    sum  = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<ভিন্ন>());
    amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<ভিন্ন>());
    d    = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
    বাবে (পূর্ণাঙ্ক i = 0; i < ElementsPerWI; i++) {
        quantized_মানসমূহ[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
template <পূর্ণাঙ্ক ElementsPerWI> সংরচনা no_quantize_q8_1 {
    শূন্য operator()(ধ্রুৱ ভিন্ন *, শূন্য *, পূর্ণাঙ্ক, পূর্ণাঙ্ক, ধ্রুৱ sycl::nd_item<1> &) ধ্রুৱ {}
};

template <পূর্ণাঙ্ক ElementsPerWI> সংরচনা quantize_and_reorder_q8_1_soa {
    __dpct_inline__ শূন্য operator()(ধ্রুৱ ভিন্ন * __restrict__ x, শূন্য * reordered_q8_tensor, ধ্রুৱ পূর্ণাঙ্ক kx,
                                    ধ্রুৱ পূর্ণাঙ্ক kx_padded, ধ্রুৱ sycl::nd_item<1> & it) ধ্রুৱ {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_মানসমূহ;
        ভিন্ন                            d   = 0.0f;
        ভিন্ন                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_মানসমূহ, d, sum, it);

        ধ্রুৱ পূর্ণাঙ্ক অঙ্ক_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / অঙ্ক_blocks_per_row;
        auto      col                = subgroup_id % অঙ্ক_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * পরিমাণ(block_q8_1);
        auto      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((ৱর্ণ *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_মানসমূহ;

        auto ds_ptr = (sycl::half2 *) ((ৱর্ণ *) reordered_q8_tensor + row_offset + kx + col * পরিমাণ(sycl::half2));
        যদি (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <পূর্ণাঙ্ক ElementsPerWI> সংরচনা quantize_q8_1 {
    __dpct_inline__ শূন্য operator()(ধ্রুৱ ভিন্ন * __restrict__ x, শূন্য * q8_tensor, ধ্রুৱ পূর্ণাঙ্ক kx, ধ্রুৱ পূর্ণাঙ্ক kx_padded,
                                    ধ্রুৱ sycl::nd_item<1> & it) ধ্রুৱ {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        ধ্রুৱ পূর্ণাঙ্ক অঙ্ক_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / অঙ্ক_blocks_per_row;
        ধ্রুৱ পূর্ণাঙ্ক pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, ElementsPerWI> quantized_মানসমূহ;
        ভিন্ন                            d   = 0.0f;
        ভিন্ন                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_মানসমূহ, d, sum, it);

        block_q8_1 * quant_ptr = (block_q8_1 *) q8_tensor;
        auto         block_id  = subgroup_id % অঙ্ক_blocks_per_row + row * pitch;

        int8_t * qs                                               = &(quant_ptr[block_id].qs[wi_id * ElementsPerWI]);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(qs) = quantized_মানসমূহ;
        যদি (wi_id == 0) {
            quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <template <পূর্ণাঙ্ক> typename quantize_f>
শূন্য quantize_row_q8_1_sycl(ধ্রুৱ ভিন্ন * x, শূন্য * vy, ধ্রুৱ পূর্ণাঙ্ক kx, ধ্রুৱ পূর্ণাঙ্ক ky, ধ্রুৱ পূর্ণাঙ্ক kx_padded,
                            dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range      = std::size_t(WARP_SIZE);
    auto অঙ্ক_quant_blocks = ky * (kx / QK8_1);
    auto global_range     = অঙ্ক_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             quantize_f<QK8_1 / WARP_SIZE>()(x, vy, kx, kx_padded, it);
                         });
}
