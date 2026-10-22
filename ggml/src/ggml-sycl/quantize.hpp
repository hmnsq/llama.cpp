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

#অন্তর্ভাব <sycl/nd_item.hpp>

#অন্তর্ভাব "ggml-sycl/dpct/helper.hpp"

template <পূর্ণসংখ্যা ElementsPerWI>
__dpct_inline__ স্থির শূন্য quantize_q8_1_impl(ধ্রুবক দশমিক * __restrict__ x,
                                               sycl::vec<int8_t, ElementsPerWI> & quantized_মানসমূহ, দশমিক & d,
                                               দশমিক & যোগ_2, ধ্রুবক sycl::nd_item<1> & it) {
    auto subgroup_id = it.get_group(0);
    auto wi_id       = it.get_local_id(0);

    sycl::vec<দশমিক, ElementsPerWI> wi_f32_vals;

    auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
    wi_f32_vals           = *reinterpret_cast<ধ্রুবক sycl::vec<দশমিক, ElementsPerWI> *>(x + float_ptr_offset);

    দশমিক amax = 0.0f;

#pragma unroll(ElementsPerWI)
    জন্য (পূর্ণসংখ্যা i = 0; i < ElementsPerWI; i++) {
        যোগ_2 += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_মানসমূহ[i] = 0;
    }
    যোগ_2  = sycl::reduce_over_group(it.get_sub_group(), যোগ_2, sycl::plus<দশমিক>());
    amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<দশমিক>());
    d    = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
    জন্য (পূর্ণসংখ্যা i = 0; i < ElementsPerWI; i++) {
        quantized_মানসমূহ[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
template <পূর্ণসংখ্যা ElementsPerWI> কাঠামো no_quantize_q8_1 {
    শূন্য operator()(ধ্রুবক দশমিক *, শূন্য *, পূর্ণসংখ্যা, পূর্ণসংখ্যা, ধ্রুবক sycl::nd_item<1> &) ধ্রুবক {}
};

template <পূর্ণসংখ্যা ElementsPerWI> কাঠামো quantize_and_reorder_q8_1_soa {
    __dpct_inline__ শূন্য operator()(ধ্রুবক দশমিক * __restrict__ x, শূন্য * reordered_q8_tensor, ধ্রুবক পূর্ণসংখ্যা kx,
                                    ধ্রুবক পূর্ণসংখ্যা kx_padded, ধ্রুবক sycl::nd_item<1> & it) ধ্রুবক {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_মানসমূহ;
        দশমিক                            d   = 0.0f;
        দশমিক                            যোগ_2 = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_মানসমূহ, d, যোগ_2, it);

        ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / অঙ্ক_blocks_per_row;
        auto      col                = subgroup_id % অঙ্ক_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * আকার(block_q8_1);
        auto      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((অক্ষর *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_মানসমূহ;

        auto ds_ptr = (sycl::half2 *) ((অক্ষর *) reordered_q8_tensor + row_offset + kx + col * আকার(sycl::half2));
        যদি (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(যোগ_2));
        }
    }
};

template <পূর্ণসংখ্যা ElementsPerWI> কাঠামো quantize_q8_1 {
    __dpct_inline__ শূন্য operator()(ধ্রুবক দশমিক * __restrict__ x, শূন্য * q8_tensor, ধ্রুবক পূর্ণসংখ্যা kx, ধ্রুবক পূর্ণসংখ্যা kx_padded,
                                    ধ্রুবক sycl::nd_item<1> & it) ধ্রুবক {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / অঙ্ক_blocks_per_row;
        ধ্রুবক পূর্ণসংখ্যা pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, ElementsPerWI> quantized_মানসমূহ;
        দশমিক                            d   = 0.0f;
        দশমিক                            যোগ_2 = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_মানসমূহ, d, যোগ_2, it);

        block_q8_1 * quant_ptr = (block_q8_1 *) q8_tensor;
        auto         block_id  = subgroup_id % অঙ্ক_blocks_per_row + row * pitch;

        int8_t * qs                                               = &(quant_ptr[block_id].qs[wi_id * ElementsPerWI]);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(qs) = quantized_মানসমূহ;
        যদি (wi_id == 0) {
            quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(যোগ_2));
        }
    }
};

template <template <পূর্ণসংখ্যা> typename quantize_f>
শূন্য quantize_row_q8_1_sycl(ধ্রুবক দশমিক * x, শূন্য * vy, ধ্রুবক পূর্ণসংখ্যা kx, ধ্রুবক পূর্ণসংখ্যা ky, ধ্রুবক পূর্ণসংখ্যা kx_padded,
                            dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_আকার == 0);
    auto local_range      = std::size_t(WARP_আকার);
    auto অঙ্ক_quant_blocks = ky * (kx / QK8_1);
    auto global_range     = অঙ্ক_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_আকার(WARP_আকার)]] {
                             quantize_f<QK8_1 / WARP_আকার>()(x, vy, kx, kx_padded, it);
                         });
}
