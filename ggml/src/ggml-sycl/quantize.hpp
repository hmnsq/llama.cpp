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

#ଅନ୍ତର୍ଭାଵ <sycl/nd_item.hpp>

#ଅନ୍ତର୍ଭାଵ "ggml-sycl/dpct/helper.hpp"

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ElementsPerWI>
__dpct_inline__ ସ୍ଥିର ଶୂନ୍ୟ quantize_q8_1_impl(ଅଚଳ ଦଶମିକ * __restrict__ x,
                                               sycl::vec<int8_t, ElementsPerWI> & quantized_ମୂଲ୍ୟଗୁଡ଼ିକ, ଦଶମିକ & d,
                                               ଦଶମିକ & sum, ଅଚଳ sycl::nd_item<1> & it) {
    auto subgroup_id = it.get_group(0);
    auto wi_id       = it.get_local_id(0);

    sycl::vec<ଦଶମିକ, ElementsPerWI> wi_f32_vals;

    auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
    wi_f32_vals           = *reinterpret_cast<ଅଚଳ sycl::vec<ଦଶମିକ, ElementsPerWI> *>(x + float_ptr_offset);

    ଦଶମିକ amax = 0.0f;

#pragma unroll(ElementsPerWI)
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ElementsPerWI; i++) {
        sum += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_ମୂଲ୍ୟଗୁଡ଼ିକ[i] = 0;
    }
    sum  = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<ଦଶମିକ>());
    amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<ଦଶମିକ>());
    d    = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ElementsPerWI; i++) {
        quantized_ମୂଲ୍ୟଗୁଡ଼ିକ[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ElementsPerWI> ସଂରଚନା no_quantize_q8_1 {
    ଶୂନ୍ୟ operator()(ଅଚଳ ଦଶମିକ *, ଶୂନ୍ୟ *, ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ଅଚଳ sycl::nd_item<1> &) ଅଚଳ {}
};

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ElementsPerWI> ସଂରଚନା quantize_and_reorder_q8_1_soa {
    __dpct_inline__ ଶୂନ୍ୟ operator()(ଅଚଳ ଦଶମିକ * __restrict__ x, ଶୂନ୍ୟ * reordered_q8_tensor, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kx,
                                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kx_padded, ଅଚଳ sycl::nd_item<1> & it) ଅଚଳ {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_ମୂଲ୍ୟଗୁଡ଼ିକ;
        ଦଶମିକ                            d   = 0.0f;
        ଦଶମିକ                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_ମୂଲ୍ୟଗୁଡ଼ିକ, d, sum, it);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / ଅଙ୍କ_blocks_per_row;
        auto      col                = subgroup_id % ଅଙ୍କ_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * ଆକାର(block_q8_1);
        auto      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((ଅକ୍ଷର *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_ମୂଲ୍ୟଗୁଡ଼ିକ;

        auto ds_ptr = (sycl::half2 *) ((ଅକ୍ଷର *) reordered_q8_tensor + row_offset + kx + col * ଆକାର(sycl::half2));
        ଯଦି (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା ElementsPerWI> ସଂରଚନା quantize_q8_1 {
    __dpct_inline__ ଶୂନ୍ୟ operator()(ଅଚଳ ଦଶମିକ * __restrict__ x, ଶୂନ୍ୟ * q8_tensor, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kx, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kx_padded,
                                    ଅଚଳ sycl::nd_item<1> & it) ଅଚଳ {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / ଅଙ୍କ_blocks_per_row;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, ElementsPerWI> quantized_ମୂଲ୍ୟଗୁଡ଼ିକ;
        ଦଶମିକ                            d   = 0.0f;
        ଦଶମିକ                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_ମୂଲ୍ୟଗୁଡ଼ିକ, d, sum, it);

        block_q8_1 * quant_ptr = (block_q8_1 *) q8_tensor;
        auto         block_id  = subgroup_id % ଅଙ୍କ_blocks_per_row + row * pitch;

        int8_t * qs                                               = &(quant_ptr[block_id].qs[wi_id * ElementsPerWI]);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(qs) = quantized_ମୂଲ୍ୟଗୁଡ଼ିକ;
        ଯଦି (wi_id == 0) {
            quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା> typename quantize_f>
ଶୂନ୍ୟ quantize_row_q8_1_sycl(ଅଚଳ ଦଶମିକ * x, ଶୂନ୍ୟ * vy, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kx, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ky, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kx_padded,
                            dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range      = std::size_t(WARP_SIZE);
    auto ଅଙ୍କ_quant_blocks = ky * (kx / QK8_1);
    auto global_range     = ଅଙ୍କ_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             quantize_f<QK8_1 / WARP_SIZE>()(x, vy, kx, kx_padded, it);
                         });
}
