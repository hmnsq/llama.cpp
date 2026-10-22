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

#અન્તર્ભાવ <sycl/nd_item.hpp>

#અન્તર્ભાવ "ggml-sycl/dpct/helper.hpp"

template <પૂર્ણાંક ElementsPerWI>
__dpct_inline__ સ્થિર શૂન્ય quantize_q8_1_impl(અચલ દશાંશ * __restrict__ x,
                                               sycl::vec<int8_t, ElementsPerWI> & quantized_કિંમતો, દશાંશ & d,
                                               દશાંશ & sum, અચલ sycl::nd_item<1> & it) {
    auto subgroup_id = it.get_group(0);
    auto wi_id       = it.get_local_id(0);

    sycl::vec<દશાંશ, ElementsPerWI> wi_f32_vals;

    auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
    wi_f32_vals           = *reinterpret_cast<અચલ sycl::vec<દશાંશ, ElementsPerWI> *>(x + float_ptr_offset);

    દશાંશ amax = 0.0f;

#pragma unroll(ElementsPerWI)
    માટે (પૂર્ણાંક i = 0; i < ElementsPerWI; i++) {
        sum += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_કિંમતો[i] = 0;
    }
    sum  = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<દશાંશ>());
    amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<દશાંશ>());
    d    = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
    માટે (પૂર્ણાંક i = 0; i < ElementsPerWI; i++) {
        quantized_કિંમતો[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
template <પૂર્ણાંક ElementsPerWI> સંરચના no_quantize_q8_1 {
    શૂન્ય operator()(અચલ દશાંશ *, શૂન્ય *, પૂર્ણાંક, પૂર્ણાંક, અચલ sycl::nd_item<1> &) અચલ {}
};

template <પૂર્ણાંક ElementsPerWI> સંરચના quantize_and_reorder_q8_1_soa {
    __dpct_inline__ શૂન્ય operator()(અચલ દશાંશ * __restrict__ x, શૂન્ય * reordered_q8_tensor, અચલ પૂર્ણાંક kx,
                                    અચલ પૂર્ણાંક kx_padded, અચલ sycl::nd_item<1> & it) અચલ {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_કિંમતો;
        દશાંશ                            d   = 0.0f;
        દશાંશ                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_કિંમતો, d, sum, it);

        અચલ પૂર્ણાંક અંક_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / અંક_blocks_per_row;
        auto      col                = subgroup_id % અંક_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * કદ(block_q8_1);
        auto      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((અક્ષર *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_કિંમતો;

        auto ds_ptr = (sycl::half2 *) ((અક્ષર *) reordered_q8_tensor + row_offset + kx + col * કદ(sycl::half2));
        જો (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <પૂર્ણાંક ElementsPerWI> સંરચના quantize_q8_1 {
    __dpct_inline__ શૂન્ય operator()(અચલ દશાંશ * __restrict__ x, શૂન્ય * q8_tensor, અચલ પૂર્ણાંક kx, અચલ પૂર્ણાંક kx_padded,
                                    અચલ sycl::nd_item<1> & it) અચલ {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        અચલ પૂર્ણાંક અંક_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / અંક_blocks_per_row;
        અચલ પૂર્ણાંક pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, ElementsPerWI> quantized_કિંમતો;
        દશાંશ                            d   = 0.0f;
        દશાંશ                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_કિંમતો, d, sum, it);

        block_q8_1 * quant_ptr = (block_q8_1 *) q8_tensor;
        auto         block_id  = subgroup_id % અંક_blocks_per_row + row * pitch;

        int8_t * qs                                               = &(quant_ptr[block_id].qs[wi_id * ElementsPerWI]);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(qs) = quantized_કિંમતો;
        જો (wi_id == 0) {
            quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <template <પૂર્ણાંક> typename quantize_f>
શૂન્ય quantize_row_q8_1_sycl(અચલ દશાંશ * x, શૂન્ય * vy, અચલ પૂર્ણાંક kx, અચલ પૂર્ણાંક ky, અચલ પૂર્ણાંક kx_padded,
                            dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range      = std::size_t(WARP_SIZE);
    auto અંક_quant_blocks = ky * (kx / QK8_1);
    auto global_range     = અંક_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             quantize_f<QK8_1 / WARP_SIZE>()(x, vy, kx, kx_padded, it);
                         });
}
