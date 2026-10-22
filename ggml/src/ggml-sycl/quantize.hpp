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

#അന്തര്ഭാവ <sycl/nd_item.hpp>

#അന്തര്ഭാവ "ggml-sycl/dpct/helper.hpp"

template <പൂർണ്ണസംഖ്യ ElementsPerWI>
__dpct_inline__ സ്ഥിതം ശൂന്യം quantize_q8_1_impl(സ്ഥിരം ദശാംശം * __restrict__ x,
                                               sycl::vec<int8_t, ElementsPerWI> & quantized_മൂല്യങ്ങൾ, ദശാംശം & d,
                                               ദശാംശം & sum, സ്ഥിരം sycl::nd_item<1> & it) {
    auto subgroup_id = it.get_group(0);
    auto wi_id       = it.get_local_id(0);

    sycl::vec<ദശാംശം, ElementsPerWI> wi_f32_vals;

    auto float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
    wi_f32_vals           = *reinterpret_cast<സ്ഥിരം sycl::vec<ദശാംശം, ElementsPerWI> *>(x + float_ptr_offset);

    ദശാംശം amax = 0.0f;

#pragma unroll(ElementsPerWI)
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ElementsPerWI; i++) {
        sum += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_മൂല്യങ്ങൾ[i] = 0;
    }
    sum  = sycl::reduce_over_group(it.get_sub_group(), sum, sycl::plus<ദശാംശം>());
    amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<ദശാംശം>());
    d    = amax == 0 ? 1 : amax / 127;

#pragma unroll(ElementsPerWI)
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ElementsPerWI; i++) {
        quantized_മൂല്യങ്ങൾ[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
template <പൂർണ്ണസംഖ്യ ElementsPerWI> ഘടന no_quantize_q8_1 {
    ശൂന്യം operator()(സ്ഥിരം ദശാംശം *, ശൂന്യം *, പൂർണ്ണസംഖ്യ, പൂർണ്ണസംഖ്യ, സ്ഥിരം sycl::nd_item<1> &) സ്ഥിരം {}
};

template <പൂർണ്ണസംഖ്യ ElementsPerWI> ഘടന quantize_and_reorder_q8_1_soa {
    __dpct_inline__ ശൂന്യം operator()(സ്ഥിരം ദശാംശം * __restrict__ x, ശൂന്യം * reordered_q8_tensor, സ്ഥിരം പൂർണ്ണസംഖ്യ kx,
                                    സ്ഥിരം പൂർണ്ണസംഖ്യ kx_padded, സ്ഥിരം sycl::nd_item<1> & it) സ്ഥിരം {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_മൂല്യങ്ങൾ;
        ദശാംശം                            d   = 0.0f;
        ദശാംശം                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_മൂല്യങ്ങൾ, d, sum, it);

        സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / അക്കം_blocks_per_row;
        auto      col                = subgroup_id % അക്കം_blocks_per_row;
        auto      row_offset         = row * (kx_padded / QK8_1) * വലുപ്പം(block_q8_1);
        auto      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        auto quant_ptr = (int8_t *) ((അക്ഷരം *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_മൂല്യങ്ങൾ;

        auto ds_ptr = (sycl::half2 *) ((അക്ഷരം *) reordered_q8_tensor + row_offset + kx + col * വലുപ്പം(sycl::half2));
        എങ്കിൽ (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <പൂർണ്ണസംഖ്യ ElementsPerWI> ഘടന quantize_q8_1 {
    __dpct_inline__ ശൂന്യം operator()(സ്ഥിരം ദശാംശം * __restrict__ x, ശൂന്യം * q8_tensor, സ്ഥിരം പൂർണ്ണസംഖ്യ kx, സ്ഥിരം പൂർണ്ണസംഖ്യ kx_padded,
                                    സ്ഥിരം sycl::nd_item<1> & it) സ്ഥിരം {
        auto subgroup_id = it.get_group(0);
        auto wi_id       = it.get_local_id(0);

        സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks_per_row = kx / QK8_1;
        auto      row                = subgroup_id / അക്കം_blocks_per_row;
        സ്ഥിരം പൂർണ്ണസംഖ്യ pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, ElementsPerWI> quantized_മൂല്യങ്ങൾ;
        ദശാംശം                            d   = 0.0f;
        ദശാംശം                            sum = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_മൂല്യങ്ങൾ, d, sum, it);

        block_q8_1 * quant_ptr = (block_q8_1 *) q8_tensor;
        auto         block_id  = subgroup_id % അക്കം_blocks_per_row + row * pitch;

        int8_t * qs                                               = &(quant_ptr[block_id].qs[wi_id * ElementsPerWI]);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(qs) = quantized_മൂല്യങ്ങൾ;
        എങ്കിൽ (wi_id == 0) {
            quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(sum));
        }
    }
};

template <template <പൂർണ്ണസംഖ്യ> typename quantize_f>
ശൂന്യം quantize_row_q8_1_sycl(സ്ഥിരം ദശാംശം * x, ശൂന്യം * vy, സ്ഥിരം പൂർണ്ണസംഖ്യ kx, സ്ഥിരം പൂർണ്ണസംഖ്യ ky, സ്ഥിരം പൂർണ്ണസംഖ്യ kx_padded,
                            dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_SIZE == 0);
    auto local_range      = std::size_t(WARP_SIZE);
    auto അക്കം_quant_blocks = ky * (kx / QK8_1);
    auto global_range     = അക്കം_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             quantize_f<QK8_1 / WARP_SIZE>()(x, vy, kx, kx_padded, it);
                         });
}
