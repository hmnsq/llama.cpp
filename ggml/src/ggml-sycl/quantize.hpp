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

#आशय once

#समावेश <sycl/nd_item.hpp>

#समावेश "ggml-sycl/dpct/helper.hpp"

ढाँचा <पूर्णांक ElementsPerWI>
__dpct_inline__ अटल व्योम quantize_q8_1_impl(स्थिर भग्न * __restrict__ x,
                                               sycl::vec<int8_t, ElementsPerWI> & quantized_मूल्यावली, भग्न & d,
                                               भग्न & योग_2, स्थिर sycl::nd_item<1> & it) {
    स्वतः subgroup_id = it.get_group(0);
    स्वतः wi_id       = it.get_local_id(0);

    sycl::vec<भग्न, ElementsPerWI> wi_f32_vals;

    स्वतः float_ptr_offset = subgroup_id * QK8_1 + ElementsPerWI * wi_id;
    wi_f32_vals           = *reinterpret_cast<स्थिर sycl::vec<भग्न, ElementsPerWI> *>(x + float_ptr_offset);

    भग्न amax = 0.0f;

#आशय unroll(ElementsPerWI)
    क्रम (पूर्णांक i = 0; i < ElementsPerWI; i++) {
        योग_2 += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::भ_असल(wi_f32_vals[i]));
        quantized_मूल्यावली[i] = 0;
    }
    योग_2  = sycl::reduce_over_group(it.get_sub_group(), योग_2, sycl::plus<भग्न>());
    amax = sycl::reduce_over_group(it.get_sub_group(), amax, sycl::maximum<भग्न>());
    d    = amax == 0 ? 1 : amax / 127;

#आशय unroll(ElementsPerWI)
    क्रम (पूर्णांक i = 0; i < ElementsPerWI; i++) {
        quantized_मूल्यावली[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
ढाँचा <पूर्णांक ElementsPerWI> काष्ठा no_quantize_q8_1 {
    व्योम चालक()(स्थिर भग्न *, व्योम *, पूर्णांक, पूर्णांक, स्थिर sycl::nd_item<1> &) स्थिर {}
};

ढाँचा <पूर्णांक ElementsPerWI> काष्ठा quantize_and_reorder_q8_1_soa {
    __dpct_inline__ व्योम चालक()(स्थिर भग्न * __restrict__ x, व्योम * reordered_q8_tensor, स्थिर पूर्णांक kx,
                                    स्थिर पूर्णांक kx_padded, स्थिर sycl::nd_item<1> & it) स्थिर {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        स्वतः subgroup_id = it.get_group(0);
        स्वतः wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, ElementsPerWI> quantized_मूल्यावली;
        भग्न                            d   = 0.0f;
        भग्न                            योग_2 = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_मूल्यावली, d, योग_2, it);

        स्थिर पूर्णांक अंक_blocks_per_row = kx / QK8_1;
        स्वतः      row                = subgroup_id / अंक_blocks_per_row;
        स्वतः      col                = subgroup_id % अंक_blocks_per_row;
        स्वतः      row_offset         = row * (kx_padded / QK8_1) * माप(block_q8_1);
        स्वतः      col_offset         = QK8_1 * col + wi_id * ElementsPerWI;

        स्वतः quant_ptr = (int8_t *) ((अक्षर *) reordered_q8_tensor + row_offset + col_offset);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(quant_ptr) = quantized_मूल्यावली;

        स्वतः ds_ptr = (sycl::half2 *) ((अक्षर *) reordered_q8_tensor + row_offset + kx + col * माप(sycl::half2));
        अगर (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(योग_2));
        }
    }
};

ढाँचा <पूर्णांक ElementsPerWI> काष्ठा quantize_q8_1 {
    __dpct_inline__ व्योम चालक()(स्थिर भग्न * __restrict__ x, व्योम * q8_tensor, स्थिर पूर्णांक kx, स्थिर पूर्णांक kx_padded,
                                    स्थिर sycl::nd_item<1> & it) स्थिर {
        स्वतः subgroup_id = it.get_group(0);
        स्वतः wi_id       = it.get_local_id(0);

        स्थिर पूर्णांक अंक_blocks_per_row = kx / QK8_1;
        स्वतः      row                = subgroup_id / अंक_blocks_per_row;
        स्थिर पूर्णांक pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, ElementsPerWI> quantized_मूल्यावली;
        भग्न                            d   = 0.0f;
        भग्न                            योग_2 = 0.0f;
        quantize_q8_1_impl<ElementsPerWI>(x, quantized_मूल्यावली, d, योग_2, it);

        block_q8_1 * quant_ptr = (block_q8_1 *) q8_tensor;
        स्वतः         block_id  = subgroup_id % अंक_blocks_per_row + row * pitch;

        int8_t * qs                                               = &(quant_ptr[block_id].qs[wi_id * ElementsPerWI]);
        *reinterpret_cast<sycl::vec<int8_t, ElementsPerWI> *>(qs) = quantized_मूल्यावली;
        अगर (wi_id == 0) {
            quant_ptr[block_id].ds = sycl::half2(sycl::half(d), sycl::half(योग_2));
        }
    }
};

ढाँचा <ढाँचा <पूर्णांक> typename quantize_f>
व्योम quantize_row_q8_1_sycl(स्थिर भग्न * x, व्योम * vy, स्थिर पूर्णांक kx, स्थिर पूर्णांक ky, स्थिर पूर्णांक kx_padded,
                            dpct::queue_ptr stream) {
    static_assert(QK8_1 % WARP_आकार == 0);
    स्वतः local_range      = std::माप_प्रकार(WARP_आकार);
    स्वतः अंक_quant_blocks = ky * (kx / QK8_1);
    स्वतः global_range     = अंक_quant_blocks * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                             quantize_f<QK8_1 / WARP_आकार>()(x, vy, kx, kx_padded, it);
                         });
}
