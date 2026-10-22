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

#هدايت once

#ضمن <sycl/nd_item.hpp>

#ضمن "ggml-sycl/dpct/helper.hpp"

سانچو <صحيح Elements_في_WI>
__dpct_inline__ ساكن فراغ quantize_q8_1_impl(ثابت عائم * __restrict__ x,
                                               sycl::vec<int8_t, Elements_في_WI> & quantized_قدرون, عائم & d,
                                               عائم & جوڙ, ثابت sycl::nd_شي<1> & it) {
    خودڪار subgroup_id = it.get_گروهه(0);
    خودڪار wi_id       = it.get_local_id(0);

    sycl::vec<عائم, Elements_في_WI> wi_f32_vals;

    خودڪار float_ptr_وٿي = subgroup_id * QK8_1 + Elements_في_WI * wi_id;
    wi_f32_vals           = *reinterpret_cast<ثابت sycl::vec<عائم, Elements_في_WI> *>(x + float_ptr_وٿي);

    عائم amax = 0.0f;

#هدايت unroll(Elements_في_WI)
    لكل (صحيح i = 0; i < Elements_في_WI; i++) {
        جوڙ += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_قدرون[i] = 0;
    }
    جوڙ  = sycl::reduce_over_گروهه(it.get_sub_گروهه(), جوڙ, sycl::plus<عائم>());
    amax = sycl::reduce_over_گروهه(it.get_sub_گروهه(), amax, sycl::وڌ_کان_وڌ<عائم>());
    d    = amax == 0 ? 1 : amax / 127;

#هدايت unroll(Elements_في_WI)
    لكل (صحيح i = 0; i < Elements_في_WI; i++) {
        quantized_قدرون[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
سانچو <صحيح Elements_في_WI> بنية no_quantize_q8_1 {
    فراغ عامل()(ثابت عائم *, فراغ *, صحيح, صحيح, ثابت sycl::nd_شي<1> &) ثابت {}
};

سانچو <صحيح Elements_في_WI> بنية quantize_and_reorder_q8_1_soa {
    __dpct_inline__ فراغ عامل()(ثابت عائم * __restrict__ x, فراغ * reordered_q8_tensor, ثابت صحيح kx,
                                    ثابت صحيح kx_padded, ثابت sycl::nd_شي<1> & it) ثابت {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        خودڪار subgroup_id = it.get_گروهه(0);
        خودڪار wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, Elements_في_WI> quantized_قدرون;
        عائم                            d   = 0.0f;
        عائم                            جوڙ = 0.0f;
        quantize_q8_1_impl<Elements_في_WI>(x, quantized_قدرون, d, جوڙ, it);

        ثابت صحيح انگ_بلاڪون_في_صف = kx / QK8_1;
        خودڪار      صف                = subgroup_id / انگ_بلاڪون_في_صف;
        خودڪار      کانو                = subgroup_id % انگ_بلاڪون_في_صف;
        خودڪار      صف_وٿي         = صف * (kx_padded / QK8_1) * حجم(بلاڪ_q8_1);
        خودڪار      کانو_وٿي         = QK8_1 * کانو + wi_id * Elements_في_WI;

        خودڪار quant_ptr = (int8_t *) ((حرف *) reordered_q8_tensor + صف_وٿي + کانو_وٿي);
        *reinterpret_cast<sycl::vec<int8_t, Elements_في_WI> *>(quant_ptr) = quantized_قدرون;

        خودڪار ds_ptr = (sycl::half2 *) ((حرف *) reordered_q8_tensor + صف_وٿي + kx + کانو * حجم(sycl::half2));
        إذا (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(جوڙ));
        }
    }
};

سانچو <صحيح Elements_في_WI> بنية quantize_q8_1 {
    __dpct_inline__ فراغ عامل()(ثابت عائم * __restrict__ x, فراغ * q8_tensor, ثابت صحيح kx, ثابت صحيح kx_padded,
                                    ثابت sycl::nd_شي<1> & it) ثابت {
        خودڪار subgroup_id = it.get_گروهه(0);
        خودڪار wi_id       = it.get_local_id(0);

        ثابت صحيح انگ_بلاڪون_في_صف = kx / QK8_1;
        خودڪار      صف                = subgroup_id / انگ_بلاڪون_في_صف;
        ثابت صحيح pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, Elements_في_WI> quantized_قدرون;
        عائم                            d   = 0.0f;
        عائم                            جوڙ = 0.0f;
        quantize_q8_1_impl<Elements_في_WI>(x, quantized_قدرون, d, جوڙ, it);

        بلاڪ_q8_1 * quant_ptr = (بلاڪ_q8_1 *) q8_tensor;
        خودڪار         بلاڪ_id  = subgroup_id % انگ_بلاڪون_في_صف + صف * pitch;

        int8_t * qs                                               = &(quant_ptr[بلاڪ_id].qs[wi_id * Elements_في_WI]);
        *reinterpret_cast<sycl::vec<int8_t, Elements_في_WI> *>(qs) = quantized_قدرون;
        إذا (wi_id == 0) {
            quant_ptr[بلاڪ_id].ds = sycl::half2(sycl::half(d), sycl::half(جوڙ));
        }
    }
};

سانچو <سانچو <صحيح> قسم_نالو quantize_f>
فراغ quantize_صف_q8_1_sycl(ثابت عائم * x, فراغ * vy, ثابت صحيح kx, ثابت صحيح ky, ثابت صحيح kx_padded,
                            dpct::قطار_ptr stream) {
    static_assert(QK8_1 % WARP_سائيز == 0);
    خودڪار local_range      = std::size_t(WARP_سائيز);
    خودڪار انگ_quant_بلاڪون = ky * (kx / QK8_1);
    خودڪار global_range     = انگ_quant_بلاڪون * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_شي<1> it) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                             quantize_f<QK8_1 / WARP_سائيز>()(x, vy, kx, kx_padded, it);
                         });
}
