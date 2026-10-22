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

#لارښوونه once

#ضمن <sycl/nd_item.hpp>

#ضمن "ggml-sycl/dpct/helper.hpp"

کينډۍ <صحيح Elements_فی_WI>
__dpct_inline__ ساكن فراغ quantize_q8_1_impl(ثابت عائم * __restrict__ x,
                                               sycl::vec<int8_t, Elements_فی_WI> & quantized_ارزښتونه, عائم & d,
                                               عائم & مجموعه, ثابت sycl::nd_توکی<1> & it) {
    خپلکاری subgroup_id = it.get_ډله(0);
    خپلکاری wi_id       = it.get_local_id(0);

    sycl::vec<عائم, Elements_فی_WI> wi_f32_vals;

    خپلکاری float_ptr_دننه_واټن = subgroup_id * QK8_1 + Elements_فی_WI * wi_id;
    wi_f32_vals           = *reinterpret_cast<ثابت sycl::vec<عائم, Elements_فی_WI> *>(x + float_ptr_دننه_واټن);

    عائم amax = 0.0f;

#لارښوونه unroll(Elements_فی_WI)
    لكل (صحيح i = 0; i < Elements_فی_WI; i++) {
        مجموعه += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_ارزښتونه[i] = 0;
    }
    مجموعه  = sycl::reduce_over_ډله(it.get_sub_ډله(), مجموعه, sycl::plus<عائم>());
    amax = sycl::reduce_over_ډله(it.get_sub_ډله(), amax, sycl::اعظمي<عائم>());
    d    = amax == 0 ? 1 : amax / 127;

#لارښوونه unroll(Elements_فی_WI)
    لكل (صحيح i = 0; i < Elements_فی_WI; i++) {
        quantized_ارزښتونه[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
کينډۍ <صحيح Elements_فی_WI> بنية no_quantize_q8_1 {
    فراغ چلوونکی()(ثابت عائم *, فراغ *, صحيح, صحيح, ثابت sycl::nd_توکی<1> &) ثابت {}
};

کينډۍ <صحيح Elements_فی_WI> بنية quantize_and_reorder_q8_1_soa {
    __dpct_inline__ فراغ چلوونکی()(ثابت عائم * __restrict__ x, فراغ * reordered_q8_tensor, ثابت صحيح kx,
                                    ثابت صحيح kx_padded, ثابت sycl::nd_توکی<1> & it) ثابت {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        خپلکاری subgroup_id = it.get_ډله(0);
        خپلکاری wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, Elements_فی_WI> quantized_ارزښتونه;
        عائم                            d   = 0.0f;
        عائم                            مجموعه = 0.0f;
        quantize_q8_1_impl<Elements_فی_WI>(x, quantized_ارزښتونه, d, مجموعه, it);

        ثابت صحيح عدد_بلاکونه_فی_لیکه = kx / QK8_1;
        خپلکاری      لیکه                = subgroup_id / عدد_بلاکونه_فی_لیکه;
        خپلکاری      ستون                = subgroup_id % عدد_بلاکونه_فی_لیکه;
        خپلکاری      لیکه_دننه_واټن         = لیکه * (kx_padded / QK8_1) * حجم(بلاک_q8_1);
        خپلکاری      ستون_دننه_واټن         = QK8_1 * ستون + wi_id * Elements_فی_WI;

        خپلکاری quant_ptr = (int8_t *) ((حرف *) reordered_q8_tensor + لیکه_دننه_واټن + ستون_دننه_واټن);
        *reinterpret_cast<sycl::vec<int8_t, Elements_فی_WI> *>(quant_ptr) = quantized_ارزښتونه;

        خپلکاری ds_ptr = (sycl::half2 *) ((حرف *) reordered_q8_tensor + لیکه_دننه_واټن + kx + ستون * حجم(sycl::half2));
        إذا (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(مجموعه));
        }
    }
};

کينډۍ <صحيح Elements_فی_WI> بنية quantize_q8_1 {
    __dpct_inline__ فراغ چلوونکی()(ثابت عائم * __restrict__ x, فراغ * q8_tensor, ثابت صحيح kx, ثابت صحيح kx_padded,
                                    ثابت sycl::nd_توکی<1> & it) ثابت {
        خپلکاری subgroup_id = it.get_ډله(0);
        خپلکاری wi_id       = it.get_local_id(0);

        ثابت صحيح عدد_بلاکونه_فی_لیکه = kx / QK8_1;
        خپلکاری      لیکه                = subgroup_id / عدد_بلاکونه_فی_لیکه;
        ثابت صحيح pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, Elements_فی_WI> quantized_ارزښتونه;
        عائم                            d   = 0.0f;
        عائم                            مجموعه = 0.0f;
        quantize_q8_1_impl<Elements_فی_WI>(x, quantized_ارزښتونه, d, مجموعه, it);

        بلاک_q8_1 * quant_ptr = (بلاک_q8_1 *) q8_tensor;
        خپلکاری         بلاک_id  = subgroup_id % عدد_بلاکونه_فی_لیکه + لیکه * pitch;

        int8_t * qs                                               = &(quant_ptr[بلاک_id].qs[wi_id * Elements_فی_WI]);
        *reinterpret_cast<sycl::vec<int8_t, Elements_فی_WI> *>(qs) = quantized_ارزښتونه;
        إذا (wi_id == 0) {
            quant_ptr[بلاک_id].ds = sycl::half2(sycl::half(d), sycl::half(مجموعه));
        }
    }
};

کينډۍ <کينډۍ <صحيح> ډول_نوم quantize_f>
فراغ quantize_لیکه_q8_1_sycl(ثابت عائم * x, فراغ * vy, ثابت صحيح kx, ثابت صحيح ky, ثابت صحيح kx_padded,
                            dpct::کتار_ptr stream) {
    static_assert(QK8_1 % WARP_اندازه == 0);
    خپلکاری local_range      = std::size_t(WARP_اندازه);
    خپلکاری عدد_quant_بلاکونه = ky * (kx / QK8_1);
    خپلکاری global_range     = عدد_quant_بلاکونه * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_توکی<1> it) [[sycl::reqd_sub_ډله_اندازه(WARP_اندازه)]] {
                             quantize_f<QK8_1 / WARP_اندازه>()(x, vy, kx, kx_padded, it);
                         });
}
