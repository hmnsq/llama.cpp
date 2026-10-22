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

#సూచన once

#అన్తర్భావ <sycl/nd_item.hpp>

#అన్తర్భావ "ggml-sycl/dpct/helper.hpp"

మూస <పూర్ణాంకం Elements_చొప్పున_WI>
__dpct_inline__ స్థిర శూన్య quantize_q8_1_impl(స్థిరరాశి దశాంశం * __restrict__ x,
                                               sycl::vec<int8_t, Elements_చొప్పున_WI> & quantized_విలువలు, దశాంశం & d,
                                               దశాంశం & కూడిక, స్థిరరాశి sycl::nd_అంశం<1> & it) {
    స్వయంచాలక subgroup_id = it.get_సమూహం(0);
    స్వయంచాలక wi_id       = it.get_local_id(0);

    sycl::vec<దశాంశం, Elements_చొప్పున_WI> wi_f32_vals;

    స్వయంచాలక float_ptr_స్థానభ్రంశం = subgroup_id * QK8_1 + Elements_చొప్పున_WI * wi_id;
    wi_f32_vals           = *reinterpret_cast<స్థిరరాశి sycl::vec<దశాంశం, Elements_చొప్పున_WI> *>(x + float_ptr_స్థానభ్రంశం);

    దశాంశం amax = 0.0f;

#సూచన unroll(Elements_చొప్పున_WI)
    కొరకు (పూర్ణాంకం i = 0; i < Elements_చొప్పున_WI; i++) {
        కూడిక += wi_f32_vals[i];
        amax                = sycl::fmax(amax, sycl::fabs(wi_f32_vals[i]));
        quantized_విలువలు[i] = 0;
    }
    కూడిక  = sycl::reduce_over_సమూహం(it.get_sub_సమూహం(), కూడిక, sycl::plus<దశాంశం>());
    amax = sycl::reduce_over_సమూహం(it.get_sub_సమూహం(), amax, sycl::గరిష్ఠ<దశాంశం>());
    d    = amax == 0 ? 1 : amax / 127;

#సూచన unroll(Elements_చొప్పున_WI)
    కొరకు (పూర్ణాంకం i = 0; i < Elements_చొప్పున_WI; i++) {
        quantized_విలువలు[i] = sycl::round(wi_f32_vals[i] / d);
    }

    d = amax == 0 ? 0 : d;
}

// No op to control codepath in ggml_sycl_op_mul_mat
మూస <పూర్ణాంకం Elements_చొప్పున_WI> నిర్మాణం no_quantize_q8_1 {
    శూన్య క్రియాకారకం()(స్థిరరాశి దశాంశం *, శూన్య *, పూర్ణాంకం, పూర్ణాంకం, స్థిరరాశి sycl::nd_అంశం<1> &) స్థిరరాశి {}
};

మూస <పూర్ణాంకం Elements_చొప్పున_WI> నిర్మాణం quantize_and_reorder_q8_1_soa {
    __dpct_inline__ శూన్య క్రియాకారకం()(స్థిరరాశి దశాంశం * __restrict__ x, శూన్య * reordered_q8_tensor, స్థిరరాశి పూర్ణాంకం kx,
                                    స్థిరరాశి పూర్ణాంకం kx_padded, స్థిరరాశి sycl::nd_అంశం<1> & it) స్థిరరాశి {
        /*
        Quantizes and reorders the resultant q8 tensor in a per row fashion
        Each sub-group calculates one quant block. i.e. QK8_1 quant values and the d and sum values
    */
        స్వయంచాలక subgroup_id = it.get_సమూహం(0);
        స్వయంచాలక wi_id       = it.get_local_id(0);

        sycl::vec<int8_t, Elements_చొప్పున_WI> quantized_విలువలు;
        దశాంశం                            d   = 0.0f;
        దశాంశం                            కూడిక = 0.0f;
        quantize_q8_1_impl<Elements_చొప్పున_WI>(x, quantized_విలువలు, d, కూడిక, it);

        స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు_చొప్పున_అడ్డువరుస = kx / QK8_1;
        స్వయంచాలక      అడ్డువరుస                = subgroup_id / అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;
        స్వయంచాలక      నిలువు                = subgroup_id % అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;
        స్వయంచాలక      అడ్డువరుస_స్థానభ్రంశం         = అడ్డువరుస * (kx_padded / QK8_1) * పరిమాణం(ఖండం_q8_1);
        స్వయంచాలక      నిలువు_స్థానభ్రంశం         = QK8_1 * నిలువు + wi_id * Elements_చొప్పున_WI;

        స్వయంచాలక quant_ptr = (int8_t *) ((అక్షరం *) reordered_q8_tensor + అడ్డువరుస_స్థానభ్రంశం + నిలువు_స్థానభ్రంశం);
        *reinterpret_cast<sycl::vec<int8_t, Elements_చొప్పున_WI> *>(quant_ptr) = quantized_విలువలు;

        స్వయంచాలక ds_ptr = (sycl::half2 *) ((అక్షరం *) reordered_q8_tensor + అడ్డువరుస_స్థానభ్రంశం + kx + నిలువు * పరిమాణం(sycl::half2));
        ఒకవేళ (wi_id == 0) {
            *ds_ptr = sycl::half2(sycl::half(d), sycl::half(కూడిక));
        }
    }
};

మూస <పూర్ణాంకం Elements_చొప్పున_WI> నిర్మాణం quantize_q8_1 {
    __dpct_inline__ శూన్య క్రియాకారకం()(స్థిరరాశి దశాంశం * __restrict__ x, శూన్య * q8_tensor, స్థిరరాశి పూర్ణాంకం kx, స్థిరరాశి పూర్ణాంకం kx_padded,
                                    స్థిరరాశి sycl::nd_అంశం<1> & it) స్థిరరాశి {
        స్వయంచాలక subgroup_id = it.get_సమూహం(0);
        స్వయంచాలక wi_id       = it.get_local_id(0);

        స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు_చొప్పున_అడ్డువరుస = kx / QK8_1;
        స్వయంచాలక      అడ్డువరుస                = subgroup_id / అంకె_ఖండాలు_చొప్పున_అడ్డువరుస;
        స్థిరరాశి పూర్ణాంకం pitch              = kx_padded / QK8_1;

        sycl::vec<int8_t, Elements_చొప్పున_WI> quantized_విలువలు;
        దశాంశం                            d   = 0.0f;
        దశాంశం                            కూడిక = 0.0f;
        quantize_q8_1_impl<Elements_చొప్పున_WI>(x, quantized_విలువలు, d, కూడిక, it);

        ఖండం_q8_1 * quant_ptr = (ఖండం_q8_1 *) q8_tensor;
        స్వయంచాలక         ఖండం_id  = subgroup_id % అంకె_ఖండాలు_చొప్పున_అడ్డువరుస + అడ్డువరుస * pitch;

        int8_t * qs                                               = &(quant_ptr[ఖండం_id].qs[wi_id * Elements_చొప్పున_WI]);
        *reinterpret_cast<sycl::vec<int8_t, Elements_చొప్పున_WI> *>(qs) = quantized_విలువలు;
        ఒకవేళ (wi_id == 0) {
            quant_ptr[ఖండం_id].ds = sycl::half2(sycl::half(d), sycl::half(కూడిక));
        }
    }
};

మూస <మూస <పూర్ణాంకం> రకం_పేరు quantize_f>
శూన్య quantize_అడ్డువరుస_q8_1_sycl(స్థిరరాశి దశాంశం * x, శూన్య * vy, స్థిరరాశి పూర్ణాంకం kx, స్థిరరాశి పూర్ణాంకం ky, స్థిరరాశి పూర్ణాంకం kx_padded,
                            dpct::వరుస_ptr stream) {
    static_assert(QK8_1 % WARP_కొలత == 0);
    స్వయంచాలక local_range      = std::size_t(WARP_కొలత);
    స్వయంచాలక అంకె_quant_ఖండాలు = ky * (kx / QK8_1);
    స్వయంచాలక global_range     = అంకె_quant_ఖండాలు * local_range;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    stream->parallel_for(sycl::nd_range<1>({ global_range }, { local_range }),
                         [=](sycl::nd_అంశం<1> it) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                             quantize_f<QK8_1 / WARP_కొలత>()(x, vy, kx, kx_padded, it);
                         });
}
