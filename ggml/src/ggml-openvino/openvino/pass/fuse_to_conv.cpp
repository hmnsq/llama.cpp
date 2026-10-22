#ضمن "fuse_to_conv.h"

#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/core/rt_info.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/convolution.hpp>
#ضمن <openvino/op/extractimagepatches.hpp>
#ضمن <openvino/op/group_conv.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <openvino/op/pad.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <openvino/pass/pattern/op/pattern.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>
#ضمن <utility>

نطاق opp = ov::pass::pattern;

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

Fuse_منزل_Conv::Fuse_منزل_Conv() {
    ثابت خودڪار m_in0 = opp::any_داخلا();
    ثابت خودڪار m_in1 = opp::any_داخلا();
    ثابت خودڪار m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    ثابت خودڪار callback = [=](opp::Matcher & m) {
        خودڪار matmul_ڳنڍ = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_ميلاپ_پاڙ());
        إذا (!matmul_ڳنڍ) {
            أرجع خطأ;
        }

        خودڪار unwrap = [](ov::Output<Node> n) {
            طالما (ov::is_type<ov::op::v0::Convert>(n.get_ڳنڍ_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_ڳنڍ_shared_ptr())) {
                n = n.get_ڳنڍ_shared_ptr()->داخلا_قدر(0);
            }
            أرجع n;
        };

        خودڪار get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ڪڍو_Image_Patches>, std::shared_ptr<ov::op::v1::Pad>> {
            خودڪار t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_ڳنڍ_shared_ptr());
            خودڪار r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_داخلا_ڳنڍ_shared_ptr(0)) : خالي_اشارو;
            خودڪار t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_داخلا_ڳنڍ_shared_ptr(0)) : خالي_اشارو;
            خودڪار eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ڪڍو_Image_Patches>(t1->get_داخلا_ڳنڍ_shared_ptr(0)) : خالي_اشارو;
            خودڪار pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_داخلا_ڳنڍ_shared_ptr(0)) : خالي_اشارو;
            أرجع {eip, pad};
        };

        منطقي وزن_is_in0 = سچ;
        خودڪار [eip, pad] = get_im2col(matmul_ڳنڍ->داخلا_قدر(1));
        ov::Output<Node> w_trace = matmul_ڳنڍ->داخلا_قدر(0);
        إذا (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_ڳنڍ->داخلا_قدر(0));
            w_trace = matmul_ڳنڍ->داخلا_قدر(1);
            وزن_is_in0 = خطأ;
            إذا (!pad) {
                أرجع خطأ;
            }
        }

        خودڪار pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_داخلا_ڳنڍ_shared_ptr(1));
        خودڪار pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_داخلا_ڳنڍ_shared_ptr(2));
        إذا (!pb_const || !pe_const) {
            أرجع خطأ;
        }

        ثابت خودڪار pb = pb_const->cast_ويڪٽر<std::ptrdiff_t>();
        ثابت خودڪار pe = pe_const->cast_ويڪٽر<std::ptrdiff_t>();
        إذا (pb.size() < 4 || pe.size() < 4) {
            أرجع خطأ;
        }

        خودڪار image_داخلا = pad->داخلا_قدر(0);
        خودڪار image_shape = image_داخلا.get_partial_shape();
        إذا (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            أرجع خطأ;
        }
        ثابت size_t IC = static_cast<size_t>(image_shape[1].get_ڊيگهائي());

        w_trace = unwrap(w_trace);
        خودڪار وزن_pshape = w_trace.get_partial_shape();
        إذا (!وزن_pshape.is_static()) {
            أرجع خطأ;
        }

        ثابت خودڪار & ws = وزن_pshape.منزل_shape();
        ثابت size_t KH = eip->get_sizes()[0];
        ثابت size_t KW = eip->get_sizes()[1];
        ثابت size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        إذا (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            إذا ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } وإلا إذا (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        ثابت منطقي is_depthwise = groups > 1 && (image_shape[0].get_ڊيگهائي() % groups == 0);

        ov::Output<Node> conv_ٻاهريون;
        size_t OC = 0;

        إذا (is_depthwise) {
            ثابت size_t N = static_cast<size_t>(image_shape[0].get_ڊيگهائي()) / groups;
            ثابت size_t IH = static_cast<size_t>(image_shape[2].get_ڊيگهائي());
            ثابت size_t IW = static_cast<size_t>(image_shape[3].get_ڊيگهائي());
            OC = groups;

            خودڪار img_shape_const = رجسٽر_new_ڳنڍ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                رجسٽر_new_ڳنڍ<ov::op::v1::Reshape>(image_داخلا, img_shape_const, خطأ);

            ثابت ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> وزن_داخلا;
            إذا (خودڪار وزن_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ڳنڍ_shared_ptr())) {
                وزن_داخلا = رجسٽر_new_ڳنڍ<ov::op::v0::Constant>(وزن_const->get_element_type(), conv_w_shape,
                                                                       وزن_const->get_معلومات_ptr());
            } وإلا {
                خودڪار shape_const = رجسٽر_new_ڳنڍ<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                وزن_داخلا = رجسٽر_new_ڳنڍ<ov::op::v1::Reshape>(w_trace, shape_const, خطأ);
            }

            إذا (وزن_داخلا.get_element_type() != image_reshaped.get_element_type()) {
                وزن_داخلا = رجسٽر_new_ڳنڍ<ov::op::v0::Convert>(وزن_داخلا, image_reshaped.get_element_type());
            }

            conv_ٻاهريون = رجسٽر_new_ڳنڍ<ov::op::v1::گروهه_Convolution>(
                image_reshaped, وزن_داخلا, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } وإلا {
            إذا ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } وإلا إذا ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } وإلا إذا (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } وإلا إذا (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } وإلا إذا (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } وإلا إذا (kernel_spatial_ic && ov::shape_سائيز(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_سائيز(ws) / kernel_spatial_ic;
            } وإلا {
                أرجع خطأ;
            }

            ثابت ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> وزن_داخلا;
            إذا (خودڪار وزن_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ڳنڍ_shared_ptr())) {
                وزن_داخلا = رجسٽر_new_ڳنڍ<ov::op::v0::Constant>(وزن_const->get_element_type(), conv_w_shape,
                                                                       وزن_const->get_معلومات_ptr());
            } وإلا {
                خودڪار shape_const = رجسٽر_new_ڳنڍ<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                وزن_داخلا = رجسٽر_new_ڳنڍ<ov::op::v1::Reshape>(w_trace, shape_const, خطأ);
            }

            إذا (وزن_داخلا.get_element_type() != image_داخلا.get_element_type()) {
                وزن_داخلا = رجسٽر_new_ڳنڍ<ov::op::v0::Convert>(وزن_داخلا, image_داخلا.get_element_type());
            }

            conv_ٻاهريون = رجسٽر_new_ڳنڍ<ov::op::v1::Convolution>(
                image_داخلا, وزن_داخلا, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr خودڪار هدف_type = ov::element::f32;
        إذا (conv_ٻاهريون.get_element_type() != هدف_type) {
            conv_ٻاهريون = رجسٽر_new_ڳنڍ<ov::op::v0::Convert>(conv_ٻاهريون, هدف_type);
        }

        std::shared_ptr<ov::op::v1::جوڙيو> جوڙيو_ڳنڍ;
        ov::Output<Node> bias_داخلا;

        خودڪار try_fuse_bias = [&](ثابت std::shared_ptr<Node> & n) {
            خودڪار جوڙيو_2 = ov::as_type_ptr<ov::op::v1::جوڙيو>(n);
            إذا (!جوڙيو_2) {
                أرجع خطأ;
            }
            لكل (size_t i = 0; i < 2; ++i) {
                إذا (ov::is_type<ov::op::v0::Constant>(جوڙيو_2->get_داخلا_ڳنڍ_shared_ptr(i))) {
                    bias_داخلا = جوڙيو_2->داخلا_قدر(i);
                    جوڙيو_ڳنڍ = جوڙيو_2;
                    أرجع سچ;
                }
            }
            أرجع خطأ;
        };

        لكل (ثابت خودڪار & استعمال_ڪندڙ : matmul_ڳنڍ->output(0).get_هدف_inputs()) {
            خودڪار n = استعمال_ڪندڙ.get_ڳنڍ()->shared_from_this();
            إذا (try_fuse_bias(n)) {
                اكسر;
            }
            إذا (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                لكل (ثابت خودڪار & ايندڙ : n->output(0).get_هدف_inputs()) {
                    إذا (try_fuse_bias(ايندڙ.get_ڳنڍ()->shared_from_this())) {
                        اكسر;
                    }
                }
            }
            إذا (جوڙيو_ڳنڍ) {
                اكسر;
            }
        }

        ov::Output<Node> final_ٻاهريون = conv_ٻاهريون;
        std::shared_ptr<Node> هدف_ڳنڍ = matmul_ڳنڍ;

        إذا (جوڙيو_ڳنڍ) {
            ov::Output<Node> bias = bias_داخلا;
            إذا (bias.get_element_type() != هدف_type) {
                bias = رجسٽر_new_ڳنڍ<ov::op::v0::Convert>(bias, هدف_type);
            }
            خودڪار bias_shape = رجسٽر_new_ڳنڍ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = رجسٽر_new_ڳنڍ<ov::op::v1::Reshape>(bias, bias_shape, خطأ);
            final_ٻاهريون = رجسٽر_new_ڳنڍ<ov::op::v1::جوڙيو>(conv_ٻاهريون, bias);
            هدف_ڳنڍ = جوڙيو_ڳنڍ;
        }

        إذا (!is_depthwise) {
            خودڪار perm = رجسٽر_new_ڳنڍ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                وزن_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_ٻاهريون = رجسٽر_new_ڳنڍ<ov::op::v1::Transpose>(final_ٻاهريون, perm);
        }

        خودڪار orig_shape = هدف_ڳنڍ->get_output_partial_shape(0);
        إذا (orig_shape.is_static() && final_ٻاهريون.get_partial_shape().is_static()) {
            إذا (ov::shape_سائيز(orig_shape.منزل_shape()) != ov::shape_سائيز(final_ٻاهريون.get_shape())) {
                أرجع خطأ;
            }
        }
        إذا (orig_shape.is_static() && final_ٻاهريون.get_partial_shape() != orig_shape) {
            خودڪار shape_const = رجسٽر_new_ڳنڍ<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.منزل_shape());
            final_ٻاهريون = رجسٽر_new_ڳنڍ<ov::op::v1::Reshape>(final_ٻاهريون, shape_const, خطأ);
        }

        خودڪار orig_type = هدف_ڳنڍ->get_output_element_type(0);
        إذا (final_ٻاهريون.get_element_type() != orig_type) {
            final_ٻاهريون = رجسٽر_new_ڳنڍ<ov::op::v0::Convert>(final_ٻاهريون, orig_type);
        }

        final_ٻاهريون.get_ڳنڍ_shared_ptr()->سيٽ_friendly_name(هدف_ڳنڍ->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_ٻاهريون.get_ڳنڍ_shared_ptr());
        ov::replace_ڳنڍ(هدف_ڳنڍ, final_ٻاهريون.get_ڳنڍ_shared_ptr());

        أرجع سچ;
    };

    رجسٽر_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
