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
    ثابت خودکار m_in0 = opp::any_داخلہ();
    ثابت خودکار m_in1 = opp::any_داخلہ();
    ثابت خودکار m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    ثابت خودکار callback = [=](opp::Matcher & m) {
        خودکار matmul_گرہ = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_میل_جڑ());
        إذا (!matmul_گرہ) {
            أرجع خطأ;
        }

        خودکار unwrap = [](ov::Output<Node> n) {
            طالما (ov::is_type<ov::op::v0::Convert>(n.get_گرہ_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_گرہ_shared_ptr())) {
                n = n.get_گرہ_shared_ptr()->داخلہ_قدر(0);
            }
            أرجع n;
        };

        خودکار get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::کڈھو_Image_Patches>, std::shared_ptr<ov::op::v1::Pad>> {
            خودکار t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_گرہ_shared_ptr());
            خودکار r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_داخلہ_گرہ_shared_ptr(0)) : خالی_سوچک;
            خودکار t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_داخلہ_گرہ_shared_ptr(0)) : خالی_سوچک;
            خودکار eip =
                t1 ? ov::as_type_ptr<ov::op::v3::کڈھو_Image_Patches>(t1->get_داخلہ_گرہ_shared_ptr(0)) : خالی_سوچک;
            خودکار pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_داخلہ_گرہ_shared_ptr(0)) : خالی_سوچک;
            أرجع {eip, pad};
        };

        منطقي بھار_is_in0 = سچ;
        خودکار [eip, pad] = get_im2col(matmul_گرہ->داخلہ_قدر(1));
        ov::Output<Node> w_trace = matmul_گرہ->داخلہ_قدر(0);
        إذا (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_گرہ->داخلہ_قدر(0));
            w_trace = matmul_گرہ->داخلہ_قدر(1);
            بھار_is_in0 = خطأ;
            إذا (!pad) {
                أرجع خطأ;
            }
        }

        خودکار pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_داخلہ_گرہ_shared_ptr(1));
        خودکار pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_داخلہ_گرہ_shared_ptr(2));
        إذا (!pb_const || !pe_const) {
            أرجع خطأ;
        }

        ثابت خودکار pb = pb_const->cast_سمتیہ<std::ptrdiff_t>();
        ثابت خودکار pe = pe_const->cast_سمتیہ<std::ptrdiff_t>();
        إذا (pb.size() < 4 || pe.size() < 4) {
            أرجع خطأ;
        }

        خودکار image_داخلہ = pad->داخلہ_قدر(0);
        خودکار image_shape = image_داخلہ.get_partial_shape();
        إذا (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            أرجع خطأ;
        }
        ثابت size_t IC = static_cast<size_t>(image_shape[1].get_لمائی());

        w_trace = unwrap(w_trace);
        خودکار بھار_pshape = w_trace.get_partial_shape();
        إذا (!بھار_pshape.is_static()) {
            أرجع خطأ;
        }

        ثابت خودکار & ws = بھار_pshape.منزل_shape();
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

        ثابت منطقي is_depthwise = groups > 1 && (image_shape[0].get_لمائی() % groups == 0);

        ov::Output<Node> conv_باہرلا;
        size_t OC = 0;

        إذا (is_depthwise) {
            ثابت size_t N = static_cast<size_t>(image_shape[0].get_لمائی()) / groups;
            ثابت size_t IH = static_cast<size_t>(image_shape[2].get_لمائی());
            ثابت size_t IW = static_cast<size_t>(image_shape[3].get_لمائی());
            OC = groups;

            خودکار img_shape_const = رجسٹر_new_گرہ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                رجسٹر_new_گرہ<ov::op::v1::Reshape>(image_داخلہ, img_shape_const, خطأ);

            ثابت ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> بھار_داخلہ;
            إذا (خودکار بھار_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_گرہ_shared_ptr())) {
                بھار_داخلہ = رجسٹر_new_گرہ<ov::op::v0::Constant>(بھار_const->get_element_type(), conv_w_shape,
                                                                       بھار_const->get_انکڑے_ptr());
            } وإلا {
                خودکار shape_const = رجسٹر_new_گرہ<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                بھار_داخلہ = رجسٹر_new_گرہ<ov::op::v1::Reshape>(w_trace, shape_const, خطأ);
            }

            إذا (بھار_داخلہ.get_element_type() != image_reshaped.get_element_type()) {
                بھار_داخلہ = رجسٹر_new_گرہ<ov::op::v0::Convert>(بھار_داخلہ, image_reshaped.get_element_type());
            }

            conv_باہرلا = رجسٹر_new_گرہ<ov::op::v1::گروہ_Convolution>(
                image_reshaped, بھار_داخلہ, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
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
            } وإلا إذا (kernel_spatial_ic && ov::shape_ناپ(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_ناپ(ws) / kernel_spatial_ic;
            } وإلا {
                أرجع خطأ;
            }

            ثابت ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> بھار_داخلہ;
            إذا (خودکار بھار_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_گرہ_shared_ptr())) {
                بھار_داخلہ = رجسٹر_new_گرہ<ov::op::v0::Constant>(بھار_const->get_element_type(), conv_w_shape,
                                                                       بھار_const->get_انکڑے_ptr());
            } وإلا {
                خودکار shape_const = رجسٹر_new_گرہ<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                بھار_داخلہ = رجسٹر_new_گرہ<ov::op::v1::Reshape>(w_trace, shape_const, خطأ);
            }

            إذا (بھار_داخلہ.get_element_type() != image_داخلہ.get_element_type()) {
                بھار_داخلہ = رجسٹر_new_گرہ<ov::op::v0::Convert>(بھار_داخلہ, image_داخلہ.get_element_type());
            }

            conv_باہرلا = رجسٹر_new_گرہ<ov::op::v1::Convolution>(
                image_داخلہ, بھار_داخلہ, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr خودکار نشانہ_type = ov::element::f32;
        إذا (conv_باہرلا.get_element_type() != نشانہ_type) {
            conv_باہرلا = رجسٹر_new_گرہ<ov::op::v0::Convert>(conv_باہرلا, نشانہ_type);
        }

        std::shared_ptr<ov::op::v1::جوڑو> جوڑو_گرہ;
        ov::Output<Node> bias_داخلہ;

        خودکار try_fuse_bias = [&](ثابت std::shared_ptr<Node> & n) {
            خودکار جوڑو_2 = ov::as_type_ptr<ov::op::v1::جوڑو>(n);
            إذا (!جوڑو_2) {
                أرجع خطأ;
            }
            لكل (size_t i = 0; i < 2; ++i) {
                إذا (ov::is_type<ov::op::v0::Constant>(جوڑو_2->get_داخلہ_گرہ_shared_ptr(i))) {
                    bias_داخلہ = جوڑو_2->داخلہ_قدر(i);
                    جوڑو_گرہ = جوڑو_2;
                    أرجع سچ;
                }
            }
            أرجع خطأ;
        };

        لكل (ثابت خودکار & کھپتکار : matmul_گرہ->output(0).get_نشانہ_inputs()) {
            خودکار n = کھپتکار.get_گرہ()->shared_from_this();
            إذا (try_fuse_bias(n)) {
                اكسر;
            }
            إذا (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                لكل (ثابت خودکار & اگلا : n->output(0).get_نشانہ_inputs()) {
                    إذا (try_fuse_bias(اگلا.get_گرہ()->shared_from_this())) {
                        اكسر;
                    }
                }
            }
            إذا (جوڑو_گرہ) {
                اكسر;
            }
        }

        ov::Output<Node> final_باہرلا = conv_باہرلا;
        std::shared_ptr<Node> نشانہ_گرہ = matmul_گرہ;

        إذا (جوڑو_گرہ) {
            ov::Output<Node> bias = bias_داخلہ;
            إذا (bias.get_element_type() != نشانہ_type) {
                bias = رجسٹر_new_گرہ<ov::op::v0::Convert>(bias, نشانہ_type);
            }
            خودکار bias_shape = رجسٹر_new_گرہ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = رجسٹر_new_گرہ<ov::op::v1::Reshape>(bias, bias_shape, خطأ);
            final_باہرلا = رجسٹر_new_گرہ<ov::op::v1::جوڑو>(conv_باہرلا, bias);
            نشانہ_گرہ = جوڑو_گرہ;
        }

        إذا (!is_depthwise) {
            خودکار perm = رجسٹر_new_گرہ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                بھار_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_باہرلا = رجسٹر_new_گرہ<ov::op::v1::Transpose>(final_باہرلا, perm);
        }

        خودکار orig_shape = نشانہ_گرہ->get_output_partial_shape(0);
        إذا (orig_shape.is_static() && final_باہرلا.get_partial_shape().is_static()) {
            إذا (ov::shape_ناپ(orig_shape.منزل_shape()) != ov::shape_ناپ(final_باہرلا.get_shape())) {
                أرجع خطأ;
            }
        }
        إذا (orig_shape.is_static() && final_باہرلا.get_partial_shape() != orig_shape) {
            خودکار shape_const = رجسٹر_new_گرہ<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.منزل_shape());
            final_باہرلا = رجسٹر_new_گرہ<ov::op::v1::Reshape>(final_باہرلا, shape_const, خطأ);
        }

        خودکار orig_type = نشانہ_گرہ->get_output_element_type(0);
        إذا (final_باہرلا.get_element_type() != orig_type) {
            final_باہرلا = رجسٹر_new_گرہ<ov::op::v0::Convert>(final_باہرلا, orig_type);
        }

        final_باہرلا.get_گرہ_shared_ptr()->سیٹ_friendly_name(نشانہ_گرہ->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_باہرلا.get_گرہ_shared_ptr());
        ov::replace_گرہ(نشانہ_گرہ, final_باہرلا.get_گرہ_shared_ptr());

        أرجع سچ;
    };

    رجسٹر_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
