#شامل "fuse_to_conv.h"

#شامل <openvino/core/graph_util.hpp>
#شامل <openvino/core/rt_info.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/convolution.hpp>
#شامل <openvino/op/extractimagepatches.hpp>
#شامل <openvino/op/group_conv.hpp>
#شامل <openvino/op/matmul.hpp>
#شامل <openvino/op/pad.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/transpose.hpp>
#شامل <openvino/pass/pattern/op/pattern.hpp>
#شامل <openvino/pass/pattern/op/wrap_type.hpp>
#شامل <utility>

نامخلا opp = ov::pass::pattern;

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا pass {

Fuse_مقصد_Conv::Fuse_مقصد_Conv() {
    ثابت خودکار m_in0 = opp::any_داخلہ();
    ثابت خودکار m_in1 = opp::any_داخلہ();
    ثابت خودکار m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    ثابت خودکار callback = [=](opp::Matcher & m) {
        خودکار matmul_گرہ = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_مطابقت_جڑ());
        اگر (!matmul_گرہ) {
            واپس جھوٹ;
        }

        خودکار unwrap = [](ov::Output<Node> n) {
            جب (ov::is_type<ov::op::v0::Convert>(n.get_گرہ_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_گرہ_shared_ptr())) {
                n = n.get_گرہ_shared_ptr()->داخلہ_قدر(0);
            }
            واپس n;
        };

        خودکار get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::نکالو_Image_Patches>, std::shared_ptr<ov::op::v1::Pad>> {
            خودکار t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_گرہ_shared_ptr());
            خودکار r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_داخلہ_گرہ_shared_ptr(0)) : خالی_اشاریہ;
            خودکار t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_داخلہ_گرہ_shared_ptr(0)) : خالی_اشاریہ;
            خودکار eip =
                t1 ? ov::as_type_ptr<ov::op::v3::نکالو_Image_Patches>(t1->get_داخلہ_گرہ_shared_ptr(0)) : خالی_اشاریہ;
            خودکار pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_داخلہ_گرہ_shared_ptr(0)) : خالی_اشاریہ;
            واپس {eip, pad};
        };

        منطقی وزن_is_in0 = سچ;
        خودکار [eip, pad] = get_im2col(matmul_گرہ->داخلہ_قدر(1));
        ov::Output<Node> w_trace = matmul_گرہ->داخلہ_قدر(0);
        اگر (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_گرہ->داخلہ_قدر(0));
            w_trace = matmul_گرہ->داخلہ_قدر(1);
            وزن_is_in0 = جھوٹ;
            اگر (!pad) {
                واپس جھوٹ;
            }
        }

        خودکار pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_داخلہ_گرہ_shared_ptr(1));
        خودکار pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_داخلہ_گرہ_shared_ptr(2));
        اگر (!pb_const || !pe_const) {
            واپس جھوٹ;
        }

        ثابت خودکار pb = pb_const->cast_سمتیہ<std::ptrdiff_t>();
        ثابت خودکار pe = pe_const->cast_سمتیہ<std::ptrdiff_t>();
        اگر (pb.size() < 4 || pe.size() < 4) {
            واپس جھوٹ;
        }

        خودکار image_داخلہ = pad->داخلہ_قدر(0);
        خودکار image_shape = image_داخلہ.get_partial_shape();
        اگر (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            واپس جھوٹ;
        }
        ثابت size_t IC = static_cast<size_t>(image_shape[1].get_لمبائی());

        w_trace = unwrap(w_trace);
        خودکار وزن_pshape = w_trace.get_partial_shape();
        اگر (!وزن_pshape.is_static()) {
            واپس جھوٹ;
        }

        ثابت خودکار & ws = وزن_pshape.مقصد_shape();
        ثابت size_t KH = eip->get_sizes()[0];
        ثابت size_t KW = eip->get_sizes()[1];
        ثابت size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        اگر (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            اگر ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } ورنہ اگر (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        ثابت منطقی is_depthwise = groups > 1 && (image_shape[0].get_لمبائی() % groups == 0);

        ov::Output<Node> conv_برآمد;
        size_t OC = 0;

        اگر (is_depthwise) {
            ثابت size_t N = static_cast<size_t>(image_shape[0].get_لمبائی()) / groups;
            ثابت size_t IH = static_cast<size_t>(image_shape[2].get_لمبائی());
            ثابت size_t IW = static_cast<size_t>(image_shape[3].get_لمبائی());
            OC = groups;

            خودکار img_shape_const = رجسٹر_new_گرہ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                رجسٹر_new_گرہ<ov::op::v1::Reshape>(image_داخلہ, img_shape_const, جھوٹ);

            ثابت ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> وزن_داخلہ;
            اگر (خودکار وزن_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_گرہ_shared_ptr())) {
                وزن_داخلہ = رجسٹر_new_گرہ<ov::op::v0::Constant>(وزن_const->get_element_type(), conv_w_shape,
                                                                       وزن_const->get_ڈیٹا_ptr());
            } ورنہ {
                خودکار shape_const = رجسٹر_new_گرہ<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                وزن_داخلہ = رجسٹر_new_گرہ<ov::op::v1::Reshape>(w_trace, shape_const, جھوٹ);
            }

            اگر (وزن_داخلہ.get_element_type() != image_reshaped.get_element_type()) {
                وزن_داخلہ = رجسٹر_new_گرہ<ov::op::v0::Convert>(وزن_داخلہ, image_reshaped.get_element_type());
            }

            conv_برآمد = رجسٹر_new_گرہ<ov::op::v1::گروہ_Convolution>(
                image_reshaped, وزن_داخلہ, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } ورنہ {
            اگر ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } ورنہ اگر ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } ورنہ اگر (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } ورنہ اگر (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } ورنہ اگر (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } ورنہ اگر (kernel_spatial_ic && ov::shape_سائز(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_سائز(ws) / kernel_spatial_ic;
            } ورنہ {
                واپس جھوٹ;
            }

            ثابت ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> وزن_داخلہ;
            اگر (خودکار وزن_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_گرہ_shared_ptr())) {
                وزن_داخلہ = رجسٹر_new_گرہ<ov::op::v0::Constant>(وزن_const->get_element_type(), conv_w_shape,
                                                                       وزن_const->get_ڈیٹا_ptr());
            } ورنہ {
                خودکار shape_const = رجسٹر_new_گرہ<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                وزن_داخلہ = رجسٹر_new_گرہ<ov::op::v1::Reshape>(w_trace, shape_const, جھوٹ);
            }

            اگر (وزن_داخلہ.get_element_type() != image_داخلہ.get_element_type()) {
                وزن_داخلہ = رجسٹر_new_گرہ<ov::op::v0::Convert>(وزن_داخلہ, image_داخلہ.get_element_type());
            }

            conv_برآمد = رجسٹر_new_گرہ<ov::op::v1::Convolution>(
                image_داخلہ, وزن_داخلہ, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr خودکار ہدف_type = ov::element::f32;
        اگر (conv_برآمد.get_element_type() != ہدف_type) {
            conv_برآمد = رجسٹر_new_گرہ<ov::op::v0::Convert>(conv_برآمد, ہدف_type);
        }

        std::shared_ptr<ov::op::v1::جوڑو> جوڑو_گرہ;
        ov::Output<Node> bias_داخلہ;

        خودکار try_fuse_bias = [&](ثابت std::shared_ptr<Node> & n) {
            خودکار جوڑو_2 = ov::as_type_ptr<ov::op::v1::جوڑو>(n);
            اگر (!جوڑو_2) {
                واپس جھوٹ;
            }
            برائے (size_t i = 0; i < 2; ++i) {
                اگر (ov::is_type<ov::op::v0::Constant>(جوڑو_2->get_داخلہ_گرہ_shared_ptr(i))) {
                    bias_داخلہ = جوڑو_2->داخلہ_قدر(i);
                    جوڑو_گرہ = جوڑو_2;
                    واپس سچ;
                }
            }
            واپس جھوٹ;
        };

        برائے (ثابت خودکار & صارف : matmul_گرہ->output(0).get_ہدف_inputs()) {
            خودکار n = صارف.get_گرہ()->shared_from_this();
            اگر (try_fuse_bias(n)) {
                توڑو;
            }
            اگر (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                برائے (ثابت خودکار & اگلا : n->output(0).get_ہدف_inputs()) {
                    اگر (try_fuse_bias(اگلا.get_گرہ()->shared_from_this())) {
                        توڑو;
                    }
                }
            }
            اگر (جوڑو_گرہ) {
                توڑو;
            }
        }

        ov::Output<Node> final_برآمد = conv_برآمد;
        std::shared_ptr<Node> ہدف_گرہ = matmul_گرہ;

        اگر (جوڑو_گرہ) {
            ov::Output<Node> bias = bias_داخلہ;
            اگر (bias.get_element_type() != ہدف_type) {
                bias = رجسٹر_new_گرہ<ov::op::v0::Convert>(bias, ہدف_type);
            }
            خودکار bias_shape = رجسٹر_new_گرہ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = رجسٹر_new_گرہ<ov::op::v1::Reshape>(bias, bias_shape, جھوٹ);
            final_برآمد = رجسٹر_new_گرہ<ov::op::v1::جوڑو>(conv_برآمد, bias);
            ہدف_گرہ = جوڑو_گرہ;
        }

        اگر (!is_depthwise) {
            خودکار perm = رجسٹر_new_گرہ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                وزن_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_برآمد = رجسٹر_new_گرہ<ov::op::v1::Transpose>(final_برآمد, perm);
        }

        خودکار orig_shape = ہدف_گرہ->get_output_partial_shape(0);
        اگر (orig_shape.is_static() && final_برآمد.get_partial_shape().is_static()) {
            اگر (ov::shape_سائز(orig_shape.مقصد_shape()) != ov::shape_سائز(final_برآمد.get_shape())) {
                واپس جھوٹ;
            }
        }
        اگر (orig_shape.is_static() && final_برآمد.get_partial_shape() != orig_shape) {
            خودکار shape_const = رجسٹر_new_گرہ<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.مقصد_shape());
            final_برآمد = رجسٹر_new_گرہ<ov::op::v1::Reshape>(final_برآمد, shape_const, جھوٹ);
        }

        خودکار orig_type = ہدف_گرہ->get_output_element_type(0);
        اگر (final_برآمد.get_element_type() != orig_type) {
            final_برآمد = رجسٹر_new_گرہ<ov::op::v0::Convert>(final_برآمد, orig_type);
        }

        final_برآمد.get_گرہ_shared_ptr()->سیٹ_friendly_name(ہدف_گرہ->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_برآمد.get_گرہ_shared_ptr());
        ov::replace_گرہ(ہدف_گرہ, final_برآمد.get_گرہ_shared_ptr());

        واپس سچ;
    };

    رجسٹر_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
