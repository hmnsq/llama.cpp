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

FuseToConv::FuseToConv() {
    ثابت auto m_in0 = opp::any_ماسوقن();
    ثابت auto m_in1 = opp::any_ماسوقن();
    ثابت auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    ثابت auto callback = [=](opp::Matcher & m) {
        auto matmul_نود = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_اکر());
        إذا (!matmul_نود) {
            أرجع خطأ;
        }

        auto unwrap = [](ov::Output<Node> n) {
            طالما (ov::is_type<ov::op::v0::Convert>(n.get_نود_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_نود_shared_ptr())) {
                n = n.get_نود_shared_ptr()->ماسوقن_نيلاي(0);
            }
            أرجع n;
        };

        auto get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ExtractImagePatches>, std::shared_ptr<ov::op::v1::Pad>> {
            auto t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_نود_shared_ptr());
            auto r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_ماسوقن_نود_shared_ptr(0)) : nullptr;
            auto t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_ماسوقن_نود_shared_ptr(0)) : nullptr;
            auto eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(t1->get_ماسوقن_نود_shared_ptr(0)) : nullptr;
            auto pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_ماسوقن_نود_shared_ptr(0)) : nullptr;
            أرجع {eip, pad};
        };

        منطقي weight_is_in0 = true;
        auto [eip, pad] = get_im2col(matmul_نود->ماسوقن_نيلاي(1));
        ov::Output<Node> w_trace = matmul_نود->ماسوقن_نيلاي(0);
        إذا (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_نود->ماسوقن_نيلاي(0));
            w_trace = matmul_نود->ماسوقن_نيلاي(1);
            weight_is_in0 = خطأ;
            إذا (!pad) {
                أرجع خطأ;
            }
        }

        auto pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_ماسوقن_نود_shared_ptr(1));
        auto pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_ماسوقن_نود_shared_ptr(2));
        إذا (!pb_const || !pe_const) {
            أرجع خطأ;
        }

        ثابت auto pb = pb_const->cast_vector<std::ptrdiff_t>();
        ثابت auto pe = pe_const->cast_vector<std::ptrdiff_t>();
        إذا (pb.size() < 4 || pe.size() < 4) {
            أرجع خطأ;
        }

        auto image_ماسوقن = pad->ماسوقن_نيلاي(0);
        auto image_shape = image_ماسوقن.get_partial_shape();
        إذا (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            أرجع خطأ;
        }
        ثابت size_t IC = static_cast<size_t>(image_shape[1].get_length());

        w_trace = unwrap(w_trace);
        auto weight_pshape = w_trace.get_partial_shape();
        إذا (!weight_pshape.is_static()) {
            أرجع خطأ;
        }

        ثابت auto & ws = weight_pshape.to_shape();
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

        ثابت منطقي is_depthwise = groups > 1 && (image_shape[0].get_length() % groups == 0);

        ov::Output<Node> conv_out;
        size_t OC = 0;

        إذا (is_depthwise) {
            ثابت size_t N = static_cast<size_t>(image_shape[0].get_length()) / groups;
            ثابت size_t IH = static_cast<size_t>(image_shape[2].get_length());
            ثابت size_t IW = static_cast<size_t>(image_shape[3].get_length());
            OC = groups;

            auto img_shape_const = register_new_نود<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                register_new_نود<ov::op::v1::Reshape>(image_ماسوقن, img_shape_const, خطأ);

            ثابت ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> weight_ماسوقن;
            إذا (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_نود_shared_ptr())) {
                weight_ماسوقن = register_new_نود<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } وإلا {
                auto shape_const = register_new_نود<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_ماسوقن = register_new_نود<ov::op::v1::Reshape>(w_trace, shape_const, خطأ);
            }

            إذا (weight_ماسوقن.get_element_type() != image_reshaped.get_element_type()) {
                weight_ماسوقن = register_new_نود<ov::op::v0::Convert>(weight_ماسوقن, image_reshaped.get_element_type());
            }

            conv_out = register_new_نود<ov::op::v1::GroupConvolution>(
                image_reshaped, weight_ماسوقن, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
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
            } وإلا إذا (kernel_spatial_ic && ov::shape_size(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_size(ws) / kernel_spatial_ic;
            } وإلا {
                أرجع خطأ;
            }

            ثابت ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> weight_ماسوقن;
            إذا (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_نود_shared_ptr())) {
                weight_ماسوقن = register_new_نود<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } وإلا {
                auto shape_const = register_new_نود<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_ماسوقن = register_new_نود<ov::op::v1::Reshape>(w_trace, shape_const, خطأ);
            }

            إذا (weight_ماسوقن.get_element_type() != image_ماسوقن.get_element_type()) {
                weight_ماسوقن = register_new_نود<ov::op::v0::Convert>(weight_ماسوقن, image_ماسوقن.get_element_type());
            }

            conv_out = register_new_نود<ov::op::v1::Convolution>(
                image_ماسوقن, weight_ماسوقن, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr auto target_type = ov::element::f32;
        إذا (conv_out.get_element_type() != target_type) {
            conv_out = register_new_نود<ov::op::v0::Convert>(conv_out, target_type);
        }

        std::shared_ptr<ov::op::v1::Add> add_نود;
        ov::Output<Node> bias_ماسوقن;

        auto try_fuse_bias = [&](ثابت std::shared_ptr<Node> & n) {
            auto add = ov::as_type_ptr<ov::op::v1::Add>(n);
            إذا (!add) {
                أرجع خطأ;
            }
            لكل (size_t i = 0; i < 2; ++i) {
                إذا (ov::is_type<ov::op::v0::Constant>(add->get_ماسوقن_نود_shared_ptr(i))) {
                    bias_ماسوقن = add->ماسوقن_نيلاي(i);
                    add_نود = add;
                    أرجع true;
                }
            }
            أرجع خطأ;
        };

        لكل (ثابت auto & consumer : matmul_نود->output(0).get_target_inputs()) {
            auto n = consumer.get_نود()->shared_from_this();
            إذا (try_fuse_bias(n)) {
                اكسر;
            }
            إذا (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                لكل (ثابت auto & next : n->output(0).get_target_inputs()) {
                    إذا (try_fuse_bias(next.get_نود()->shared_from_this())) {
                        اكسر;
                    }
                }
            }
            إذا (add_نود) {
                اكسر;
            }
        }

        ov::Output<Node> final_out = conv_out;
        std::shared_ptr<Node> target_نود = matmul_نود;

        إذا (add_نود) {
            ov::Output<Node> bias = bias_ماسوقن;
            إذا (bias.get_element_type() != target_type) {
                bias = register_new_نود<ov::op::v0::Convert>(bias, target_type);
            }
            auto bias_shape = register_new_نود<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = register_new_نود<ov::op::v1::Reshape>(bias, bias_shape, خطأ);
            final_out = register_new_نود<ov::op::v1::Add>(conv_out, bias);
            target_نود = add_نود;
        }

        إذا (!is_depthwise) {
            auto perm = register_new_نود<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                weight_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_out = register_new_نود<ov::op::v1::Transpose>(final_out, perm);
        }

        auto orig_shape = target_نود->get_output_partial_shape(0);
        إذا (orig_shape.is_static() && final_out.get_partial_shape().is_static()) {
            إذا (ov::shape_size(orig_shape.to_shape()) != ov::shape_size(final_out.get_shape())) {
                أرجع خطأ;
            }
        }
        إذا (orig_shape.is_static() && final_out.get_partial_shape() != orig_shape) {
            auto shape_const = register_new_نود<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.to_shape());
            final_out = register_new_نود<ov::op::v1::Reshape>(final_out, shape_const, خطأ);
        }

        auto orig_type = target_نود->get_output_element_type(0);
        إذا (final_out.get_element_type() != orig_type) {
            final_out = register_new_نود<ov::op::v0::Convert>(final_out, orig_type);
        }

        final_out.get_نود_shared_ptr()->set_friendly_name(target_نود->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_out.get_نود_shared_ptr());
        ov::replace_نود(target_نود, final_out.get_نود_shared_ptr());

        أرجع true;
    };

    register_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
