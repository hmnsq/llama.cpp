#अन्तर्भाव "fuse_to_conv.h"

#अन्तर्भाव <openvino/core/graph_util.hpp>
#अन्तर्भाव <openvino/core/rt_info.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/convolution.hpp>
#अन्तर्भाव <openvino/op/extractimagepatches.hpp>
#अन्तर्भाव <openvino/op/group_conv.hpp>
#अन्तर्भाव <openvino/op/matmul.hpp>
#अन्तर्भाव <openvino/op/pad.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/pattern.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/wrap_type.hpp>
#अन्तर्भाव <utility>

नामाकाश opp = ov::pass::pattern;

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश pass {

FuseToConv::FuseToConv() {
    ध्रुव auto m_in0 = opp::any_निवेशः();
    ध्रुव auto m_in1 = opp::any_निवेशः();
    ध्रुव auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    ध्रुव auto callback = [=](opp::Matcher & m) {
        auto matmul_ग्रन्थिः = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_मूलम्());
        जइ (!matmul_ग्रन्थिः) {
            प्रत्यागम मिच्छ;
        }

        auto unwrap = [](ov::Output<Node> n) {
            जाव (ov::is_type<ov::op::v0::Convert>(n.get_ग्रन्थिः_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_ग्रन्थिः_shared_ptr())) {
                n = n.get_ग्रन्थिः_shared_ptr()->निवेशः_मानम्(0);
            }
            प्रत्यागम n;
        };

        auto get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ExtractImagePatches>, std::shared_ptr<ov::op::v1::Pad>> {
            auto t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_ग्रन्थिः_shared_ptr());
            auto r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_निवेशः_ग्रन्थिः_shared_ptr(0)) : nullptr;
            auto t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_निवेशः_ग्रन्थिः_shared_ptr(0)) : nullptr;
            auto eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(t1->get_निवेशः_ग्रन्थिः_shared_ptr(0)) : nullptr;
            auto pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_निवेशः_ग्रन्थिः_shared_ptr(0)) : nullptr;
            प्रत्यागम {eip, pad};
        };

        तार्किक weight_is_in0 = सच्च;
        auto [eip, pad] = get_im2col(matmul_ग्रन्थिः->निवेशः_मानम्(1));
        ov::Output<Node> w_trace = matmul_ग्रन्थिः->निवेशः_मानम्(0);
        जइ (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_ग्रन्थिः->निवेशः_मानम्(0));
            w_trace = matmul_ग्रन्थिः->निवेशः_मानम्(1);
            weight_is_in0 = मिच्छ;
            जइ (!pad) {
                प्रत्यागम मिच्छ;
            }
        }

        auto pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_निवेशः_ग्रन्थिः_shared_ptr(1));
        auto pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_निवेशः_ग्रन्थिः_shared_ptr(2));
        जइ (!pb_const || !pe_const) {
            प्रत्यागम मिच्छ;
        }

        ध्रुव auto pb = pb_const->cast_vector<std::ptrdiff_t>();
        ध्रुव auto pe = pe_const->cast_vector<std::ptrdiff_t>();
        जइ (pb.size() < 4 || pe.size() < 4) {
            प्रत्यागम मिच्छ;
        }

        auto image_निवेशः = pad->निवेशः_मानम्(0);
        auto image_shape = image_निवेशः.get_partial_shape();
        जइ (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            प्रत्यागम मिच्छ;
        }
        ध्रुव size_t IC = static_cast<size_t>(image_shape[1].get_length());

        w_trace = unwrap(w_trace);
        auto weight_pshape = w_trace.get_partial_shape();
        जइ (!weight_pshape.is_static()) {
            प्रत्यागम मिच्छ;
        }

        ध्रुव auto & ws = weight_pshape.to_shape();
        ध्रुव size_t KH = eip->get_sizes()[0];
        ध्रुव size_t KW = eip->get_sizes()[1];
        ध्रुव size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        जइ (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            जइ ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } अण्णहा जइ (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        ध्रुव तार्किक is_depthwise = groups > 1 && (image_shape[0].get_length() % groups == 0);

        ov::Output<Node> conv_out;
        size_t OC = 0;

        जइ (is_depthwise) {
            ध्रुव size_t N = static_cast<size_t>(image_shape[0].get_length()) / groups;
            ध्रुव size_t IH = static_cast<size_t>(image_shape[2].get_length());
            ध्रुव size_t IW = static_cast<size_t>(image_shape[3].get_length());
            OC = groups;

            auto img_shape_const = register_new_ग्रन्थिः<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                register_new_ग्रन्थिः<ov::op::v1::Reshape>(image_निवेशः, img_shape_const, मिच्छ);

            ध्रुव ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> weight_निवेशः;
            जइ (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ग्रन्थिः_shared_ptr())) {
                weight_निवेशः = register_new_ग्रन्थिः<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } अण्णहा {
                auto shape_const = register_new_ग्रन्थिः<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_निवेशः = register_new_ग्रन्थिः<ov::op::v1::Reshape>(w_trace, shape_const, मिच्छ);
            }

            जइ (weight_निवेशः.get_element_type() != image_reshaped.get_element_type()) {
                weight_निवेशः = register_new_ग्रन्थिः<ov::op::v0::Convert>(weight_निवेशः, image_reshaped.get_element_type());
            }

            conv_out = register_new_ग्रन्थिः<ov::op::v1::GroupConvolution>(
                image_reshaped, weight_निवेशः, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } अण्णहा {
            जइ ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } अण्णहा जइ ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } अण्णहा जइ (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } अण्णहा जइ (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } अण्णहा जइ (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } अण्णहा जइ (kernel_spatial_ic && ov::shape_size(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_size(ws) / kernel_spatial_ic;
            } अण्णहा {
                प्रत्यागम मिच्छ;
            }

            ध्रुव ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> weight_निवेशः;
            जइ (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ग्रन्थिः_shared_ptr())) {
                weight_निवेशः = register_new_ग्रन्थिः<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } अण्णहा {
                auto shape_const = register_new_ग्रन्थिः<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_निवेशः = register_new_ग्रन्थिः<ov::op::v1::Reshape>(w_trace, shape_const, मिच्छ);
            }

            जइ (weight_निवेशः.get_element_type() != image_निवेशः.get_element_type()) {
                weight_निवेशः = register_new_ग्रन्थिः<ov::op::v0::Convert>(weight_निवेशः, image_निवेशः.get_element_type());
            }

            conv_out = register_new_ग्रन्थिः<ov::op::v1::Convolution>(
                image_निवेशः, weight_निवेशः, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr auto target_type = ov::element::f32;
        जइ (conv_out.get_element_type() != target_type) {
            conv_out = register_new_ग्रन्थिः<ov::op::v0::Convert>(conv_out, target_type);
        }

        std::shared_ptr<ov::op::v1::Add> add_ग्रन्थिः;
        ov::Output<Node> bias_निवेशः;

        auto try_fuse_bias = [&](ध्रुव std::shared_ptr<Node> & n) {
            auto add = ov::as_type_ptr<ov::op::v1::Add>(n);
            जइ (!add) {
                प्रत्यागम मिच्छ;
            }
            प्रति (size_t i = 0; i < 2; ++i) {
                जइ (ov::is_type<ov::op::v0::Constant>(add->get_निवेशः_ग्रन्थिः_shared_ptr(i))) {
                    bias_निवेशः = add->निवेशः_मानम्(i);
                    add_ग्रन्थिः = add;
                    प्रत्यागम सच्च;
                }
            }
            प्रत्यागम मिच्छ;
        };

        प्रति (ध्रुव auto & consumer : matmul_ग्रन्थिः->output(0).get_target_inputs()) {
            auto n = consumer.get_ग्रन्थिः()->shared_from_this();
            जइ (try_fuse_bias(n)) {
                भङ्ग;
            }
            जइ (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                प्रति (ध्रुव auto & next : n->output(0).get_target_inputs()) {
                    जइ (try_fuse_bias(next.get_ग्रन्थिः()->shared_from_this())) {
                        भङ्ग;
                    }
                }
            }
            जइ (add_ग्रन्थिः) {
                भङ्ग;
            }
        }

        ov::Output<Node> final_out = conv_out;
        std::shared_ptr<Node> target_ग्रन्थिः = matmul_ग्रन्थिः;

        जइ (add_ग्रन्थिः) {
            ov::Output<Node> bias = bias_निवेशः;
            जइ (bias.get_element_type() != target_type) {
                bias = register_new_ग्रन्थिः<ov::op::v0::Convert>(bias, target_type);
            }
            auto bias_shape = register_new_ग्रन्थिः<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = register_new_ग्रन्थिः<ov::op::v1::Reshape>(bias, bias_shape, मिच्छ);
            final_out = register_new_ग्रन्थिः<ov::op::v1::Add>(conv_out, bias);
            target_ग्रन्थिः = add_ग्रन्थिः;
        }

        जइ (!is_depthwise) {
            auto perm = register_new_ग्रन्थिः<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                weight_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_out = register_new_ग्रन्थिः<ov::op::v1::Transpose>(final_out, perm);
        }

        auto orig_shape = target_ग्रन्थिः->get_output_partial_shape(0);
        जइ (orig_shape.is_static() && final_out.get_partial_shape().is_static()) {
            जइ (ov::shape_size(orig_shape.to_shape()) != ov::shape_size(final_out.get_shape())) {
                प्रत्यागम मिच्छ;
            }
        }
        जइ (orig_shape.is_static() && final_out.get_partial_shape() != orig_shape) {
            auto shape_const = register_new_ग्रन्थिः<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.to_shape());
            final_out = register_new_ग्रन्थिः<ov::op::v1::Reshape>(final_out, shape_const, मिच्छ);
        }

        auto orig_type = target_ग्रन्थिः->get_output_element_type(0);
        जइ (final_out.get_element_type() != orig_type) {
            final_out = register_new_ग्रन्थिः<ov::op::v0::Convert>(final_out, orig_type);
        }

        final_out.get_ग्रन्थिः_shared_ptr()->set_friendly_name(target_ग्रन्थिः->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_out.get_ग्रन्थिः_shared_ptr());
        ov::replace_ग्रन्थिः(target_ग्रन्थिः, final_out.get_ग्रन्थिः_shared_ptr());

        प्रत्यागम सच्च;
    };

    register_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
