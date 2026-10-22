#অন্তর্ভাব "fuse_to_conv.h"

#অন্তর্ভাব <openvino/core/graph_util.hpp>
#অন্তর্ভাব <openvino/core/rt_info.hpp>
#অন্তর্ভাব <openvino/op/add.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/convert.hpp>
#অন্তর্ভাব <openvino/op/convolution.hpp>
#অন্তর্ভাব <openvino/op/extractimagepatches.hpp>
#অন্তর্ভাব <openvino/op/group_conv.hpp>
#অন্তর্ভাব <openvino/op/matmul.hpp>
#অন্তর্ভাব <openvino/op/pad.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/transpose.hpp>
#অন্তর্ভাব <openvino/pass/pattern/op/pattern.hpp>
#অন্তর্ভাব <openvino/pass/pattern/op/wrap_type.hpp>
#অন্তর্ভাব <utility>

নামাকাশ opp = ov::pass::pattern;

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ pass {

FuseToConv::FuseToConv() {
    ধ্রুব auto m_in0 = opp::any_প্রবেশ();
    ধ্রুব auto m_in1 = opp::any_প্রবেশ();
    ধ্রুব auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    ধ্রুব auto callback = [=](opp::Matcher & m) {
        auto matmul_গ্রন্থি = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_মূল());
        যদি (!matmul_গ্রন্থি) {
            প্রত্যাগম অসত্য;
        }

        auto unwrap = [](ov::Output<Node> n) {
            যাবত্ (ov::is_type<ov::op::v0::Convert>(n.get_গ্রন্থি_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_গ্রন্থি_shared_ptr())) {
                n = n.get_গ্রন্থি_shared_ptr()->প্রবেশ_মান(0);
            }
            প্রত্যাগম n;
        };

        auto get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ExtractImagePatches>, std::shared_ptr<ov::op::v1::Pad>> {
            auto t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_গ্রন্থি_shared_ptr());
            auto r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_প্রবেশ_গ্রন্থি_shared_ptr(0)) : nullptr;
            auto t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_প্রবেশ_গ্রন্থি_shared_ptr(0)) : nullptr;
            auto eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(t1->get_প্রবেশ_গ্রন্থি_shared_ptr(0)) : nullptr;
            auto pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_প্রবেশ_গ্রন্থি_shared_ptr(0)) : nullptr;
            প্রত্যাগম {eip, pad};
        };

        তার্কিক weight_is_in0 = সত্য;
        auto [eip, pad] = get_im2col(matmul_গ্রন্থি->প্রবেশ_মান(1));
        ov::Output<Node> w_trace = matmul_গ্রন্থি->প্রবেশ_মান(0);
        যদি (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_গ্রন্থি->প্রবেশ_মান(0));
            w_trace = matmul_গ্রন্থি->প্রবেশ_মান(1);
            weight_is_in0 = অসত্য;
            যদি (!pad) {
                প্রত্যাগম অসত্য;
            }
        }

        auto pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_প্রবেশ_গ্রন্থি_shared_ptr(1));
        auto pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_প্রবেশ_গ্রন্থি_shared_ptr(2));
        যদি (!pb_const || !pe_const) {
            প্রত্যাগম অসত্য;
        }

        ধ্রুব auto pb = pb_const->cast_vector<std::ptrdiff_t>();
        ধ্রুব auto pe = pe_const->cast_vector<std::ptrdiff_t>();
        যদি (pb.size() < 4 || pe.size() < 4) {
            প্রত্যাগম অসত্য;
        }

        auto image_প্রবেশ = pad->প্রবেশ_মান(0);
        auto image_shape = image_প্রবেশ.get_partial_shape();
        যদি (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            প্রত্যাগম অসত্য;
        }
        ধ্রুব size_t IC = static_cast<size_t>(image_shape[1].get_length());

        w_trace = unwrap(w_trace);
        auto weight_pshape = w_trace.get_partial_shape();
        যদি (!weight_pshape.is_static()) {
            প্রত্যাগম অসত্য;
        }

        ধ্রুব auto & ws = weight_pshape.to_shape();
        ধ্রুব size_t KH = eip->get_sizes()[0];
        ধ্রুব size_t KW = eip->get_sizes()[1];
        ধ্রুব size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        যদি (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            যদি ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } অন্যথা যদি (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        ধ্রুব তার্কিক is_depthwise = groups > 1 && (image_shape[0].get_length() % groups == 0);

        ov::Output<Node> conv_out;
        size_t OC = 0;

        যদি (is_depthwise) {
            ধ্রুব size_t N = static_cast<size_t>(image_shape[0].get_length()) / groups;
            ধ্রুব size_t IH = static_cast<size_t>(image_shape[2].get_length());
            ধ্রুব size_t IW = static_cast<size_t>(image_shape[3].get_length());
            OC = groups;

            auto img_shape_const = register_new_গ্রন্থি<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                register_new_গ্রন্থি<ov::op::v1::Reshape>(image_প্রবেশ, img_shape_const, অসত্য);

            ধ্রুব ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> weight_প্রবেশ;
            যদি (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_গ্রন্থি_shared_ptr())) {
                weight_প্রবেশ = register_new_গ্রন্থি<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } অন্যথা {
                auto shape_const = register_new_গ্রন্থি<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_প্রবেশ = register_new_গ্রন্থি<ov::op::v1::Reshape>(w_trace, shape_const, অসত্য);
            }

            যদি (weight_প্রবেশ.get_element_type() != image_reshaped.get_element_type()) {
                weight_প্রবেশ = register_new_গ্রন্থি<ov::op::v0::Convert>(weight_প্রবেশ, image_reshaped.get_element_type());
            }

            conv_out = register_new_গ্রন্থি<ov::op::v1::GroupConvolution>(
                image_reshaped, weight_প্রবেশ, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } অন্যথা {
            যদি ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } অন্যথা যদি ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } অন্যথা যদি (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } অন্যথা যদি (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } অন্যথা যদি (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } অন্যথা যদি (kernel_spatial_ic && ov::shape_size(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_size(ws) / kernel_spatial_ic;
            } অন্যথা {
                প্রত্যাগম অসত্য;
            }

            ধ্রুব ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> weight_প্রবেশ;
            যদি (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_গ্রন্থি_shared_ptr())) {
                weight_প্রবেশ = register_new_গ্রন্থি<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } অন্যথা {
                auto shape_const = register_new_গ্রন্থি<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_প্রবেশ = register_new_গ্রন্থি<ov::op::v1::Reshape>(w_trace, shape_const, অসত্য);
            }

            যদি (weight_প্রবেশ.get_element_type() != image_প্রবেশ.get_element_type()) {
                weight_প্রবেশ = register_new_গ্রন্থি<ov::op::v0::Convert>(weight_প্রবেশ, image_প্রবেশ.get_element_type());
            }

            conv_out = register_new_গ্রন্থি<ov::op::v1::Convolution>(
                image_প্রবেশ, weight_প্রবেশ, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr auto target_type = ov::element::f32;
        যদি (conv_out.get_element_type() != target_type) {
            conv_out = register_new_গ্রন্থি<ov::op::v0::Convert>(conv_out, target_type);
        }

        std::shared_ptr<ov::op::v1::Add> add_গ্রন্থি;
        ov::Output<Node> bias_প্রবেশ;

        auto try_fuse_bias = [&](ধ্রুব std::shared_ptr<Node> & n) {
            auto add = ov::as_type_ptr<ov::op::v1::Add>(n);
            যদি (!add) {
                প্রত্যাগম অসত্য;
            }
            প্রতি (size_t i = 0; i < 2; ++i) {
                যদি (ov::is_type<ov::op::v0::Constant>(add->get_প্রবেশ_গ্রন্থি_shared_ptr(i))) {
                    bias_প্রবেশ = add->প্রবেশ_মান(i);
                    add_গ্রন্থি = add;
                    প্রত্যাগম সত্য;
                }
            }
            প্রত্যাগম অসত্য;
        };

        প্রতি (ধ্রুব auto & consumer : matmul_গ্রন্থি->output(0).get_target_inputs()) {
            auto n = consumer.get_গ্রন্থি()->shared_from_this();
            যদি (try_fuse_bias(n)) {
                ভঙ্গ;
            }
            যদি (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                প্রতি (ধ্রুব auto & next : n->output(0).get_target_inputs()) {
                    যদি (try_fuse_bias(next.get_গ্রন্থি()->shared_from_this())) {
                        ভঙ্গ;
                    }
                }
            }
            যদি (add_গ্রন্থি) {
                ভঙ্গ;
            }
        }

        ov::Output<Node> final_out = conv_out;
        std::shared_ptr<Node> target_গ্রন্থি = matmul_গ্রন্থি;

        যদি (add_গ্রন্থি) {
            ov::Output<Node> bias = bias_প্রবেশ;
            যদি (bias.get_element_type() != target_type) {
                bias = register_new_গ্রন্থি<ov::op::v0::Convert>(bias, target_type);
            }
            auto bias_shape = register_new_গ্রন্থি<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = register_new_গ্রন্থি<ov::op::v1::Reshape>(bias, bias_shape, অসত্য);
            final_out = register_new_গ্রন্থি<ov::op::v1::Add>(conv_out, bias);
            target_গ্রন্থি = add_গ্রন্থি;
        }

        যদি (!is_depthwise) {
            auto perm = register_new_গ্রন্থি<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                weight_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_out = register_new_গ্রন্থি<ov::op::v1::Transpose>(final_out, perm);
        }

        auto orig_shape = target_গ্রন্থি->get_output_partial_shape(0);
        যদি (orig_shape.is_static() && final_out.get_partial_shape().is_static()) {
            যদি (ov::shape_size(orig_shape.to_shape()) != ov::shape_size(final_out.get_shape())) {
                প্রত্যাগম অসত্য;
            }
        }
        যদি (orig_shape.is_static() && final_out.get_partial_shape() != orig_shape) {
            auto shape_const = register_new_গ্রন্থি<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.to_shape());
            final_out = register_new_গ্রন্থি<ov::op::v1::Reshape>(final_out, shape_const, অসত্য);
        }

        auto orig_type = target_গ্রন্থি->get_output_element_type(0);
        যদি (final_out.get_element_type() != orig_type) {
            final_out = register_new_গ্রন্থি<ov::op::v0::Convert>(final_out, orig_type);
        }

        final_out.get_গ্রন্থি_shared_ptr()->set_friendly_name(target_গ্রন্থি->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_out.get_গ্রন্থি_shared_ptr());
        ov::replace_গ্রন্থি(target_গ্রন্থি, final_out.get_গ্রন্থি_shared_ptr());

        প্রত্যাগম সত্য;
    };

    register_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
