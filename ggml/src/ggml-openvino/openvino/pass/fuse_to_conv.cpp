#ಅನ್ತರ್ಭಾವ "fuse_to_conv.h"

#ಅನ್ತರ್ಭಾವ <openvino/core/graph_util.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/rt_info.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convolution.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/extractimagepatches.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/group_conv.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/matmul.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/pad.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/pattern.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/wrap_type.hpp>
#ಅನ್ತರ್ಭಾವ <utility>

ನಾಮಾಕಾಶ opp = ov::pass::pattern;

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ pass {

FuseToConv::FuseToConv() {
    ಧ್ರುವ auto m_in0 = opp::any_ಒಳಹರಿವು();
    ಧ್ರುವ auto m_in1 = opp::any_ಒಳಹರಿವು();
    ಧ್ರುವ auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    ಧ್ರುವ auto callback = [=](opp::Matcher & m) {
        auto matmul_ಗಂಟು = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_ಬೇರು());
        ಯದಿ (!matmul_ಗಂಟು) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        auto unwrap = [](ov::Output<Node> n) {
            ಯಾವತ್ (ov::is_type<ov::op::v0::Convert>(n.get_ಗಂಟು_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_ಗಂಟು_shared_ptr())) {
                n = n.get_ಗಂಟು_shared_ptr()->ಒಳಹರಿವು_ಮೌಲ್ಯ(0);
            }
            ಪ್ರತ್ಯಾಗಮ n;
        };

        auto get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ExtractImagePatches>, std::shared_ptr<ov::op::v1::Pad>> {
            auto t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_ಗಂಟು_shared_ptr());
            auto r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0)) : nullptr;
            auto t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0)) : nullptr;
            auto eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(t1->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0)) : nullptr;
            auto pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0)) : nullptr;
            ಪ್ರತ್ಯಾಗಮ {eip, pad};
        };

        ತಾರ್ಕಿಕ weight_is_in0 = ಸತ್ಯ;
        auto [eip, pad] = get_im2col(matmul_ಗಂಟು->ಒಳಹರಿವು_ಮೌಲ್ಯ(1));
        ov::Output<Node> w_trace = matmul_ಗಂಟು->ಒಳಹರಿವು_ಮೌಲ್ಯ(0);
        ಯದಿ (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_ಗಂಟು->ಒಳಹರಿವು_ಮೌಲ್ಯ(0));
            w_trace = matmul_ಗಂಟು->ಒಳಹರಿವು_ಮೌಲ್ಯ(1);
            weight_is_in0 = ಅಸತ್ಯ;
            ಯದಿ (!pad) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }

        auto pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(1));
        auto pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(2));
        ಯದಿ (!pb_const || !pe_const) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ auto pb = pb_const->cast_vector<std::ptrdiff_t>();
        ಧ್ರುವ auto pe = pe_const->cast_vector<std::ptrdiff_t>();
        ಯದಿ (pb.size() < 4 || pe.size() < 4) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        auto image_ಒಳಹರಿವು = pad->ಒಳಹರಿವು_ಮೌಲ್ಯ(0);
        auto image_shape = image_ಒಳಹರಿವು.get_partial_shape();
        ಯದಿ (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಧ್ರುವ size_t IC = static_cast<size_t>(image_shape[1].get_length());

        w_trace = unwrap(w_trace);
        auto weight_pshape = w_trace.get_partial_shape();
        ಯದಿ (!weight_pshape.is_static()) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ auto & ws = weight_pshape.to_shape();
        ಧ್ರುವ size_t KH = eip->get_sizes()[0];
        ಧ್ರುವ size_t KW = eip->get_sizes()[1];
        ಧ್ರುವ size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        ಯದಿ (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            ಯದಿ ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } ಅನ್ಯಥಾ ಯದಿ (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        ಧ್ರುವ ತಾರ್ಕಿಕ is_depthwise = groups > 1 && (image_shape[0].get_length() % groups == 0);

        ov::Output<Node> conv_out;
        size_t OC = 0;

        ಯದಿ (is_depthwise) {
            ಧ್ರುವ size_t N = static_cast<size_t>(image_shape[0].get_length()) / groups;
            ಧ್ರುವ size_t IH = static_cast<size_t>(image_shape[2].get_length());
            ಧ್ರುವ size_t IW = static_cast<size_t>(image_shape[3].get_length());
            OC = groups;

            auto img_shape_const = register_new_ಗಂಟು<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                register_new_ಗಂಟು<ov::op::v1::Reshape>(image_ಒಳಹರಿವು, img_shape_const, ಅಸತ್ಯ);

            ಧ್ರುವ ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> weight_ಒಳಹರಿವು;
            ಯದಿ (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ಗಂಟು_shared_ptr())) {
                weight_ಒಳಹರಿವು = register_new_ಗಂಟು<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } ಅನ್ಯಥಾ {
                auto shape_const = register_new_ಗಂಟು<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_ಒಳಹರಿವು = register_new_ಗಂಟು<ov::op::v1::Reshape>(w_trace, shape_const, ಅಸತ್ಯ);
            }

            ಯದಿ (weight_ಒಳಹರಿವು.get_element_type() != image_reshaped.get_element_type()) {
                weight_ಒಳಹರಿವು = register_new_ಗಂಟು<ov::op::v0::Convert>(weight_ಒಳಹರಿವು, image_reshaped.get_element_type());
            }

            conv_out = register_new_ಗಂಟು<ov::op::v1::GroupConvolution>(
                image_reshaped, weight_ಒಳಹರಿವು, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } ಅನ್ಯಥಾ {
            ಯದಿ ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } ಅನ್ಯಥಾ ಯದಿ ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } ಅನ್ಯಥಾ ಯದಿ (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } ಅನ್ಯಥಾ ಯದಿ (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } ಅನ್ಯಥಾ ಯದಿ (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } ಅನ್ಯಥಾ ಯದಿ (kernel_spatial_ic && ov::shape_size(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_size(ws) / kernel_spatial_ic;
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }

            ಧ್ರುವ ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> weight_ಒಳಹರಿವು;
            ಯದಿ (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ಗಂಟು_shared_ptr())) {
                weight_ಒಳಹರಿವು = register_new_ಗಂಟು<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } ಅನ್ಯಥಾ {
                auto shape_const = register_new_ಗಂಟು<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_ಒಳಹರಿವು = register_new_ಗಂಟು<ov::op::v1::Reshape>(w_trace, shape_const, ಅಸತ್ಯ);
            }

            ಯದಿ (weight_ಒಳಹರಿವು.get_element_type() != image_ಒಳಹರಿವು.get_element_type()) {
                weight_ಒಳಹರಿವು = register_new_ಗಂಟು<ov::op::v0::Convert>(weight_ಒಳಹರಿವು, image_ಒಳಹರಿವು.get_element_type());
            }

            conv_out = register_new_ಗಂಟು<ov::op::v1::Convolution>(
                image_ಒಳಹರಿವು, weight_ಒಳಹರಿವು, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr auto target_type = ov::element::f32;
        ಯದಿ (conv_out.get_element_type() != target_type) {
            conv_out = register_new_ಗಂಟು<ov::op::v0::Convert>(conv_out, target_type);
        }

        std::shared_ptr<ov::op::v1::Add> add_ಗಂಟು;
        ov::Output<Node> bias_ಒಳಹರಿವು;

        auto try_fuse_bias = [&](ಧ್ರುವ std::shared_ptr<Node> & n) {
            auto add = ov::as_type_ptr<ov::op::v1::Add>(n);
            ಯದಿ (!add) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
            ಪ್ರತಿ (size_t i = 0; i < 2; ++i) {
                ಯದಿ (ov::is_type<ov::op::v0::Constant>(add->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(i))) {
                    bias_ಒಳಹರಿವು = add->ಒಳಹರಿವು_ಮೌಲ್ಯ(i);
                    add_ಗಂಟು = add;
                    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
                }
            }
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        };

        ಪ್ರತಿ (ಧ್ರುವ auto & consumer : matmul_ಗಂಟು->output(0).get_target_inputs()) {
            auto n = consumer.get_ಗಂಟು()->shared_from_this();
            ಯದಿ (try_fuse_bias(n)) {
                ಭಙ್ಗ;
            }
            ಯದಿ (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                ಪ್ರತಿ (ಧ್ರುವ auto & next : n->output(0).get_target_inputs()) {
                    ಯದಿ (try_fuse_bias(next.get_ಗಂಟು()->shared_from_this())) {
                        ಭಙ್ಗ;
                    }
                }
            }
            ಯದಿ (add_ಗಂಟು) {
                ಭಙ್ಗ;
            }
        }

        ov::Output<Node> final_out = conv_out;
        std::shared_ptr<Node> target_ಗಂಟು = matmul_ಗಂಟು;

        ಯದಿ (add_ಗಂಟು) {
            ov::Output<Node> bias = bias_ಒಳಹರಿವು;
            ಯದಿ (bias.get_element_type() != target_type) {
                bias = register_new_ಗಂಟು<ov::op::v0::Convert>(bias, target_type);
            }
            auto bias_shape = register_new_ಗಂಟು<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = register_new_ಗಂಟು<ov::op::v1::Reshape>(bias, bias_shape, ಅಸತ್ಯ);
            final_out = register_new_ಗಂಟು<ov::op::v1::Add>(conv_out, bias);
            target_ಗಂಟು = add_ಗಂಟು;
        }

        ಯದಿ (!is_depthwise) {
            auto perm = register_new_ಗಂಟು<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                weight_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_out = register_new_ಗಂಟು<ov::op::v1::Transpose>(final_out, perm);
        }

        auto orig_shape = target_ಗಂಟು->get_output_partial_shape(0);
        ಯದಿ (orig_shape.is_static() && final_out.get_partial_shape().is_static()) {
            ಯದಿ (ov::shape_size(orig_shape.to_shape()) != ov::shape_size(final_out.get_shape())) {
                ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            }
        }
        ಯದಿ (orig_shape.is_static() && final_out.get_partial_shape() != orig_shape) {
            auto shape_const = register_new_ಗಂಟು<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.to_shape());
            final_out = register_new_ಗಂಟು<ov::op::v1::Reshape>(final_out, shape_const, ಅಸತ್ಯ);
        }

        auto orig_type = target_ಗಂಟು->get_output_element_type(0);
        ಯದಿ (final_out.get_element_type() != orig_type) {
            final_out = register_new_ಗಂಟು<ov::op::v0::Convert>(final_out, orig_type);
        }

        final_out.get_ಗಂಟು_shared_ptr()->set_friendly_name(target_ಗಂಟು->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_out.get_ಗಂಟು_shared_ptr());
        ov::replace_ಗಂಟು(target_ಗಂಟು, final_out.get_ಗಂಟು_shared_ptr());

        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    };

    register_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
