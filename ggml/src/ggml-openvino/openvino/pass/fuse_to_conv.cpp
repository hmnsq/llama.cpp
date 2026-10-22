#ଅନ୍ତର୍ଭାଵ "fuse_to_conv.h"

#ଅନ୍ତର୍ଭାଵ <openvino/core/graph_util.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/core/rt_info.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/add.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/constant.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convert.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/convolution.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/extractimagepatches.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/group_conv.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/matmul.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/pad.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/reshape.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/op/transpose.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/pass/pattern/op/pattern.hpp>
#ଅନ୍ତର୍ଭାଵ <openvino/pass/pattern/op/wrap_type.hpp>
#ଅନ୍ତର୍ଭାଵ <utility>

ନାମାକାଶ opp = ov::pass::pattern;

ନାମାକାଶ ov {
ନାମାକାଶ frontend {
ନାମାକାଶ ggml {
ନାମାକାଶ pass {

FuseToConv::FuseToConv() {
    ଅଚଳ auto m_in0 = opp::any_ନିବେଶ();
    ଅଚଳ auto m_in1 = opp::any_ନିବେଶ();
    ଅଚଳ auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    ଅଚଳ auto callback = [=](opp::Matcher & m) {
        auto matmul_ଗଣ୍ଠି = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_ମୂଳ());
        ଯଦି (!matmul_ଗଣ୍ଠି) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        auto unwrap = [](ov::Output<Node> n) {
            ଯେତେବେଳେ (ov::is_type<ov::op::v0::Convert>(n.get_ଗଣ୍ଠି_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_ଗଣ୍ଠି_shared_ptr())) {
                n = n.get_ଗଣ୍ଠି_shared_ptr()->ନିବେଶ_ମୂଲ୍ୟ(0);
            }
            ଫେରନ୍ତୁ n;
        };

        auto get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ExtractImagePatches>, std::shared_ptr<ov::op::v1::Pad>> {
            auto t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_ଗଣ୍ଠି_shared_ptr());
            auto r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_ନିବେଶ_ଗଣ୍ଠି_shared_ptr(0)) : nullptr;
            auto t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_ନିବେଶ_ଗଣ୍ଠି_shared_ptr(0)) : nullptr;
            auto eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(t1->get_ନିବେଶ_ଗଣ୍ଠି_shared_ptr(0)) : nullptr;
            auto pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_ନିବେଶ_ଗଣ୍ଠି_shared_ptr(0)) : nullptr;
            ଫେରନ୍ତୁ {eip, pad};
        };

        ତାର୍କିକ weight_is_in0 = ସତ୍ୟ;
        auto [eip, pad] = get_im2col(matmul_ଗଣ୍ଠି->ନିବେଶ_ମୂଲ୍ୟ(1));
        ov::Output<Node> w_trace = matmul_ଗଣ୍ଠି->ନିବେଶ_ମୂଲ୍ୟ(0);
        ଯଦି (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_ଗଣ୍ଠି->ନିବେଶ_ମୂଲ୍ୟ(0));
            w_trace = matmul_ଗଣ୍ଠି->ନିବେଶ_ମୂଲ୍ୟ(1);
            weight_is_in0 = ମିଥ୍ୟା;
            ଯଦି (!pad) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }

        auto pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_ନିବେଶ_ଗଣ୍ଠି_shared_ptr(1));
        auto pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_ନିବେଶ_ଗଣ୍ଠି_shared_ptr(2));
        ଯଦି (!pb_const || !pe_const) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ auto pb = pb_const->cast_vector<std::ptrdiff_t>();
        ଅଚଳ auto pe = pe_const->cast_vector<std::ptrdiff_t>();
        ଯଦି (pb.size() < 4 || pe.size() < 4) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        auto image_ନିବେଶ = pad->ନିବେଶ_ମୂଲ୍ୟ(0);
        auto image_shape = image_ନିବେଶ.get_partial_shape();
        ଯଦି (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଅଚଳ size_t IC = static_cast<size_t>(image_shape[1].get_length());

        w_trace = unwrap(w_trace);
        auto weight_pshape = w_trace.get_partial_shape();
        ଯଦି (!weight_pshape.is_static()) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ଅଚଳ auto & ws = weight_pshape.to_shape();
        ଅଚଳ size_t KH = eip->get_sizes()[0];
        ଅଚଳ size_t KW = eip->get_sizes()[1];
        ଅଚଳ size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        ଯଦି (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            ଯଦି ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } ଅନ୍ୟଥା ଯଦି (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        ଅଚଳ ତାର୍କିକ is_depthwise = groups > 1 && (image_shape[0].get_length() % groups == 0);

        ov::Output<Node> conv_out;
        size_t OC = 0;

        ଯଦି (is_depthwise) {
            ଅଚଳ size_t N = static_cast<size_t>(image_shape[0].get_length()) / groups;
            ଅଚଳ size_t IH = static_cast<size_t>(image_shape[2].get_length());
            ଅଚଳ size_t IW = static_cast<size_t>(image_shape[3].get_length());
            OC = groups;

            auto img_shape_const = register_new_ଗଣ୍ଠି<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                register_new_ଗଣ୍ଠି<ov::op::v1::Reshape>(image_ନିବେଶ, img_shape_const, ମିଥ୍ୟା);

            ଅଚଳ ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> weight_ନିବେଶ;
            ଯଦି (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ଗଣ୍ଠି_shared_ptr())) {
                weight_ନିବେଶ = register_new_ଗଣ୍ଠି<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } ଅନ୍ୟଥା {
                auto shape_const = register_new_ଗଣ୍ଠି<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_ନିବେଶ = register_new_ଗଣ୍ଠି<ov::op::v1::Reshape>(w_trace, shape_const, ମିଥ୍ୟା);
            }

            ଯଦି (weight_ନିବେଶ.get_element_type() != image_reshaped.get_element_type()) {
                weight_ନିବେଶ = register_new_ଗଣ୍ଠି<ov::op::v0::Convert>(weight_ନିବେଶ, image_reshaped.get_element_type());
            }

            conv_out = register_new_ଗଣ୍ଠି<ov::op::v1::GroupConvolution>(
                image_reshaped, weight_ନିବେଶ, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } ଅନ୍ୟଥା {
            ଯଦି ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } ଅନ୍ୟଥା ଯଦି ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } ଅନ୍ୟଥା ଯଦି (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } ଅନ୍ୟଥା ଯଦି (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } ଅନ୍ୟଥା ଯଦି (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } ଅନ୍ୟଥା ଯଦି (kernel_spatial_ic && ov::shape_size(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_size(ws) / kernel_spatial_ic;
            } ଅନ୍ୟଥା {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }

            ଅଚଳ ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> weight_ନିବେଶ;
            ଯଦି (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ଗଣ୍ଠି_shared_ptr())) {
                weight_ନିବେଶ = register_new_ଗଣ୍ଠି<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } ଅନ୍ୟଥା {
                auto shape_const = register_new_ଗଣ୍ଠି<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_ନିବେଶ = register_new_ଗଣ୍ଠି<ov::op::v1::Reshape>(w_trace, shape_const, ମିଥ୍ୟା);
            }

            ଯଦି (weight_ନିବେଶ.get_element_type() != image_ନିବେଶ.get_element_type()) {
                weight_ନିବେଶ = register_new_ଗଣ୍ଠି<ov::op::v0::Convert>(weight_ନିବେଶ, image_ନିବେଶ.get_element_type());
            }

            conv_out = register_new_ଗଣ୍ଠି<ov::op::v1::Convolution>(
                image_ନିବେଶ, weight_ନିବେଶ, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr auto target_type = ov::element::f32;
        ଯଦି (conv_out.get_element_type() != target_type) {
            conv_out = register_new_ଗଣ୍ଠି<ov::op::v0::Convert>(conv_out, target_type);
        }

        std::shared_ptr<ov::op::v1::Add> add_ଗଣ୍ଠି;
        ov::Output<Node> bias_ନିବେଶ;

        auto try_fuse_bias = [&](ଅଚଳ std::shared_ptr<Node> & n) {
            auto add = ov::as_type_ptr<ov::op::v1::Add>(n);
            ଯଦି (!add) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ପାଇଁ (size_t i = 0; i < 2; ++i) {
                ଯଦି (ov::is_type<ov::op::v0::Constant>(add->get_ନିବେଶ_ଗଣ୍ଠି_shared_ptr(i))) {
                    bias_ନିବେଶ = add->ନିବେଶ_ମୂଲ୍ୟ(i);
                    add_ଗଣ୍ଠି = add;
                    ଫେରନ୍ତୁ ସତ୍ୟ;
                }
            }
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        };

        ପାଇଁ (ଅଚଳ auto & consumer : matmul_ଗଣ୍ଠି->output(0).get_target_inputs()) {
            auto n = consumer.get_ଗଣ୍ଠି()->shared_from_this();
            ଯଦି (try_fuse_bias(n)) {
                ଭାଙ୍ଗ;
            }
            ଯଦି (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                ପାଇଁ (ଅଚଳ auto & next : n->output(0).get_target_inputs()) {
                    ଯଦି (try_fuse_bias(next.get_ଗଣ୍ଠି()->shared_from_this())) {
                        ଭାଙ୍ଗ;
                    }
                }
            }
            ଯଦି (add_ଗଣ୍ଠି) {
                ଭାଙ୍ଗ;
            }
        }

        ov::Output<Node> final_out = conv_out;
        std::shared_ptr<Node> target_ଗଣ୍ଠି = matmul_ଗଣ୍ଠି;

        ଯଦି (add_ଗଣ୍ଠି) {
            ov::Output<Node> bias = bias_ନିବେଶ;
            ଯଦି (bias.get_element_type() != target_type) {
                bias = register_new_ଗଣ୍ଠି<ov::op::v0::Convert>(bias, target_type);
            }
            auto bias_shape = register_new_ଗଣ୍ଠି<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = register_new_ଗଣ୍ଠି<ov::op::v1::Reshape>(bias, bias_shape, ମିଥ୍ୟା);
            final_out = register_new_ଗଣ୍ଠି<ov::op::v1::Add>(conv_out, bias);
            target_ଗଣ୍ଠି = add_ଗଣ୍ଠି;
        }

        ଯଦି (!is_depthwise) {
            auto perm = register_new_ଗଣ୍ଠି<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                weight_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_out = register_new_ଗଣ୍ଠି<ov::op::v1::Transpose>(final_out, perm);
        }

        auto orig_shape = target_ଗଣ୍ଠି->get_output_partial_shape(0);
        ଯଦି (orig_shape.is_static() && final_out.get_partial_shape().is_static()) {
            ଯଦି (ov::shape_size(orig_shape.to_shape()) != ov::shape_size(final_out.get_shape())) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        }
        ଯଦି (orig_shape.is_static() && final_out.get_partial_shape() != orig_shape) {
            auto shape_const = register_new_ଗଣ୍ଠି<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.to_shape());
            final_out = register_new_ଗଣ୍ଠି<ov::op::v1::Reshape>(final_out, shape_const, ମିଥ୍ୟା);
        }

        auto orig_type = target_ଗଣ୍ଠି->get_output_element_type(0);
        ଯଦି (final_out.get_element_type() != orig_type) {
            final_out = register_new_ଗଣ୍ଠି<ov::op::v0::Convert>(final_out, orig_type);
        }

        final_out.get_ଗଣ୍ଠି_shared_ptr()->set_friendly_name(target_ଗଣ୍ଠି->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_out.get_ଗଣ୍ଠି_shared_ptr());
        ov::replace_ଗଣ୍ଠି(target_ଗଣ୍ଠି, final_out.get_ଗଣ୍ଠି_shared_ptr());

        ଫେରନ୍ତୁ ସତ୍ୟ;
    };

    register_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
