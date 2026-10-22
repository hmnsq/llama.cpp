#அந்தர்பாவ "fuse_to_conv.h"

#அந்தர்பாவ <openvino/core/graph_util.hpp>
#அந்தர்பாவ <openvino/core/rt_info.hpp>
#அந்தர்பாவ <openvino/op/add.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/convert.hpp>
#அந்தர்பாவ <openvino/op/convolution.hpp>
#அந்தர்பாவ <openvino/op/extractimagepatches.hpp>
#அந்தர்பாவ <openvino/op/group_conv.hpp>
#அந்தர்பாவ <openvino/op/matmul.hpp>
#அந்தர்பாவ <openvino/op/pad.hpp>
#அந்தர்பாவ <openvino/op/reshape.hpp>
#அந்தர்பாவ <openvino/op/transpose.hpp>
#அந்தர்பாவ <openvino/pass/pattern/op/pattern.hpp>
#அந்தர்பாவ <openvino/pass/pattern/op/wrap_type.hpp>
#அந்தர்பாவ <utility>

நாமாகாஶ opp = ov::pass::pattern;

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ pass {

FuseToConv::FuseToConv() {
    மாறிலி auto m_in0 = opp::any_உள்ளீடு();
    மாறிலி auto m_in1 = opp::any_உள்ளீடு();
    மாறிலி auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    மாறிலி auto callback = [=](opp::Matcher & m) {
        auto matmul_முனை = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_வேர்());
        எனில் (!matmul_முனை) {
            திரும்பு பொய்;
        }

        auto unwrap = [](ov::Output<Node> n) {
            போது (ov::is_type<ov::op::v0::Convert>(n.get_முனை_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_முனை_shared_ptr())) {
                n = n.get_முனை_shared_ptr()->உள்ளீடு_மதிப்பு(0);
            }
            திரும்பு n;
        };

        auto get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ExtractImagePatches>, std::shared_ptr<ov::op::v1::Pad>> {
            auto t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_முனை_shared_ptr());
            auto r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_உள்ளீடு_முனை_shared_ptr(0)) : nullptr;
            auto t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_உள்ளீடு_முனை_shared_ptr(0)) : nullptr;
            auto eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(t1->get_உள்ளீடு_முனை_shared_ptr(0)) : nullptr;
            auto pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_உள்ளீடு_முனை_shared_ptr(0)) : nullptr;
            திரும்பு {eip, pad};
        };

        தருக்க weight_is_in0 = உண்மை;
        auto [eip, pad] = get_im2col(matmul_முனை->உள்ளீடு_மதிப்பு(1));
        ov::Output<Node> w_trace = matmul_முனை->உள்ளீடு_மதிப்பு(0);
        எனில் (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_முனை->உள்ளீடு_மதிப்பு(0));
            w_trace = matmul_முனை->உள்ளீடு_மதிப்பு(1);
            weight_is_in0 = பொய்;
            எனில் (!pad) {
                திரும்பு பொய்;
            }
        }

        auto pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_உள்ளீடு_முனை_shared_ptr(1));
        auto pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_உள்ளீடு_முனை_shared_ptr(2));
        எனில் (!pb_const || !pe_const) {
            திரும்பு பொய்;
        }

        மாறிலி auto pb = pb_const->cast_vector<std::ptrdiff_t>();
        மாறிலி auto pe = pe_const->cast_vector<std::ptrdiff_t>();
        எனில் (pb.size() < 4 || pe.size() < 4) {
            திரும்பு பொய்;
        }

        auto image_உள்ளீடு = pad->உள்ளீடு_மதிப்பு(0);
        auto image_shape = image_உள்ளீடு.get_partial_shape();
        எனில் (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            திரும்பு பொய்;
        }
        மாறிலி size_t IC = static_cast<size_t>(image_shape[1].get_length());

        w_trace = unwrap(w_trace);
        auto weight_pshape = w_trace.get_partial_shape();
        எனில் (!weight_pshape.is_static()) {
            திரும்பு பொய்;
        }

        மாறிலி auto & ws = weight_pshape.to_shape();
        மாறிலி size_t KH = eip->get_sizes()[0];
        மாறிலி size_t KW = eip->get_sizes()[1];
        மாறிலி size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        எனில் (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            எனில் ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } இல்லையெனில் எனில் (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        மாறிலி தருக்க is_depthwise = groups > 1 && (image_shape[0].get_length() % groups == 0);

        ov::Output<Node> conv_out;
        size_t OC = 0;

        எனில் (is_depthwise) {
            மாறிலி size_t N = static_cast<size_t>(image_shape[0].get_length()) / groups;
            மாறிலி size_t IH = static_cast<size_t>(image_shape[2].get_length());
            மாறிலி size_t IW = static_cast<size_t>(image_shape[3].get_length());
            OC = groups;

            auto img_shape_const = register_new_முனை<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                register_new_முனை<ov::op::v1::Reshape>(image_உள்ளீடு, img_shape_const, பொய்);

            மாறிலி ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> weight_உள்ளீடு;
            எனில் (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_முனை_shared_ptr())) {
                weight_உள்ளீடு = register_new_முனை<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } இல்லையெனில் {
                auto shape_const = register_new_முனை<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_உள்ளீடு = register_new_முனை<ov::op::v1::Reshape>(w_trace, shape_const, பொய்);
            }

            எனில் (weight_உள்ளீடு.get_element_type() != image_reshaped.get_element_type()) {
                weight_உள்ளீடு = register_new_முனை<ov::op::v0::Convert>(weight_உள்ளீடு, image_reshaped.get_element_type());
            }

            conv_out = register_new_முனை<ov::op::v1::GroupConvolution>(
                image_reshaped, weight_உள்ளீடு, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } இல்லையெனில் {
            எனில் ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } இல்லையெனில் எனில் ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } இல்லையெனில் எனில் (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } இல்லையெனில் எனில் (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } இல்லையெனில் எனில் (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } இல்லையெனில் எனில் (kernel_spatial_ic && ov::shape_size(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_size(ws) / kernel_spatial_ic;
            } இல்லையெனில் {
                திரும்பு பொய்;
            }

            மாறிலி ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> weight_உள்ளீடு;
            எனில் (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_முனை_shared_ptr())) {
                weight_உள்ளீடு = register_new_முனை<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } இல்லையெனில் {
                auto shape_const = register_new_முனை<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_உள்ளீடு = register_new_முனை<ov::op::v1::Reshape>(w_trace, shape_const, பொய்);
            }

            எனில் (weight_உள்ளீடு.get_element_type() != image_உள்ளீடு.get_element_type()) {
                weight_உள்ளீடு = register_new_முனை<ov::op::v0::Convert>(weight_உள்ளீடு, image_உள்ளீடு.get_element_type());
            }

            conv_out = register_new_முனை<ov::op::v1::Convolution>(
                image_உள்ளீடு, weight_உள்ளீடு, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr auto target_type = ov::element::f32;
        எனில் (conv_out.get_element_type() != target_type) {
            conv_out = register_new_முனை<ov::op::v0::Convert>(conv_out, target_type);
        }

        std::shared_ptr<ov::op::v1::Add> add_முனை;
        ov::Output<Node> bias_உள்ளீடு;

        auto try_fuse_bias = [&](மாறிலி std::shared_ptr<Node> & n) {
            auto add = ov::as_type_ptr<ov::op::v1::Add>(n);
            எனில் (!add) {
                திரும்பு பொய்;
            }
            க்கு (size_t i = 0; i < 2; ++i) {
                எனில் (ov::is_type<ov::op::v0::Constant>(add->get_உள்ளீடு_முனை_shared_ptr(i))) {
                    bias_உள்ளீடு = add->உள்ளீடு_மதிப்பு(i);
                    add_முனை = add;
                    திரும்பு உண்மை;
                }
            }
            திரும்பு பொய்;
        };

        க்கு (மாறிலி auto & consumer : matmul_முனை->output(0).get_target_inputs()) {
            auto n = consumer.get_முனை()->shared_from_this();
            எனில் (try_fuse_bias(n)) {
                நிறுத்து;
            }
            எனில் (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                க்கு (மாறிலி auto & next : n->output(0).get_target_inputs()) {
                    எனில் (try_fuse_bias(next.get_முனை()->shared_from_this())) {
                        நிறுத்து;
                    }
                }
            }
            எனில் (add_முனை) {
                நிறுத்து;
            }
        }

        ov::Output<Node> final_out = conv_out;
        std::shared_ptr<Node> target_முனை = matmul_முனை;

        எனில் (add_முனை) {
            ov::Output<Node> bias = bias_உள்ளீடு;
            எனில் (bias.get_element_type() != target_type) {
                bias = register_new_முனை<ov::op::v0::Convert>(bias, target_type);
            }
            auto bias_shape = register_new_முனை<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = register_new_முனை<ov::op::v1::Reshape>(bias, bias_shape, பொய்);
            final_out = register_new_முனை<ov::op::v1::Add>(conv_out, bias);
            target_முனை = add_முனை;
        }

        எனில் (!is_depthwise) {
            auto perm = register_new_முனை<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                weight_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_out = register_new_முனை<ov::op::v1::Transpose>(final_out, perm);
        }

        auto orig_shape = target_முனை->get_output_partial_shape(0);
        எனில் (orig_shape.is_static() && final_out.get_partial_shape().is_static()) {
            எனில் (ov::shape_size(orig_shape.to_shape()) != ov::shape_size(final_out.get_shape())) {
                திரும்பு பொய்;
            }
        }
        எனில் (orig_shape.is_static() && final_out.get_partial_shape() != orig_shape) {
            auto shape_const = register_new_முனை<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.to_shape());
            final_out = register_new_முனை<ov::op::v1::Reshape>(final_out, shape_const, பொய்);
        }

        auto orig_type = target_முனை->get_output_element_type(0);
        எனில் (final_out.get_element_type() != orig_type) {
            final_out = register_new_முனை<ov::op::v0::Convert>(final_out, orig_type);
        }

        final_out.get_முனை_shared_ptr()->set_friendly_name(target_முனை->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_out.get_முனை_shared_ptr());
        ov::replace_முனை(target_முனை, final_out.get_முனை_shared_ptr());

        திரும்பு உண்மை;
    };

    register_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
