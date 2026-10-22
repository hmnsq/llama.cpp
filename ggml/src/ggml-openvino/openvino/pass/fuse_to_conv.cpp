#അന്തര്ഭാവ "fuse_to_conv.h"

#അന്തര്ഭാവ <openvino/core/graph_util.hpp>
#അന്തര്ഭാവ <openvino/core/rt_info.hpp>
#അന്തര്ഭാവ <openvino/op/add.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/convert.hpp>
#അന്തര്ഭാവ <openvino/op/convolution.hpp>
#അന്തര്ഭാവ <openvino/op/extractimagepatches.hpp>
#അന്തര്ഭാവ <openvino/op/group_conv.hpp>
#അന്തര്ഭാവ <openvino/op/matmul.hpp>
#അന്തര്ഭാവ <openvino/op/pad.hpp>
#അന്തര്ഭാവ <openvino/op/reshape.hpp>
#അന്തര്ഭാവ <openvino/op/transpose.hpp>
#അന്തര്ഭാവ <openvino/pass/pattern/op/pattern.hpp>
#അന്തര്ഭാവ <openvino/pass/pattern/op/wrap_type.hpp>
#അന്തര്ഭാവ <utility>

നാമാകാശ opp = ov::pass::pattern;

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ pass {

FuseToConv::FuseToConv() {
    സ്ഥിരം auto m_in0 = opp::any_നിവേശം();
    സ്ഥിരം auto m_in1 = opp::any_നിവേശം();
    സ്ഥിരം auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    സ്ഥിരം auto callback = [=](opp::Matcher & m) {
        auto matmul_കണ്ണി = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_വേര്());
        എങ്കിൽ (!matmul_കണ്ണി) {
            തിരികെ അസത്യം;
        }

        auto unwrap = [](ov::Output<Node> n) {
            ആകുമ്പോൾ (ov::is_type<ov::op::v0::Convert>(n.get_കണ്ണി_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_കണ്ണി_shared_ptr())) {
                n = n.get_കണ്ണി_shared_ptr()->നിവേശം_മൂല്യം(0);
            }
            തിരികെ n;
        };

        auto get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ExtractImagePatches>, std::shared_ptr<ov::op::v1::Pad>> {
            auto t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_കണ്ണി_shared_ptr());
            auto r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_നിവേശം_കണ്ണി_shared_ptr(0)) : nullptr;
            auto t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_നിവേശം_കണ്ണി_shared_ptr(0)) : nullptr;
            auto eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(t1->get_നിവേശം_കണ്ണി_shared_ptr(0)) : nullptr;
            auto pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_നിവേശം_കണ്ണി_shared_ptr(0)) : nullptr;
            തിരികെ {eip, pad};
        };

        തർക്കം weight_is_in0 = സത്യം;
        auto [eip, pad] = get_im2col(matmul_കണ്ണി->നിവേശം_മൂല്യം(1));
        ov::Output<Node> w_trace = matmul_കണ്ണി->നിവേശം_മൂല്യം(0);
        എങ്കിൽ (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_കണ്ണി->നിവേശം_മൂല്യം(0));
            w_trace = matmul_കണ്ണി->നിവേശം_മൂല്യം(1);
            weight_is_in0 = അസത്യം;
            എങ്കിൽ (!pad) {
                തിരികെ അസത്യം;
            }
        }

        auto pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_നിവേശം_കണ്ണി_shared_ptr(1));
        auto pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_നിവേശം_കണ്ണി_shared_ptr(2));
        എങ്കിൽ (!pb_const || !pe_const) {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം auto pb = pb_const->cast_vector<std::ptrdiff_t>();
        സ്ഥിരം auto pe = pe_const->cast_vector<std::ptrdiff_t>();
        എങ്കിൽ (pb.size() < 4 || pe.size() < 4) {
            തിരികെ അസത്യം;
        }

        auto image_നിവേശം = pad->നിവേശം_മൂല്യം(0);
        auto image_shape = image_നിവേശം.get_partial_shape();
        എങ്കിൽ (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            തിരികെ അസത്യം;
        }
        സ്ഥിരം size_t IC = static_cast<size_t>(image_shape[1].get_length());

        w_trace = unwrap(w_trace);
        auto weight_pshape = w_trace.get_partial_shape();
        എങ്കിൽ (!weight_pshape.is_static()) {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം auto & ws = weight_pshape.to_shape();
        സ്ഥിരം size_t KH = eip->get_sizes()[0];
        സ്ഥിരം size_t KW = eip->get_sizes()[1];
        സ്ഥിരം size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        എങ്കിൽ (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            എങ്കിൽ ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } അല്ലെങ്കിൽ എങ്കിൽ (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        സ്ഥിരം തർക്കം is_depthwise = groups > 1 && (image_shape[0].get_length() % groups == 0);

        ov::Output<Node> conv_out;
        size_t OC = 0;

        എങ്കിൽ (is_depthwise) {
            സ്ഥിരം size_t N = static_cast<size_t>(image_shape[0].get_length()) / groups;
            സ്ഥിരം size_t IH = static_cast<size_t>(image_shape[2].get_length());
            സ്ഥിരം size_t IW = static_cast<size_t>(image_shape[3].get_length());
            OC = groups;

            auto img_shape_const = register_new_കണ്ണി<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                register_new_കണ്ണി<ov::op::v1::Reshape>(image_നിവേശം, img_shape_const, അസത്യം);

            സ്ഥിരം ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> weight_നിവേശം;
            എങ്കിൽ (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_കണ്ണി_shared_ptr())) {
                weight_നിവേശം = register_new_കണ്ണി<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } അല്ലെങ്കിൽ {
                auto shape_const = register_new_കണ്ണി<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_നിവേശം = register_new_കണ്ണി<ov::op::v1::Reshape>(w_trace, shape_const, അസത്യം);
            }

            എങ്കിൽ (weight_നിവേശം.get_element_type() != image_reshaped.get_element_type()) {
                weight_നിവേശം = register_new_കണ്ണി<ov::op::v0::Convert>(weight_നിവേശം, image_reshaped.get_element_type());
            }

            conv_out = register_new_കണ്ണി<ov::op::v1::GroupConvolution>(
                image_reshaped, weight_നിവേശം, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } അല്ലെങ്കിൽ {
            എങ്കിൽ ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } അല്ലെങ്കിൽ എങ്കിൽ ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } അല്ലെങ്കിൽ എങ്കിൽ (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } അല്ലെങ്കിൽ എങ്കിൽ (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } അല്ലെങ്കിൽ എങ്കിൽ (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } അല്ലെങ്കിൽ എങ്കിൽ (kernel_spatial_ic && ov::shape_size(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_size(ws) / kernel_spatial_ic;
            } അല്ലെങ്കിൽ {
                തിരികെ അസത്യം;
            }

            സ്ഥിരം ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> weight_നിവേശം;
            എങ്കിൽ (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_കണ്ണി_shared_ptr())) {
                weight_നിവേശം = register_new_കണ്ണി<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } അല്ലെങ്കിൽ {
                auto shape_const = register_new_കണ്ണി<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_നിവേശം = register_new_കണ്ണി<ov::op::v1::Reshape>(w_trace, shape_const, അസത്യം);
            }

            എങ്കിൽ (weight_നിവേശം.get_element_type() != image_നിവേശം.get_element_type()) {
                weight_നിവേശം = register_new_കണ്ണി<ov::op::v0::Convert>(weight_നിവേശം, image_നിവേശം.get_element_type());
            }

            conv_out = register_new_കണ്ണി<ov::op::v1::Convolution>(
                image_നിവേശം, weight_നിവേശം, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr auto target_type = ov::element::f32;
        എങ്കിൽ (conv_out.get_element_type() != target_type) {
            conv_out = register_new_കണ്ണി<ov::op::v0::Convert>(conv_out, target_type);
        }

        std::shared_ptr<ov::op::v1::Add> add_കണ്ണി;
        ov::Output<Node> bias_നിവേശം;

        auto try_fuse_bias = [&](സ്ഥിരം std::shared_ptr<Node> & n) {
            auto add = ov::as_type_ptr<ov::op::v1::Add>(n);
            എങ്കിൽ (!add) {
                തിരികെ അസത്യം;
            }
            വേണ്ടി (size_t i = 0; i < 2; ++i) {
                എങ്കിൽ (ov::is_type<ov::op::v0::Constant>(add->get_നിവേശം_കണ്ണി_shared_ptr(i))) {
                    bias_നിവേശം = add->നിവേശം_മൂല്യം(i);
                    add_കണ്ണി = add;
                    തിരികെ സത്യം;
                }
            }
            തിരികെ അസത്യം;
        };

        വേണ്ടി (സ്ഥിരം auto & consumer : matmul_കണ്ണി->output(0).get_target_inputs()) {
            auto n = consumer.get_കണ്ണി()->shared_from_this();
            എങ്കിൽ (try_fuse_bias(n)) {
                നിർത്തുക;
            }
            എങ്കിൽ (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                വേണ്ടി (സ്ഥിരം auto & next : n->output(0).get_target_inputs()) {
                    എങ്കിൽ (try_fuse_bias(next.get_കണ്ണി()->shared_from_this())) {
                        നിർത്തുക;
                    }
                }
            }
            എങ്കിൽ (add_കണ്ണി) {
                നിർത്തുക;
            }
        }

        ov::Output<Node> final_out = conv_out;
        std::shared_ptr<Node> target_കണ്ണി = matmul_കണ്ണി;

        എങ്കിൽ (add_കണ്ണി) {
            ov::Output<Node> bias = bias_നിവേശം;
            എങ്കിൽ (bias.get_element_type() != target_type) {
                bias = register_new_കണ്ണി<ov::op::v0::Convert>(bias, target_type);
            }
            auto bias_shape = register_new_കണ്ണി<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = register_new_കണ്ണി<ov::op::v1::Reshape>(bias, bias_shape, അസത്യം);
            final_out = register_new_കണ്ണി<ov::op::v1::Add>(conv_out, bias);
            target_കണ്ണി = add_കണ്ണി;
        }

        എങ്കിൽ (!is_depthwise) {
            auto perm = register_new_കണ്ണി<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                weight_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_out = register_new_കണ്ണി<ov::op::v1::Transpose>(final_out, perm);
        }

        auto orig_shape = target_കണ്ണി->get_output_partial_shape(0);
        എങ്കിൽ (orig_shape.is_static() && final_out.get_partial_shape().is_static()) {
            എങ്കിൽ (ov::shape_size(orig_shape.to_shape()) != ov::shape_size(final_out.get_shape())) {
                തിരികെ അസത്യം;
            }
        }
        എങ്കിൽ (orig_shape.is_static() && final_out.get_partial_shape() != orig_shape) {
            auto shape_const = register_new_കണ്ണി<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.to_shape());
            final_out = register_new_കണ്ണി<ov::op::v1::Reshape>(final_out, shape_const, അസത്യം);
        }

        auto orig_type = target_കണ്ണി->get_output_element_type(0);
        എങ്കിൽ (final_out.get_element_type() != orig_type) {
            final_out = register_new_കണ്ണി<ov::op::v0::Convert>(final_out, orig_type);
        }

        final_out.get_കണ്ണി_shared_ptr()->set_friendly_name(target_കണ്ണി->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_out.get_കണ്ണി_shared_ptr());
        ov::replace_കണ്ണി(target_കണ്ണി, final_out.get_കണ്ണി_shared_ptr());

        തിരികെ സത്യം;
    };

    register_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
