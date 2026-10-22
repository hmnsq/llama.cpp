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
    अचर स्वतः m_in0 = opp::any_आगत();
    अचर स्वतः m_in1 = opp::any_आगत();
    अचर स्वतः m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    अचर स्वतः callback = [=](opp::Matcher & m) {
        स्वतः matmul_गाँठो = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_जरा());
        यदि (!matmul_गाँठो) {
            फर्काऊ असत्य;
        }

        स्वतः unwrap = [](ov::Output<Node> n) {
            जबसम्म (ov::is_type<ov::op::v0::Convert>(n.get_गाँठो_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_गाँठो_shared_ptr())) {
                n = n.get_गाँठो_shared_ptr()->आगत_मान(0);
            }
            फर्काऊ n;
        };

        स्वतः get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ExtractImagePatches>, std::shared_ptr<ov::op::v1::Pad>> {
            स्वतः t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_गाँठो_shared_ptr());
            स्वतः r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_आगत_गाँठो_shared_ptr(0)) : nullptr;
            स्वतः t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_आगत_गाँठो_shared_ptr(0)) : nullptr;
            स्वतः eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(t1->get_आगत_गाँठो_shared_ptr(0)) : nullptr;
            स्वतः pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_आगत_गाँठो_shared_ptr(0)) : nullptr;
            फर्काऊ {eip, pad};
        };

        तार्किक weight_is_in0 = सत्य;
        स्वतः [eip, pad] = get_im2col(matmul_गाँठो->आगत_मान(1));
        ov::Output<Node> w_trace = matmul_गाँठो->आगत_मान(0);
        यदि (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_गाँठो->आगत_मान(0));
            w_trace = matmul_गाँठो->आगत_मान(1);
            weight_is_in0 = असत्य;
            यदि (!pad) {
                फर्काऊ असत्य;
            }
        }

        स्वतः pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_आगत_गाँठो_shared_ptr(1));
        स्वतः pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_आगत_गाँठो_shared_ptr(2));
        यदि (!pb_const || !pe_const) {
            फर्काऊ असत्य;
        }

        अचर स्वतः pb = pb_const->cast_vector<std::सूचक_भेद_प्रकार>();
        अचर स्वतः pe = pe_const->cast_vector<std::सूचक_भेद_प्रकार>();
        यदि (pb.size() < 4 || pe.size() < 4) {
            फर्काऊ असत्य;
        }

        स्वतः image_आगत = pad->आगत_मान(0);
        स्वतः image_shape = image_आगत.get_partial_shape();
        यदि (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            फर्काऊ असत्य;
        }
        अचर माप_प्रकार IC = static_cast<माप_प्रकार>(image_shape[1].get_length());

        w_trace = unwrap(w_trace);
        स्वतः weight_pshape = w_trace.get_partial_shape();
        यदि (!weight_pshape.is_static()) {
            फर्काऊ असत्य;
        }

        अचर स्वतः & ws = weight_pshape.to_shape();
        अचर माप_प्रकार KH = eip->get_sizes()[0];
        अचर माप_प्रकार KW = eip->get_sizes()[1];
        अचर माप_प्रकार kernel_spatial_ic = IC * KH * KW;

        माप_प्रकार groups = 0;
        यदि (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            यदि ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } अन्यथा यदि (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        अचर तार्किक is_depthwise = groups > 1 && (image_shape[0].get_length() % groups == 0);

        ov::Output<Node> conv_out;
        माप_प्रकार OC = 0;

        यदि (is_depthwise) {
            अचर माप_प्रकार N = static_cast<माप_प्रकार>(image_shape[0].get_length()) / groups;
            अचर माप_प्रकार IH = static_cast<माप_प्रकार>(image_shape[2].get_length());
            अचर माप_प्रकार IW = static_cast<माप_प्रकार>(image_shape[3].get_length());
            OC = groups;

            स्वतः img_shape_const = register_new_गाँठो<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                register_new_गाँठो<ov::op::v1::Reshape>(image_आगत, img_shape_const, असत्य);

            अचर ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> weight_आगत;
            यदि (स्वतः weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_गाँठो_shared_ptr())) {
                weight_आगत = register_new_गाँठो<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } अन्यथा {
                स्वतः shape_const = register_new_गाँठो<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_आगत = register_new_गाँठो<ov::op::v1::Reshape>(w_trace, shape_const, असत्य);
            }

            यदि (weight_आगत.get_element_type() != image_reshaped.get_element_type()) {
                weight_आगत = register_new_गाँठो<ov::op::v0::Convert>(weight_आगत, image_reshaped.get_element_type());
            }

            conv_out = register_new_गाँठो<ov::op::v1::GroupConvolution>(
                image_reshaped, weight_आगत, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } अन्यथा {
            यदि ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } अन्यथा यदि ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } अन्यथा यदि (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } अन्यथा यदि (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } अन्यथा यदि (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } अन्यथा यदि (kernel_spatial_ic && ov::shape_size(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_size(ws) / kernel_spatial_ic;
            } अन्यथा {
                फर्काऊ असत्य;
            }

            अचर ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> weight_आगत;
            यदि (स्वतः weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_गाँठो_shared_ptr())) {
                weight_आगत = register_new_गाँठो<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } अन्यथा {
                स्वतः shape_const = register_new_गाँठो<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_आगत = register_new_गाँठो<ov::op::v1::Reshape>(w_trace, shape_const, असत्य);
            }

            यदि (weight_आगत.get_element_type() != image_आगत.get_element_type()) {
                weight_आगत = register_new_गाँठो<ov::op::v0::Convert>(weight_आगत, image_आगत.get_element_type());
            }

            conv_out = register_new_गाँठो<ov::op::v1::Convolution>(
                image_आगत, weight_आगत, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr स्वतः target_type = ov::element::f32;
        यदि (conv_out.get_element_type() != target_type) {
            conv_out = register_new_गाँठो<ov::op::v0::Convert>(conv_out, target_type);
        }

        std::shared_ptr<ov::op::v1::Add> add_गाँठो;
        ov::Output<Node> bias_आगत;

        स्वतः try_fuse_bias = [&](अचर std::shared_ptr<Node> & n) {
            स्वतः add = ov::as_type_ptr<ov::op::v1::Add>(n);
            यदि (!add) {
                फर्काऊ असत्य;
            }
            लागि (माप_प्रकार i = 0; i < 2; ++i) {
                यदि (ov::is_type<ov::op::v0::Constant>(add->get_आगत_गाँठो_shared_ptr(i))) {
                    bias_आगत = add->आगत_मान(i);
                    add_गाँठो = add;
                    फर्काऊ सत्य;
                }
            }
            फर्काऊ असत्य;
        };

        लागि (अचर स्वतः & consumer : matmul_गाँठो->output(0).get_target_inputs()) {
            स्वतः n = consumer.get_गाँठो()->shared_from_this();
            यदि (try_fuse_bias(n)) {
                रोक;
            }
            यदि (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                लागि (अचर स्वतः & next : n->output(0).get_target_inputs()) {
                    यदि (try_fuse_bias(next.get_गाँठो()->shared_from_this())) {
                        रोक;
                    }
                }
            }
            यदि (add_गाँठो) {
                रोक;
            }
        }

        ov::Output<Node> final_out = conv_out;
        std::shared_ptr<Node> target_गाँठो = matmul_गाँठो;

        यदि (add_गाँठो) {
            ov::Output<Node> bias = bias_आगत;
            यदि (bias.get_element_type() != target_type) {
                bias = register_new_गाँठो<ov::op::v0::Convert>(bias, target_type);
            }
            स्वतः bias_shape = register_new_गाँठो<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = register_new_गाँठो<ov::op::v1::Reshape>(bias, bias_shape, असत्य);
            final_out = register_new_गाँठो<ov::op::v1::Add>(conv_out, bias);
            target_गाँठो = add_गाँठो;
        }

        यदि (!is_depthwise) {
            स्वतः perm = register_new_गाँठो<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                weight_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_out = register_new_गाँठो<ov::op::v1::Transpose>(final_out, perm);
        }

        स्वतः orig_shape = target_गाँठो->get_output_partial_shape(0);
        यदि (orig_shape.is_static() && final_out.get_partial_shape().is_static()) {
            यदि (ov::shape_size(orig_shape.to_shape()) != ov::shape_size(final_out.get_shape())) {
                फर्काऊ असत्य;
            }
        }
        यदि (orig_shape.is_static() && final_out.get_partial_shape() != orig_shape) {
            स्वतः shape_const = register_new_गाँठो<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.to_shape());
            final_out = register_new_गाँठो<ov::op::v1::Reshape>(final_out, shape_const, असत्य);
        }

        स्वतः orig_type = target_गाँठो->get_output_element_type(0);
        यदि (final_out.get_element_type() != orig_type) {
            final_out = register_new_गाँठो<ov::op::v0::Convert>(final_out, orig_type);
        }

        final_out.get_गाँठो_shared_ptr()->set_friendly_name(target_गाँठो->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_out.get_गाँठो_shared_ptr());
        ov::replace_गाँठो(target_गाँठो, final_out.get_गाँठो_shared_ptr());

        फर्काऊ सत्य;
    };

    register_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
