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

Fuse_गन्तव्यम्_Conv::Fuse_गन्तव्यम्_Conv() {
    ध्रुव स्वचालित m_in0 = opp::any_निवेशः();
    ध्रुव स्वचालित m_in1 = opp::any_निवेशः();
    ध्रुव स्वचालित m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    ध्रुव स्वचालित callback = [=](opp::Matcher & m) {
        स्वचालित matmul_ग्रन्थिः = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_मेलनम्_मूलम्());
        यदि (!matmul_ग्रन्थिः) {
            प्रत्यागम असत्य;
        }

        स्वचालित unwrap = [](ov::Output<Node> n) {
            यावत् (ov::is_type<ov::op::v0::Convert>(n.get_ग्रन्थिः_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_ग्रन्थिः_shared_ptr())) {
                n = n.get_ग्रन्थिः_shared_ptr()->निवेशः_मानम्(0);
            }
            प्रत्यागम n;
        };

        स्वचालित get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ExtractImagePatches>, std::shared_ptr<ov::op::v1::Pad>> {
            स्वचालित t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_ग्रन्थिः_shared_ptr());
            स्वचालित r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_निवेशः_ग्रन्थिः_shared_ptr(0)) : शून्यसूचकः;
            स्वचालित t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_निवेशः_ग्रन्थिः_shared_ptr(0)) : शून्यसूचकः;
            स्वचालित eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(t1->get_निवेशः_ग्रन्थिः_shared_ptr(0)) : शून्यसूचकः;
            स्वचालित pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_निवेशः_ग्रन्थिः_shared_ptr(0)) : शून्यसूचकः;
            प्रत्यागम {eip, pad};
        };

        तार्किक weight_is_in0 = सत्य;
        स्वचालित [eip, pad] = get_im2col(matmul_ग्रन्थिः->निवेशः_मानम्(1));
        ov::Output<Node> w_trace = matmul_ग्रन्थिः->निवेशः_मानम्(0);
        यदि (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_ग्रन्थिः->निवेशः_मानम्(0));
            w_trace = matmul_ग्रन्थिः->निवेशः_मानम्(1);
            weight_is_in0 = असत्य;
            यदि (!pad) {
                प्रत्यागम असत्य;
            }
        }

        स्वचालित pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_निवेशः_ग्रन्थिः_shared_ptr(1));
        स्वचालित pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_निवेशः_ग्रन्थिः_shared_ptr(2));
        यदि (!pb_const || !pe_const) {
            प्रत्यागम असत्य;
        }

        ध्रुव स्वचालित pb = pb_const->cast_सदिशम्<std::ptrdiff_t>();
        ध्रुव स्वचालित pe = pe_const->cast_सदिशम्<std::ptrdiff_t>();
        यदि (pb.size() < 4 || pe.size() < 4) {
            प्रत्यागम असत्य;
        }

        स्वचालित image_निवेशः = pad->निवेशः_मानम्(0);
        स्वचालित image_shape = image_निवेशः.get_partial_shape();
        यदि (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            प्रत्यागम असत्य;
        }
        ध्रुव size_t IC = static_cast<size_t>(image_shape[1].get_दीर्घता());

        w_trace = unwrap(w_trace);
        स्वचालित weight_pshape = w_trace.get_partial_shape();
        यदि (!weight_pshape.is_static()) {
            प्रत्यागम असत्य;
        }

        ध्रुव स्वचालित & ws = weight_pshape.गन्तव्यम्_shape();
        ध्रुव size_t KH = eip->get_sizes()[0];
        ध्रुव size_t KW = eip->get_sizes()[1];
        ध्रुव size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        यदि (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            यदि ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } अन्यथा यदि (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        ध्रुव तार्किक is_depthwise = groups > 1 && (image_shape[0].get_दीर्घता() % groups == 0);

        ov::Output<Node> conv_बहिर्गमनम्;
        size_t OC = 0;

        यदि (is_depthwise) {
            ध्रुव size_t N = static_cast<size_t>(image_shape[0].get_दीर्घता()) / groups;
            ध्रुव size_t IH = static_cast<size_t>(image_shape[2].get_दीर्घता());
            ध्रुव size_t IW = static_cast<size_t>(image_shape[3].get_दीर्घता());
            OC = groups;

            स्वचालित img_shape_const = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v1::Reshape>(image_निवेशः, img_shape_const, असत्य);

            ध्रुव ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> weight_निवेशः;
            यदि (स्वचालित weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ग्रन्थिः_shared_ptr())) {
                weight_निवेशः = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_दत्त_ptr());
            } अन्यथा {
                स्वचालित shape_const = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_निवेशः = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v1::Reshape>(w_trace, shape_const, असत्य);
            }

            यदि (weight_निवेशः.get_element_type() != image_reshaped.get_element_type()) {
                weight_निवेशः = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v0::Convert>(weight_निवेशः, image_reshaped.get_element_type());
            }

            conv_बहिर्गमनम् = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v1::गणः_Convolution>(
                image_reshaped, weight_निवेशः, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
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
            } अन्यथा यदि (kernel_spatial_ic && ov::shape_आकार(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_आकार(ws) / kernel_spatial_ic;
            } अन्यथा {
                प्रत्यागम असत्य;
            }

            ध्रुव ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> weight_निवेशः;
            यदि (स्वचालित weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ग्रन्थिः_shared_ptr())) {
                weight_निवेशः = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_दत्त_ptr());
            } अन्यथा {
                स्वचालित shape_const = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_निवेशः = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v1::Reshape>(w_trace, shape_const, असत्य);
            }

            यदि (weight_निवेशः.get_element_type() != image_निवेशः.get_element_type()) {
                weight_निवेशः = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v0::Convert>(weight_निवेशः, image_निवेशः.get_element_type());
            }

            conv_बहिर्गमनम् = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v1::Convolution>(
                image_निवेशः, weight_निवेशः, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr स्वचालित target_type = ov::element::f32;
        यदि (conv_बहिर्गमनम्.get_element_type() != target_type) {
            conv_बहिर्गमनम् = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v0::Convert>(conv_बहिर्गमनम्, target_type);
        }

        std::shared_ptr<ov::op::v1::सङ्कलय> सङ्कलय_ग्रन्थिः;
        ov::Output<Node> bias_निवेशः;

        स्वचालित try_fuse_bias = [&](ध्रुव std::shared_ptr<Node> & n) {
            स्वचालित सङ्कलय_2 = ov::as_type_ptr<ov::op::v1::सङ्कलय>(n);
            यदि (!सङ्कलय_2) {
                प्रत्यागम असत्य;
            }
            प्रति (size_t i = 0; i < 2; ++i) {
                यदि (ov::is_type<ov::op::v0::Constant>(सङ्कलय_2->get_निवेशः_ग्रन्थिः_shared_ptr(i))) {
                    bias_निवेशः = सङ्कलय_2->निवेशः_मानम्(i);
                    सङ्कलय_ग्रन्थिः = सङ्कलय_2;
                    प्रत्यागम सत्य;
                }
            }
            प्रत्यागम असत्य;
        };

        प्रति (ध्रुव स्वचालित & उपभोक्ता : matmul_ग्रन्थिः->output(0).get_target_inputs()) {
            स्वचालित n = उपभोक्ता.get_ग्रन्थिः()->shared_from_this();
            यदि (try_fuse_bias(n)) {
                भङ्ग;
            }
            यदि (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                प्रति (ध्रुव स्वचालित & अग्र : n->output(0).get_target_inputs()) {
                    यदि (try_fuse_bias(अग्र.get_ग्रन्थिः()->shared_from_this())) {
                        भङ्ग;
                    }
                }
            }
            यदि (सङ्कलय_ग्रन्थिः) {
                भङ्ग;
            }
        }

        ov::Output<Node> final_बहिर्गमनम् = conv_बहिर्गमनम्;
        std::shared_ptr<Node> target_ग्रन्थिः = matmul_ग्रन्थिः;

        यदि (सङ्कलय_ग्रन्थिः) {
            ov::Output<Node> bias = bias_निवेशः;
            यदि (bias.get_element_type() != target_type) {
                bias = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v0::Convert>(bias, target_type);
            }
            स्वचालित bias_shape = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v1::Reshape>(bias, bias_shape, असत्य);
            final_बहिर्गमनम् = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v1::सङ्कलय>(conv_बहिर्गमनम्, bias);
            target_ग्रन्थिः = सङ्कलय_ग्रन्थिः;
        }

        यदि (!is_depthwise) {
            स्वचालित perm = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                weight_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_बहिर्गमनम् = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v1::Transpose>(final_बहिर्गमनम्, perm);
        }

        स्वचालित orig_shape = target_ग्रन्थिः->get_output_partial_shape(0);
        यदि (orig_shape.is_static() && final_बहिर्गमनम्.get_partial_shape().is_static()) {
            यदि (ov::shape_आकार(orig_shape.गन्तव्यम्_shape()) != ov::shape_आकार(final_बहिर्गमनम्.get_shape())) {
                प्रत्यागम असत्य;
            }
        }
        यदि (orig_shape.is_static() && final_बहिर्गमनम्.get_partial_shape() != orig_shape) {
            स्वचालित shape_const = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.गन्तव्यम्_shape());
            final_बहिर्गमनम् = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v1::Reshape>(final_बहिर्गमनम्, shape_const, असत्य);
        }

        स्वचालित orig_type = target_ग्रन्थिः->get_output_element_type(0);
        यदि (final_बहिर्गमनम्.get_element_type() != orig_type) {
            final_बहिर्गमनम् = पञ्जिकाकोशः_new_ग्रन्थिः<ov::op::v0::Convert>(final_बहिर्गमनम्, orig_type);
        }

        final_बहिर्गमनम्.get_ग्रन्थिः_shared_ptr()->समुच्चयः_friendly_name(target_ग्रन्थिः->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_बहिर्गमनम्.get_ग्रन्थिः_shared_ptr());
        ov::replace_ग्रन्थिः(target_ग्रन्थिः, final_बहिर्गमनम्.get_ग्रन्थिः_shared_ptr());

        प्रत्यागम सत्य;
    };

    पञ्जिकाकोशः_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
