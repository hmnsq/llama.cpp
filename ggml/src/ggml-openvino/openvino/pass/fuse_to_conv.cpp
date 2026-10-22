#समावेश "fuse_to_conv.h"

#समावेश <openvino/core/graph_util.hpp>
#समावेश <openvino/core/rt_info.hpp>
#समावेश <openvino/op/add.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/convolution.hpp>
#समावेश <openvino/op/extractimagepatches.hpp>
#समावेश <openvino/op/group_conv.hpp>
#समावेश <openvino/op/matmul.hpp>
#समावेश <openvino/op/pad.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/transpose.hpp>
#समावेश <openvino/pass/pattern/op/pattern.hpp>
#समावेश <openvino/pass/pattern/op/wrap_type.hpp>
#समावेश <utility>

namespace opp = ov::pass::pattern;

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

FuseToConv::FuseToConv() {
    स्थिर स्वतः m_in0 = opp::any_निवेश();
    स्थिर स्वतः m_in1 = opp::any_निवेश();
    स्थिर स्वतः m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    स्थिर स्वतः callback = [=](opp::Matcher & m) {
        स्वतः matmul_गाँठ = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_मूल());
        अगर (!matmul_गाँठ) {
            वापस false;
        }

        स्वतः unwrap = [](ov::Output<Node> n) {
            जबतक (ov::is_type<ov::op::v0::Convert>(n.get_गाँठ_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_गाँठ_shared_ptr())) {
                n = n.get_गाँठ_shared_ptr()->निवेश_मूल्य(0);
            }
            वापस n;
        };

        स्वतः get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ExtractImagePatches>, std::shared_ptr<ov::op::v1::Pad>> {
            स्वतः t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_गाँठ_shared_ptr());
            स्वतः r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_निवेश_गाँठ_shared_ptr(0)) : nullptr;
            स्वतः t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_निवेश_गाँठ_shared_ptr(0)) : nullptr;
            स्वतः eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(t1->get_निवेश_गाँठ_shared_ptr(0)) : nullptr;
            स्वतः pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_निवेश_गाँठ_shared_ptr(0)) : nullptr;
            वापस {eip, pad};
        };

        bool weight_is_in0 = true;
        स्वतः [eip, pad] = get_im2col(matmul_गाँठ->निवेश_मूल्य(1));
        ov::Output<Node> w_trace = matmul_गाँठ->निवेश_मूल्य(0);
        अगर (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_गाँठ->निवेश_मूल्य(0));
            w_trace = matmul_गाँठ->निवेश_मूल्य(1);
            weight_is_in0 = false;
            अगर (!pad) {
                वापस false;
            }
        }

        स्वतः pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_निवेश_गाँठ_shared_ptr(1));
        स्वतः pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_निवेश_गाँठ_shared_ptr(2));
        अगर (!pb_const || !pe_const) {
            वापस false;
        }

        स्थिर स्वतः pb = pb_const->cast_vector<std::सूचक_भेद_प्रकार>();
        स्थिर स्वतः pe = pe_const->cast_vector<std::सूचक_भेद_प्रकार>();
        अगर (pb.size() < 4 || pe.size() < 4) {
            वापस false;
        }

        स्वतः image_निवेश = pad->निवेश_मूल्य(0);
        स्वतः image_shape = image_निवेश.get_partial_shape();
        अगर (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            वापस false;
        }
        स्थिर माप_प्रकार IC = static_cast<माप_प्रकार>(image_shape[1].get_लंबाई());

        w_trace = unwrap(w_trace);
        स्वतः weight_pshape = w_trace.get_partial_shape();
        अगर (!weight_pshape.is_static()) {
            वापस false;
        }

        स्थिर स्वतः & खाली = weight_pshape.to_shape();
        स्थिर माप_प्रकार KH = eip->get_sizes()[0];
        स्थिर माप_प्रकार KW = eip->get_sizes()[1];
        स्थिर माप_प्रकार kernel_spatial_ic = IC * KH * KW;

        माप_प्रकार groups = 0;
        अगर (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            अगर ((खाली.size() == 4 && खाली[1] == 1 && खाली[2] == KH && खाली[3] == KW && खाली[0] > 1) ||
                (खाली.size() == 2 && खाली[1] == KH * KW && खाली[0] > 1) ||
                (खाली.size() == 3 && खाली[1] == 1 && खाली[2] == KH * KW && खाली[0] > 1)) {
                groups = खाली[0];
            } अन्यथा अगर (खाली.size() == 4 && खाली[0] == 1 && खाली[2] == 1 && खाली[3] == KH * KW && खाली[1] > 1) {
                groups = खाली[1];
            }
        }

        स्थिर bool is_depthwise = groups > 1 && (image_shape[0].get_लंबाई() % groups == 0);

        ov::Output<Node> conv_out;
        माप_प्रकार OC = 0;

        अगर (is_depthwise) {
            स्थिर माप_प्रकार N = static_cast<माप_प्रकार>(image_shape[0].get_लंबाई()) / groups;
            स्थिर माप_प्रकार IH = static_cast<माप_प्रकार>(image_shape[2].get_लंबाई());
            स्थिर माप_प्रकार IW = static_cast<माप_प्रकार>(image_shape[3].get_लंबाई());
            OC = groups;

            स्वतः img_shape_const = register_new_गाँठ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                register_new_गाँठ<ov::op::v1::Reshape>(image_निवेश, img_shape_const, false);

            स्थिर ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> weight_निवेश;
            अगर (स्वतः weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_गाँठ_shared_ptr())) {
                weight_निवेश = register_new_गाँठ<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_तथ्य_ptr());
            } अन्यथा {
                स्वतः shape_const = register_new_गाँठ<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_निवेश = register_new_गाँठ<ov::op::v1::Reshape>(w_trace, shape_const, false);
            }

            अगर (weight_निवेश.get_element_type() != image_reshaped.get_element_type()) {
                weight_निवेश = register_new_गाँठ<ov::op::v0::Convert>(weight_निवेश, image_reshaped.get_element_type());
            }

            conv_out = register_new_गाँठ<ov::op::v1::GroupConvolution>(
                image_reshaped, weight_निवेश, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } अन्यथा {
            अगर ((खाली.size() == 4 && खाली[1] == IC && खाली[2] == KH && खाली[3] == KW) ||
                (खाली.size() == 3 && खाली[1] == IC && खाली[2] == KW) || (खाली.size() == 2 && खाली[1] == kernel_spatial_ic)) {
                OC = खाली[0];
            } अन्यथा अगर ((खाली.size() == 4 && खाली[0] == 1 && खाली[2] == IC && खाली[3] == KW) ||
                       (खाली.size() == 2 && खाली[0] == kernel_spatial_ic)) {
                OC = खाली[1];
            } अन्यथा अगर (खाली.size() == 3 && खाली[0] == 1 && खाली[1] == IC && खाली[2] == KW) {
                OC = 1;
            } अन्यथा अगर (खाली.size() == 4 && खाली[3] == kernel_spatial_ic) {
                OC = खाली[2];
            } अन्यथा अगर (खाली.size() == 4 && खाली[2] == kernel_spatial_ic) {
                OC = खाली[3];
            } अन्यथा अगर (kernel_spatial_ic && ov::shape_आकार(खाली) % kernel_spatial_ic == 0) {
                OC = ov::shape_आकार(खाली) / kernel_spatial_ic;
            } अन्यथा {
                वापस false;
            }

            स्थिर ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> weight_निवेश;
            अगर (स्वतः weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_गाँठ_shared_ptr())) {
                weight_निवेश = register_new_गाँठ<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_तथ्य_ptr());
            } अन्यथा {
                स्वतः shape_const = register_new_गाँठ<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_निवेश = register_new_गाँठ<ov::op::v1::Reshape>(w_trace, shape_const, false);
            }

            अगर (weight_निवेश.get_element_type() != image_निवेश.get_element_type()) {
                weight_निवेश = register_new_गाँठ<ov::op::v0::Convert>(weight_निवेश, image_निवेश.get_element_type());
            }

            conv_out = register_new_गाँठ<ov::op::v1::Convolution>(
                image_निवेश, weight_निवेश, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr स्वतः target_type = ov::element::f32;
        अगर (conv_out.get_element_type() != target_type) {
            conv_out = register_new_गाँठ<ov::op::v0::Convert>(conv_out, target_type);
        }

        std::shared_ptr<ov::op::v1::Add> add_गाँठ;
        ov::Output<Node> bias_निवेश;

        स्वतः try_fuse_bias = [&](स्थिर std::shared_ptr<Node> & n) {
            स्वतः add = ov::as_type_ptr<ov::op::v1::Add>(n);
            अगर (!add) {
                वापस false;
            }
            क्रम (माप_प्रकार i = 0; i < 2; ++i) {
                अगर (ov::is_type<ov::op::v0::Constant>(add->get_निवेश_गाँठ_shared_ptr(i))) {
                    bias_निवेश = add->निवेश_मूल्य(i);
                    add_गाँठ = add;
                    वापस true;
                }
            }
            वापस false;
        };

        क्रम (स्थिर स्वतः & consumer : matmul_गाँठ->output(0).get_target_inputs()) {
            स्वतः n = consumer.get_गाँठ()->shared_from_this();
            अगर (try_fuse_bias(n)) {
                अवरोध;
            }
            अगर (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                क्रम (स्थिर स्वतः & अग्र : n->output(0).get_target_inputs()) {
                    अगर (try_fuse_bias(अग्र.get_गाँठ()->shared_from_this())) {
                        अवरोध;
                    }
                }
            }
            अगर (add_गाँठ) {
                अवरोध;
            }
        }

        ov::Output<Node> final_out = conv_out;
        std::shared_ptr<Node> target_गाँठ = matmul_गाँठ;

        अगर (add_गाँठ) {
            ov::Output<Node> bias = bias_निवेश;
            अगर (bias.get_element_type() != target_type) {
                bias = register_new_गाँठ<ov::op::v0::Convert>(bias, target_type);
            }
            स्वतः bias_shape = register_new_गाँठ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = register_new_गाँठ<ov::op::v1::Reshape>(bias, bias_shape, false);
            final_out = register_new_गाँठ<ov::op::v1::Add>(conv_out, bias);
            target_गाँठ = add_गाँठ;
        }

        अगर (!is_depthwise) {
            स्वतः perm = register_new_गाँठ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                weight_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_out = register_new_गाँठ<ov::op::v1::Transpose>(final_out, perm);
        }

        स्वतः orig_shape = target_गाँठ->get_output_partial_shape(0);
        अगर (orig_shape.is_static() && final_out.get_partial_shape().is_static()) {
            अगर (ov::shape_आकार(orig_shape.to_shape()) != ov::shape_आकार(final_out.get_shape())) {
                वापस false;
            }
        }
        अगर (orig_shape.is_static() && final_out.get_partial_shape() != orig_shape) {
            स्वतः shape_const = register_new_गाँठ<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.to_shape());
            final_out = register_new_गाँठ<ov::op::v1::Reshape>(final_out, shape_const, false);
        }

        स्वतः orig_type = target_गाँठ->get_output_element_type(0);
        अगर (final_out.get_element_type() != orig_type) {
            final_out = register_new_गाँठ<ov::op::v0::Convert>(final_out, orig_type);
        }

        final_out.get_गाँठ_shared_ptr()->set_friendly_name(target_गाँठ->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_out.get_गाँठ_shared_ptr());
        ov::replace_गाँठ(target_गाँठ, final_out.get_गाँठ_shared_ptr());

        वापस true;
    };

    register_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
