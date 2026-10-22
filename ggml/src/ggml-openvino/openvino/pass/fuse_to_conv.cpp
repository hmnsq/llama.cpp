#include "fuse_to_conv.h"

#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/convolution.hpp>
#include <openvino/op/extractimagepatches.hpp>
#include <openvino/op/group_conv.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/pad.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/pass/pattern/op/pattern.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <utility>

namespace opp = ov::pass::pattern;

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

FuseToConv::FuseToConv() {
    const auto m_in0 = opp::any_input();
    const auto m_in1 = opp::any_input();
    const auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    const auto callback = [=](opp::Matcher & m) {
        auto matmul_node = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_root());
        གལ་སྲིད (!matmul_node) {
            ལོག false;
        }

        auto unwrap = [](ov::Output<Node> n) {
            སྐབས (ov::is_type<ov::op::v0::Convert>(n.get_node_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_node_shared_ptr())) {
                n = n.get_node_shared_ptr()->input_value(0);
            }
            ལོག n;
        };

        auto get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ExtractImagePatches>, std::shared_ptr<ov::op::v1::Pad>> {
            auto t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_node_shared_ptr());
            auto r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_input_node_shared_ptr(0)) : nullptr;
            auto t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_input_node_shared_ptr(0)) : nullptr;
            auto eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(t1->get_input_node_shared_ptr(0)) : nullptr;
            auto pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_input_node_shared_ptr(0)) : nullptr;
            ལོག {eip, pad};
        };

        bool weight_is_in0 = true;
        auto [eip, pad] = get_im2col(matmul_node->input_value(1));
        ov::Output<Node> w_trace = matmul_node->input_value(0);
        གལ་སྲིད (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_node->input_value(0));
            w_trace = matmul_node->input_value(1);
            weight_is_in0 = false;
            གལ་སྲིད (!pad) {
                ལོག false;
            }
        }

        auto pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_input_node_shared_ptr(1));
        auto pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_input_node_shared_ptr(2));
        གལ་སྲིད (!pb_const || !pe_const) {
            ལོག false;
        }

        const auto pb = pb_const->cast_vector<std::ptrdiff_t>();
        const auto pe = pe_const->cast_vector<std::ptrdiff_t>();
        གལ་སྲིད (pb.size() < 4 || pe.size() < 4) {
            ལོག false;
        }

        auto image_input = pad->input_value(0);
        auto image_shape = image_input.get_partial_shape();
        གལ་སྲིད (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            ལོག false;
        }
        const size_t IC = static_cast<size_t>(image_shape[1].get_length());

        w_trace = unwrap(w_trace);
        auto weight_pshape = w_trace.get_partial_shape();
        གལ་སྲིད (!weight_pshape.is_static()) {
            ལོག false;
        }

        const auto & ws = weight_pshape.to_shape();
        const size_t KH = eip->get_sizes()[0];
        const size_t KW = eip->get_sizes()[1];
        const size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        གལ་སྲིད (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            གལ་སྲིད ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } དེ་མིན གལ་སྲིད (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        const bool is_depthwise = groups > 1 && (image_shape[0].get_length() % groups == 0);

        ov::Output<Node> conv_out;
        size_t OC = 0;

        གལ་སྲིད (is_depthwise) {
            const size_t N = static_cast<size_t>(image_shape[0].get_length()) / groups;
            const size_t IH = static_cast<size_t>(image_shape[2].get_length());
            const size_t IW = static_cast<size_t>(image_shape[3].get_length());
            OC = groups;

            auto img_shape_const = register_new_node<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                register_new_node<ov::op::v1::Reshape>(image_input, img_shape_const, false);

            const ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> weight_input;
            གལ་སྲིད (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_node_shared_ptr())) {
                weight_input = register_new_node<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } དེ་མིན {
                auto shape_const = register_new_node<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_input = register_new_node<ov::op::v1::Reshape>(w_trace, shape_const, false);
            }

            གལ་སྲིད (weight_input.get_element_type() != image_reshaped.get_element_type()) {
                weight_input = register_new_node<ov::op::v0::Convert>(weight_input, image_reshaped.get_element_type());
            }

            conv_out = register_new_node<ov::op::v1::GroupConvolution>(
                image_reshaped, weight_input, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } དེ་མིན {
            གལ་སྲིད ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } དེ་མིན གལ་སྲིད ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } དེ་མིན གལ་སྲིད (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } དེ་མིན གལ་སྲིད (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } དེ་མིན གལ་སྲིད (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } དེ་མིན གལ་སྲིད (kernel_spatial_ic && ov::shape_size(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_size(ws) / kernel_spatial_ic;
            } དེ་མིན {
                ལོག false;
            }

            const ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> weight_input;
            གལ་སྲིད (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_node_shared_ptr())) {
                weight_input = register_new_node<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } དེ་མིན {
                auto shape_const = register_new_node<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_input = register_new_node<ov::op::v1::Reshape>(w_trace, shape_const, false);
            }

            གལ་སྲིད (weight_input.get_element_type() != image_input.get_element_type()) {
                weight_input = register_new_node<ov::op::v0::Convert>(weight_input, image_input.get_element_type());
            }

            conv_out = register_new_node<ov::op::v1::Convolution>(
                image_input, weight_input, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr auto target_type = ov::element::f32;
        གལ་སྲིད (conv_out.get_element_type() != target_type) {
            conv_out = register_new_node<ov::op::v0::Convert>(conv_out, target_type);
        }

        std::shared_ptr<ov::op::v1::Add> add_node;
        ov::Output<Node> bias_input;

        auto try_fuse_bias = [&](const std::shared_ptr<Node> & n) {
            auto add = ov::as_type_ptr<ov::op::v1::Add>(n);
            གལ་སྲིད (!add) {
                ལོག false;
            }
            དོན་ལུ (size_t i = 0; i < 2; ++i) {
                གལ་སྲིད (ov::is_type<ov::op::v0::Constant>(add->get_input_node_shared_ptr(i))) {
                    bias_input = add->input_value(i);
                    add_node = add;
                    ལོག true;
                }
            }
            ལོག false;
        };

        དོན་ལུ (const auto & consumer : matmul_node->output(0).get_target_inputs()) {
            auto n = consumer.get_node()->shared_from_this();
            གལ་སྲིད (try_fuse_bias(n)) {
                bha_nga;
            }
            གལ་སྲིད (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                དོན་ལུ (const auto & next : n->output(0).get_target_inputs()) {
                    གལ་སྲིད (try_fuse_bias(next.get_node()->shared_from_this())) {
                        bha_nga;
                    }
                }
            }
            གལ་སྲིད (add_node) {
                bha_nga;
            }
        }

        ov::Output<Node> final_out = conv_out;
        std::shared_ptr<Node> target_node = matmul_node;

        གལ་སྲིད (add_node) {
            ov::Output<Node> bias = bias_input;
            གལ་སྲིད (bias.get_element_type() != target_type) {
                bias = register_new_node<ov::op::v0::Convert>(bias, target_type);
            }
            auto bias_shape = register_new_node<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = register_new_node<ov::op::v1::Reshape>(bias, bias_shape, false);
            final_out = register_new_node<ov::op::v1::Add>(conv_out, bias);
            target_node = add_node;
        }

        གལ་སྲིད (!is_depthwise) {
            auto perm = register_new_node<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                weight_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_out = register_new_node<ov::op::v1::Transpose>(final_out, perm);
        }

        auto orig_shape = target_node->get_output_partial_shape(0);
        གལ་སྲིད (orig_shape.is_static() && final_out.get_partial_shape().is_static()) {
            གལ་སྲིད (ov::shape_size(orig_shape.to_shape()) != ov::shape_size(final_out.get_shape())) {
                ལོག false;
            }
        }
        གལ་སྲིད (orig_shape.is_static() && final_out.get_partial_shape() != orig_shape) {
            auto shape_const = register_new_node<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.to_shape());
            final_out = register_new_node<ov::op::v1::Reshape>(final_out, shape_const, false);
        }

        auto orig_type = target_node->get_output_element_type(0);
        གལ་སྲིད (final_out.get_element_type() != orig_type) {
            final_out = register_new_node<ov::op::v0::Convert>(final_out, orig_type);
        }

        final_out.get_node_shared_ptr()->set_friendly_name(target_node->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_out.get_node_shared_ptr());
        ov::replace_node(target_node, final_out.get_node_shared_ptr());

        ལོག true;
    };

    register_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
