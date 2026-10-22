#અન્તર્ભાવ "fuse_to_conv.h"

#અન્તર્ભાવ <openvino/core/graph_util.hpp>
#અન્તર્ભાવ <openvino/core/rt_info.hpp>
#અન્તર્ભાવ <openvino/op/add.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>
#અન્તર્ભાવ <openvino/op/convolution.hpp>
#અન્તર્ભાવ <openvino/op/extractimagepatches.hpp>
#અન્તર્ભાવ <openvino/op/group_conv.hpp>
#અન્તર્ભાવ <openvino/op/matmul.hpp>
#અન્તર્ભાવ <openvino/op/pad.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/transpose.hpp>
#અન્તર્ભાવ <openvino/pass/pattern/op/pattern.hpp>
#અન્તર્ભાવ <openvino/pass/pattern/op/wrap_type.hpp>
#અન્તર્ભાવ <utility>

નામાકાશ opp = ov::pass::pattern;

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ pass {

FuseToConv::FuseToConv() {
    અચલ auto m_in0 = opp::any_આગત();
    અચલ auto m_in1 = opp::any_આગત();
    અચલ auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    અચલ auto callback = [=](opp::Matcher & m) {
        auto matmul_ગાંઠ = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_મૂળ());
        જો (!matmul_ગાંઠ) {
            પાછા ખોટું;
        }

        auto unwrap = [](ov::Output<Node> n) {
            જ્યારે (ov::is_type<ov::op::v0::Convert>(n.get_ગાંઠ_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_ગાંઠ_shared_ptr())) {
                n = n.get_ગાંઠ_shared_ptr()->આગત_કિંમત(0);
            }
            પાછા n;
        };

        auto get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ExtractImagePatches>, std::shared_ptr<ov::op::v1::Pad>> {
            auto t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_ગાંઠ_shared_ptr());
            auto r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_આગત_ગાંઠ_shared_ptr(0)) : nullptr;
            auto t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_આગત_ગાંઠ_shared_ptr(0)) : nullptr;
            auto eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(t1->get_આગત_ગાંઠ_shared_ptr(0)) : nullptr;
            auto pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_આગત_ગાંઠ_shared_ptr(0)) : nullptr;
            પાછા {eip, pad};
        };

        તાર્કિક weight_is_in0 = સાચું;
        auto [eip, pad] = get_im2col(matmul_ગાંઠ->આગત_કિંમત(1));
        ov::Output<Node> w_trace = matmul_ગાંઠ->આગત_કિંમત(0);
        જો (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_ગાંઠ->આગત_કિંમત(0));
            w_trace = matmul_ગાંઠ->આગત_કિંમત(1);
            weight_is_in0 = ખોટું;
            જો (!pad) {
                પાછા ખોટું;
            }
        }

        auto pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_આગત_ગાંઠ_shared_ptr(1));
        auto pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_આગત_ગાંઠ_shared_ptr(2));
        જો (!pb_const || !pe_const) {
            પાછા ખોટું;
        }

        અચલ auto pb = pb_const->cast_vector<std::ptrdiff_t>();
        અચલ auto pe = pe_const->cast_vector<std::ptrdiff_t>();
        જો (pb.size() < 4 || pe.size() < 4) {
            પાછા ખોટું;
        }

        auto image_આગત = pad->આગત_કિંમત(0);
        auto image_shape = image_આગત.get_partial_shape();
        જો (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            પાછા ખોટું;
        }
        અચલ size_t IC = static_cast<size_t>(image_shape[1].get_length());

        w_trace = unwrap(w_trace);
        auto weight_pshape = w_trace.get_partial_shape();
        જો (!weight_pshape.is_static()) {
            પાછા ખોટું;
        }

        અચલ auto & ws = weight_pshape.to_shape();
        અચલ size_t KH = eip->get_sizes()[0];
        અચલ size_t KW = eip->get_sizes()[1];
        અચલ size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        જો (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            જો ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } નહીંતર જો (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        અચલ તાર્કિક is_depthwise = groups > 1 && (image_shape[0].get_length() % groups == 0);

        ov::Output<Node> conv_out;
        size_t OC = 0;

        જો (is_depthwise) {
            અચલ size_t N = static_cast<size_t>(image_shape[0].get_length()) / groups;
            અચલ size_t IH = static_cast<size_t>(image_shape[2].get_length());
            અચલ size_t IW = static_cast<size_t>(image_shape[3].get_length());
            OC = groups;

            auto img_shape_const = register_new_ગાંઠ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                register_new_ગાંઠ<ov::op::v1::Reshape>(image_આગત, img_shape_const, ખોટું);

            અચલ ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> weight_આગત;
            જો (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ગાંઠ_shared_ptr())) {
                weight_આગત = register_new_ગાંઠ<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } નહીંતર {
                auto shape_const = register_new_ગાંઠ<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_આગત = register_new_ગાંઠ<ov::op::v1::Reshape>(w_trace, shape_const, ખોટું);
            }

            જો (weight_આગત.get_element_type() != image_reshaped.get_element_type()) {
                weight_આગત = register_new_ગાંઠ<ov::op::v0::Convert>(weight_આગત, image_reshaped.get_element_type());
            }

            conv_out = register_new_ગાંઠ<ov::op::v1::GroupConvolution>(
                image_reshaped, weight_આગત, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } નહીંતર {
            જો ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } નહીંતર જો ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } નહીંતર જો (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } નહીંતર જો (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } નહીંતર જો (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } નહીંતર જો (kernel_spatial_ic && ov::shape_size(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_size(ws) / kernel_spatial_ic;
            } નહીંતર {
                પાછા ખોટું;
            }

            અચલ ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> weight_આગત;
            જો (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ગાંઠ_shared_ptr())) {
                weight_આગત = register_new_ગાંઠ<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } નહીંતર {
                auto shape_const = register_new_ગાંઠ<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_આગત = register_new_ગાંઠ<ov::op::v1::Reshape>(w_trace, shape_const, ખોટું);
            }

            જો (weight_આગત.get_element_type() != image_આગત.get_element_type()) {
                weight_આગત = register_new_ગાંઠ<ov::op::v0::Convert>(weight_આગત, image_આગત.get_element_type());
            }

            conv_out = register_new_ગાંઠ<ov::op::v1::Convolution>(
                image_આગત, weight_આગત, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr auto target_type = ov::element::f32;
        જો (conv_out.get_element_type() != target_type) {
            conv_out = register_new_ગાંઠ<ov::op::v0::Convert>(conv_out, target_type);
        }

        std::shared_ptr<ov::op::v1::Add> add_ગાંઠ;
        ov::Output<Node> bias_આગત;

        auto try_fuse_bias = [&](અચલ std::shared_ptr<Node> & n) {
            auto add = ov::as_type_ptr<ov::op::v1::Add>(n);
            જો (!add) {
                પાછા ખોટું;
            }
            માટે (size_t i = 0; i < 2; ++i) {
                જો (ov::is_type<ov::op::v0::Constant>(add->get_આગત_ગાંઠ_shared_ptr(i))) {
                    bias_આગત = add->આગત_કિંમત(i);
                    add_ગાંઠ = add;
                    પાછા સાચું;
                }
            }
            પાછા ખોટું;
        };

        માટે (અચલ auto & consumer : matmul_ગાંઠ->output(0).get_target_inputs()) {
            auto n = consumer.get_ગાંઠ()->shared_from_this();
            જો (try_fuse_bias(n)) {
                તોડો;
            }
            જો (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                માટે (અચલ auto & next : n->output(0).get_target_inputs()) {
                    જો (try_fuse_bias(next.get_ગાંઠ()->shared_from_this())) {
                        તોડો;
                    }
                }
            }
            જો (add_ગાંઠ) {
                તોડો;
            }
        }

        ov::Output<Node> final_out = conv_out;
        std::shared_ptr<Node> target_ગાંઠ = matmul_ગાંઠ;

        જો (add_ગાંઠ) {
            ov::Output<Node> bias = bias_આગત;
            જો (bias.get_element_type() != target_type) {
                bias = register_new_ગાંઠ<ov::op::v0::Convert>(bias, target_type);
            }
            auto bias_shape = register_new_ગાંઠ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = register_new_ગાંઠ<ov::op::v1::Reshape>(bias, bias_shape, ખોટું);
            final_out = register_new_ગાંઠ<ov::op::v1::Add>(conv_out, bias);
            target_ગાંઠ = add_ગાંઠ;
        }

        જો (!is_depthwise) {
            auto perm = register_new_ગાંઠ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                weight_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_out = register_new_ગાંઠ<ov::op::v1::Transpose>(final_out, perm);
        }

        auto orig_shape = target_ગાંઠ->get_output_partial_shape(0);
        જો (orig_shape.is_static() && final_out.get_partial_shape().is_static()) {
            જો (ov::shape_size(orig_shape.to_shape()) != ov::shape_size(final_out.get_shape())) {
                પાછા ખોટું;
            }
        }
        જો (orig_shape.is_static() && final_out.get_partial_shape() != orig_shape) {
            auto shape_const = register_new_ગાંઠ<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.to_shape());
            final_out = register_new_ગાંઠ<ov::op::v1::Reshape>(final_out, shape_const, ખોટું);
        }

        auto orig_type = target_ગાંઠ->get_output_element_type(0);
        જો (final_out.get_element_type() != orig_type) {
            final_out = register_new_ગાંઠ<ov::op::v0::Convert>(final_out, orig_type);
        }

        final_out.get_ગાંઠ_shared_ptr()->set_friendly_name(target_ગાંઠ->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_out.get_ગાંઠ_shared_ptr());
        ov::replace_ગાંઠ(target_ગાંઠ, final_out.get_ગાંઠ_shared_ptr());

        પાછા સાચું;
    };

    register_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
