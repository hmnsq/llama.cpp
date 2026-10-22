#شامل "fuse_to_conv.h"

#شامل <openvino/core/graph_util.hpp>
#شامل <openvino/core/rt_info.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/convert.hpp>
#شامل <openvino/op/convolution.hpp>
#شامل <openvino/op/extractimagepatches.hpp>
#شامل <openvino/op/group_conv.hpp>
#شامل <openvino/op/matmul.hpp>
#شامل <openvino/op/pad.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/transpose.hpp>
#شامل <openvino/pass/pattern/op/pattern.hpp>
#شامل <openvino/pass/pattern/op/wrap_type.hpp>
#شامل <utility>

فضاینام opp = ov::pass::pattern;

فضاینام ov {
فضاینام frontend {
فضاینام ggml {
فضاینام pass {

FuseToConv::FuseToConv() {
    ثابت auto m_in0 = opp::any_درون_داد();
    ثابت auto m_in1 = opp::any_درون_داد();
    ثابت auto m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    ثابت auto callback = [=](opp::Matcher & m) {
        auto matmul_گره = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_match_ریشه());
        اگر (!matmul_گره) {
            بازگشت نادرست;
        }

        auto unwrap = [](ov::Output<Node> n) {
            وقتی (ov::is_type<ov::op::v0::Convert>(n.get_گره_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_گره_shared_ptr())) {
                n = n.get_گره_shared_ptr()->درون_داد_مقدار(0);
            }
            بازگشت n;
        };

        auto get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ExtractImagePatches>, std::shared_ptr<ov::op::v1::Pad>> {
            auto t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_گره_shared_ptr());
            auto r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_درون_داد_گره_shared_ptr(0)) : nullptr;
            auto t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_درون_داد_گره_shared_ptr(0)) : nullptr;
            auto eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ExtractImagePatches>(t1->get_درون_داد_گره_shared_ptr(0)) : nullptr;
            auto pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_درون_داد_گره_shared_ptr(0)) : nullptr;
            بازگشت {eip, pad};
        };

        منطقی weight_is_in0 = درست;
        auto [eip, pad] = get_im2col(matmul_گره->درون_داد_مقدار(1));
        ov::Output<Node> w_trace = matmul_گره->درون_داد_مقدار(0);
        اگر (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_گره->درون_داد_مقدار(0));
            w_trace = matmul_گره->درون_داد_مقدار(1);
            weight_is_in0 = نادرست;
            اگر (!pad) {
                بازگشت نادرست;
            }
        }

        auto pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_درون_داد_گره_shared_ptr(1));
        auto pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_درون_داد_گره_shared_ptr(2));
        اگر (!pb_const || !pe_const) {
            بازگشت نادرست;
        }

        ثابت auto pb = pb_const->cast_vector<std::ptrdiff_t>();
        ثابت auto pe = pe_const->cast_vector<std::ptrdiff_t>();
        اگر (pb.size() < 4 || pe.size() < 4) {
            بازگشت نادرست;
        }

        auto image_درون_داد = pad->درون_داد_مقدار(0);
        auto image_shape = image_درون_داد.get_partial_shape();
        اگر (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            بازگشت نادرست;
        }
        ثابت size_t IC = static_cast<size_t>(image_shape[1].get_length());

        w_trace = unwrap(w_trace);
        auto weight_pshape = w_trace.get_partial_shape();
        اگر (!weight_pshape.is_static()) {
            بازگشت نادرست;
        }

        ثابت auto & ws = weight_pshape.to_shape();
        ثابت size_t KH = eip->get_sizes()[0];
        ثابت size_t KW = eip->get_sizes()[1];
        ثابت size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        اگر (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            اگر ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } وگرنه اگر (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        ثابت منطقی is_depthwise = groups > 1 && (image_shape[0].get_length() % groups == 0);

        ov::Output<Node> conv_out;
        size_t OC = 0;

        اگر (is_depthwise) {
            ثابت size_t N = static_cast<size_t>(image_shape[0].get_length()) / groups;
            ثابت size_t IH = static_cast<size_t>(image_shape[2].get_length());
            ثابت size_t IW = static_cast<size_t>(image_shape[3].get_length());
            OC = groups;

            auto img_shape_const = register_new_گره<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                register_new_گره<ov::op::v1::Reshape>(image_درون_داد, img_shape_const, نادرست);

            ثابت ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> weight_درون_داد;
            اگر (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_گره_shared_ptr())) {
                weight_درون_داد = register_new_گره<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } وگرنه {
                auto shape_const = register_new_گره<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_درون_داد = register_new_گره<ov::op::v1::Reshape>(w_trace, shape_const, نادرست);
            }

            اگر (weight_درون_داد.get_element_type() != image_reshaped.get_element_type()) {
                weight_درون_داد = register_new_گره<ov::op::v0::Convert>(weight_درون_داد, image_reshaped.get_element_type());
            }

            conv_out = register_new_گره<ov::op::v1::GroupConvolution>(
                image_reshaped, weight_درون_داد, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } وگرنه {
            اگر ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } وگرنه اگر ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } وگرنه اگر (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } وگرنه اگر (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } وگرنه اگر (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } وگرنه اگر (kernel_spatial_ic && ov::shape_size(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_size(ws) / kernel_spatial_ic;
            } وگرنه {
                بازگشت نادرست;
            }

            ثابت ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> weight_درون_داد;
            اگر (auto weight_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_گره_shared_ptr())) {
                weight_درون_داد = register_new_گره<ov::op::v0::Constant>(weight_const->get_element_type(), conv_w_shape,
                                                                       weight_const->get_data_ptr());
            } وگرنه {
                auto shape_const = register_new_گره<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                weight_درون_داد = register_new_گره<ov::op::v1::Reshape>(w_trace, shape_const, نادرست);
            }

            اگر (weight_درون_داد.get_element_type() != image_درون_داد.get_element_type()) {
                weight_درون_داد = register_new_گره<ov::op::v0::Convert>(weight_درون_داد, image_درون_داد.get_element_type());
            }

            conv_out = register_new_گره<ov::op::v1::Convolution>(
                image_درون_داد, weight_درون_داد, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr auto target_type = ov::element::f32;
        اگر (conv_out.get_element_type() != target_type) {
            conv_out = register_new_گره<ov::op::v0::Convert>(conv_out, target_type);
        }

        std::shared_ptr<ov::op::v1::Add> add_گره;
        ov::Output<Node> bias_درون_داد;

        auto try_fuse_bias = [&](ثابت std::shared_ptr<Node> & n) {
            auto add = ov::as_type_ptr<ov::op::v1::Add>(n);
            اگر (!add) {
                بازگشت نادرست;
            }
            برای (size_t i = 0; i < 2; ++i) {
                اگر (ov::is_type<ov::op::v0::Constant>(add->get_درون_داد_گره_shared_ptr(i))) {
                    bias_درون_داد = add->درون_داد_مقدار(i);
                    add_گره = add;
                    بازگشت درست;
                }
            }
            بازگشت نادرست;
        };

        برای (ثابت auto & consumer : matmul_گره->output(0).get_target_inputs()) {
            auto n = consumer.get_گره()->shared_from_this();
            اگر (try_fuse_bias(n)) {
                بشکن;
            }
            اگر (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                برای (ثابت auto & next : n->output(0).get_target_inputs()) {
                    اگر (try_fuse_bias(next.get_گره()->shared_from_this())) {
                        بشکن;
                    }
                }
            }
            اگر (add_گره) {
                بشکن;
            }
        }

        ov::Output<Node> final_out = conv_out;
        std::shared_ptr<Node> target_گره = matmul_گره;

        اگر (add_گره) {
            ov::Output<Node> bias = bias_درون_داد;
            اگر (bias.get_element_type() != target_type) {
                bias = register_new_گره<ov::op::v0::Convert>(bias, target_type);
            }
            auto bias_shape = register_new_گره<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = register_new_گره<ov::op::v1::Reshape>(bias, bias_shape, نادرست);
            final_out = register_new_گره<ov::op::v1::Add>(conv_out, bias);
            target_گره = add_گره;
        }

        اگر (!is_depthwise) {
            auto perm = register_new_گره<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                weight_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_out = register_new_گره<ov::op::v1::Transpose>(final_out, perm);
        }

        auto orig_shape = target_گره->get_output_partial_shape(0);
        اگر (orig_shape.is_static() && final_out.get_partial_shape().is_static()) {
            اگر (ov::shape_size(orig_shape.to_shape()) != ov::shape_size(final_out.get_shape())) {
                بازگشت نادرست;
            }
        }
        اگر (orig_shape.is_static() && final_out.get_partial_shape() != orig_shape) {
            auto shape_const = register_new_گره<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.to_shape());
            final_out = register_new_گره<ov::op::v1::Reshape>(final_out, shape_const, نادرست);
        }

        auto orig_type = target_گره->get_output_element_type(0);
        اگر (final_out.get_element_type() != orig_type) {
            final_out = register_new_گره<ov::op::v0::Convert>(final_out, orig_type);
        }

        final_out.get_گره_shared_ptr()->set_friendly_name(target_گره->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_out.get_گره_shared_ptr());
        ov::replace_گره(target_گره, final_out.get_گره_shared_ptr());

        بازگشت درست;
    };

    register_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
