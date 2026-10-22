#అన్తర్భావ "fuse_to_conv.h"

#అన్తర్భావ <openvino/core/graph_util.hpp>
#అన్తర్భావ <openvino/core/rt_info.hpp>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/convolution.hpp>
#అన్తర్భావ <openvino/op/extractimagepatches.hpp>
#అన్తర్భావ <openvino/op/group_conv.hpp>
#అన్తర్భావ <openvino/op/matmul.hpp>
#అన్తర్భావ <openvino/op/pad.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/transpose.hpp>
#అన్తర్భావ <openvino/pass/pattern/op/pattern.hpp>
#అన్తర్భావ <openvino/pass/pattern/op/wrap_type.hpp>
#అన్తర్భావ <utility>

నామాకాశ opp = ov::pass::pattern;

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ pass {

Fuse_గమ్యం_Conv::Fuse_గమ్యం_Conv() {
    స్థిరరాశి స్వయంచాలక m_in0 = opp::any_ఆదానం();
    స్థిరరాశి స్వయంచాలక m_in1 = opp::any_ఆదానం();
    స్థిరరాశి స్వయంచాలక m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    స్థిరరాశి స్వయంచాలక callback = [=](opp::Matcher & m) {
        స్వయంచాలక matmul_కణుపు = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_సరిపోలిక_మూలం());
        ఒకవేళ (!matmul_కణుపు) {
            తిరిగి అబద్ధం;
        }

        స్వయంచాలక unwrap = [](ov::Output<Node> n) {
            అయితే (ov::is_type<ov::op::v0::Convert>(n.get_కణుపు_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_కణుపు_shared_ptr())) {
                n = n.get_కణుపు_shared_ptr()->ఆదానం_విలువ(0);
            }
            తిరిగి n;
        };

        స్వయంచాలక get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::వెలికితీయి_Image_Patches>, std::shared_ptr<ov::op::v1::Pad>> {
            స్వయంచాలక t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_కణుపు_shared_ptr());
            స్వయంచాలక r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_ఆదానం_కణుపు_shared_ptr(0)) : శూన్యసూచిక;
            స్వయంచాలక t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_ఆదానం_కణుపు_shared_ptr(0)) : శూన్యసూచిక;
            స్వయంచాలక eip =
                t1 ? ov::as_type_ptr<ov::op::v3::వెలికితీయి_Image_Patches>(t1->get_ఆదానం_కణుపు_shared_ptr(0)) : శూన్యసూచిక;
            స్వయంచాలక pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_ఆదానం_కణుపు_shared_ptr(0)) : శూన్యసూచిక;
            తిరిగి {eip, pad};
        };

        తార్కిక బరువు_is_in0 = నిజం;
        స్వయంచాలక [eip, pad] = get_im2col(matmul_కణుపు->ఆదానం_విలువ(1));
        ov::Output<Node> w_trace = matmul_కణుపు->ఆదానం_విలువ(0);
        ఒకవేళ (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_కణుపు->ఆదానం_విలువ(0));
            w_trace = matmul_కణుపు->ఆదానం_విలువ(1);
            బరువు_is_in0 = అబద్ధం;
            ఒకవేళ (!pad) {
                తిరిగి అబద్ధం;
            }
        }

        స్వయంచాలక pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_ఆదానం_కణుపు_shared_ptr(1));
        స్వయంచాలక pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_ఆదానం_కణుపు_shared_ptr(2));
        ఒకవేళ (!pb_const || !pe_const) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి స్వయంచాలక pb = pb_const->cast_సదిశ<std::ptrdiff_t>();
        స్థిరరాశి స్వయంచాలక pe = pe_const->cast_సదిశ<std::ptrdiff_t>();
        ఒకవేళ (pb.size() < 4 || pe.size() < 4) {
            తిరిగి అబద్ధం;
        }

        స్వయంచాలక image_ఆదానం = pad->ఆదానం_విలువ(0);
        స్వయంచాలక image_shape = image_ఆదానం.get_partial_shape();
        ఒకవేళ (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            తిరిగి అబద్ధం;
        }
        స్థిరరాశి size_t IC = static_cast<size_t>(image_shape[1].get_నిడివి());

        w_trace = unwrap(w_trace);
        స్వయంచాలక బరువు_pshape = w_trace.get_partial_shape();
        ఒకవేళ (!బరువు_pshape.is_static()) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి స్వయంచాలక & ws = బరువు_pshape.గమ్యం_shape();
        స్థిరరాశి size_t KH = eip->get_sizes()[0];
        స్థిరరాశి size_t KW = eip->get_sizes()[1];
        స్థిరరాశి size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        ఒకవేళ (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            ఒకవేళ ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } లేకపోతే ఒకవేళ (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        స్థిరరాశి తార్కిక is_depthwise = groups > 1 && (image_shape[0].get_నిడివి() % groups == 0);

        ov::Output<Node> conv_బహిర్గతం;
        size_t OC = 0;

        ఒకవేళ (is_depthwise) {
            స్థిరరాశి size_t N = static_cast<size_t>(image_shape[0].get_నిడివి()) / groups;
            స్థిరరాశి size_t IH = static_cast<size_t>(image_shape[2].get_నిడివి());
            స్థిరరాశి size_t IW = static_cast<size_t>(image_shape[3].get_నిడివి());
            OC = groups;

            స్వయంచాలక img_shape_const = రిజిస్టరు_new_కణుపు<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                రిజిస్టరు_new_కణుపు<ov::op::v1::Reshape>(image_ఆదానం, img_shape_const, అబద్ధం);

            స్థిరరాశి ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> బరువు_ఆదానం;
            ఒకవేళ (స్వయంచాలక బరువు_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_కణుపు_shared_ptr())) {
                బరువు_ఆదానం = రిజిస్టరు_new_కణుపు<ov::op::v0::Constant>(బరువు_const->get_element_type(), conv_w_shape,
                                                                       బరువు_const->get_దత్తాంశం_ptr());
            } లేకపోతే {
                స్వయంచాలక shape_const = రిజిస్టరు_new_కణుపు<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                బరువు_ఆదానం = రిజిస్టరు_new_కణుపు<ov::op::v1::Reshape>(w_trace, shape_const, అబద్ధం);
            }

            ఒకవేళ (బరువు_ఆదానం.get_element_type() != image_reshaped.get_element_type()) {
                బరువు_ఆదానం = రిజిస్టరు_new_కణుపు<ov::op::v0::Convert>(బరువు_ఆదానం, image_reshaped.get_element_type());
            }

            conv_బహిర్గతం = రిజిస్టరు_new_కణుపు<ov::op::v1::సమూహం_Convolution>(
                image_reshaped, బరువు_ఆదానం, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } లేకపోతే {
            ఒకవేళ ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } లేకపోతే ఒకవేళ ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } లేకపోతే ఒకవేళ (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } లేకపోతే ఒకవేళ (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } లేకపోతే ఒకవేళ (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } లేకపోతే ఒకవేళ (kernel_spatial_ic && ov::shape_కొలత(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_కొలత(ws) / kernel_spatial_ic;
            } లేకపోతే {
                తిరిగి అబద్ధం;
            }

            స్థిరరాశి ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> బరువు_ఆదానం;
            ఒకవేళ (స్వయంచాలక బరువు_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_కణుపు_shared_ptr())) {
                బరువు_ఆదానం = రిజిస్టరు_new_కణుపు<ov::op::v0::Constant>(బరువు_const->get_element_type(), conv_w_shape,
                                                                       బరువు_const->get_దత్తాంశం_ptr());
            } లేకపోతే {
                స్వయంచాలక shape_const = రిజిస్టరు_new_కణుపు<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                బరువు_ఆదానం = రిజిస్టరు_new_కణుపు<ov::op::v1::Reshape>(w_trace, shape_const, అబద్ధం);
            }

            ఒకవేళ (బరువు_ఆదానం.get_element_type() != image_ఆదానం.get_element_type()) {
                బరువు_ఆదానం = రిజిస్టరు_new_కణుపు<ov::op::v0::Convert>(బరువు_ఆదానం, image_ఆదానం.get_element_type());
            }

            conv_బహిర్గతం = రిజిస్టరు_new_కణుపు<ov::op::v1::Convolution>(
                image_ఆదానం, బరువు_ఆదానం, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr స్వయంచాలక లక్ష్యం_type = ov::element::f32;
        ఒకవేళ (conv_బహిర్గతం.get_element_type() != లక్ష్యం_type) {
            conv_బహిర్గతం = రిజిస్టరు_new_కణుపు<ov::op::v0::Convert>(conv_బహిర్గతం, లక్ష్యం_type);
        }

        std::shared_ptr<ov::op::v1::కూడు> కూడు_కణుపు;
        ov::Output<Node> bias_ఆదానం;

        స్వయంచాలక try_fuse_bias = [&](స్థిరరాశి std::shared_ptr<Node> & n) {
            స్వయంచాలక కూడు_2 = ov::as_type_ptr<ov::op::v1::కూడు>(n);
            ఒకవేళ (!కూడు_2) {
                తిరిగి అబద్ధం;
            }
            కొరకు (size_t i = 0; i < 2; ++i) {
                ఒకవేళ (ov::is_type<ov::op::v0::Constant>(కూడు_2->get_ఆదానం_కణుపు_shared_ptr(i))) {
                    bias_ఆదానం = కూడు_2->ఆదానం_విలువ(i);
                    కూడు_కణుపు = కూడు_2;
                    తిరిగి నిజం;
                }
            }
            తిరిగి అబద్ధం;
        };

        కొరకు (స్థిరరాశి స్వయంచాలక & వినియోగదారు : matmul_కణుపు->output(0).get_లక్ష్యం_inputs()) {
            స్వయంచాలక n = వినియోగదారు.get_కణుపు()->shared_from_this();
            ఒకవేళ (try_fuse_bias(n)) {
                ఆపు;
            }
            ఒకవేళ (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                కొరకు (స్థిరరాశి స్వయంచాలక & తదుపరి : n->output(0).get_లక్ష్యం_inputs()) {
                    ఒకవేళ (try_fuse_bias(తదుపరి.get_కణుపు()->shared_from_this())) {
                        ఆపు;
                    }
                }
            }
            ఒకవేళ (కూడు_కణుపు) {
                ఆపు;
            }
        }

        ov::Output<Node> final_బహిర్గతం = conv_బహిర్గతం;
        std::shared_ptr<Node> లక్ష్యం_కణుపు = matmul_కణుపు;

        ఒకవేళ (కూడు_కణుపు) {
            ov::Output<Node> bias = bias_ఆదానం;
            ఒకవేళ (bias.get_element_type() != లక్ష్యం_type) {
                bias = రిజిస్టరు_new_కణుపు<ov::op::v0::Convert>(bias, లక్ష్యం_type);
            }
            స్వయంచాలక bias_shape = రిజిస్టరు_new_కణుపు<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = రిజిస్టరు_new_కణుపు<ov::op::v1::Reshape>(bias, bias_shape, అబద్ధం);
            final_బహిర్గతం = రిజిస్టరు_new_కణుపు<ov::op::v1::కూడు>(conv_బహిర్గతం, bias);
            లక్ష్యం_కణుపు = కూడు_కణుపు;
        }

        ఒకవేళ (!is_depthwise) {
            స్వయంచాలక perm = రిజిస్టరు_new_కణుపు<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                బరువు_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_బహిర్గతం = రిజిస్టరు_new_కణుపు<ov::op::v1::Transpose>(final_బహిర్గతం, perm);
        }

        స్వయంచాలక orig_shape = లక్ష్యం_కణుపు->get_output_partial_shape(0);
        ఒకవేళ (orig_shape.is_static() && final_బహిర్గతం.get_partial_shape().is_static()) {
            ఒకవేళ (ov::shape_కొలత(orig_shape.గమ్యం_shape()) != ov::shape_కొలత(final_బహిర్గతం.get_shape())) {
                తిరిగి అబద్ధం;
            }
        }
        ఒకవేళ (orig_shape.is_static() && final_బహిర్గతం.get_partial_shape() != orig_shape) {
            స్వయంచాలక shape_const = రిజిస్టరు_new_కణుపు<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.గమ్యం_shape());
            final_బహిర్గతం = రిజిస్టరు_new_కణుపు<ov::op::v1::Reshape>(final_బహిర్గతం, shape_const, అబద్ధం);
        }

        స్వయంచాలక orig_type = లక్ష్యం_కణుపు->get_output_element_type(0);
        ఒకవేళ (final_బహిర్గతం.get_element_type() != orig_type) {
            final_బహిర్గతం = రిజిస్టరు_new_కణుపు<ov::op::v0::Convert>(final_బహిర్గతం, orig_type);
        }

        final_బహిర్గతం.get_కణుపు_shared_ptr()->సమితి_friendly_name(లక్ష్యం_కణుపు->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_బహిర్గతం.get_కణుపు_shared_ptr());
        ov::replace_కణుపు(లక్ష్యం_కణుపు, final_బహిర్గతం.get_కణుపు_shared_ptr());

        తిరిగి నిజం;
    };

    రిజిస్టరు_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
