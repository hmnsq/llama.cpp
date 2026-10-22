#ضمن "fuse_to_conv.h"

#ضمن <openvino/core/graph_util.hpp>
#ضمن <openvino/core/rt_info.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/convolution.hpp>
#ضمن <openvino/op/extractimagepatches.hpp>
#ضمن <openvino/op/group_conv.hpp>
#ضمن <openvino/op/matmul.hpp>
#ضمن <openvino/op/pad.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <openvino/pass/pattern/op/pattern.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>
#ضمن <utility>

نطاق opp = ov::pass::pattern;

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

Fuse_منزل_Conv::Fuse_منزل_Conv() {
    ثابت خپلکاری m_in0 = opp::any_ننوت();
    ثابت خپلکاری m_in1 = opp::any_ننوت();
    ثابت خپلکاری m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    ثابت خپلکاری callback = [=](opp::Matcher & m) {
        خپلکاری matmul_غوټه = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_سمون_ریښه());
        إذا (!matmul_غوټه) {
            أرجع خطأ;
        }

        خپلکاری unwrap = [](ov::Output<Node> n) {
            طالما (ov::is_type<ov::op::v0::Convert>(n.get_غوټه_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_غوټه_shared_ptr())) {
                n = n.get_غوټه_shared_ptr()->ننوت_ارزښت(0);
            }
            أرجع n;
        };

        خپلکاری get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ایستل_Image_Patches>, std::shared_ptr<ov::op::v1::Pad>> {
            خپلکاری t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_غوټه_shared_ptr());
            خپلکاری r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_ننوت_غوټه_shared_ptr(0)) : تش_اشاره;
            خپلکاری t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_ننوت_غوټه_shared_ptr(0)) : تش_اشاره;
            خپلکاری eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ایستل_Image_Patches>(t1->get_ننوت_غوټه_shared_ptr(0)) : تش_اشاره;
            خپلکاری pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_ننوت_غوټه_shared_ptr(0)) : تش_اشاره;
            أرجع {eip, pad};
        };

        منطقي وزن_is_in0 = ریښتیا;
        خپلکاری [eip, pad] = get_im2col(matmul_غوټه->ننوت_ارزښت(1));
        ov::Output<Node> w_trace = matmul_غوټه->ننوت_ارزښت(0);
        إذا (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_غوټه->ننوت_ارزښت(0));
            w_trace = matmul_غوټه->ننوت_ارزښت(1);
            وزن_is_in0 = خطأ;
            إذا (!pad) {
                أرجع خطأ;
            }
        }

        خپلکاری pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_ننوت_غوټه_shared_ptr(1));
        خپلکاری pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_ننوت_غوټه_shared_ptr(2));
        إذا (!pb_const || !pe_const) {
            أرجع خطأ;
        }

        ثابت خپلکاری pb = pb_const->cast_ویکټور<std::ptrdiff_t>();
        ثابت خپلکاری pe = pe_const->cast_ویکټور<std::ptrdiff_t>();
        إذا (pb.size() < 4 || pe.size() < 4) {
            أرجع خطأ;
        }

        خپلکاری image_ننوت = pad->ننوت_ارزښت(0);
        خپلکاری image_shape = image_ننوت.get_partial_shape();
        إذا (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            أرجع خطأ;
        }
        ثابت size_t IC = static_cast<size_t>(image_shape[1].get_اوږدتیا());

        w_trace = unwrap(w_trace);
        خپلکاری وزن_pshape = w_trace.get_partial_shape();
        إذا (!وزن_pshape.is_static()) {
            أرجع خطأ;
        }

        ثابت خپلکاری & ws = وزن_pshape.منزل_shape();
        ثابت size_t KH = eip->get_sizes()[0];
        ثابت size_t KW = eip->get_sizes()[1];
        ثابت size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        إذا (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            إذا ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } وإلا إذا (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        ثابت منطقي is_depthwise = groups > 1 && (image_shape[0].get_اوږدتیا() % groups == 0);

        ov::Output<Node> conv_بهر;
        size_t OC = 0;

        إذا (is_depthwise) {
            ثابت size_t N = static_cast<size_t>(image_shape[0].get_اوږدتیا()) / groups;
            ثابت size_t IH = static_cast<size_t>(image_shape[2].get_اوږدتیا());
            ثابت size_t IW = static_cast<size_t>(image_shape[3].get_اوږدتیا());
            OC = groups;

            خپلکاری img_shape_const = ثبتوونکی_new_غوټه<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                ثبتوونکی_new_غوټه<ov::op::v1::Reshape>(image_ننوت, img_shape_const, خطأ);

            ثابت ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> وزن_ننوت;
            إذا (خپلکاری وزن_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_غوټه_shared_ptr())) {
                وزن_ننوت = ثبتوونکی_new_غوټه<ov::op::v0::Constant>(وزن_const->get_element_type(), conv_w_shape,
                                                                       وزن_const->get_معلومات_ptr());
            } وإلا {
                خپلکاری shape_const = ثبتوونکی_new_غوټه<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                وزن_ننوت = ثبتوونکی_new_غوټه<ov::op::v1::Reshape>(w_trace, shape_const, خطأ);
            }

            إذا (وزن_ننوت.get_element_type() != image_reshaped.get_element_type()) {
                وزن_ننوت = ثبتوونکی_new_غوټه<ov::op::v0::Convert>(وزن_ننوت, image_reshaped.get_element_type());
            }

            conv_بهر = ثبتوونکی_new_غوټه<ov::op::v1::ډله_Convolution>(
                image_reshaped, وزن_ننوت, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } وإلا {
            إذا ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } وإلا إذا ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } وإلا إذا (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } وإلا إذا (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } وإلا إذا (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } وإلا إذا (kernel_spatial_ic && ov::shape_اندازه(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_اندازه(ws) / kernel_spatial_ic;
            } وإلا {
                أرجع خطأ;
            }

            ثابت ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> وزن_ننوت;
            إذا (خپلکاری وزن_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_غوټه_shared_ptr())) {
                وزن_ننوت = ثبتوونکی_new_غوټه<ov::op::v0::Constant>(وزن_const->get_element_type(), conv_w_shape,
                                                                       وزن_const->get_معلومات_ptr());
            } وإلا {
                خپلکاری shape_const = ثبتوونکی_new_غوټه<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                وزن_ننوت = ثبتوونکی_new_غوټه<ov::op::v1::Reshape>(w_trace, shape_const, خطأ);
            }

            إذا (وزن_ننوت.get_element_type() != image_ننوت.get_element_type()) {
                وزن_ننوت = ثبتوونکی_new_غوټه<ov::op::v0::Convert>(وزن_ننوت, image_ننوت.get_element_type());
            }

            conv_بهر = ثبتوونکی_new_غوټه<ov::op::v1::Convolution>(
                image_ننوت, وزن_ننوت, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr خپلکاری موخه_type = ov::element::f32;
        إذا (conv_بهر.get_element_type() != موخه_type) {
            conv_بهر = ثبتوونکی_new_غوټه<ov::op::v0::Convert>(conv_بهر, موخه_type);
        }

        std::shared_ptr<ov::op::v1::جمع_کړه> جمع_کړه_غوټه;
        ov::Output<Node> bias_ننوت;

        خپلکاری try_fuse_bias = [&](ثابت std::shared_ptr<Node> & n) {
            خپلکاری جمع_کړه_2 = ov::as_type_ptr<ov::op::v1::جمع_کړه>(n);
            إذا (!جمع_کړه_2) {
                أرجع خطأ;
            }
            لكل (size_t i = 0; i < 2; ++i) {
                إذا (ov::is_type<ov::op::v0::Constant>(جمع_کړه_2->get_ننوت_غوټه_shared_ptr(i))) {
                    bias_ننوت = جمع_کړه_2->ننوت_ارزښت(i);
                    جمع_کړه_غوټه = جمع_کړه_2;
                    أرجع ریښتیا;
                }
            }
            أرجع خطأ;
        };

        لكل (ثابت خپلکاری & مصرفوونکی : matmul_غوټه->output(0).get_موخه_inputs()) {
            خپلکاری n = مصرفوونکی.get_غوټه()->shared_from_this();
            إذا (try_fuse_bias(n)) {
                اكسر;
            }
            إذا (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                لكل (ثابت خپلکاری & بل : n->output(0).get_موخه_inputs()) {
                    إذا (try_fuse_bias(بل.get_غوټه()->shared_from_this())) {
                        اكسر;
                    }
                }
            }
            إذا (جمع_کړه_غوټه) {
                اكسر;
            }
        }

        ov::Output<Node> final_بهر = conv_بهر;
        std::shared_ptr<Node> موخه_غوټه = matmul_غوټه;

        إذا (جمع_کړه_غوټه) {
            ov::Output<Node> bias = bias_ننوت;
            إذا (bias.get_element_type() != موخه_type) {
                bias = ثبتوونکی_new_غوټه<ov::op::v0::Convert>(bias, موخه_type);
            }
            خپلکاری bias_shape = ثبتوونکی_new_غوټه<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = ثبتوونکی_new_غوټه<ov::op::v1::Reshape>(bias, bias_shape, خطأ);
            final_بهر = ثبتوونکی_new_غوټه<ov::op::v1::جمع_کړه>(conv_بهر, bias);
            موخه_غوټه = جمع_کړه_غوټه;
        }

        إذا (!is_depthwise) {
            خپلکاری perm = ثبتوونکی_new_غوټه<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                وزن_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_بهر = ثبتوونکی_new_غوټه<ov::op::v1::Transpose>(final_بهر, perm);
        }

        خپلکاری orig_shape = موخه_غوټه->get_output_partial_shape(0);
        إذا (orig_shape.is_static() && final_بهر.get_partial_shape().is_static()) {
            إذا (ov::shape_اندازه(orig_shape.منزل_shape()) != ov::shape_اندازه(final_بهر.get_shape())) {
                أرجع خطأ;
            }
        }
        إذا (orig_shape.is_static() && final_بهر.get_partial_shape() != orig_shape) {
            خپلکاری shape_const = ثبتوونکی_new_غوټه<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.منزل_shape());
            final_بهر = ثبتوونکی_new_غوټه<ov::op::v1::Reshape>(final_بهر, shape_const, خطأ);
        }

        خپلکاری orig_type = موخه_غوټه->get_output_element_type(0);
        إذا (final_بهر.get_element_type() != orig_type) {
            final_بهر = ثبتوونکی_new_غوټه<ov::op::v0::Convert>(final_بهر, orig_type);
        }

        final_بهر.get_غوټه_shared_ptr()->سېټ_friendly_name(موخه_غوټه->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_بهر.get_غوټه_shared_ptr());
        ov::replace_غوټه(موخه_غوټه, final_بهر.get_غوټه_shared_ptr());

        أرجع ریښتیا;
    };

    ثبتوونکی_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
