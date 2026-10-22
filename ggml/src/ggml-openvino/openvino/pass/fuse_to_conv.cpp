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

Fuse_مقصد_Conv::Fuse_مقصد_Conv() {
    ثابت خودکار m_in0 = opp::any_درون_داد();
    ثابت خودکار m_in1 = opp::any_درون_داد();
    ثابت خودکار m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    ثابت خودکار callback = [=](opp::Matcher & m) {
        خودکار matmul_گره = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_تطابق_ریشه());
        إذا (!matmul_گره) {
            أرجع خطأ;
        }

        خودکار unwrap = [](ov::Output<Node> n) {
            طالما (ov::is_type<ov::op::v0::Convert>(n.get_گره_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_گره_shared_ptr())) {
                n = n.get_گره_shared_ptr()->درون_داد_مقدار(0);
            }
            أرجع n;
        };

        خودکار get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::استخراج_Image_Patches>, std::shared_ptr<ov::op::v1::Pad>> {
            خودکار t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_گره_shared_ptr());
            خودکار r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_درون_داد_گره_shared_ptr(0)) : اشاره_تهی;
            خودکار t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_درون_داد_گره_shared_ptr(0)) : اشاره_تهی;
            خودکار eip =
                t1 ? ov::as_type_ptr<ov::op::v3::استخراج_Image_Patches>(t1->get_درون_داد_گره_shared_ptr(0)) : اشاره_تهی;
            خودکار pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_درون_داد_گره_shared_ptr(0)) : اشاره_تهی;
            أرجع {eip, pad};
        };

        منطقي وزن_is_in0 = درست;
        خودکار [eip, pad] = get_im2col(matmul_گره->درون_داد_مقدار(1));
        ov::Output<Node> w_trace = matmul_گره->درون_داد_مقدار(0);
        إذا (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_گره->درون_داد_مقدار(0));
            w_trace = matmul_گره->درون_داد_مقدار(1);
            وزن_is_in0 = خطأ;
            إذا (!pad) {
                أرجع خطأ;
            }
        }

        خودکار pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_درون_داد_گره_shared_ptr(1));
        خودکار pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_درون_داد_گره_shared_ptr(2));
        إذا (!pb_const || !pe_const) {
            أرجع خطأ;
        }

        ثابت خودکار pb = pb_const->cast_وکتور<std::ptrdiff_t>();
        ثابت خودکار pe = pe_const->cast_وکتور<std::ptrdiff_t>();
        إذا (pb.size() < 4 || pe.size() < 4) {
            أرجع خطأ;
        }

        خودکار image_درون_داد = pad->درون_داد_مقدار(0);
        خودکار image_shape = image_درون_داد.get_partial_shape();
        إذا (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            أرجع خطأ;
        }
        ثابت size_t IC = static_cast<size_t>(image_shape[1].get_درازی());

        w_trace = unwrap(w_trace);
        خودکار وزن_pshape = w_trace.get_partial_shape();
        إذا (!وزن_pshape.is_static()) {
            أرجع خطأ;
        }

        ثابت خودکار & ws = وزن_pshape.مقصد_shape();
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

        ثابت منطقي is_depthwise = groups > 1 && (image_shape[0].get_درازی() % groups == 0);

        ov::Output<Node> conv_بیرون;
        size_t OC = 0;

        إذا (is_depthwise) {
            ثابت size_t N = static_cast<size_t>(image_shape[0].get_درازی()) / groups;
            ثابت size_t IH = static_cast<size_t>(image_shape[2].get_درازی());
            ثابت size_t IW = static_cast<size_t>(image_shape[3].get_درازی());
            OC = groups;

            خودکار img_shape_const = ثبات_new_گره<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                ثبات_new_گره<ov::op::v1::Reshape>(image_درون_داد, img_shape_const, خطأ);

            ثابت ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> وزن_درون_داد;
            إذا (خودکار وزن_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_گره_shared_ptr())) {
                وزن_درون_داد = ثبات_new_گره<ov::op::v0::Constant>(وزن_const->get_element_type(), conv_w_shape,
                                                                       وزن_const->get_داده_ptr());
            } وإلا {
                خودکار shape_const = ثبات_new_گره<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                وزن_درون_داد = ثبات_new_گره<ov::op::v1::Reshape>(w_trace, shape_const, خطأ);
            }

            إذا (وزن_درون_داد.get_element_type() != image_reshaped.get_element_type()) {
                وزن_درون_داد = ثبات_new_گره<ov::op::v0::Convert>(وزن_درون_داد, image_reshaped.get_element_type());
            }

            conv_بیرون = ثبات_new_گره<ov::op::v1::گروه_Convolution>(
                image_reshaped, وزن_درون_داد, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
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

            ov::Output<Node> وزن_درون_داد;
            إذا (خودکار وزن_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_گره_shared_ptr())) {
                وزن_درون_داد = ثبات_new_گره<ov::op::v0::Constant>(وزن_const->get_element_type(), conv_w_shape,
                                                                       وزن_const->get_داده_ptr());
            } وإلا {
                خودکار shape_const = ثبات_new_گره<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                وزن_درون_داد = ثبات_new_گره<ov::op::v1::Reshape>(w_trace, shape_const, خطأ);
            }

            إذا (وزن_درون_داد.get_element_type() != image_درون_داد.get_element_type()) {
                وزن_درون_داد = ثبات_new_گره<ov::op::v0::Convert>(وزن_درون_داد, image_درون_داد.get_element_type());
            }

            conv_بیرون = ثبات_new_گره<ov::op::v1::Convolution>(
                image_درون_داد, وزن_درون_داد, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr خودکار هدف_type = ov::element::f32;
        إذا (conv_بیرون.get_element_type() != هدف_type) {
            conv_بیرون = ثبات_new_گره<ov::op::v0::Convert>(conv_بیرون, هدف_type);
        }

        std::shared_ptr<ov::op::v1::جمع_کن> جمع_کن_گره;
        ov::Output<Node> bias_درون_داد;

        خودکار try_fuse_bias = [&](ثابت std::shared_ptr<Node> & n) {
            خودکار جمع_کن_2 = ov::as_type_ptr<ov::op::v1::جمع_کن>(n);
            إذا (!جمع_کن_2) {
                أرجع خطأ;
            }
            لكل (size_t i = 0; i < 2; ++i) {
                إذا (ov::is_type<ov::op::v0::Constant>(جمع_کن_2->get_درون_داد_گره_shared_ptr(i))) {
                    bias_درون_داد = جمع_کن_2->درون_داد_مقدار(i);
                    جمع_کن_گره = جمع_کن_2;
                    أرجع درست;
                }
            }
            أرجع خطأ;
        };

        لكل (ثابت خودکار & مصرفکننده : matmul_گره->output(0).get_هدف_inputs()) {
            خودکار n = مصرفکننده.get_گره()->shared_from_this();
            إذا (try_fuse_bias(n)) {
                اكسر;
            }
            إذا (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                لكل (ثابت خودکار & بعدی : n->output(0).get_هدف_inputs()) {
                    إذا (try_fuse_bias(بعدی.get_گره()->shared_from_this())) {
                        اكسر;
                    }
                }
            }
            إذا (جمع_کن_گره) {
                اكسر;
            }
        }

        ov::Output<Node> final_بیرون = conv_بیرون;
        std::shared_ptr<Node> هدف_گره = matmul_گره;

        إذا (جمع_کن_گره) {
            ov::Output<Node> bias = bias_درون_داد;
            إذا (bias.get_element_type() != هدف_type) {
                bias = ثبات_new_گره<ov::op::v0::Convert>(bias, هدف_type);
            }
            خودکار bias_shape = ثبات_new_گره<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = ثبات_new_گره<ov::op::v1::Reshape>(bias, bias_shape, خطأ);
            final_بیرون = ثبات_new_گره<ov::op::v1::جمع_کن>(conv_بیرون, bias);
            هدف_گره = جمع_کن_گره;
        }

        إذا (!is_depthwise) {
            خودکار perm = ثبات_new_گره<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                وزن_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_بیرون = ثبات_new_گره<ov::op::v1::Transpose>(final_بیرون, perm);
        }

        خودکار orig_shape = هدف_گره->get_output_partial_shape(0);
        إذا (orig_shape.is_static() && final_بیرون.get_partial_shape().is_static()) {
            إذا (ov::shape_اندازه(orig_shape.مقصد_shape()) != ov::shape_اندازه(final_بیرون.get_shape())) {
                أرجع خطأ;
            }
        }
        إذا (orig_shape.is_static() && final_بیرون.get_partial_shape() != orig_shape) {
            خودکار shape_const = ثبات_new_گره<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.مقصد_shape());
            final_بیرون = ثبات_new_گره<ov::op::v1::Reshape>(final_بیرون, shape_const, خطأ);
        }

        خودکار orig_type = هدف_گره->get_output_element_type(0);
        إذا (final_بیرون.get_element_type() != orig_type) {
            final_بیرون = ثبات_new_گره<ov::op::v0::Convert>(final_بیرون, orig_type);
        }

        final_بیرون.get_گره_shared_ptr()->مجموعه_friendly_name(هدف_گره->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_بیرون.get_گره_shared_ptr());
        ov::replace_گره(هدف_گره, final_بیرون.get_گره_shared_ptr());

        أرجع درست;
    };

    ثبات_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
