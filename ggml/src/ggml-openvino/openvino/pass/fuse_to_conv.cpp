#ಅನ್ತರ್ಭಾವ "fuse_to_conv.h"

#ಅನ್ತರ್ಭಾವ <openvino/core/graph_util.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/rt_info.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convolution.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/extractimagepatches.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/group_conv.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/matmul.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/pad.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/pattern.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/wrap_type.hpp>
#ಅನ್ತರ್ಭಾವ <utility>

ನಾಮಾಕಾಶ opp = ov::pass::pattern;

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ pass {

Fuse_ಗಮ್ಯ_Conv::Fuse_ಗಮ್ಯ_Conv() {
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_in0 = opp::any_ಒಳಹರಿವು();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_in1 = opp::any_ಒಳಹರಿವು();
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ callback = [=](opp::Matcher & m) {
        ಸ್ವಯಂಚಾಲಿತ matmul_ಗಂಟು = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_ಹೊಂದಿಕೆ_ಬೇರು());
        ಆದರೆ (!matmul_ಗಂಟು) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ವಯಂಚಾಲಿತ unwrap = [](ov::Output<Node> n) {
            ಆಗ (ov::is_type<ov::op::v0::Convert>(n.get_ಗಂಟು_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_ಗಂಟು_shared_ptr())) {
                n = n.get_ಗಂಟು_shared_ptr()->ಒಳಹರಿವು_ಮೌಲ್ಯ(0);
            }
            ಹಿಂತಿರುಗಿ n;
        };

        ಸ್ವಯಂಚಾಲಿತ get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ಹೊರತೆಗೆ_Image_Patches>, std::shared_ptr<ov::op::v1::Pad>> {
            ಸ್ವಯಂಚಾಲಿತ t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_ಗಂಟು_shared_ptr());
            ಸ್ವಯಂಚಾಲಿತ r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0)) : ಶೂನ್ಯಸೂಚಕ;
            ಸ್ವಯಂಚಾಲಿತ t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0)) : ಶೂನ್ಯಸೂಚಕ;
            ಸ್ವಯಂಚಾಲಿತ eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ಹೊರತೆಗೆ_Image_Patches>(t1->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0)) : ಶೂನ್ಯಸೂಚಕ;
            ಸ್ವಯಂಚಾಲಿತ pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(0)) : ಶೂನ್ಯಸೂಚಕ;
            ಹಿಂತಿರುಗಿ {eip, pad};
        };

        ತಾರ್ಕಿಕ ತೂಕ_is_in0 = ನಿಜ;
        ಸ್ವಯಂಚಾಲಿತ [eip, pad] = get_im2col(matmul_ಗಂಟು->ಒಳಹರಿವು_ಮೌಲ್ಯ(1));
        ov::Output<Node> w_trace = matmul_ಗಂಟು->ಒಳಹರಿವು_ಮೌಲ್ಯ(0);
        ಆದರೆ (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_ಗಂಟು->ಒಳಹರಿವು_ಮೌಲ್ಯ(0));
            w_trace = matmul_ಗಂಟು->ಒಳಹರಿವು_ಮೌಲ್ಯ(1);
            ತೂಕ_is_in0 = ಸುಳ್ಳು;
            ಆದರೆ (!pad) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }

        ಸ್ವಯಂಚಾಲಿತ pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(1));
        ಸ್ವಯಂಚಾಲಿತ pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(2));
        ಆದರೆ (!pb_const || !pe_const) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ pb = pb_const->cast_ಸದಿಶ<std::ptrdiff_t>();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ pe = pe_const->cast_ಸದಿಶ<std::ptrdiff_t>();
        ಆದರೆ (pb.size() < 4 || pe.size() < 4) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ವಯಂಚಾಲಿತ image_ಒಳಹರಿವು = pad->ಒಳಹರಿವು_ಮೌಲ್ಯ(0);
        ಸ್ವಯಂಚಾಲಿತ image_shape = image_ಒಳಹರಿವು.get_partial_shape();
        ಆದರೆ (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ size_t IC = static_cast<size_t>(image_shape[1].get_ಉದ್ದಳತೆ());

        w_trace = unwrap(w_trace);
        ಸ್ವಯಂಚಾಲಿತ ತೂಕ_pshape = w_trace.get_partial_shape();
        ಆದರೆ (!ತೂಕ_pshape.is_static()) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ws = ತೂಕ_pshape.ಗಮ್ಯ_shape();
        ಸ್ಥಿರಾಂಕ size_t KH = eip->get_sizes()[0];
        ಸ್ಥಿರಾಂಕ size_t KW = eip->get_sizes()[1];
        ಸ್ಥಿರಾಂಕ size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        ಆದರೆ (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            ಆದರೆ ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_depthwise = groups > 1 && (image_shape[0].get_ಉದ್ದಳತೆ() % groups == 0);

        ov::Output<Node> conv_ಹೊರಹೋಗು;
        size_t OC = 0;

        ಆದರೆ (is_depthwise) {
            ಸ್ಥಿರಾಂಕ size_t N = static_cast<size_t>(image_shape[0].get_ಉದ್ದಳತೆ()) / groups;
            ಸ್ಥಿರಾಂಕ size_t IH = static_cast<size_t>(image_shape[2].get_ಉದ್ದಳತೆ());
            ಸ್ಥಿರಾಂಕ size_t IW = static_cast<size_t>(image_shape[3].get_ಉದ್ದಳತೆ());
            OC = groups;

            ಸ್ವಯಂಚಾಲಿತ img_shape_const = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v1::Reshape>(image_ಒಳಹರಿವು, img_shape_const, ಸುಳ್ಳು);

            ಸ್ಥಿರಾಂಕ ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> ತೂಕ_ಒಳಹರಿವು;
            ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ ತೂಕ_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ಗಂಟು_shared_ptr())) {
                ತೂಕ_ಒಳಹರಿವು = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Constant>(ತೂಕ_const->get_element_type(), conv_w_shape,
                                                                       ತೂಕ_const->get_ದತ್ತಾಂಶ_ptr());
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ವಯಂಚಾಲಿತ shape_const = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                ತೂಕ_ಒಳಹರಿವು = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v1::Reshape>(w_trace, shape_const, ಸುಳ್ಳು);
            }

            ಆದರೆ (ತೂಕ_ಒಳಹರಿವು.get_element_type() != image_reshaped.get_element_type()) {
                ತೂಕ_ಒಳಹರಿವು = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Convert>(ತೂಕ_ಒಳಹರಿವು, image_reshaped.get_element_type());
            }

            conv_ಹೊರಹೋಗು = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v1::ಗುಂಪು_Convolution>(
                image_reshaped, ತೂಕ_ಒಳಹರಿವು, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kernel_spatial_ic && ov::shape_ಗಾತ್ರ(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_ಗಾತ್ರ(ws) / kernel_spatial_ic;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }

            ಸ್ಥಿರಾಂಕ ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> ತೂಕ_ಒಳಹರಿವು;
            ಆದರೆ (ಸ್ವಯಂಚಾಲಿತ ತೂಕ_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ಗಂಟು_shared_ptr())) {
                ತೂಕ_ಒಳಹರಿವು = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Constant>(ತೂಕ_const->get_element_type(), conv_w_shape,
                                                                       ತೂಕ_const->get_ದತ್ತಾಂಶ_ptr());
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ವಯಂಚಾಲಿತ shape_const = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                ತೂಕ_ಒಳಹರಿವು = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v1::Reshape>(w_trace, shape_const, ಸುಳ್ಳು);
            }

            ಆದರೆ (ತೂಕ_ಒಳಹರಿವು.get_element_type() != image_ಒಳಹರಿವು.get_element_type()) {
                ತೂಕ_ಒಳಹರಿವು = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Convert>(ತೂಕ_ಒಳಹರಿವು, image_ಒಳಹರಿವು.get_element_type());
            }

            conv_ಹೊರಹೋಗು = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v1::Convolution>(
                image_ಒಳಹರಿವು, ತೂಕ_ಒಳಹರಿವು, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr ಸ್ವಯಂಚಾಲಿತ ಗುರಿ_type = ov::element::f32;
        ಆದರೆ (conv_ಹೊರಹೋಗು.get_element_type() != ಗುರಿ_type) {
            conv_ಹೊರಹೋಗು = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Convert>(conv_ಹೊರಹೋಗು, ಗುರಿ_type);
        }

        std::shared_ptr<ov::op::v1::ಕೂಡಿಸು> ಕೂಡಿಸು_ಗಂಟು;
        ov::Output<Node> bias_ಒಳಹರಿವು;

        ಸ್ವಯಂಚಾಲಿತ try_fuse_bias = [&](ಸ್ಥಿರಾಂಕ std::shared_ptr<Node> & n) {
            ಸ್ವಯಂಚಾಲಿತ ಕೂಡಿಸು_2 = ov::as_type_ptr<ov::op::v1::ಕೂಡಿಸು>(n);
            ಆದರೆ (!ಕೂಡಿಸು_2) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಗಾಗಿ (size_t i = 0; i < 2; ++i) {
                ಆದರೆ (ov::is_type<ov::op::v0::Constant>(ಕೂಡಿಸು_2->get_ಒಳಹರಿವು_ಗಂಟು_shared_ptr(i))) {
                    bias_ಒಳಹರಿವು = ಕೂಡಿಸು_2->ಒಳಹರಿವು_ಮೌಲ್ಯ(i);
                    ಕೂಡಿಸು_ಗಂಟು = ಕೂಡಿಸು_2;
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }
            }
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        };

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಗ್ರಾಹಕ : matmul_ಗಂಟು->output(0).get_ಗುರಿ_inputs()) {
            ಸ್ವಯಂಚಾಲಿತ n = ಗ್ರಾಹಕ.get_ಗಂಟು()->shared_from_this();
            ಆದರೆ (try_fuse_bias(n)) {
                ನಿಲ್ಲಿಸು;
            }
            ಆದರೆ (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಮುಂದಿನ : n->output(0).get_ಗುರಿ_inputs()) {
                    ಆದರೆ (try_fuse_bias(ಮುಂದಿನ.get_ಗಂಟು()->shared_from_this())) {
                        ನಿಲ್ಲಿಸು;
                    }
                }
            }
            ಆದರೆ (ಕೂಡಿಸು_ಗಂಟು) {
                ನಿಲ್ಲಿಸು;
            }
        }

        ov::Output<Node> final_ಹೊರಹೋಗು = conv_ಹೊರಹೋಗು;
        std::shared_ptr<Node> ಗುರಿ_ಗಂಟು = matmul_ಗಂಟು;

        ಆದರೆ (ಕೂಡಿಸು_ಗಂಟು) {
            ov::Output<Node> bias = bias_ಒಳಹರಿವು;
            ಆದರೆ (bias.get_element_type() != ಗುರಿ_type) {
                bias = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Convert>(bias, ಗುರಿ_type);
            }
            ಸ್ವಯಂಚಾಲಿತ bias_shape = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v1::Reshape>(bias, bias_shape, ಸುಳ್ಳು);
            final_ಹೊರಹೋಗು = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v1::ಕೂಡಿಸು>(conv_ಹೊರಹೋಗು, bias);
            ಗುರಿ_ಗಂಟು = ಕೂಡಿಸು_ಗಂಟು;
        }

        ಆದರೆ (!is_depthwise) {
            ಸ್ವಯಂಚಾಲಿತ perm = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                ತೂಕ_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_ಹೊರಹೋಗು = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v1::Transpose>(final_ಹೊರಹೋಗು, perm);
        }

        ಸ್ವಯಂಚಾಲಿತ orig_shape = ಗುರಿ_ಗಂಟು->get_output_partial_shape(0);
        ಆದರೆ (orig_shape.is_static() && final_ಹೊರಹೋಗು.get_partial_shape().is_static()) {
            ಆದರೆ (ov::shape_ಗಾತ್ರ(orig_shape.ಗಮ್ಯ_shape()) != ov::shape_ಗಾತ್ರ(final_ಹೊರಹೋಗು.get_shape())) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
        ಆದರೆ (orig_shape.is_static() && final_ಹೊರಹೋಗು.get_partial_shape() != orig_shape) {
            ಸ್ವಯಂಚಾಲಿತ shape_const = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.ಗಮ್ಯ_shape());
            final_ಹೊರಹೋಗು = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v1::Reshape>(final_ಹೊರಹೋಗು, shape_const, ಸುಳ್ಳು);
        }

        ಸ್ವಯಂಚಾಲಿತ orig_type = ಗುರಿ_ಗಂಟು->get_output_element_type(0);
        ಆದರೆ (final_ಹೊರಹೋಗು.get_element_type() != orig_type) {
            final_ಹೊರಹೋಗು = ರಿಜಿಸ್ಟರ್_new_ಗಂಟು<ov::op::v0::Convert>(final_ಹೊರಹೋಗು, orig_type);
        }

        final_ಹೊರಹೋಗು.get_ಗಂಟು_shared_ptr()->ಗಣ_friendly_name(ಗುರಿ_ಗಂಟು->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_ಹೊರಹೋಗು.get_ಗಂಟು_shared_ptr());
        ov::replace_ಗಂಟು(ಗುರಿ_ಗಂಟು, final_ಹೊರಹೋಗು.get_ಗಂಟು_shared_ptr());

        ಹಿಂತಿರುಗಿ ನಿಜ;
    };

    ರಿಜಿಸ್ಟರ್_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
