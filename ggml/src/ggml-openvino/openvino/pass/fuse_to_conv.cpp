#ਅਨ੍ਤਰ੍ਭਾਵ "fuse_to_conv.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/graph_util.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/rt_info.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convolution.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/extractimagepatches.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/group_conv.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/matmul.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/pad.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/transpose.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/pattern/op/pattern.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/pass/pattern/op/wrap_type.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <utility>

ਨਾਮਾਕਾਸ਼ opp = ov::pass::pattern;

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ pass {

Fuse_ਮੰਜ਼ਿਲ_Conv::Fuse_ਮੰਜ਼ਿਲ_Conv() {
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_in0 = opp::any_ਨਿਵੇਸ਼();
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_in1 = opp::any_ਨਿਵੇਸ਼();
    ਅਚਲ ਸਵੈਚਾਲਿਤ m_matmul = opp::wrap_type<ov::op::v0::MatMul>({m_in0, m_in1});

    ਅਚਲ ਸਵੈਚਾਲਿਤ callback = [=](opp::Matcher & m) {
        ਸਵੈਚਾਲਿਤ matmul_ਗੰਢ = ov::as_type_ptr<ov::op::v0::MatMul>(m.get_ਮੇਲ_ਜੜ੍ਹ());
        ਜੇ (!matmul_ਗੰਢ) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਸਵੈਚਾਲਿਤ unwrap = [](ov::Output<Node> n) {
            ਜਦੋਂ (ov::is_type<ov::op::v0::Convert>(n.get_ਗੰਢ_shared_ptr()) ||
                   ov::is_type<ov::op::v1::Reshape>(n.get_ਗੰਢ_shared_ptr())) {
                n = n.get_ਗੰਢ_shared_ptr()->ਨਿਵੇਸ਼_ਮੁੱਲ(0);
            }
            ਵਾਪਸ n;
        };

        ਸਵੈਚਾਲਿਤ get_im2col = [&](ov::Output<Node> trace)
            -> std::pair<std::shared_ptr<ov::op::v3::ਕੱਢੋ_Image_Patches>, std::shared_ptr<ov::op::v1::Pad>> {
            ਸਵੈਚਾਲਿਤ t2 = ov::as_type_ptr<ov::op::v1::Transpose>(unwrap(std::move(trace)).get_ਗੰਢ_shared_ptr());
            ਸਵੈਚਾਲਿਤ r1 = t2 ? ov::as_type_ptr<ov::op::v1::Reshape>(t2->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0)) : ਖਾਲੀ_ਸੂਚਕ;
            ਸਵੈਚਾਲਿਤ t1 = r1 ? ov::as_type_ptr<ov::op::v1::Transpose>(r1->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0)) : ਖਾਲੀ_ਸੂਚਕ;
            ਸਵੈਚਾਲਿਤ eip =
                t1 ? ov::as_type_ptr<ov::op::v3::ਕੱਢੋ_Image_Patches>(t1->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0)) : ਖਾਲੀ_ਸੂਚਕ;
            ਸਵੈਚਾਲਿਤ pad = eip ? ov::as_type_ptr<ov::op::v1::Pad>(eip->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(0)) : ਖਾਲੀ_ਸੂਚਕ;
            ਵਾਪਸ {eip, pad};
        };

        ਤਾਰਕਿਕ ਭਾਰ_is_in0 = ਸੱਚ;
        ਸਵੈਚਾਲਿਤ [eip, pad] = get_im2col(matmul_ਗੰਢ->ਨਿਵੇਸ਼_ਮੁੱਲ(1));
        ov::Output<Node> w_trace = matmul_ਗੰਢ->ਨਿਵੇਸ਼_ਮੁੱਲ(0);
        ਜੇ (!pad) {
            std::tie(eip, pad) = get_im2col(matmul_ਗੰਢ->ਨਿਵੇਸ਼_ਮੁੱਲ(0));
            w_trace = matmul_ਗੰਢ->ਨਿਵੇਸ਼_ਮੁੱਲ(1);
            ਭਾਰ_is_in0 = ਝੂਠ;
            ਜੇ (!pad) {
                ਵਾਪਸ ਝੂਠ;
            }
        }

        ਸਵੈਚਾਲਿਤ pb_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(1));
        ਸਵੈਚਾਲਿਤ pe_const = ov::as_type_ptr<ov::op::v0::Constant>(pad->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(2));
        ਜੇ (!pb_const || !pe_const) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਸਵੈਚਾਲਿਤ pb = pb_const->cast_ਸਦਿਸ਼<std::ptrdiff_t>();
        ਅਚਲ ਸਵੈਚਾਲਿਤ pe = pe_const->cast_ਸਦਿਸ਼<std::ptrdiff_t>();
        ਜੇ (pb.size() < 4 || pe.size() < 4) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਸਵੈਚਾਲਿਤ image_ਨਿਵੇਸ਼ = pad->ਨਿਵੇਸ਼_ਮੁੱਲ(0);
        ਸਵੈਚਾਲਿਤ image_shape = image_ਨਿਵੇਸ਼.get_partial_shape();
        ਜੇ (image_shape.rank() != 4 || image_shape[1].is_dynamic()) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਅਚਲ size_t IC = static_cast<size_t>(image_shape[1].get_ਲੰਮਾਈ());

        w_trace = unwrap(w_trace);
        ਸਵੈਚਾਲਿਤ ਭਾਰ_pshape = w_trace.get_partial_shape();
        ਜੇ (!ਭਾਰ_pshape.is_static()) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ਸਵੈਚਾਲਿਤ & ws = ਭਾਰ_pshape.ਮੰਜ਼ਿਲ_shape();
        ਅਚਲ size_t KH = eip->get_sizes()[0];
        ਅਚਲ size_t KW = eip->get_sizes()[1];
        ਅਚਲ size_t kernel_spatial_ic = IC * KH * KW;

        size_t groups = 0;
        ਜੇ (IC == 1 && image_shape[0].is_static() && image_shape[2].is_static() && image_shape[3].is_static()) {
            ਜੇ ((ws.size() == 4 && ws[1] == 1 && ws[2] == KH && ws[3] == KW && ws[0] > 1) ||
                (ws.size() == 2 && ws[1] == KH * KW && ws[0] > 1) ||
                (ws.size() == 3 && ws[1] == 1 && ws[2] == KH * KW && ws[0] > 1)) {
                groups = ws[0];
            } ਨਹੀਂਤਾਂ ਜੇ (ws.size() == 4 && ws[0] == 1 && ws[2] == 1 && ws[3] == KH * KW && ws[1] > 1) {
                groups = ws[1];
            }
        }

        ਅਚਲ ਤਾਰਕਿਕ is_depthwise = groups > 1 && (image_shape[0].get_ਲੰਮਾਈ() % groups == 0);

        ov::Output<Node> conv_ਬਾਹਰਲਾ;
        size_t OC = 0;

        ਜੇ (is_depthwise) {
            ਅਚਲ size_t N = static_cast<size_t>(image_shape[0].get_ਲੰਮਾਈ()) / groups;
            ਅਚਲ size_t IH = static_cast<size_t>(image_shape[2].get_ਲੰਮਾਈ());
            ਅਚਲ size_t IW = static_cast<size_t>(image_shape[3].get_ਲੰਮਾਈ());
            OC = groups;

            ਸਵੈਚਾਲਿਤ img_shape_const = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                std::vector<int64_t>{static_cast<int64_t>(N), static_cast<int64_t>(groups), static_cast<int64_t>(IH),
                                     static_cast<int64_t>(IW)});
            ov::Output<Node> image_reshaped =
                ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v1::Reshape>(image_ਨਿਵੇਸ਼, img_shape_const, ਝੂਠ);

            ਅਚਲ ov::Shape conv_w_shape = {groups, 1, 1, KH, KW};
            ov::Output<Node> ਭਾਰ_ਨਿਵੇਸ਼;
            ਜੇ (ਸਵੈਚਾਲਿਤ ਭਾਰ_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ਗੰਢ_shared_ptr())) {
                ਭਾਰ_ਨਿਵੇਸ਼ = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Constant>(ਭਾਰ_const->get_element_type(), conv_w_shape,
                                                                       ਭਾਰ_const->get_ਅੰਕੜੇ_ptr());
            } ਨਹੀਂਤਾਂ {
                ਸਵੈਚਾਲਿਤ shape_const = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{5},
                    std::vector<int64_t>{static_cast<int64_t>(groups), 1, 1, static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                ਭਾਰ_ਨਿਵੇਸ਼ = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v1::Reshape>(w_trace, shape_const, ਝੂਠ);
            }

            ਜੇ (ਭਾਰ_ਨਿਵੇਸ਼.get_element_type() != image_reshaped.get_element_type()) {
                ਭਾਰ_ਨਿਵੇਸ਼ = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Convert>(ਭਾਰ_ਨਿਵੇਸ਼, image_reshaped.get_element_type());
            }

            conv_ਬਾਹਰਲਾ = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v1::ਸਮੂਹ_Convolution>(
                image_reshaped, ਭਾਰ_ਨਿਵੇਸ਼, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        } ਨਹੀਂਤਾਂ {
            ਜੇ ((ws.size() == 4 && ws[1] == IC && ws[2] == KH && ws[3] == KW) ||
                (ws.size() == 3 && ws[1] == IC && ws[2] == KW) || (ws.size() == 2 && ws[1] == kernel_spatial_ic)) {
                OC = ws[0];
            } ਨਹੀਂਤਾਂ ਜੇ ((ws.size() == 4 && ws[0] == 1 && ws[2] == IC && ws[3] == KW) ||
                       (ws.size() == 2 && ws[0] == kernel_spatial_ic)) {
                OC = ws[1];
            } ਨਹੀਂਤਾਂ ਜੇ (ws.size() == 3 && ws[0] == 1 && ws[1] == IC && ws[2] == KW) {
                OC = 1;
            } ਨਹੀਂਤਾਂ ਜੇ (ws.size() == 4 && ws[3] == kernel_spatial_ic) {
                OC = ws[2];
            } ਨਹੀਂਤਾਂ ਜੇ (ws.size() == 4 && ws[2] == kernel_spatial_ic) {
                OC = ws[3];
            } ਨਹੀਂਤਾਂ ਜੇ (kernel_spatial_ic && ov::shape_ਮਾਪ(ws) % kernel_spatial_ic == 0) {
                OC = ov::shape_ਮਾਪ(ws) / kernel_spatial_ic;
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ ਝੂਠ;
            }

            ਅਚਲ ov::Shape conv_w_shape = {OC, IC, KH, KW};

            ov::Output<Node> ਭਾਰ_ਨਿਵੇਸ਼;
            ਜੇ (ਸਵੈਚਾਲਿਤ ਭਾਰ_const = ov::as_type_ptr<ov::op::v0::Constant>(w_trace.get_ਗੰਢ_shared_ptr())) {
                ਭਾਰ_ਨਿਵੇਸ਼ = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Constant>(ਭਾਰ_const->get_element_type(), conv_w_shape,
                                                                       ਭਾਰ_const->get_ਅੰਕੜੇ_ptr());
            } ਨਹੀਂਤਾਂ {
                ਸਵੈਚਾਲਿਤ shape_const = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{4},
                    std::vector<int64_t>{static_cast<int64_t>(OC), static_cast<int64_t>(IC), static_cast<int64_t>(KH),
                                         static_cast<int64_t>(KW)});
                ਭਾਰ_ਨਿਵੇਸ਼ = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v1::Reshape>(w_trace, shape_const, ਝੂਠ);
            }

            ਜੇ (ਭਾਰ_ਨਿਵੇਸ਼.get_element_type() != image_ਨਿਵੇਸ਼.get_element_type()) {
                ਭਾਰ_ਨਿਵੇਸ਼ = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Convert>(ਭਾਰ_ਨਿਵੇਸ਼, image_ਨਿਵੇਸ਼.get_element_type());
            }

            conv_ਬਾਹਰਲਾ = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v1::Convolution>(
                image_ਨਿਵੇਸ਼, ਭਾਰ_ਨਿਵੇਸ਼, eip->get_strides(), ov::CoordinateDiff{pb[2], pb[3]},
                ov::CoordinateDiff{pe[2], pe[3]}, ov::Strides{eip->get_rates()[0], eip->get_rates()[1]},
                ov::op::PadType::EXPLICIT);
        }

        constexpr ਸਵੈਚਾਲਿਤ ਨਿਸ਼ਾਨਾ_type = ov::element::f32;
        ਜੇ (conv_ਬਾਹਰਲਾ.get_element_type() != ਨਿਸ਼ਾਨਾ_type) {
            conv_ਬਾਹਰਲਾ = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Convert>(conv_ਬਾਹਰਲਾ, ਨਿਸ਼ਾਨਾ_type);
        }

        std::shared_ptr<ov::op::v1::ਜੋੜੋ> ਜੋੜੋ_ਗੰਢ;
        ov::Output<Node> bias_ਨਿਵੇਸ਼;

        ਸਵੈਚਾਲਿਤ try_fuse_bias = [&](ਅਚਲ std::shared_ptr<Node> & n) {
            ਸਵੈਚਾਲਿਤ ਜੋੜੋ_2 = ov::as_type_ptr<ov::op::v1::ਜੋੜੋ>(n);
            ਜੇ (!ਜੋੜੋ_2) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਲਈ (size_t i = 0; i < 2; ++i) {
                ਜੇ (ov::is_type<ov::op::v0::Constant>(ਜੋੜੋ_2->get_ਨਿਵੇਸ਼_ਗੰਢ_shared_ptr(i))) {
                    bias_ਨਿਵੇਸ਼ = ਜੋੜੋ_2->ਨਿਵੇਸ਼_ਮੁੱਲ(i);
                    ਜੋੜੋ_ਗੰਢ = ਜੋੜੋ_2;
                    ਵਾਪਸ ਸੱਚ;
                }
            }
            ਵਾਪਸ ਝੂਠ;
        };

        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਖਪਤਕਾਰ : matmul_ਗੰਢ->output(0).get_ਨਿਸ਼ਾਨਾ_inputs()) {
            ਸਵੈਚਾਲਿਤ n = ਖਪਤਕਾਰ.get_ਗੰਢ()->shared_from_this();
            ਜੇ (try_fuse_bias(n)) {
                ਰੋਕੋ;
            }
            ਜੇ (ov::is_type<ov::op::v0::Convert>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
                ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਅਗਲਾ : n->output(0).get_ਨਿਸ਼ਾਨਾ_inputs()) {
                    ਜੇ (try_fuse_bias(ਅਗਲਾ.get_ਗੰਢ()->shared_from_this())) {
                        ਰੋਕੋ;
                    }
                }
            }
            ਜੇ (ਜੋੜੋ_ਗੰਢ) {
                ਰੋਕੋ;
            }
        }

        ov::Output<Node> final_ਬਾਹਰਲਾ = conv_ਬਾਹਰਲਾ;
        std::shared_ptr<Node> ਨਿਸ਼ਾਨਾ_ਗੰਢ = matmul_ਗੰਢ;

        ਜੇ (ਜੋੜੋ_ਗੰਢ) {
            ov::Output<Node> bias = bias_ਨਿਵੇਸ਼;
            ਜੇ (bias.get_element_type() != ਨਿਸ਼ਾਨਾ_type) {
                bias = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Convert>(bias, ਨਿਸ਼ਾਨਾ_type);
            }
            ਸਵੈਚਾਲਿਤ bias_shape = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, static_cast<int64_t>(OC), 1, 1});
            bias = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v1::Reshape>(bias, bias_shape, ਝੂਠ);
            final_ਬਾਹਰਲਾ = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v1::ਜੋੜੋ>(conv_ਬਾਹਰਲਾ, bias);
            ਨਿਸ਼ਾਨਾ_ਗੰਢ = ਜੋੜੋ_ਗੰਢ;
        }

        ਜੇ (!is_depthwise) {
            ਸਵੈਚਾਲਿਤ perm = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{4},
                ਭਾਰ_is_in0 ? std::vector<int64_t>{1, 0, 2, 3} : std::vector<int64_t>{0, 2, 3, 1});
            final_ਬਾਹਰਲਾ = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v1::Transpose>(final_ਬਾਹਰਲਾ, perm);
        }

        ਸਵੈਚਾਲਿਤ orig_shape = ਨਿਸ਼ਾਨਾ_ਗੰਢ->get_output_partial_shape(0);
        ਜੇ (orig_shape.is_static() && final_ਬਾਹਰਲਾ.get_partial_shape().is_static()) {
            ਜੇ (ov::shape_ਮਾਪ(orig_shape.ਮੰਜ਼ਿਲ_shape()) != ov::shape_ਮਾਪ(final_ਬਾਹਰਲਾ.get_shape())) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
        ਜੇ (orig_shape.is_static() && final_ਬਾਹਰਲਾ.get_partial_shape() != orig_shape) {
            ਸਵੈਚਾਲਿਤ shape_const = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()},
                                                                       orig_shape.ਮੰਜ਼ਿਲ_shape());
            final_ਬਾਹਰਲਾ = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v1::Reshape>(final_ਬਾਹਰਲਾ, shape_const, ਝੂਠ);
        }

        ਸਵੈਚਾਲਿਤ orig_type = ਨਿਸ਼ਾਨਾ_ਗੰਢ->get_output_element_type(0);
        ਜੇ (final_ਬਾਹਰਲਾ.get_element_type() != orig_type) {
            final_ਬਾਹਰਲਾ = ਰਜਿਸਟਰ_new_ਗੰਢ<ov::op::v0::Convert>(final_ਬਾਹਰਲਾ, orig_type);
        }

        final_ਬਾਹਰਲਾ.get_ਗੰਢ_shared_ptr()->ਸੈੱਟ_friendly_name(ਨਿਸ਼ਾਨਾ_ਗੰਢ->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), final_ਬਾਹਰਲਾ.get_ਗੰਢ_shared_ptr());
        ov::replace_ਗੰਢ(ਨਿਸ਼ਾਨਾ_ਗੰਢ, final_ਬਾਹਰਲਾ.get_ਗੰਢ_shared_ptr());

        ਵਾਪਸ ਸੱਚ;
    };

    ਰਜਿਸਟਰ_matcher(std::make_shared<opp::Matcher>(m_matmul, "ov::frontend::ggml::pass::FuseToConv"), callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
