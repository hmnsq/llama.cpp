#అన్తర్భావ "utils.h"

#అన్తర్భావ <cmath>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <ctime>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/clamp.hpp>
#అన్తర్భావ <openvino/op/convert.hpp>
#అన్తర్భావ <openvino/op/cos.hpp>
#అన్తర్భావ <openvino/op/divide.hpp>
#అన్తర్భావ <openvino/op/gather.hpp>
#అన్తర్భావ <openvino/op/maximum.hpp>
#అన్తర్భావ <openvino/op/multiply.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/shape_of.hpp>
#అన్తర్భావ <openvino/op/sin.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/split.hpp>
#అన్తర్భావ <openvino/op/squeeze.hpp>
#అన్తర్భావ <openvino/op/subtract.hpp>
#అన్తర్భావ <openvino/op/transpose.hpp>
#అన్తర్భావ <string>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {

శూన్య అంకె_inputs_check(స్థిరరాశి కణుపు_Context & context, size_t min_inputs, size_t max_inputs) {
    స్వయంచాలక ఆదానం_కొలత = context.get_ఆదానం_కొలత();
    FRONT_చివర_OP_CONVERSION_CHECK(ఆదానం_కొలత >= min_inputs, "Got less inputs than expected");
    FRONT_చివర_OP_CONVERSION_CHECK(ఆదానం_కొలత <= max_inputs, "Got more inputs than expected");
}

పూర్ణాంకం non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    పూర్ణాంకం dim = nb.size() - 1;
    size_t బైట్లు = nb[dim];
    కొరకు (పూర్ణాంకం i = dim; i > 0; i--) {
        బైట్లు *= ne[i];
        ఒకవేళ (బైట్లు != nb[i - 1]) {
            తిరిగి i;
        }
    }
    తిరిగి 0;
}

std::shared_ptr<ov::Node> get_dimensions(స్థిరరాశి std::shared_ptr<ov::op::v3::Shape_యొక్క> & shape,
                                         స్థిరరాశి std::vector<పూర్ణాంకం> & dims) {
    ఉపయోగించి నామాకాశ ov::op;
    స్థిరరాశి స్వయంచాలక zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    స్థిరరాశి స్వయంచాలక dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    తిరిగి std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(స్థిరరాశి std::shared_ptr<ov::Node> & కణుపు, స్థిరరాశి std::vector<పూర్ణాంకం> & dims) {
    తిరిగి get_dimensions(std::make_shared<ov::op::v3::Shape_యొక్క>(కణుపు), dims);
}

Output_సదిశ rename_outputs_with_suffix(స్థిరరాశి Output_సదిశ & outputs, స్థిరరాశి std::string & suffix) {
    కొరకు (స్థిరరాశి స్వయంచాలక & output : outputs) {
        స్వయంచాలక కణుపు = output.get_కణుపు_shared_ptr();
        std::string name = కణుపు->get_friendly_name();
        name += "_";
        name += suffix;
        కణుపు->సమితి_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    తిరిగి outputs;
}

నామాకాశ {
ov::Output<ov::Node> rope_yarn_ramp_mix(పూర్ణాంకం అంకె_elements, స్థిరరాశి దశాంశం corr_dims[2], దశాంశం ext_గుణకం) {
    std::vector<దశాంశం> dim_ids_vec(అంకె_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    స్వయంచాలక dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) అంకె_elements}, dim_ids_vec);
    స్వయంచాలక corr_దిగువ = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    స్వయంచాలక corr_ఎగువ = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    స్వయంచాలక denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::వ్యవకలనం>(corr_ఎగువ, corr_దిగువ),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    స్వయంచాలక ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::వ్యవకలనం>(dim_ids, corr_దిగువ), denom);
    స్వయంచాలక ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    స్వయంచాలక one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    స్వయంచాలక ramp_inverted = std::make_shared<ov::op::v1::వ్యవకలనం>(one, ramp_clamped);
    స్వయంచాలక ext_గుణకం_కణుపు = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_గుణకం});
    స్వయంచాలక ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_గుణకం_కణుపు);
    తిరిగి ramp_mix;
}

దశాంశం ggml_rope_yarn_corr_dim(పూర్ణాంకం n_dims, పూర్ణాంకం n_ctx_orig, దశాంశం n_rot, దశాంశం పునాది) {
#నిర్వచితం_కాకపోతే M_PI
#    define M_PI 3.14159265358979323846
#అయితే_ముగింపు
    తిరిగి n_dims * logf(n_ctx_orig / (n_rot * 2 * (దశాంశం) M_PI)) / (2 * logf(పునాది));
}

శూన్య ggml_rope_yarn_corr_dims(పూర్ణాంకం n_dims,
                              పూర్ణాంకం n_ctx_orig,
                              దశాంశం freq_పునాది,
                              దశాంశం beta_fast,
                              దశాంశం beta_slow,
                              దశాంశం dims[2]) {
    దశాంశం start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_పునాది));
    దశాంశం end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_పునాది));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<దశాంశం>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_బరువు,
                                                           పూర్ణాంకం mode,
                                                           తార్కిక stateful,
                                                           int64_t తల_dim) {
    constexpr పూర్ణాంకం TYPE_IMROPE = 2;
    constexpr పూర్ణాంకం TYPE_VISION = 3;
    constexpr పూర్ణాంకం TYPE_MROPE = 4;

    స్థిరరాశి తార్కిక is_imrope = (mode == TYPE_IMROPE);
    స్థిరరాశి తార్కిక is_vision = (mode == TYPE_VISION);
    స్థిరరాశి తార్కిక is_mrope = (mode == TYPE_MROPE);
    స్థిరరాశి తార్కిక is_multi = is_imrope || is_vision || is_mrope;

    ఒకవేళ (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        స్వయంచాలక pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } లేకపోతే ఒకవేళ (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        స్వయంచాలక pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, నిజం);
        స్వయంచాలక pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } లేకపోతే {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        స్వయంచాలక pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    దశాంశం freq_పునాది;
    దశాంశం freq_scale;
    దశాంశం ext_గుణకం;
    దశాంశం attn_గుణకం;
    దశాంశం beta_fast;
    దశాంశం beta_slow;
    స్థిరరాశి పూర్ణాంకం n_dims = rope_params[1];
    స్థిరరాశి size_t n_dims_half = n_dims >> 1;
    స్థిరరాశి పూర్ణాంకం n_ctx_orig = rope_params[4];
    జ్ఞాపకం_నకలు(&freq_పునాది, rope_params + 5, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&freq_scale, rope_params + 6, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&ext_గుణకం, rope_params + 7, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&attn_గుణకం, rope_params + 8, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&beta_fast, rope_params + 9, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&beta_slow, rope_params + 10, పరిమాణం(దశాంశం));

    స్థిరరాశి దశాంశం theta_scale = powf(freq_పునాది, -2.0f / n_dims);

    Output<Node> theta;
    దశాంశం mscale = attn_గుణకం;
    ఒకవేళ (is_multi) {
        స్థిరరాశి size_t అంకె_జతలు = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (తల_dim / 2)) : n_dims_half;
        పూర్ణాంకం sections[4];
        జ్ఞాపకం_నకలు(sections, rope_params + 11, పరిమాణం(పూర్ణాంకం) * 4);
        పూర్ణాంకం sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        ఒకవేళ (sect_dims <= 0) {
            sect_dims = అంకె_జతలు;
        }
        పూర్ణాంకం sec_w = sections[0] + sections[1];
        పూర్ణాంకం sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(అంకె_జతలు);
        std::vector<దశాంశం> గుణకం(అంకె_జతలు);
        కొరకు (size_t j = 0; j < అంకె_జతలు; j++) {
            పూర్ణాంకం sector = j % sect_dims;
            పూర్ణాంకం plane = 0;
            ఒకవేళ (is_imrope) {
                ఒకవేళ (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } లేకపోతే ఒకవేళ (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } లేకపోతే ఒకవేళ (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } లేకపోతే {
                    plane = 3;
                }
                గుణకం[j] = std::ఘాతం(theta_scale, j);
            } లేకపోతే ఒకవేళ (is_vision) {
                పూర్ణాంకం p_idx = 0;
                ఒకవేళ (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } లేకపోతే ఒకవేళ (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } లేకపోతే ఒకవేళ (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } లేకపోతే {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                గుణకం[j] = std::ఘాతం(theta_scale, p_idx);
            } లేకపోతే {
                ఒకవేళ (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } లేకపోతే ఒకవేళ (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } లేకపోతే ఒకవేళ (sector >= sec_e) {
                    plane = 3;
                } లేకపోతే {
                    plane = 0;
                }
                గుణకం[j] = std::ఘాతం(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        స్వయంచాలక gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{అంకె_జతలు}, gather_indices);
        స్వయంచాలక gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> గుణకం_కణుపు =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{అంకె_జతలు}, గుణకం);
        ఒకవేళ (rope_freqs_బరువు) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::ముక్క>(
                rope_freqs_బరువు,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) అంకె_జతలు}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_బరువు->get_output_partial_shape(0).rank().get_నిడివి() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) అంకె_జతలు}),
                అబద్ధం);
            గుణకం_కణుపు = std::make_shared<ov::op::v1::Divide>(గుణకం_కణుపు, rope_factors);
        }
        స్వయంచాలక theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, గుణకం_కణుపు);
        స్వయంచాలక theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        ఒకవేళ (ext_గుణకం == 0.0f) {
            theta = theta_interp;
        } లేకపోతే {
            దశాంశం corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_పునాది, beta_fast, beta_slow, corr_dims);
            స్వయంచాలక ramp_mix = rope_yarn_ramp_mix(అంకె_జతలు, corr_dims, ext_గుణకం);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            స్వయంచాలక one_minus_ramp = std::make_shared<ov::op::v1::వ్యవకలనం>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::కూడు>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::సంవర్గమానం(1.0f / freq_scale));
        }
    } లేకపోతే {
        std::vector<దశాంశం> గుణకం(n_dims_half);
        Output<Node> freq_factors;
        దశాంశం corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_పునాది, beta_fast, beta_slow, corr_dims);
        గుణకం[0] = 1.0f;
        కొరకు (size_t i = 1; i < గుణకం.size(); i++) {
            గుణకం[i] = theta_scale * గుణకం[i - 1];
        }
        ఒకవేళ (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, గుణకం.size()}, గుణకం);
        } లేకపోతే {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, గుణకం.size()}, గుణకం);
        }
        ఒకవేళ (rope_freqs_బరువు) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::ముక్క>(
                rope_freqs_బరువు,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_బరువు->get_output_partial_shape(0).rank().get_నిడివి() - 1}));
            ఒకవేళ (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    అబద్ధం);
            } లేకపోతే {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    అబద్ధం);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        స్వయంచాలక theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        స్వయంచాలక theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        ఒకవేళ (ext_గుణకం == 0.0f) {
            theta = theta_interp;
        } లేకపోతే {
            స్వయంచాలక ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_గుణకం);
            Output<Node> one;
            ఒకవేళ (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } లేకపోతే {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            }
            స్వయంచాలక one_minus_ramp = std::make_shared<ov::op::v1::వ్యవకలనం>(one, ramp_mix);

            theta =
                std::make_shared<ov::op::v1::కూడు>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                                                  std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::సంవర్గమానం(1.0f / freq_scale));
        }
    }

    Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
    Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);

    ఒకవేళ (mscale != 1.0f) {
        స్వయంచాలక mscale_కణుపు = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_కణుపు);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_కణుపు);
    }

    తిరిగి std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_ఆదానం(స్థిరరాశి కణుపు_Context & context, పూర్ణాంకం ఆదానం_సూచీ, పూర్ణాంకం ముక్క_len, పూర్ణాంకం axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    స్వయంచాలక ఆదానం = context.get_ఆదానం(ఆదానం_సూచీ);
    స్వయంచాలక * op_params = (size_t *) context.get_ఆదానం_op_params(ఆదానం_సూచీ);
    స్వయంచాలక src_stride = context.get_ఆదానం_stride(ఆదానం_సూచీ);

    int64_t ముక్క_ప్రారంభం = op_params[0] / src_stride[3];
    ఒకవేళ (ముక్క_len == 0) {
        ముక్క_len = context.get_ఆదానం_shape(ఆదానం_సూచీ)[3].get_నిడివి();
    }
    int64_t ముక్క_చివర = ముక్క_ప్రారంభం + ముక్క_len;

    స్వయంచాలక begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {ముక్క_ప్రారంభం});
    స్వయంచాలక end = ov::op::v0::Constant::create(ov::element::i64, {1}, {ముక్క_చివర});
    స్వయంచాలక stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    ఒకవేళ (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } లేకపోతే {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    స్వయంచాలక sliced = std::make_shared<ov::op::v8::ముక్క>(ఆదానం, begin, end, stride, axes);
    తిరిగి sliced;
}

ov::Output<ov::Node> process_view_ఆదానం_new(స్థిరరాశి కణుపు_Context & context, పూర్ణాంకం ఆదానం_సూచీ) {
    స్వయంచాలక ఆదానం = context.get_ఆదానం(ఆదానం_సూచీ);

    // Check if this input has view inputs
    size_t view_ఆదానం_కొలత = context.get_view_ఆదానం_కొలత(ఆదానం_సూచీ);
    ఒకవేళ (view_ఆదానం_కొలత == 0) {
        // No view inputs, return the input as is
        తిరిగి ఆదానం;
    }

    // If translate_view already resolved this VIEW (produced a Slice), the input
    // will already have the expected shape — skip re-slicing.
    //
    // Two notions of "matches" are accepted per axis:
    //   - both dims static and equal, OR
    //   - both dims dynamic.
    // The dynamic case matters for the MoE expert-plane views: translate_view now emits a
    // DYNAMIC-token slice (so the token dim is not frozen). An all-static-only check would
    // see the dynamic token dim, decide the shapes "don't match", and fall through to
    // re-slice/flatten the already-resolved view (a Reshape to the full flattened
    // n_expert_used*n_embd tail, which then conflicts with the single-plane input). Treat a
    // dynamic-vs-dynamic axis as matching so the already-resolved view is reused as-is.
    //
    // A third case matters for split-model MoE fragments: translate_view resolves the
    // expert-plane view against the fragment's INPUT parameter. When the graph is split
    // the token axis of that parameter may already be concrete (static n_tokens) even
    // though get_view_input_ov_shape() still reports it as dynamic (-1). The resolved
    // view is then static [1,1,n_tokens,n_embd] while `expected` is [1,1,?,n_embd].
    // An "expected dynamic, actual static" axis is a valid concretization of the SAME
    // resolved view, so treat it as matching too. Falling through to process_single_view
    // here would re-slice/re-flatten the already-resolved single-plane view against the
    // recorded (multi-plane) source strides and emit a constant-target Reshape whose baked
    // dims no longer divide the concretized input -> "dimensions do not evenly divide".
    // A fourth case matters for stateful execution: `expected` comes from ggml metadata and is
    // always GGML_MAX_DIMS ranks, but stateful drops the leading size-1 batch dim, so the
    // resolved view is one rank lower. Compare the common trailing dims and require the
    // leading expected dims we skip to be 1. Without this the rank test below never matches on
    // the stateful path and every already-resolved MoE expert-plane view is re-sliced.
    స్వయంచాలక expected_ov_shape = context.get_view_ఆదానం_ov_shape(ఆదానం_సూచీ, 0);
    స్వయంచాలక actual_shape = ఆదానం.get_partial_shape();
    ఒకవేళ (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_నిడివి() >= actual_shape.rank().get_నిడివి()) {
        స్థిరరాశి int64_t n_actual = actual_shape.rank().get_నిడివి();
        స్థిరరాశి int64_t జరుపు = expected_ov_shape.rank().get_నిడివి() - n_actual;
        తార్కిక shapes_సరిపోలిక = నిజం;
        కొరకు (int64_t i = 0; i < జరుపు; ++i) {
            ఒకవేళ (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_నిడివి() != 1) {
                shapes_సరిపోలిక = అబద్ధం;
                ఆపు;
            }
        }
        కొరకు (int64_t i = 0; i < n_actual && shapes_సరిపోలిక; ++i) {
            స్థిరరాశి స్వయంచాలక & ఘాతాంకం = expected_ov_shape[i + జరుపు];
            స్థిరరాశి తార్కిక both_dynamic = ఘాతాంకం.is_dynamic() && actual_shape[i].is_dynamic();
            స్థిరరాశి తార్కిక both_static_equal =
                ఘాతాంకం.is_static() && actual_shape[i].is_static() && ఘాతాంకం == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            స్థిరరాశి తార్కిక expected_dyn_actual_static = ఘాతాంకం.is_dynamic() && actual_shape[i].is_static();
            ఒకవేళ (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_సరిపోలిక = అబద్ధం;
                ఆపు;
            }
        }
        ఒకవేళ (shapes_సరిపోలిక) {
            తిరిగి ఆదానం;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    ఒకవేళ (context.is_static() && view_ఆదానం_కొలత == 1) {
        స్వయంచాలక view_stride_v = context.get_view_ఆదానం_stride(ఆదానం_సూచీ, 0);
        స్వయంచాలక view_src_stride_v = context.get_view_ఆదానం_src_stride(ఆదానం_సూచీ, 0);
        స్వయంచాలక view_ggml_shape = context.get_view_ఆదానం_ggml_shape(ఆదానం_సూచీ, 0);
        స్వయంచాలక view_src_ggml_shape = context.get_view_ఆదానం_src_ggml_shape(ఆదానం_సూచీ, 0);
        స్వయంచాలక view_స్థానభ్రంశం = context.get_view_ఆదానం_స్థానభ్రంశం(ఆదానం_సూచీ, 0);
        స్వయంచాలక view_src_స్థానభ్రంశం = context.get_view_ఆదానం_src_స్థానభ్రంశం(ఆదానం_సూచీ, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<పూర్ణాంకం> diff_dims;
        ఒకవేళ (view_src_ggml_shape.size() == ndims) {
            కొరకు (size_t i = 0; i < ndims; ++i) {
                ఒకవేళ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<పూర్ణాంకం>(i));
                }
            }
        }

        ఒకవేళ (diff_dims.size() == 1) {
            పూర్ణాంకం విభజించు_dim = diff_dims[0];
            int64_t అంకె_splits = static_cast<int64_t>(view_src_ggml_shape[విభజించు_dim]);
            int64_t chunk_కొలత = static_cast<int64_t>(view_ggml_shape[విభజించు_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            ఒకవేళ (chunk_కొలత == 1 && అంకె_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                తార్కిక suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                ఒకవేళ (suffix_ok) {
                    కొరకు (size_t i = static_cast<size_t>(విభజించు_dim) + 1; i < ndims; ++i) {
                        ఒకవేళ (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = అబద్ధం;
                            ఆపు;
                        }
                    }
                }

                ఒకవేళ (suffix_ok && view_src_stride_v[విభజించు_dim] > 0) {
                    size_t relative_స్థానభ్రంశం = view_స్థానభ్రంశం >= view_src_స్థానభ్రంశం ? view_స్థానభ్రంశం - view_src_స్థానభ్రంశం : 0;
                    int64_t విభజించు_సూచీ = static_cast<int64_t>(relative_స్థానభ్రంశం / view_src_stride_v[విభజించు_dim]);

                    ఒకవేళ (విభజించు_సూచీ >= 0 && విభజించు_సూచీ < అంకె_splits) {
                        స్వయంచాలక src_కణుపు = ఆదానం.get_కణుపు_shared_ptr();
                        std::string rt_కీలకం = "split_dim_" + std::to_string(విభజించు_dim);
                        స్వయంచాలక & rt_info = src_కణుపు->get_rt_info();

                        ఒకవేళ (rt_info.find(rt_కీలకం) == rt_info.end()) {
                            స్వయంచాలక axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(విభజించు_dim)});
                            స్వయంచాలక విభజించు_కణుపు =
                                std::make_shared<ov::op::v1::విభజించు_2>(ఆదానం, axis_const, static_cast<size_t>(అంకె_splits));
                            విభజించు_కణుపు->సమితి_friendly_name(src_కణుపు->get_friendly_name() + "_split");
                            rt_info[rt_కీలకం] = విభజించు_కణుపు;
                        }

                        స్వయంచాలక విభజించు_కణుపు = rt_info[rt_కీలకం].as<std::shared_ptr<ov::op::v1::విభజించు_2>>();
                        తిరిగి విభజించు_కణుపు->output(static_cast<size_t>(విభజించు_సూచీ));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    స్వయంచాలక process_single_view =
        [](ov::Output<ov::Node> current, size_t view_స్థానభ్రంశం, స్థిరరాశి std::vector<size_t> & view_stride,
           స్థిరరాశి ov::Shape & view_ggml_shape, స్థిరరాశి ov::PartialShape & view_ov_shape, స్థిరరాశి std::string & view_name,
           size_t view_src_స్థానభ్రంశం, స్థిరరాశి std::vector<size_t> & view_src_stride, స్థిరరాశి ov::Shape & view_src_ggml_shape,
           స్థిరరాశి ov::PartialShape & view_src_ov_shape, స్థిరరాశి std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        స్థిరరాశి స్వయంచాలక current_rank = current.get_partial_shape().rank();
        స్థిరరాశి పూర్ణాంకం axis_జరుపు = current_rank.is_static() ?
                                    static_cast<పూర్ణాంకం>(view_stride.size()) - static_cast<పూర్ణాంకం>(current_rank.get_నిడివి()) :
                                    0;

        స్వయంచాలక build_reshape_నమూనా = [](స్థిరరాశి ov::PartialShape & లక్ష్యం_ov_shape,
                                        స్థిరరాశి ov::Shape & లక్ష్యం_ggml_shape) -> std::vector<int64_t> {
            స్థిరరాశి size_t ndims = లక్ష్యం_ggml_shape.size();
            std::vector<int64_t> reshape_నమూనా(ndims);
            size_t dynamic_dims = 0;

            ఒకవేళ (లక్ష్యం_ov_shape.rank().is_static() &&
                లక్ష్యం_ov_shape.rank().get_నిడివి() == static_cast<int64_t>(ndims)) {
                కొరకు (size_t i = 0; i < ndims; ++i) {
                    ఒకవేళ (లక్ష్యం_ov_shape[i].is_static()) {
                        reshape_నమూనా[i] = లక్ష్యం_ov_shape[i].get_నిడివి();
                    } లేకపోతే {
                        reshape_నమూనా[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } లేకపోతే {
                dynamic_dims = 2;
            }

            ఒకవేళ (dynamic_dims > 1) {
                కొరకు (size_t i = 0; i < ndims; ++i) {
                    reshape_నమూనా[i] = static_cast<int64_t>(లక్ష్యం_ggml_shape[i]);
                }
            }

            తిరిగి reshape_నమూనా;
        };

        స్వయంచాలక build_ఉపసర్గ_తోక_reshape_నమూనా = [](స్థిరరాశి ov::PartialShape & లక్ష్యం_ov_shape,
                                                    స్థిరరాశి ov::Shape & లక్ష్యం_ggml_shape, size_t ఉపసర్గ_dims,
                                                    int64_t తోక_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_నమూనా(ఉపసర్గ_dims + 1);
            size_t dynamic_dims = 0;

            ఒకవేళ (లక్ష్యం_ov_shape.rank().is_static() &&
                లక్ష్యం_ov_shape.rank().get_నిడివి() == static_cast<int64_t>(లక్ష్యం_ggml_shape.size())) {
                కొరకు (size_t i = 0; i < ఉపసర్గ_dims; ++i) {
                    ఒకవేళ (లక్ష్యం_ov_shape[i].is_static()) {
                        reshape_నమూనా[i] = లక్ష్యం_ov_shape[i].get_నిడివి();
                    } లేకపోతే {
                        reshape_నమూనా[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } లేకపోతే {
                dynamic_dims = 2;
            }

            ఒకవేళ (dynamic_dims > 1) {
                కొరకు (size_t i = 0; i < ఉపసర్గ_dims; ++i) {
                    reshape_నమూనా[i] = static_cast<int64_t>(లక్ష్యం_ggml_shape[i]);
                }
            }

            reshape_నమూనా[ఉపసర్గ_dims] = తోక_dim;
            తిరిగి reshape_నమూనా;
        };

        తార్కిక same_stride = view_stride.size() == view_src_stride.size();
        ఒకవేళ (same_stride) {
            కొరకు (size_t i = 0; i < view_stride.size(); ++i) {
                ఒకవేళ (view_stride[i] != view_src_stride[i]) {
                    same_stride = అబద్ధం;
                    ఆపు;
                }
            }
        }

        తార్కిక same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        ఒకవేళ (same_ggml_shape) {
            కొరకు (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                ఒకవేళ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = అబద్ధం;
                    ఆపు;
                }
            }
        }

        ఒకవేళ (same_stride && same_ggml_shape) {
            తిరిగి current;
        }

        ఒకవేళ (same_stride) {
            స్థిరరాశి size_t relative_స్థానభ్రంశం = view_స్థానభ్రంశం >= view_src_స్థానభ్రంశం ? view_స్థానభ్రంశం - view_src_స్థానభ్రంశం : 0;
            స్థిరరాశి size_t ndims = view_stride.size();

            std::vector<పూర్ణాంకం> diff_dims;
            ఒకవేళ (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                కొరకు (size_t i = 0; i < ndims; ++i) {
                    ఒకవేళ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<పూర్ణాంకం>(i));
                    }
                }
            }

            ఒకవేళ (diff_dims.size() == 1) {
                స్థిరరాశి పూర్ణాంకం ముక్క_dim = diff_dims[0];
                స్థిరరాశి int64_t dim_కొలత = static_cast<int64_t>(view_src_ggml_shape[ముక్క_dim]);

                ఒకవేళ (view_stride[ముక్క_dim] > 0 && relative_స్థానభ్రంశం % view_stride[ముక్క_dim] == 0) {
                    స్థిరరాశి int64_t begin_val = static_cast<int64_t>((relative_స్థానభ్రంశం / view_stride[ముక్క_dim]) %
                                                                   static_cast<size_t>(dim_కొలత));
                    స్థిరరాశి int64_t చివర_val = begin_val + static_cast<int64_t>(view_ggml_shape[ముక్క_dim]);

                    ఒకవేళ (begin_val >= 0 && చివర_val <= dim_కొలత) {
                        స్వయంచాలక sliced = std::make_shared<ov::op::v8::ముక్క>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {చివర_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {ముక్క_dim - axis_జరుపు}));

                        ఒకవేళ (view_ov_shape.is_static()) {
                            స్వయంచాలక reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.గమ్యం_shape()),
                                అబద్ధం);
                            reshaped->సమితి_friendly_name(view_name);
                            తిరిగి reshaped;
                        }

                        sliced->సమితి_friendly_name(view_name);
                        తిరిగి sliced;
                    }
                }

                int64_t తోక_src_elems = 1;
                int64_t తోక_dst_elems = 1;
                కొరకు (size_t i = ముక్క_dim; i < ndims; ++i) {
                    తోక_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    తోక_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                స్థిరరాశి size_t elem_stride = view_stride[ndims - 1];
                int64_t తోక_begin = 0;
                ఒకవేళ (elem_stride > 0) {
                    తోక_begin =
                        static_cast<int64_t>((relative_స్థానభ్రంశం / elem_stride) % static_cast<size_t>(తోక_src_elems));
                }
                స్థిరరాశి int64_t తోక_చివర = తోక_begin + తోక_dst_elems;

                ఒకవేళ (తోక_begin >= 0 && తోక_చివర <= తోక_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(ముక్క_dim);
                    కొరకు (పూర్ణాంకం i = 0; i < ముక్క_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(తోక_src_elems);
                    స్థిరరాశి size_t flat_ndims = flat_shape.size();

                    స్వయంచాలక flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), అబద్ధం);

                    స్వయంచాలక sliced = std::make_shared<ov::op::v8::ముక్క>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {తోక_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {తోక_చివర}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {ముక్క_dim}));

                    ఒకవేళ (view_ov_shape.is_static()) {
                        స్వయంచాలక reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.గమ్యం_shape()),
                            అబద్ధం);
                        reshaped->సమితి_friendly_name(view_name);
                        తిరిగి reshaped;
                    }

                    sliced->సమితి_friendly_name(view_name);
                    తిరిగి sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> అడుగు(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t మిగిలిన_స్థానభ్రంశం = relative_స్థానభ్రంశం;
            కొరకు (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                ఒకవేళ (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(మిగిలిన_స్థానభ్రంశం / view_stride[i]);
                    మిగిలిన_స్థానభ్రంశం %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            తార్కిక లోపల_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            ఒకవేళ (లోపల_bounds) {
                కొరకు (size_t i = 0; i < ndims; ++i) {
                    ఒకవేళ (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        లోపల_bounds = అబద్ధం;
                        ఆపు;
                    }
                }
            }

            ఒకవేళ (లోపల_bounds && మిగిలిన_స్థానభ్రంశం == 0) {
                స్వయంచాలక sliced = std::make_shared<ov::op::v8::ముక్క>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, అడుగు),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->సమితి_friendly_name(view_name);
                తిరిగి sliced;
            }
        } లేకపోతే {
            తార్కిక same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            స్థిరరాశి size_t relative_స్థానభ్రంశం = view_స్థానభ్రంశం >= view_src_స్థానభ్రంశం ? view_స్థానభ్రంశం - view_src_స్థానభ్రంశం : 0;

            ఒకవేళ (same_rank) {
                స్థిరరాశి size_t ndims = view_ggml_shape.size();
                std::vector<పూర్ణాంకం> diff_dims;
                కొరకు (size_t i = 0; i < ndims; ++i) {
                    ఒకవేళ (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<పూర్ణాంకం>(i));
                    }
                }

                ఒకవేళ (diff_dims.size() == 1) {
                    స్థిరరాశి size_t ముక్క_dim = static_cast<size_t>(diff_dims[0]);
                    తార్కిక suffix_stride_సరిపోలిక = నిజం;
                    కొరకు (size_t i = ముక్క_dim + 1; i < ndims; ++i) {
                        ఒకవేళ (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_సరిపోలిక = అబద్ధం;
                            ఆపు;
                        }
                    }

                    ఒకవేళ (suffix_stride_సరిపోలిక && view_src_stride[ముక్క_dim] > 0 &&
                        relative_స్థానభ్రంశం % view_src_stride[ముక్క_dim] == 0) {
                        స్థిరరాశి int64_t begin_val = static_cast<int64_t>(relative_స్థానభ్రంశం / view_src_stride[ముక్క_dim]);
                        స్థిరరాశి int64_t చివర_val = begin_val + static_cast<int64_t>(view_ggml_shape[ముక్క_dim]);
                        స్థిరరాశి int64_t dim_కొలత = static_cast<int64_t>(view_src_ggml_shape[ముక్క_dim]);

                        ఒకవేళ (begin_val >= 0 && చివర_val <= dim_కొలత) {
                            స్వయంచాలక sliced = std::make_shared<ov::op::v8::ముక్క>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {చివర_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(ముక్క_dim)}));
                            sliced->సమితి_friendly_name(view_name);
                            తిరిగి sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            ఒకవేళ (same_rank) {
                కొరకు (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            తార్కిక same_అంకె_elements = same_rank && view_elems == src_elems;

            ఒకవేళ (same_rank && relative_స్థానభ్రంశం == 0 && same_అంకె_elements) {
                స్వయంచాలక reshape_నమూనా = build_reshape_నమూనా(view_ov_shape, view_ggml_shape);

                స్వయంచాలక reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_నమూనా.size()}, reshape_నమూనా),
                    అబద్ధం);
                reshaped->సమితి_friendly_name(view_name);
                తిరిగి reshaped;
            }

            ఒకవేళ (same_rank) {
                స్థిరరాశి size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                తార్కిక is_regular_ముక్క = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> అడుగు(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t మిగిలిన_స్థానభ్రంశం = relative_స్థానభ్రంశం;

                ఒకవేళ (is_regular_ముక్క) {
                    కొరకు (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        ఒకవేళ (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_ముక్క = అబద్ధం;
                            ఆపు;
                        }

                        అడుగు[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        ఒకవేళ (అడుగు[i] <= 0) {
                            is_regular_ముక్క = అబద్ధం;
                            ఆపు;
                        }

                        begin[i] = static_cast<int64_t>(మిగిలిన_స్థానభ్రంశం / view_src_stride[i]);
                        మిగిలిన_స్థానభ్రంశం %= view_src_stride[i];

                        ఒకవేళ (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            కొనసాగు;
                        }

                        end[i] = begin[i] + అడుగు[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        ఒకవేళ (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_ముక్క = అబద్ధం;
                            ఆపు;
                        }
                    }
                }

                ఒకవేళ (is_regular_ముక్క && మిగిలిన_స్థానభ్రంశం == 0) {
                    స్వయంచాలక sliced = std::make_shared<ov::op::v8::ముక్క>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, అడుగు),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->సమితి_friendly_name(view_name);
                    తిరిగి sliced;
                }

                స్థిరరాశి size_t elem_stride = view_src_stride.back();
                స్థిరరాశి తార్కిక aligned_స్థానభ్రంశం = elem_stride > 0 && relative_స్థానభ్రంశం % elem_stride == 0;

                ఒకవేళ (aligned_స్థానభ్రంశం) {
                    size_t suffix_ప్రారంభం = 0;
                    size_t expected_stride = elem_stride;
                    కొరకు (పూర్ణాంకం i = static_cast<పూర్ణాంకం>(ndims) - 1; i >= 0; --i) {
                        ఒకవేళ (view_stride[i] != expected_stride) {
                            suffix_ప్రారంభం = static_cast<size_t>(i + 1);
                            ఆపు;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t ఉపసర్గ_elems = 1;
                    size_t suffix_elems = 1;
                    కొరకు (size_t i = 0; i < suffix_ప్రారంభం; ++i) {
                        ఉపసర్గ_elems *= view_ggml_shape[i];
                    }
                    కొరకు (size_t i = suffix_ప్రారంభం; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    ఒకవేళ (ఉపసర్గ_elems > 0 && src_elems % ఉపసర్గ_elems == 0) {
                        స్థిరరాశి size_t src_తోక_elems = src_elems / ఉపసర్గ_elems;
                        స్థిరరాశి int64_t తోక_begin = static_cast<int64_t>(relative_స్థానభ్రంశం / elem_stride);
                        స్థిరరాశి int64_t తోక_చివర = తోక_begin + static_cast<int64_t>(suffix_elems);

                        ఒకవేళ (తోక_begin >= 0 && తోక_చివర <= static_cast<int64_t>(src_తోక_elems)) {
                            స్వయంచాలక ఉపసర్గ_తోక_నమూనా = build_ఉపసర్గ_తోక_reshape_నమూనా(
                                view_ov_shape, view_ggml_shape, suffix_ప్రారంభం, static_cast<int64_t>(src_తోక_elems));

                            స్వయంచాలక ఉపసర్గ_తోక = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {ఉపసర్గ_తోక_నమూనా.size()},
                                                             ఉపసర్గ_తోక_నమూనా),
                                అబద్ధం);

                            ov::Output<ov::Node> selected = ఉపసర్గ_తోక;
                            ఒకవేళ (తోక_begin != 0 || తోక_చివర != static_cast<int64_t>(src_తోక_elems)) {
                                selected = std::make_shared<ov::op::v8::ముక్క>(
                                    ఉపసర్గ_తోక, ov::op::v0::Constant::create(ov::element::i64, {1}, {తోక_begin}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {తోక_చివర}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {static_cast<int64_t>(suffix_ప్రారంభం)}));
                            }

                            స్వయంచాలక reshape_నమూనా = build_reshape_నమూనా(view_ov_shape, view_ggml_shape);
                            స్వయంచాలక reshaped = std::make_shared<ov::op::v1::Reshape>(
                                selected,
                                ov::op::v0::Constant::create(ov::element::i64, {reshape_నమూనా.size()},
                                                             reshape_నమూనా),
                                అబద్ధం);
                            reshaped->సమితి_friendly_name(view_name);
                            తిరిగి reshaped;
                        }
                    }
                }
            }

            తిరిగి current;
        }

        (శూన్య) view_name;
        (శూన్య) view_src_ov_shape;
        (శూన్య) view_src_name;

        తిరిగి current;
    };

    // Special case: ggml collapses VIEW-of-VIEW chains so that `view_offs` is always an
    // ABSOLUTE offset from the true root allocation, regardless of how many VIEW levels
    // are in between (see ggml_new_tensor_impl). `src[0]` is still the immediate op-graph
    // parent though, which can be a DIFFERENT (already narrowed) VIEW with the SAME ggml
    // shape as this one but a different absolute offset -- e.g. a per-layer deepstack
    // slice `view_2d(embd, n_embd, n_tokens, embd->nb[1], layer*n_embd*sizeof(float))`
    // whose src[0] ("embd") is itself already a zero-offset VIEW of the true root (the
    // padded embedding). Chaining through "embd" here would try to re-slice an already
    // 2-narrowed tensor using a root-relative offset, going out of bounds and silently
    // falling back to a no-op (returning the wrong, already-resolved sibling slice).
    // Detect this (same shape as the immediate src, but different absolute offset) and
    // re-slice directly from the untouched root using the innermost view's absolute
    // offset against the ROOT's own shape/stride instead of chaining through src[0].
    {
        స్వయంచాలక innermost_స్థానభ్రంశం = context.get_view_ఆదానం_స్థానభ్రంశం(ఆదానం_సూచీ, 0);
        స్వయంచాలక innermost_src_స్థానభ్రంశం = context.get_view_ఆదానం_src_స్థానభ్రంశం(ఆదానం_సూచీ, 0);
        స్వయంచాలక innermost_shape = context.get_view_ఆదానం_ggml_shape(ఆదానం_సూచీ, 0);
        స్వయంచాలక innermost_src_shape = context.get_view_ఆదానం_src_ggml_shape(ఆదానం_సూచీ, 0);
        ఒకవేళ (innermost_స్థానభ్రంశం != innermost_src_స్థానభ్రంశం && innermost_shape == innermost_src_shape) {
            size_t మూలం_view_idx = view_ఆదానం_కొలత - 1;
            స్వయంచాలక మూలం_ggml_shape = context.get_view_ఆదానం_src_ggml_shape(ఆదానం_సూచీ, మూలం_view_idx);
            స్వయంచాలక మూలం_stride = context.get_view_ఆదానం_src_stride(ఆదానం_సూచీ, మూలం_view_idx);
            స్వయంచాలక మూలం_స్థానభ్రంశం = context.get_view_ఆదానం_src_స్థానభ్రంశం(ఆదానం_సూచీ, మూలం_view_idx);
            స్వయంచాలక మూలం_ov_shape = context.get_view_ఆదానం_src_ov_shape(ఆదానం_సూచీ, మూలం_view_idx);
            స్వయంచాలక మూలం_name = context.get_view_ఆదానం_src_name(ఆదానం_సూచీ, మూలం_view_idx);
            స్వయంచాలక innermost_stride = context.get_view_ఆదానం_stride(ఆదానం_సూచీ, 0);
            స్వయంచాలక innermost_ov_shape = context.get_view_ఆదానం_ov_shape(ఆదానం_సూచీ, 0);
            స్వయంచాలక innermost_name = context.get_view_ఆదానం_name(ఆదానం_సూచీ, 0);

            తిరిగి process_single_view(ఆదానం, innermost_స్థానభ్రంశం, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, మూలం_స్థానభ్రంశం, మూలం_stride, మూలం_ggml_shape, మూలం_ov_shape,
                                       మూలం_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = ఆదానం;

    // Process each view in reverse order (from base to current)
    కొరకు (పూర్ణాంకం view_idx = view_ఆదానం_కొలత - 1; view_idx >= 0; view_idx--) {
        స్వయంచాలక view_స్థానభ్రంశం = context.get_view_ఆదానం_స్థానభ్రంశం(ఆదానం_సూచీ, view_idx);
        స్వయంచాలక view_stride = context.get_view_ఆదానం_stride(ఆదానం_సూచీ, view_idx);
        స్వయంచాలక view_ggml_shape = context.get_view_ఆదానం_ggml_shape(ఆదానం_సూచీ, view_idx);
        స్వయంచాలక view_ov_shape = context.get_view_ఆదానం_ov_shape(ఆదానం_సూచీ, view_idx);
        స్వయంచాలక view_name = context.get_view_ఆదానం_name(ఆదానం_సూచీ, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        స్వయంచాలక view_src_స్థానభ్రంశం = context.get_view_ఆదానం_src_స్థానభ్రంశం(ఆదానం_సూచీ, view_idx);
        స్వయంచాలక view_src_stride = context.get_view_ఆదానం_src_stride(ఆదానం_సూచీ, view_idx);
        స్వయంచాలక view_src_ggml_shape = context.get_view_ఆదానం_src_ggml_shape(ఆదానం_సూచీ, view_idx);
        స్వయంచాలక view_src_ov_shape = context.get_view_ఆదానం_src_ov_shape(ఆదానం_సూచీ, view_idx);
        స్వయంచాలక view_src_name = context.get_view_ఆదానం_src_name(ఆదానం_సూచీ, view_idx);
        // print source view info
        // std::cout << "View " << view_idx << ": source name = " << view_src_name
        //           << ", source offset = " << view_src_offset << ", source stride = [" << view_src_stride[0] << ","
        //           << view_src_stride[1] << "," << view_src_stride[2] << "," << view_src_stride[3]
        //           << "], source ggml shape = [" << view_src_ggml_shape[0] << "," << view_src_ggml_shape[1] << ","
        //           << view_src_ggml_shape[2] << "," << view_src_ggml_shape[3]
        //           << "], source ov shape = " << view_src_ov_shape << std::endl;

        current = process_single_view(current, view_స్థానభ్రంశం, view_stride, view_ggml_shape, view_ov_shape, view_name,
                                      view_src_స్థానభ్రంశం, view_src_stride, view_src_ggml_shape, view_src_ov_shape,
                                      view_src_name);
    }

    తిరిగి current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
