#अन्तर्भाव "utils.h"

#अन्तर्भाव <cmath>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <ctime>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/clamp.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/cos.hpp>
#अन्तर्भाव <openvino/op/divide.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/maximum.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/sin.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/split.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/subtract.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <string>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {

शून्य अङ्कः_inputs_check(ध्रुव ग्रन्थिः_Context & context, size_t min_inputs, size_t max_inputs) {
    स्वचालित निवेशः_आकार = context.get_निवेशः_आकार();
    FRONT_अन्तः_OP_CONVERSION_CHECK(निवेशः_आकार >= min_inputs, "Got less inputs than expected");
    FRONT_अन्तः_OP_CONVERSION_CHECK(निवेशः_आकार <= max_inputs, "Got more inputs than expected");
}

पूर्णाङ्क non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    पूर्णाङ्क dim = nb.size() - 1;
    size_t अष्टकानि = nb[dim];
    प्रति (पूर्णाङ्क i = dim; i > 0; i--) {
        अष्टकानि *= ne[i];
        यदि (अष्टकानि != nb[i - 1]) {
            प्रत्यागम i;
        }
    }
    प्रत्यागम 0;
}

std::shared_ptr<ov::Node> get_dimensions(ध्रुव std::shared_ptr<ov::op::v3::Shape_सम्बन्धि> & shape,
                                         ध्रुव std::vector<पूर्णाङ्क> & dims) {
    उपयुज्य नामाकाश ov::op;
    ध्रुव स्वचालित zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ध्रुव स्वचालित dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    प्रत्यागम std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(ध्रुव std::shared_ptr<ov::Node> & ग्रन्थिः, ध्रुव std::vector<पूर्णाङ्क> & dims) {
    प्रत्यागम get_dimensions(std::make_shared<ov::op::v3::Shape_सम्बन्धि>(ग्रन्थिः), dims);
}

Output_सदिशम् rename_outputs_with_suffix(ध्रुव Output_सदिशम् & outputs, ध्रुव std::string & suffix) {
    प्रति (ध्रुव स्वचालित & output : outputs) {
        स्वचालित ग्रन्थिः = output.get_ग्रन्थिः_shared_ptr();
        std::string name = ग्रन्थिः->get_friendly_name();
        name += "_";
        name += suffix;
        ग्रन्थिः->समुच्चयः_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    प्रत्यागम outputs;
}

नामाकाश {
ov::Output<ov::Node> rope_yarn_ramp_mix(पूर्णाङ्क अङ्कः_elements, ध्रुव भिन्न corr_dims[2], भिन्न ext_गुणकः) {
    std::vector<भिन्न> dim_ids_vec(अङ्कः_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    स्वचालित dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) अङ्कः_elements}, dim_ids_vec);
    स्वचालित corr_नीचः = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    स्वचालित corr_उच्चः = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    स्वचालित denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::व्यवकलय>(corr_उच्चः, corr_नीचः),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    स्वचालित ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::व्यवकलय>(dim_ids, corr_नीचः), denom);
    स्वचालित ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    स्वचालित one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    स्वचालित ramp_inverted = std::make_shared<ov::op::v1::व्यवकलय>(one, ramp_clamped);
    स्वचालित ext_गुणकः_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_गुणकः});
    स्वचालित ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_गुणकः_ग्रन्थिः);
    प्रत्यागम ramp_mix;
}

भिन्न ggml_rope_yarn_corr_dim(पूर्णाङ्क n_dims, पूर्णाङ्क n_ctx_orig, भिन्न n_rot, भिन्न आधारः) {
#यदि_अपरिभाषितम् M_PI
#    define M_PI 3.14159265358979323846
#यदि_अन्तः
    प्रत्यागम n_dims * logf(n_ctx_orig / (n_rot * 2 * (भिन्न) M_PI)) / (2 * logf(आधारः));
}

शून्य ggml_rope_yarn_corr_dims(पूर्णाङ्क n_dims,
                              पूर्णाङ्क n_ctx_orig,
                              भिन्न freq_आधारः,
                              भिन्न beta_fast,
                              भिन्न beta_slow,
                              भिन्न dims[2]) {
    भिन्न start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_आधारः));
    भिन्न end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_आधारः));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<भिन्न>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight,
                                                           पूर्णाङ्क mode,
                                                           तार्किक stateful,
                                                           int64_t शीर्ष_dim) {
    constexpr पूर्णाङ्क TYPE_IMROPE = 2;
    constexpr पूर्णाङ्क TYPE_VISION = 3;
    constexpr पूर्णाङ्क TYPE_MROPE = 4;

    ध्रुव तार्किक is_imrope = (mode == TYPE_IMROPE);
    ध्रुव तार्किक is_vision = (mode == TYPE_VISION);
    ध्रुव तार्किक is_mrope = (mode == TYPE_MROPE);
    ध्रुव तार्किक is_multi = is_imrope || is_vision || is_mrope;

    यदि (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        स्वचालित pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } अन्यथा यदि (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        स्वचालित pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, सत्य);
        स्वचालित pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } अन्यथा {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        स्वचालित pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    भिन्न freq_आधारः;
    भिन्न freq_scale;
    भिन्न ext_गुणकः;
    भिन्न attn_गुणकः;
    भिन्न beta_fast;
    भिन्न beta_slow;
    ध्रुव पूर्णाङ्क n_dims = rope_params[1];
    ध्रुव size_t n_dims_half = n_dims >> 1;
    ध्रुव पूर्णाङ्क n_ctx_orig = rope_params[4];
    स्मृतिं_प्रतिलिख(&freq_आधारः, rope_params + 5, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&freq_scale, rope_params + 6, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&ext_गुणकः, rope_params + 7, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&attn_गुणकः, rope_params + 8, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&beta_fast, rope_params + 9, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&beta_slow, rope_params + 10, परिमाण(भिन्न));

    ध्रुव भिन्न theta_scale = powf(freq_आधारः, -2.0f / n_dims);

    Output<Node> theta;
    भिन्न mscale = attn_गुणकः;
    यदि (is_multi) {
        ध्रुव size_t अङ्कः_युग्मानि = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (शीर्ष_dim / 2)) : n_dims_half;
        पूर्णाङ्क sections[4];
        स्मृतिं_प्रतिलिख(sections, rope_params + 11, परिमाण(पूर्णाङ्क) * 4);
        पूर्णाङ्क sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        यदि (sect_dims <= 0) {
            sect_dims = अङ्कः_युग्मानि;
        }
        पूर्णाङ्क sec_w = sections[0] + sections[1];
        पूर्णाङ्क sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(अङ्कः_युग्मानि);
        std::vector<भिन्न> गुणकः(अङ्कः_युग्मानि);
        प्रति (size_t j = 0; j < अङ्कः_युग्मानि; j++) {
            पूर्णाङ्क sector = j % sect_dims;
            पूर्णाङ्क plane = 0;
            यदि (is_imrope) {
                यदि (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } अन्यथा यदि (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } अन्यथा यदि (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } अन्यथा {
                    plane = 3;
                }
                गुणकः[j] = std::घातः(theta_scale, j);
            } अन्यथा यदि (is_vision) {
                पूर्णाङ्क p_idx = 0;
                यदि (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } अन्यथा यदि (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } अन्यथा यदि (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } अन्यथा {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                गुणकः[j] = std::घातः(theta_scale, p_idx);
            } अन्यथा {
                यदि (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } अन्यथा यदि (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } अन्यथा यदि (sector >= sec_e) {
                    plane = 3;
                } अन्यथा {
                    plane = 0;
                }
                गुणकः[j] = std::घातः(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        स्वचालित gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{अङ्कः_युग्मानि}, gather_indices);
        स्वचालित gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> गुणकः_ग्रन्थिः =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{अङ्कः_युग्मानि}, गुणकः);
        यदि (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::खण्डिका>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) अङ्कः_युग्मानि}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_दीर्घता() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) अङ्कः_युग्मानि}),
                असत्य);
            गुणकः_ग्रन्थिः = std::make_shared<ov::op::v1::Divide>(गुणकः_ग्रन्थिः, rope_factors);
        }
        स्वचालित theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, गुणकः_ग्रन्थिः);
        स्वचालित theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        यदि (ext_गुणकः == 0.0f) {
            theta = theta_interp;
        } अन्यथा {
            भिन्न corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_आधारः, beta_fast, beta_slow, corr_dims);
            स्वचालित ramp_mix = rope_yarn_ramp_mix(अङ्कः_युग्मानि, corr_dims, ext_गुणकः);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            स्वचालित one_minus_ramp = std::make_shared<ov::op::v1::व्यवकलय>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::सङ्कलय>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::लघुगणकः(1.0f / freq_scale));
        }
    } अन्यथा {
        std::vector<भिन्न> गुणकः(n_dims_half);
        Output<Node> freq_factors;
        भिन्न corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_आधारः, beta_fast, beta_slow, corr_dims);
        गुणकः[0] = 1.0f;
        प्रति (size_t i = 1; i < गुणकः.size(); i++) {
            गुणकः[i] = theta_scale * गुणकः[i - 1];
        }
        यदि (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, गुणकः.size()}, गुणकः);
        } अन्यथा {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, गुणकः.size()}, गुणकः);
        }
        यदि (rope_freqs_weight) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::खण्डिका>(
                rope_freqs_weight,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_weight->get_output_partial_shape(0).rank().get_दीर्घता() - 1}));
            यदि (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    असत्य);
            } अन्यथा {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    असत्य);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        स्वचालित theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        स्वचालित theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        यदि (ext_गुणकः == 0.0f) {
            theta = theta_interp;
        } अन्यथा {
            स्वचालित ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_गुणकः);
            Output<Node> one;
            यदि (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } अन्यथा {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            }
            स्वचालित one_minus_ramp = std::make_shared<ov::op::v1::व्यवकलय>(one, ramp_mix);

            theta =
                std::make_shared<ov::op::v1::सङ्कलय>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                                                  std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::लघुगणकः(1.0f / freq_scale));
        }
    }

    Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
    Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);

    यदि (mscale != 1.0f) {
        स्वचालित mscale_ग्रन्थिः = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_ग्रन्थिः);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_ग्रन्थिः);
    }

    प्रत्यागम std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_निवेशः(ध्रुव ग्रन्थिः_Context & context, पूर्णाङ्क निवेशः_सूचकांक, पूर्णाङ्क खण्डिका_len, पूर्णाङ्क axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    स्वचालित निवेशः = context.get_निवेशः(निवेशः_सूचकांक);
    स्वचालित * op_params = (size_t *) context.get_निवेशः_op_params(निवेशः_सूचकांक);
    स्वचालित src_stride = context.get_निवेशः_stride(निवेशः_सूचकांक);

    int64_t खण्डिका_start = op_params[0] / src_stride[3];
    यदि (खण्डिका_len == 0) {
        खण्डिका_len = context.get_निवेशः_shape(निवेशः_सूचकांक)[3].get_दीर्घता();
    }
    int64_t खण्डिका_अन्तः = खण्डिका_start + खण्डिका_len;

    स्वचालित begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {खण्डिका_start});
    स्वचालित end = ov::op::v0::Constant::create(ov::element::i64, {1}, {खण्डिका_अन्तः});
    स्वचालित stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    यदि (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } अन्यथा {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    स्वचालित sliced = std::make_shared<ov::op::v8::खण्डिका>(निवेशः, begin, end, stride, axes);
    प्रत्यागम sliced;
}

ov::Output<ov::Node> process_view_निवेशः_new(ध्रुव ग्रन्थिः_Context & context, पूर्णाङ्क निवेशः_सूचकांक) {
    स्वचालित निवेशः = context.get_निवेशः(निवेशः_सूचकांक);

    // Check if this input has view inputs
    size_t view_निवेशः_आकार = context.get_view_निवेशः_आकार(निवेशः_सूचकांक);
    यदि (view_निवेशः_आकार == 0) {
        // No view inputs, return the input as is
        प्रत्यागम निवेशः;
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
    स्वचालित expected_ov_shape = context.get_view_निवेशः_ov_shape(निवेशः_सूचकांक, 0);
    स्वचालित actual_shape = निवेशः.get_partial_shape();
    यदि (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_दीर्घता() >= actual_shape.rank().get_दीर्घता()) {
        ध्रुव int64_t n_actual = actual_shape.rank().get_दीर्घता();
        ध्रुव int64_t सरणम् = expected_ov_shape.rank().get_दीर्घता() - n_actual;
        तार्किक shapes_मेलनम् = सत्य;
        प्रति (int64_t i = 0; i < सरणम्; ++i) {
            यदि (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_दीर्घता() != 1) {
                shapes_मेलनम् = असत्य;
                भङ्ग;
            }
        }
        प्रति (int64_t i = 0; i < n_actual && shapes_मेलनम्; ++i) {
            ध्रुव स्वचालित & घातांकः = expected_ov_shape[i + सरणम्];
            ध्रुव तार्किक both_dynamic = घातांकः.is_dynamic() && actual_shape[i].is_dynamic();
            ध्रुव तार्किक both_static_equal =
                घातांकः.is_static() && actual_shape[i].is_static() && घातांकः == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            ध्रुव तार्किक expected_dyn_actual_static = घातांकः.is_dynamic() && actual_shape[i].is_static();
            यदि (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_मेलनम् = असत्य;
                भङ्ग;
            }
        }
        यदि (shapes_मेलनम्) {
            प्रत्यागम निवेशः;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    यदि (context.is_static() && view_निवेशः_आकार == 1) {
        स्वचालित view_stride_v = context.get_view_निवेशः_stride(निवेशः_सूचकांक, 0);
        स्वचालित view_src_stride_v = context.get_view_निवेशः_src_stride(निवेशः_सूचकांक, 0);
        स्वचालित view_ggml_shape = context.get_view_निवेशः_ggml_shape(निवेशः_सूचकांक, 0);
        स्वचालित view_src_ggml_shape = context.get_view_निवेशः_src_ggml_shape(निवेशः_सूचकांक, 0);
        स्वचालित view_विस्थापनम् = context.get_view_निवेशः_विस्थापनम्(निवेशः_सूचकांक, 0);
        स्वचालित view_src_विस्थापनम् = context.get_view_निवेशः_src_विस्थापनम्(निवेशः_सूचकांक, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<पूर्णाङ्क> diff_dims;
        यदि (view_src_ggml_shape.size() == ndims) {
            प्रति (size_t i = 0; i < ndims; ++i) {
                यदि (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<पूर्णाङ्क>(i));
                }
            }
        }

        यदि (diff_dims.size() == 1) {
            पूर्णाङ्क विभज_dim = diff_dims[0];
            int64_t अङ्कः_splits = static_cast<int64_t>(view_src_ggml_shape[विभज_dim]);
            int64_t chunk_आकार = static_cast<int64_t>(view_ggml_shape[विभज_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            यदि (chunk_आकार == 1 && अङ्कः_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                तार्किक suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                यदि (suffix_ok) {
                    प्रति (size_t i = static_cast<size_t>(विभज_dim) + 1; i < ndims; ++i) {
                        यदि (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = असत्य;
                            भङ्ग;
                        }
                    }
                }

                यदि (suffix_ok && view_src_stride_v[विभज_dim] > 0) {
                    size_t relative_विस्थापनम् = view_विस्थापनम् >= view_src_विस्थापनम् ? view_विस्थापनम् - view_src_विस्थापनम् : 0;
                    int64_t विभज_सूचकांक = static_cast<int64_t>(relative_विस्थापनम् / view_src_stride_v[विभज_dim]);

                    यदि (विभज_सूचकांक >= 0 && विभज_सूचकांक < अङ्कः_splits) {
                        स्वचालित src_ग्रन्थिः = निवेशः.get_ग्रन्थिः_shared_ptr();
                        std::string rt_कुञ्चिका = "split_dim_" + std::to_string(विभज_dim);
                        स्वचालित & rt_info = src_ग्रन्थिः->get_rt_info();

                        यदि (rt_info.find(rt_कुञ्चिका) == rt_info.end()) {
                            स्वचालित axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(विभज_dim)});
                            स्वचालित विभज_ग्रन्थिः =
                                std::make_shared<ov::op::v1::विभज_2>(निवेशः, axis_const, static_cast<size_t>(अङ्कः_splits));
                            विभज_ग्रन्थिः->समुच्चयः_friendly_name(src_ग्रन्थिः->get_friendly_name() + "_split");
                            rt_info[rt_कुञ्चिका] = विभज_ग्रन्थिः;
                        }

                        स्वचालित विभज_ग्रन्थिः = rt_info[rt_कुञ्चिका].as<std::shared_ptr<ov::op::v1::विभज_2>>();
                        प्रत्यागम विभज_ग्रन्थिः->output(static_cast<size_t>(विभज_सूचकांक));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    स्वचालित process_single_view =
        [](ov::Output<ov::Node> current, size_t view_विस्थापनम्, ध्रुव std::vector<size_t> & view_stride,
           ध्रुव ov::Shape & view_ggml_shape, ध्रुव ov::PartialShape & view_ov_shape, ध्रुव std::string & view_name,
           size_t view_src_विस्थापनम्, ध्रुव std::vector<size_t> & view_src_stride, ध्रुव ov::Shape & view_src_ggml_shape,
           ध्रुव ov::PartialShape & view_src_ov_shape, ध्रुव std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        ध्रुव स्वचालित current_rank = current.get_partial_shape().rank();
        ध्रुव पूर्णाङ्क axis_सरणम् = current_rank.is_static() ?
                                    static_cast<पूर्णाङ्क>(view_stride.size()) - static_cast<पूर्णाङ्क>(current_rank.get_दीर्घता()) :
                                    0;

        स्वचालित build_reshape_आदर्शः = [](ध्रुव ov::PartialShape & target_ov_shape,
                                        ध्रुव ov::Shape & target_ggml_shape) -> std::vector<int64_t> {
            ध्रुव size_t ndims = target_ggml_shape.size();
            std::vector<int64_t> reshape_आदर्शः(ndims);
            size_t dynamic_dims = 0;

            यदि (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_दीर्घता() == static_cast<int64_t>(ndims)) {
                प्रति (size_t i = 0; i < ndims; ++i) {
                    यदि (target_ov_shape[i].is_static()) {
                        reshape_आदर्शः[i] = target_ov_shape[i].get_दीर्घता();
                    } अन्यथा {
                        reshape_आदर्शः[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } अन्यथा {
                dynamic_dims = 2;
            }

            यदि (dynamic_dims > 1) {
                प्रति (size_t i = 0; i < ndims; ++i) {
                    reshape_आदर्शः[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            प्रत्यागम reshape_आदर्शः;
        };

        स्वचालित build_उपसर्गः_पुच्छ_reshape_आदर्शः = [](ध्रुव ov::PartialShape & target_ov_shape,
                                                    ध्रुव ov::Shape & target_ggml_shape, size_t उपसर्गः_dims,
                                                    int64_t पुच्छ_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_आदर्शः(उपसर्गः_dims + 1);
            size_t dynamic_dims = 0;

            यदि (target_ov_shape.rank().is_static() &&
                target_ov_shape.rank().get_दीर्घता() == static_cast<int64_t>(target_ggml_shape.size())) {
                प्रति (size_t i = 0; i < उपसर्गः_dims; ++i) {
                    यदि (target_ov_shape[i].is_static()) {
                        reshape_आदर्शः[i] = target_ov_shape[i].get_दीर्घता();
                    } अन्यथा {
                        reshape_आदर्शः[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } अन्यथा {
                dynamic_dims = 2;
            }

            यदि (dynamic_dims > 1) {
                प्रति (size_t i = 0; i < उपसर्गः_dims; ++i) {
                    reshape_आदर्शः[i] = static_cast<int64_t>(target_ggml_shape[i]);
                }
            }

            reshape_आदर्शः[उपसर्गः_dims] = पुच्छ_dim;
            प्रत्यागम reshape_आदर्शः;
        };

        तार्किक same_stride = view_stride.size() == view_src_stride.size();
        यदि (same_stride) {
            प्रति (size_t i = 0; i < view_stride.size(); ++i) {
                यदि (view_stride[i] != view_src_stride[i]) {
                    same_stride = असत्य;
                    भङ्ग;
                }
            }
        }

        तार्किक same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        यदि (same_ggml_shape) {
            प्रति (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                यदि (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = असत्य;
                    भङ्ग;
                }
            }
        }

        यदि (same_stride && same_ggml_shape) {
            प्रत्यागम current;
        }

        यदि (same_stride) {
            ध्रुव size_t relative_विस्थापनम् = view_विस्थापनम् >= view_src_विस्थापनम् ? view_विस्थापनम् - view_src_विस्थापनम् : 0;
            ध्रुव size_t ndims = view_stride.size();

            std::vector<पूर्णाङ्क> diff_dims;
            यदि (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                प्रति (size_t i = 0; i < ndims; ++i) {
                    यदि (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<पूर्णाङ्क>(i));
                    }
                }
            }

            यदि (diff_dims.size() == 1) {
                ध्रुव पूर्णाङ्क खण्डिका_dim = diff_dims[0];
                ध्रुव int64_t dim_आकार = static_cast<int64_t>(view_src_ggml_shape[खण्डिका_dim]);

                यदि (view_stride[खण्डिका_dim] > 0 && relative_विस्थापनम् % view_stride[खण्डिका_dim] == 0) {
                    ध्रुव int64_t begin_val = static_cast<int64_t>((relative_विस्थापनम् / view_stride[खण्डिका_dim]) %
                                                                   static_cast<size_t>(dim_आकार));
                    ध्रुव int64_t अन्तः_val = begin_val + static_cast<int64_t>(view_ggml_shape[खण्डिका_dim]);

                    यदि (begin_val >= 0 && अन्तः_val <= dim_आकार) {
                        स्वचालित sliced = std::make_shared<ov::op::v8::खण्डिका>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {अन्तः_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {खण्डिका_dim - axis_सरणम्}));

                        यदि (view_ov_shape.is_static()) {
                            स्वचालित reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.गन्तव्यम्_shape()),
                                असत्य);
                            reshaped->समुच्चयः_friendly_name(view_name);
                            प्रत्यागम reshaped;
                        }

                        sliced->समुच्चयः_friendly_name(view_name);
                        प्रत्यागम sliced;
                    }
                }

                int64_t पुच्छ_src_elems = 1;
                int64_t पुच्छ_dst_elems = 1;
                प्रति (size_t i = खण्डिका_dim; i < ndims; ++i) {
                    पुच्छ_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    पुच्छ_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                ध्रुव size_t elem_stride = view_stride[ndims - 1];
                int64_t पुच्छ_begin = 0;
                यदि (elem_stride > 0) {
                    पुच्छ_begin =
                        static_cast<int64_t>((relative_विस्थापनम् / elem_stride) % static_cast<size_t>(पुच्छ_src_elems));
                }
                ध्रुव int64_t पुच्छ_अन्तः = पुच्छ_begin + पुच्छ_dst_elems;

                यदि (पुच्छ_begin >= 0 && पुच्छ_अन्तः <= पुच्छ_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(खण्डिका_dim);
                    प्रति (पूर्णाङ्क i = 0; i < खण्डिका_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(पुच्छ_src_elems);
                    ध्रुव size_t flat_ndims = flat_shape.size();

                    स्वचालित flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), असत्य);

                    स्वचालित sliced = std::make_shared<ov::op::v8::खण्डिका>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {पुच्छ_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {पुच्छ_अन्तः}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {खण्डिका_dim}));

                    यदि (view_ov_shape.is_static()) {
                        स्वचालित reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.गन्तव्यम्_shape()),
                            असत्य);
                        reshaped->समुच्चयः_friendly_name(view_name);
                        प्रत्यागम reshaped;
                    }

                    sliced->समुच्चयः_friendly_name(view_name);
                    प्रत्यागम sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> चरणः(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t remaining_विस्थापनम् = relative_विस्थापनम्;
            प्रति (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                यदि (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(remaining_विस्थापनम् / view_stride[i]);
                    remaining_विस्थापनम् %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            तार्किक अन्तर्_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            यदि (अन्तर्_bounds) {
                प्रति (size_t i = 0; i < ndims; ++i) {
                    यदि (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        अन्तर्_bounds = असत्य;
                        भङ्ग;
                    }
                }
            }

            यदि (अन्तर्_bounds && remaining_विस्थापनम् == 0) {
                स्वचालित sliced = std::make_shared<ov::op::v8::खण्डिका>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, चरणः),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->समुच्चयः_friendly_name(view_name);
                प्रत्यागम sliced;
            }
        } अन्यथा {
            तार्किक same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            ध्रुव size_t relative_विस्थापनम् = view_विस्थापनम् >= view_src_विस्थापनम् ? view_विस्थापनम् - view_src_विस्थापनम् : 0;

            यदि (same_rank) {
                ध्रुव size_t ndims = view_ggml_shape.size();
                std::vector<पूर्णाङ्क> diff_dims;
                प्रति (size_t i = 0; i < ndims; ++i) {
                    यदि (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<पूर्णाङ्क>(i));
                    }
                }

                यदि (diff_dims.size() == 1) {
                    ध्रुव size_t खण्डिका_dim = static_cast<size_t>(diff_dims[0]);
                    तार्किक suffix_stride_मेलनम् = सत्य;
                    प्रति (size_t i = खण्डिका_dim + 1; i < ndims; ++i) {
                        यदि (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_मेलनम् = असत्य;
                            भङ्ग;
                        }
                    }

                    यदि (suffix_stride_मेलनम् && view_src_stride[खण्डिका_dim] > 0 &&
                        relative_विस्थापनम् % view_src_stride[खण्डिका_dim] == 0) {
                        ध्रुव int64_t begin_val = static_cast<int64_t>(relative_विस्थापनम् / view_src_stride[खण्डिका_dim]);
                        ध्रुव int64_t अन्तः_val = begin_val + static_cast<int64_t>(view_ggml_shape[खण्डिका_dim]);
                        ध्रुव int64_t dim_आकार = static_cast<int64_t>(view_src_ggml_shape[खण्डिका_dim]);

                        यदि (begin_val >= 0 && अन्तः_val <= dim_आकार) {
                            स्वचालित sliced = std::make_shared<ov::op::v8::खण्डिका>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {अन्तः_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(खण्डिका_dim)}));
                            sliced->समुच्चयः_friendly_name(view_name);
                            प्रत्यागम sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            यदि (same_rank) {
                प्रति (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            तार्किक same_अङ्कः_elements = same_rank && view_elems == src_elems;

            यदि (same_rank && relative_विस्थापनम् == 0 && same_अङ्कः_elements) {
                स्वचालित reshape_आदर्शः = build_reshape_आदर्शः(view_ov_shape, view_ggml_shape);

                स्वचालित reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_आदर्शः.size()}, reshape_आदर्शः),
                    असत्य);
                reshaped->समुच्चयः_friendly_name(view_name);
                प्रत्यागम reshaped;
            }

            यदि (same_rank) {
                ध्रुव size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                तार्किक is_regular_खण्डिका = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> चरणः(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t remaining_विस्थापनम् = relative_विस्थापनम्;

                यदि (is_regular_खण्डिका) {
                    प्रति (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        यदि (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_खण्डिका = असत्य;
                            भङ्ग;
                        }

                        चरणः[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        यदि (चरणः[i] <= 0) {
                            is_regular_खण्डिका = असत्य;
                            भङ्ग;
                        }

                        begin[i] = static_cast<int64_t>(remaining_विस्थापनम् / view_src_stride[i]);
                        remaining_विस्थापनम् %= view_src_stride[i];

                        यदि (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            अनुवर्त;
                        }

                        end[i] = begin[i] + चरणः[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        यदि (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_खण्डिका = असत्य;
                            भङ्ग;
                        }
                    }
                }

                यदि (is_regular_खण्डिका && remaining_विस्थापनम् == 0) {
                    स्वचालित sliced = std::make_shared<ov::op::v8::खण्डिका>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, चरणः),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->समुच्चयः_friendly_name(view_name);
                    प्रत्यागम sliced;
                }

                ध्रुव size_t elem_stride = view_src_stride.back();
                ध्रुव तार्किक aligned_विस्थापनम् = elem_stride > 0 && relative_विस्थापनम् % elem_stride == 0;

                यदि (aligned_विस्थापनम्) {
                    size_t suffix_start = 0;
                    size_t expected_stride = elem_stride;
                    प्रति (पूर्णाङ्क i = static_cast<पूर्णाङ्क>(ndims) - 1; i >= 0; --i) {
                        यदि (view_stride[i] != expected_stride) {
                            suffix_start = static_cast<size_t>(i + 1);
                            भङ्ग;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t उपसर्गः_elems = 1;
                    size_t suffix_elems = 1;
                    प्रति (size_t i = 0; i < suffix_start; ++i) {
                        उपसर्गः_elems *= view_ggml_shape[i];
                    }
                    प्रति (size_t i = suffix_start; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    यदि (उपसर्गः_elems > 0 && src_elems % उपसर्गः_elems == 0) {
                        ध्रुव size_t src_पुच्छ_elems = src_elems / उपसर्गः_elems;
                        ध्रुव int64_t पुच्छ_begin = static_cast<int64_t>(relative_विस्थापनम् / elem_stride);
                        ध्रुव int64_t पुच्छ_अन्तः = पुच्छ_begin + static_cast<int64_t>(suffix_elems);

                        यदि (पुच्छ_begin >= 0 && पुच्छ_अन्तः <= static_cast<int64_t>(src_पुच्छ_elems)) {
                            स्वचालित उपसर्गः_पुच्छ_आदर्शः = build_उपसर्गः_पुच्छ_reshape_आदर्शः(
                                view_ov_shape, view_ggml_shape, suffix_start, static_cast<int64_t>(src_पुच्छ_elems));

                            स्वचालित उपसर्गः_पुच्छ = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {उपसर्गः_पुच्छ_आदर्शः.size()},
                                                             उपसर्गः_पुच्छ_आदर्शः),
                                असत्य);

                            ov::Output<ov::Node> selected = उपसर्गः_पुच्छ;
                            यदि (पुच्छ_begin != 0 || पुच्छ_अन्तः != static_cast<int64_t>(src_पुच्छ_elems)) {
                                selected = std::make_shared<ov::op::v8::खण्डिका>(
                                    उपसर्गः_पुच्छ, ov::op::v0::Constant::create(ov::element::i64, {1}, {पुच्छ_begin}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {पुच्छ_अन्तः}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {static_cast<int64_t>(suffix_start)}));
                            }

                            स्वचालित reshape_आदर्शः = build_reshape_आदर्शः(view_ov_shape, view_ggml_shape);
                            स्वचालित reshaped = std::make_shared<ov::op::v1::Reshape>(
                                selected,
                                ov::op::v0::Constant::create(ov::element::i64, {reshape_आदर्शः.size()},
                                                             reshape_आदर्शः),
                                असत्य);
                            reshaped->समुच्चयः_friendly_name(view_name);
                            प्रत्यागम reshaped;
                        }
                    }
                }
            }

            प्रत्यागम current;
        }

        (शून्य) view_name;
        (शून्य) view_src_ov_shape;
        (शून्य) view_src_name;

        प्रत्यागम current;
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
        स्वचालित innermost_विस्थापनम् = context.get_view_निवेशः_विस्थापनम्(निवेशः_सूचकांक, 0);
        स्वचालित innermost_src_विस्थापनम् = context.get_view_निवेशः_src_विस्थापनम्(निवेशः_सूचकांक, 0);
        स्वचालित innermost_shape = context.get_view_निवेशः_ggml_shape(निवेशः_सूचकांक, 0);
        स्वचालित innermost_src_shape = context.get_view_निवेशः_src_ggml_shape(निवेशः_सूचकांक, 0);
        यदि (innermost_विस्थापनम् != innermost_src_विस्थापनम् && innermost_shape == innermost_src_shape) {
            size_t मूलम्_view_idx = view_निवेशः_आकार - 1;
            स्वचालित मूलम्_ggml_shape = context.get_view_निवेशः_src_ggml_shape(निवेशः_सूचकांक, मूलम्_view_idx);
            स्वचालित मूलम्_stride = context.get_view_निवेशः_src_stride(निवेशः_सूचकांक, मूलम्_view_idx);
            स्वचालित मूलम्_विस्थापनम् = context.get_view_निवेशः_src_विस्थापनम्(निवेशः_सूचकांक, मूलम्_view_idx);
            स्वचालित मूलम्_ov_shape = context.get_view_निवेशः_src_ov_shape(निवेशः_सूचकांक, मूलम्_view_idx);
            स्वचालित मूलम्_name = context.get_view_निवेशः_src_name(निवेशः_सूचकांक, मूलम्_view_idx);
            स्वचालित innermost_stride = context.get_view_निवेशः_stride(निवेशः_सूचकांक, 0);
            स्वचालित innermost_ov_shape = context.get_view_निवेशः_ov_shape(निवेशः_सूचकांक, 0);
            स्वचालित innermost_name = context.get_view_निवेशः_name(निवेशः_सूचकांक, 0);

            प्रत्यागम process_single_view(निवेशः, innermost_विस्थापनम्, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, मूलम्_विस्थापनम्, मूलम्_stride, मूलम्_ggml_shape, मूलम्_ov_shape,
                                       मूलम्_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = निवेशः;

    // Process each view in reverse order (from base to current)
    प्रति (पूर्णाङ्क view_idx = view_निवेशः_आकार - 1; view_idx >= 0; view_idx--) {
        स्वचालित view_विस्थापनम् = context.get_view_निवेशः_विस्थापनम्(निवेशः_सूचकांक, view_idx);
        स्वचालित view_stride = context.get_view_निवेशः_stride(निवेशः_सूचकांक, view_idx);
        स्वचालित view_ggml_shape = context.get_view_निवेशः_ggml_shape(निवेशः_सूचकांक, view_idx);
        स्वचालित view_ov_shape = context.get_view_निवेशः_ov_shape(निवेशः_सूचकांक, view_idx);
        स्वचालित view_name = context.get_view_निवेशः_name(निवेशः_सूचकांक, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        स्वचालित view_src_विस्थापनम् = context.get_view_निवेशः_src_विस्थापनम्(निवेशः_सूचकांक, view_idx);
        स्वचालित view_src_stride = context.get_view_निवेशः_src_stride(निवेशः_सूचकांक, view_idx);
        स्वचालित view_src_ggml_shape = context.get_view_निवेशः_src_ggml_shape(निवेशः_सूचकांक, view_idx);
        स्वचालित view_src_ov_shape = context.get_view_निवेशः_src_ov_shape(निवेशः_सूचकांक, view_idx);
        स्वचालित view_src_name = context.get_view_निवेशः_src_name(निवेशः_सूचकांक, view_idx);
        // print source view info
        // std::cout << "View " << view_idx << ": source name = " << view_src_name
        //           << ", source offset = " << view_src_offset << ", source stride = [" << view_src_stride[0] << ","
        //           << view_src_stride[1] << "," << view_src_stride[2] << "," << view_src_stride[3]
        //           << "], source ggml shape = [" << view_src_ggml_shape[0] << "," << view_src_ggml_shape[1] << ","
        //           << view_src_ggml_shape[2] << "," << view_src_ggml_shape[3]
        //           << "], source ov shape = " << view_src_ov_shape << std::endl;

        current = process_single_view(current, view_विस्थापनम्, view_stride, view_ggml_shape, view_ov_shape, view_name,
                                      view_src_विस्थापनम्, view_src_stride, view_src_ggml_shape, view_src_ov_shape,
                                      view_src_name);
    }

    प्रत्यागम current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
