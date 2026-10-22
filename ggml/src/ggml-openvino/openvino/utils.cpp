#ضمن "utils.h"

#ضمن <cmath>
#ضمن <cstddef>
#ضمن <ctime>
#ضمن <memory>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/clamp.hpp>
#ضمن <openvino/op/convert.hpp>
#ضمن <openvino/op/cos.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/gather.hpp>
#ضمن <openvino/op/maximum.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/shape_of.hpp>
#ضمن <openvino/op/sin.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/split.hpp>
#ضمن <openvino/op/squeeze.hpp>
#ضمن <openvino/op/subtract.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <string>

نطاق ov {
نطاق frontend {
نطاق ggml {

فراغ عدد_inputs_check(ثابت غوټه_Context & context, size_t min_inputs, size_t max_inputs) {
    خپلکاری ننوت_اندازه = context.get_ننوت_اندازه();
    FRONT_پای_ټکی_OP_CONVERSION_CHECK(ننوت_اندازه >= min_inputs, "Got less inputs than expected");
    FRONT_پای_ټکی_OP_CONVERSION_CHECK(ننوت_اندازه <= max_inputs, "Got more inputs than expected");
}

صحيح non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb) {
    صحيح dim = nb.size() - 1;
    size_t بایټونه = nb[dim];
    لكل (صحيح i = dim; i > 0; i--) {
        بایټونه *= ne[i];
        إذا (بایټونه != nb[i - 1]) {
            أرجع i;
        }
    }
    أرجع 0;
}

std::shared_ptr<ov::Node> get_dimensions(ثابت std::shared_ptr<ov::op::v3::Shape_د> & shape,
                                         ثابت std::vector<صحيح> & dims) {
    کارول نطاق ov::op;
    ثابت خپلکاری zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ثابت خپلکاری dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    أرجع std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(ثابت std::shared_ptr<ov::Node> & غوټه, ثابت std::vector<صحيح> & dims) {
    أرجع get_dimensions(std::make_shared<ov::op::v3::Shape_د>(غوټه), dims);
}

Output_ویکټور rename_outputs_with_suffix(ثابت Output_ویکټور & outputs, ثابت std::string & suffix) {
    لكل (ثابت خپلکاری & output : outputs) {
        خپلکاری غوټه = output.get_غوټه_shared_ptr();
        std::string name = غوټه->get_friendly_name();
        name += "_";
        name += suffix;
        غوټه->سېټ_friendly_name(name);
        // Uncomment to dump every node's inferred shape (used to hunt down dynamic dims on NPU).
        // std::cout << name << "  " << output.get_partial_shape() << std::endl;
    }
    أرجع outputs;
}

نطاق {
ov::Output<ov::Node> rope_yarn_ramp_mix(صحيح عدد_elements, ثابت عائم corr_dims[2], عائم ext_ضریب) {
    std::vector<عائم> dim_ids_vec(عدد_elements);
    std::iota(dim_ids_vec.begin(), dim_ids_vec.end(), 0.0f);
    خپلکاری dim_ids = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, (size_t) عدد_elements}, dim_ids_vec);
    خپلکاری corr_ټیټ = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[0]});
    خپلکاری corr_لوړ = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {corr_dims[1]});
    خپلکاری denom = std::make_shared<ov::op::v1::Maximum>(
        std::make_shared<ov::op::v1::کمول>(corr_لوړ, corr_ټیټ),
        ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {0.001f}));
    خپلکاری ramp_y =
        std::make_shared<ov::op::v1::Divide>(std::make_shared<ov::op::v1::کمول>(dim_ids, corr_ټیټ), denom);
    خپلکاری ramp_clamped = std::make_shared<ov::op::v0::Clamp>(ramp_y, 0.0f, 1.0f);
    // rope_yarn_ramp returns (1 - clamp(y)), so invert before scaling
    خپلکاری one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
    خپلکاری ramp_inverted = std::make_shared<ov::op::v1::کمول>(one, ramp_clamped);
    خپلکاری ext_ضریب_غوټه = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {ext_ضریب});
    خپلکاری ramp_mix = std::make_shared<ov::op::v1::Multiply>(ramp_inverted, ext_ضریب_غوټه);
    أرجع ramp_mix;
}

عائم ggml_rope_yarn_corr_dim(صحيح n_dims, صحيح n_ctx_orig, عائم n_rot, عائم بنسټ) {
#که_نه_تعریف_شوی M_PI
#    define M_PI 3.14159265358979323846
#د_که_پای
    أرجع n_dims * logf(n_ctx_orig / (n_rot * 2 * (عائم) M_PI)) / (2 * logf(بنسټ));
}

فراغ ggml_rope_yarn_corr_dims(صحيح n_dims,
                              صحيح n_ctx_orig,
                              عائم freq_بنسټ,
                              عائم beta_fast,
                              عائم beta_slow,
                              عائم dims[2]) {
    عائم start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_بنسټ));
    عائم end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_بنسټ));
    dims[0] = std::max(0.0f, start);
    dims[1] = std::min(static_cast<عائم>(n_dims - 1), end);
}
}  // namespace

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_وزن,
                                                           صحيح mode,
                                                           منطقي stateful,
                                                           int64_t سر_dim) {
    constexpr صحيح TYPE_IMROPE = 2;
    constexpr صحيح TYPE_VISION = 3;
    constexpr صحيح TYPE_MROPE = 4;

    ثابت منطقي is_imrope = (mode == TYPE_IMROPE);
    ثابت منطقي is_vision = (mode == TYPE_VISION);
    ثابت منطقي is_mrope = (mode == TYPE_MROPE);
    ثابت منطقي is_multi = is_imrope || is_vision || is_mrope;

    إذا (stateful) {
        inp_pos =
            std::make_shared<ov::op::v0::Squeeze>(inp_pos, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        خپلکاری pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{2, 1, 0});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    } وإلا إذا (is_multi) {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        خپلکاری pos_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, 4, -1});
        inp_pos = std::make_shared<ov::op::v1::Reshape>(inp_pos, pos_shape, ریښتیا);
        خپلکاری pos_transpose_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 1, 2, 4, 3});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_transpose_shape);
    } وإلا {
        inp_pos = std::make_shared<ov::op::v0::Convert>(inp_pos, ov::element::f32);
        خپلکاری pos_perm =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        inp_pos = std::make_shared<ov::op::v1::Transpose>(inp_pos, pos_perm);
    }

    عائم freq_بنسټ;
    عائم freq_scale;
    عائم ext_ضریب;
    عائم attn_ضریب;
    عائم beta_fast;
    عائم beta_slow;
    ثابت صحيح n_dims = rope_params[1];
    ثابت size_t n_dims_half = n_dims >> 1;
    ثابت صحيح n_ctx_orig = rope_params[4];
    حافظه_نقل(&freq_بنسټ, rope_params + 5, حجم(عائم));
    حافظه_نقل(&freq_scale, rope_params + 6, حجم(عائم));
    حافظه_نقل(&ext_ضریب, rope_params + 7, حجم(عائم));
    حافظه_نقل(&attn_ضریب, rope_params + 8, حجم(عائم));
    حافظه_نقل(&beta_fast, rope_params + 9, حجم(عائم));
    حافظه_نقل(&beta_slow, rope_params + 10, حجم(عائم));

    ثابت عائم theta_scale = powf(freq_بنسټ, -2.0f / n_dims);

    Output<Node> theta;
    عائم mscale = attn_ضریب;
    إذا (is_multi) {
        ثابت size_t عدد_جوړې = is_vision ? (n_dims > 0 ? (size_t) n_dims : (size_t) (سر_dim / 2)) : n_dims_half;
        صحيح sections[4];
        حافظه_نقل(sections, rope_params + 11, حجم(صحيح) * 4);
        صحيح sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        إذا (sect_dims <= 0) {
            sect_dims = عدد_جوړې;
        }
        صحيح sec_w = sections[0] + sections[1];
        صحيح sec_e = sec_w + sections[2];
        std::vector<int64_t> gather_indices(عدد_جوړې);
        std::vector<عائم> ضریب(عدد_جوړې);
        لكل (size_t j = 0; j < عدد_جوړې; j++) {
            صحيح sector = j % sect_dims;
            صحيح plane = 0;
            إذا (is_imrope) {
                إذا (sector % 3 == 1 && sector < 3 * sections[1]) {
                    plane = 1;
                } وإلا إذا (sector % 3 == 2 && sector < 3 * sections[2]) {
                    plane = 2;
                } وإلا إذا (sector % 3 == 0 && sector < 3 * sections[0]) {
                    plane = 0;
                } وإلا {
                    plane = 3;
                }
                ضریب[j] = std::توان(theta_scale, j);
            } وإلا إذا (is_vision) {
                صحيح p_idx = 0;
                إذا (sector < sections[0]) {
                    plane = 0;
                    p_idx = sector;
                } وإلا إذا (sector < sections[0] + sections[1]) {
                    plane = 1;
                    p_idx = sector - sections[0];
                } وإلا إذا (sector < sec_w + sections[2]) {
                    plane = 2;
                    p_idx = sector - sec_w;
                } وإلا {
                    plane = 3;
                    p_idx = sector - sec_e;
                }
                ضریب[j] = std::توان(theta_scale, p_idx);
            } وإلا {
                إذا (sector >= sections[0] && sector < sec_w) {
                    plane = 1;
                } وإلا إذا (sector >= sec_w && sector < sec_e) {
                    plane = 2;
                } وإلا إذا (sector >= sec_e) {
                    plane = 3;
                } وإلا {
                    plane = 0;
                }
                ضریب[j] = std::توان(theta_scale, j);
            }
            gather_indices[j] = plane;
        }
        خپلکاری gather_indices_const =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{عدد_جوړې}, gather_indices);
        خپلکاری gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {4});
        inp_pos = std::make_shared<ov::op::v8::Gather>(inp_pos, gather_indices_const, gather_axis);
        Output<Node> ضریب_غوټه =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{عدد_جوړې}, ضریب);
        إذا (rope_freqs_وزن) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::ټوټه>(
                rope_freqs_وزن,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) عدد_جوړې}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_وزن->get_output_partial_shape(0).rank().get_اوږدتیا() - 1}));
            rope_factors = std::make_shared<ov::op::v1::Reshape>(
                rope_factors,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) عدد_جوړې}),
                خطأ);
            ضریب_غوټه = std::make_shared<ov::op::v1::Divide>(ضریب_غوټه, rope_factors);
        }
        خپلکاری theta_extrap = std::make_shared<ov::op::v1::Multiply>(inp_pos, ضریب_غوټه);
        خپلکاری theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));
        إذا (ext_ضریب == 0.0f) {
            theta = theta_interp;
        } وإلا {
            عائم corr_dims[2];
            ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_بنسټ, beta_fast, beta_slow, corr_dims);
            خپلکاری ramp_mix = rope_yarn_ramp_mix(عدد_جوړې, corr_dims, ext_ضریب);
            Output<Node> one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            خپلکاری one_minus_ramp = std::make_shared<ov::op::v1::کمول>(one, ramp_mix);
            theta = std::make_shared<ov::op::v1::جمع_کړه>(
                std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::لوګاریتم(1.0f / freq_scale));
        }
    } وإلا {
        std::vector<عائم> ضریب(n_dims_half);
        Output<Node> freq_factors;
        عائم corr_dims[2];
        ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_بنسټ, beta_fast, beta_slow, corr_dims);
        ضریب[0] = 1.0f;
        لكل (size_t i = 1; i < ضریب.size(); i++) {
            ضریب[i] = theta_scale * ضریب[i - 1];
        }
        إذا (stateful) {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, ضریب.size()}, ضریب);
        } وإلا {
            freq_factors =
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1, ضریب.size()}, ضریب);
        }
        إذا (rope_freqs_وزن) {
            Output<Node> rope_factors = std::make_shared<ov::op::v8::ټوټه>(
                rope_freqs_وزن,
                ov::op::v0::Constant::create(ov::element::i64, {1}, {0}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) n_dims_half}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                ov::op::v0::Constant::create(ov::element::i64, {1}, {rope_freqs_وزن->get_output_partial_shape(0).rank().get_اوږدتیا() - 1}));
            إذا (stateful) {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {3}, {(int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    خطأ);
            } وإلا {
                rope_factors = std::make_shared<ov::op::v1::Reshape>(
                    rope_factors,
                    ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) 1, (int64_t) 1, (int64_t) n_dims_half}),
                    خطأ);
            }
            freq_factors = std::make_shared<ov::op::v1::Divide>(freq_factors, rope_factors);
        }

        خپلکاری theta_extrap = std::make_shared<ov::op::v1::Multiply>(freq_factors, inp_pos);
        خپلکاری theta_interp = std::make_shared<ov::op::v1::Multiply>(
            theta_extrap, ov::op::v0::Constant::create(ov::element::f32, {1}, {freq_scale}));

        إذا (ext_ضریب == 0.0f) {
            theta = theta_interp;
        } وإلا {
            خپلکاری ramp_mix = rope_yarn_ramp_mix(n_dims_half, corr_dims, ext_ضریب);
            Output<Node> one;
            إذا (stateful) {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1}, {1.0f});
            } وإلا {
                one = ov::op::v0::Constant::create(ov::element::f32, Shape{1, 1, 1, 1}, {1.0f});
            }
            خپلکاری one_minus_ramp = std::make_shared<ov::op::v1::کمول>(one, ramp_mix);

            theta =
                std::make_shared<ov::op::v1::جمع_کړه>(std::make_shared<ov::op::v1::Multiply>(theta_interp, one_minus_ramp),
                                                  std::make_shared<ov::op::v1::Multiply>(theta_extrap, ramp_mix));
            mscale *= (1.0f + 0.1f * std::لوګاریتم(1.0f / freq_scale));
        }
    }

    Output<Node> cos_theta = std::make_shared<ov::op::v0::Cos>(theta);
    Output<Node> sin_theta = std::make_shared<ov::op::v0::Sin>(theta);

    إذا (mscale != 1.0f) {
        خپلکاری mscale_غوټه = ov::op::v0::Constant::create(ov::element::f32, Shape{}, {mscale});

        cos_theta = std::make_shared<ov::op::v1::Multiply>(cos_theta, mscale_غوټه);
        sin_theta = std::make_shared<ov::op::v1::Multiply>(sin_theta, mscale_غوټه);
    }

    أرجع std::make_pair(sin_theta, cos_theta);
}

ov::Output<ov::Node> process_view_ننوت(ثابت غوټه_Context & context, صحيح ننوت_شاخص, صحيح ټوټه_len, صحيح axis) {
    // Only works for VIEW operations that does a non-strided slice with optinal reshape on the slice result.
    // The function only does the slice part, the reshape (if any) should be handled by the caller.
    // Default axis is -1, which means slicing the last dimension.
    // If the VIEW reshapes the result, `slice_len` should be provided
    خپلکاری ننوت = context.get_ننوت(ننوت_شاخص);
    خپلکاری * op_params = (size_t *) context.get_ننوت_op_params(ننوت_شاخص);
    خپلکاری src_stride = context.get_ننوت_stride(ننوت_شاخص);

    int64_t ټوټه_پیل = op_params[0] / src_stride[3];
    إذا (ټوټه_len == 0) {
        ټوټه_len = context.get_ننوت_shape(ننوت_شاخص)[3].get_اوږدتیا();
    }
    int64_t ټوټه_پای_ټکی = ټوټه_پیل + ټوټه_len;

    خپلکاری begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {ټوټه_پیل});
    خپلکاری end = ov::op::v0::Constant::create(ov::element::i64, {1}, {ټوټه_پای_ټکی});
    خپلکاری stride = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    ov::Output<ov::Node> axes;
    إذا (axis == -1) {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {context.is_stateful() ? 2 : 3});
    } وإلا {
        axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
    }
    خپلکاری sliced = std::make_shared<ov::op::v8::ټوټه>(ننوت, begin, end, stride, axes);
    أرجع sliced;
}

ov::Output<ov::Node> process_view_ننوت_new(ثابت غوټه_Context & context, صحيح ننوت_شاخص) {
    خپلکاری ننوت = context.get_ننوت(ننوت_شاخص);

    // Check if this input has view inputs
    size_t view_ننوت_اندازه = context.get_view_ننوت_اندازه(ننوت_شاخص);
    إذا (view_ننوت_اندازه == 0) {
        // No view inputs, return the input as is
        أرجع ننوت;
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
    خپلکاری expected_ov_shape = context.get_view_ننوت_ov_shape(ننوت_شاخص, 0);
    خپلکاری actual_shape = ننوت.get_partial_shape();
    إذا (expected_ov_shape.rank().is_static() && actual_shape.rank().is_static() &&
        expected_ov_shape.rank().get_اوږدتیا() >= actual_shape.rank().get_اوږدتیا()) {
        ثابت int64_t n_actual = actual_shape.rank().get_اوږدتیا();
        ثابت int64_t ښويدنه = expected_ov_shape.rank().get_اوږدتیا() - n_actual;
        منطقي shapes_سمون = ریښتیا;
        لكل (int64_t i = 0; i < ښويدنه; ++i) {
            إذا (!expected_ov_shape[i].is_static() || expected_ov_shape[i].get_اوږدتیا() != 1) {
                shapes_سمون = خطأ;
                اكسر;
            }
        }
        لكل (int64_t i = 0; i < n_actual && shapes_سمون; ++i) {
            ثابت خپلکاری & توانیز_تابع = expected_ov_shape[i + ښويدنه];
            ثابت منطقي both_dynamic = توانیز_تابع.is_dynamic() && actual_shape[i].is_dynamic();
            ثابت منطقي both_static_equal =
                توانیز_تابع.is_static() && actual_shape[i].is_static() && توانیز_تابع == actual_shape[i];
            // expected dynamic, actual static: the resolved view already carries the
            // concrete size for this fragment; reuse it rather than re-materializing.
            ثابت منطقي expected_dyn_actual_static = توانیز_تابع.is_dynamic() && actual_shape[i].is_static();
            إذا (!both_dynamic && !both_static_equal && !expected_dyn_actual_static) {
                shapes_سمون = خطأ;
                اكسر;
            }
        }
        إذا (shapes_سمون) {
            أرجع ننوت;
        }
    }

    // In static mode, use Split instead of Slice for single-dimension reductions.
    // This ensures NPUW's FOLD doesn't parametrize per-layer slice indices (which
    // would introduce dynamic shapes). A shared Split node sits outside the repeated
    // subgraph boundary; each layer receives one of its output ports.
    إذا (context.is_static() && view_ننوت_اندازه == 1) {
        خپلکاری view_stride_v = context.get_view_ننوت_stride(ننوت_شاخص, 0);
        خپلکاری view_src_stride_v = context.get_view_ننوت_src_stride(ننوت_شاخص, 0);
        خپلکاری view_ggml_shape = context.get_view_ننوت_ggml_shape(ننوت_شاخص, 0);
        خپلکاری view_src_ggml_shape = context.get_view_ننوت_src_ggml_shape(ننوت_شاخص, 0);
        خپلکاری view_دننه_واټن = context.get_view_ننوت_دننه_واټن(ننوت_شاخص, 0);
        خپلکاری view_src_دننه_واټن = context.get_view_ننوت_src_دننه_واټن(ننوت_شاخص, 0);

        size_t ndims = view_ggml_shape.size();
        std::vector<صحيح> diff_dims;
        إذا (view_src_ggml_shape.size() == ndims) {
            لكل (size_t i = 0; i < ndims; ++i) {
                إذا (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    diff_dims.push_back(static_cast<صحيح>(i));
                }
            }
        }

        إذا (diff_dims.size() == 1) {
            صحيح وېشل_dim = diff_dims[0];
            int64_t عدد_splits = static_cast<int64_t>(view_src_ggml_shape[وېشل_dim]);
            int64_t chunk_اندازه = static_cast<int64_t>(view_ggml_shape[وېشل_dim]);

            // Only apply when slicing exactly 1 element from a multi-element dimension
            إذا (chunk_اندازه == 1 && عدد_splits > 1) {
                // Check suffix strides match (dimensions after split_dim)
                منطقي suffix_ok = view_stride_v.size() == view_src_stride_v.size();
                إذا (suffix_ok) {
                    لكل (size_t i = static_cast<size_t>(وېشل_dim) + 1; i < ndims; ++i) {
                        إذا (view_stride_v[i] != view_src_stride_v[i]) {
                            suffix_ok = خطأ;
                            اكسر;
                        }
                    }
                }

                إذا (suffix_ok && view_src_stride_v[وېشل_dim] > 0) {
                    size_t relative_دننه_واټن = view_دننه_واټن >= view_src_دننه_واټن ? view_دننه_واټن - view_src_دننه_واټن : 0;
                    int64_t وېشل_شاخص = static_cast<int64_t>(relative_دننه_واټن / view_src_stride_v[وېشل_dim]);

                    إذا (وېشل_شاخص >= 0 && وېشل_شاخص < عدد_splits) {
                        خپلکاری src_غوټه = ننوت.get_غوټه_shared_ptr();
                        std::string rt_کلی = "split_dim_" + std::to_string(وېشل_dim);
                        خپلکاری & rt_info = src_غوټه->get_rt_info();

                        إذا (rt_info.find(rt_کلی) == rt_info.end()) {
                            خپلکاری axis_const =
                                ov::op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(وېشل_dim)});
                            خپلکاری وېشل_غوټه =
                                std::make_shared<ov::op::v1::وېشل_2>(ننوت, axis_const, static_cast<size_t>(عدد_splits));
                            وېشل_غوټه->سېټ_friendly_name(src_غوټه->get_friendly_name() + "_split");
                            rt_info[rt_کلی] = وېشل_غوټه;
                        }

                        خپلکاری وېشل_غوټه = rt_info[rt_کلی].as<std::shared_ptr<ov::op::v1::وېشل_2>>();
                        أرجع وېشل_غوټه->output(static_cast<size_t>(وېشل_شاخص));
                    }
                }
            }
        }
    }

    // Lambda function to process a single view operation
    خپلکاری process_single_view =
        [](ov::Output<ov::Node> current, size_t view_دننه_واټن, ثابت std::vector<size_t> & view_stride,
           ثابت ov::Shape & view_ggml_shape, ثابت ov::PartialShape & view_ov_shape, ثابت std::string & view_name,
           size_t view_src_دننه_واټن, ثابت std::vector<size_t> & view_src_stride, ثابت ov::Shape & view_src_ggml_shape,
           ثابت ov::PartialShape & view_src_ov_shape, ثابت std::string & view_src_name) -> ov::Output<ov::Node> {
        // Stateful execution drops a leading size-1 axis, so `current` can be one rank lower
        // than the ggml shape metadata (view_stride/view_ggml_shape, always GGML_MAX_DIMS
        // entries) assumes. Shift any axis index built from that metadata down by the
        // difference before using it as an OV Slice axis.
        ثابت خپلکاری current_rank = current.get_partial_shape().rank();
        ثابت صحيح axis_ښويدنه = current_rank.is_static() ?
                                    static_cast<صحيح>(view_stride.size()) - static_cast<صحيح>(current_rank.get_اوږدتیا()) :
                                    0;

        خپلکاری build_reshape_نمونه = [](ثابت ov::PartialShape & موخه_ov_shape,
                                        ثابت ov::Shape & موخه_ggml_shape) -> std::vector<int64_t> {
            ثابت size_t ndims = موخه_ggml_shape.size();
            std::vector<int64_t> reshape_نمونه(ndims);
            size_t dynamic_dims = 0;

            إذا (موخه_ov_shape.rank().is_static() &&
                موخه_ov_shape.rank().get_اوږدتیا() == static_cast<int64_t>(ndims)) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    إذا (موخه_ov_shape[i].is_static()) {
                        reshape_نمونه[i] = موخه_ov_shape[i].get_اوږدتیا();
                    } وإلا {
                        reshape_نمونه[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } وإلا {
                dynamic_dims = 2;
            }

            إذا (dynamic_dims > 1) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    reshape_نمونه[i] = static_cast<int64_t>(موخه_ggml_shape[i]);
                }
            }

            أرجع reshape_نمونه;
        };

        خپلکاری build_مختاړی_لکۍ_reshape_نمونه = [](ثابت ov::PartialShape & موخه_ov_shape,
                                                    ثابت ov::Shape & موخه_ggml_shape, size_t مختاړی_dims,
                                                    int64_t لکۍ_dim) -> std::vector<int64_t> {
            std::vector<int64_t> reshape_نمونه(مختاړی_dims + 1);
            size_t dynamic_dims = 0;

            إذا (موخه_ov_shape.rank().is_static() &&
                موخه_ov_shape.rank().get_اوږدتیا() == static_cast<int64_t>(موخه_ggml_shape.size())) {
                لكل (size_t i = 0; i < مختاړی_dims; ++i) {
                    إذا (موخه_ov_shape[i].is_static()) {
                        reshape_نمونه[i] = موخه_ov_shape[i].get_اوږدتیا();
                    } وإلا {
                        reshape_نمونه[i] = -1;
                        ++dynamic_dims;
                    }
                }
            } وإلا {
                dynamic_dims = 2;
            }

            إذا (dynamic_dims > 1) {
                لكل (size_t i = 0; i < مختاړی_dims; ++i) {
                    reshape_نمونه[i] = static_cast<int64_t>(موخه_ggml_shape[i]);
                }
            }

            reshape_نمونه[مختاړی_dims] = لکۍ_dim;
            أرجع reshape_نمونه;
        };

        منطقي same_stride = view_stride.size() == view_src_stride.size();
        إذا (same_stride) {
            لكل (size_t i = 0; i < view_stride.size(); ++i) {
                إذا (view_stride[i] != view_src_stride[i]) {
                    same_stride = خطأ;
                    اكسر;
                }
            }
        }

        منطقي same_ggml_shape = view_ggml_shape.size() == view_src_ggml_shape.size();
        إذا (same_ggml_shape) {
            لكل (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                إذا (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                    same_ggml_shape = خطأ;
                    اكسر;
                }
            }
        }

        إذا (same_stride && same_ggml_shape) {
            أرجع current;
        }

        إذا (same_stride) {
            ثابت size_t relative_دننه_واټن = view_دننه_واټن >= view_src_دننه_واټن ? view_دننه_واټن - view_src_دننه_واټن : 0;
            ثابت size_t ndims = view_stride.size();

            std::vector<صحيح> diff_dims;
            إذا (view_ggml_shape.size() == ndims && view_src_ggml_shape.size() == ndims) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    إذا (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<صحيح>(i));
                    }
                }
            }

            إذا (diff_dims.size() == 1) {
                ثابت صحيح ټوټه_dim = diff_dims[0];
                ثابت int64_t dim_اندازه = static_cast<int64_t>(view_src_ggml_shape[ټوټه_dim]);

                إذا (view_stride[ټوټه_dim] > 0 && relative_دننه_واټن % view_stride[ټوټه_dim] == 0) {
                    ثابت int64_t begin_val = static_cast<int64_t>((relative_دننه_واټن / view_stride[ټوټه_dim]) %
                                                                   static_cast<size_t>(dim_اندازه));
                    ثابت int64_t پای_ټکی_val = begin_val + static_cast<int64_t>(view_ggml_shape[ټوټه_dim]);

                    إذا (begin_val >= 0 && پای_ټکی_val <= dim_اندازه) {
                        خپلکاری sliced = std::make_shared<ov::op::v8::ټوټه>(
                            current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {پای_ټکی_val}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                            ov::op::v0::Constant::create(ov::element::i64, {1}, {ټوټه_dim - axis_ښويدنه}));

                        إذا (view_ov_shape.is_static()) {
                            خپلکاری reshaped = std::make_shared<ov::op::v1::Reshape>(
                                sliced,
                                ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.منزل_shape()),
                                خطأ);
                            reshaped->سېټ_friendly_name(view_name);
                            أرجع reshaped;
                        }

                        sliced->سېټ_friendly_name(view_name);
                        أرجع sliced;
                    }
                }

                int64_t لکۍ_src_elems = 1;
                int64_t لکۍ_dst_elems = 1;
                لكل (size_t i = ټوټه_dim; i < ndims; ++i) {
                    لکۍ_src_elems *= static_cast<int64_t>(view_src_ggml_shape[i]);
                    لکۍ_dst_elems *= static_cast<int64_t>(view_ggml_shape[i]);
                }

                ثابت size_t elem_stride = view_stride[ndims - 1];
                int64_t لکۍ_begin = 0;
                إذا (elem_stride > 0) {
                    لکۍ_begin =
                        static_cast<int64_t>((relative_دننه_واټن / elem_stride) % static_cast<size_t>(لکۍ_src_elems));
                }
                ثابت int64_t لکۍ_پای_ټکی = لکۍ_begin + لکۍ_dst_elems;

                إذا (لکۍ_begin >= 0 && لکۍ_پای_ټکی <= لکۍ_src_elems) {
                    std::vector<int64_t> flat_shape;
                    flat_shape.reserve(ټوټه_dim);
                    لكل (صحيح i = 0; i < ټوټه_dim; ++i) {
                        flat_shape.push_back(static_cast<int64_t>(view_src_ggml_shape[i]));
                    }
                    flat_shape.push_back(لکۍ_src_elems);
                    ثابت size_t flat_ndims = flat_shape.size();

                    خپلکاری flat = std::make_shared<ov::op::v1::Reshape>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {flat_ndims}, flat_shape), خطأ);

                    خپلکاری sliced = std::make_shared<ov::op::v8::ټوټه>(
                        flat, ov::op::v0::Constant::create(ov::element::i64, {1}, {لکۍ_begin}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {لکۍ_پای_ټکی}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {ټوټه_dim}));

                    إذا (view_ov_shape.is_static()) {
                        خپلکاری reshaped = std::make_shared<ov::op::v1::Reshape>(
                            sliced, ov::op::v0::Constant::create(ov::element::i64, {ndims}, view_ov_shape.منزل_shape()),
                            خطأ);
                        reshaped->سېټ_friendly_name(view_name);
                        أرجع reshaped;
                    }

                    sliced->سېټ_friendly_name(view_name);
                    أرجع sliced;
                }
            }

            std::vector<int64_t> begin(ndims, 0);
            std::vector<int64_t> end(ndims, 0);
            std::vector<int64_t> ګام(ndims, 1);
            std::vector<int64_t> axes(ndims, 0);

            size_t پاتې_شوی_دننه_واټن = relative_دننه_واټن;
            لكل (size_t i = 0; i < ndims; ++i) {
                axes[i] = static_cast<int64_t>(i);
                إذا (view_stride[i] > 0) {
                    begin[i] = static_cast<int64_t>(پاتې_شوی_دننه_واټن / view_stride[i]);
                    پاتې_شوی_دننه_واټن %= view_stride[i];
                }
                end[i] = begin[i] + static_cast<int64_t>(view_ggml_shape[i]);
            }

            منطقي دننه_bounds = view_src_ggml_shape.size() == ndims && view_ggml_shape.size() == ndims;
            إذا (دننه_bounds) {
                لكل (size_t i = 0; i < ndims; ++i) {
                    إذا (end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                        دننه_bounds = خطأ;
                        اكسر;
                    }
                }
            }

            إذا (دننه_bounds && پاتې_شوی_دننه_واټن == 0) {
                خپلکاری sliced = std::make_shared<ov::op::v8::ټوټه>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, ګام),
                    ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                sliced->سېټ_friendly_name(view_name);
                أرجع sliced;
            }
        } وإلا {
            منطقي same_rank = view_stride.size() == view_src_stride.size() &&
                             view_ggml_shape.size() == view_src_ggml_shape.size() &&
                             view_stride.size() == view_ggml_shape.size();
            ثابت size_t relative_دننه_واټن = view_دننه_واټن >= view_src_دننه_واټن ? view_دننه_واټن - view_src_دننه_واټن : 0;

            إذا (same_rank) {
                ثابت size_t ndims = view_ggml_shape.size();
                std::vector<صحيح> diff_dims;
                لكل (size_t i = 0; i < ndims; ++i) {
                    إذا (view_ggml_shape[i] != view_src_ggml_shape[i]) {
                        diff_dims.push_back(static_cast<صحيح>(i));
                    }
                }

                إذا (diff_dims.size() == 1) {
                    ثابت size_t ټوټه_dim = static_cast<size_t>(diff_dims[0]);
                    منطقي suffix_stride_سمون = ریښتیا;
                    لكل (size_t i = ټوټه_dim + 1; i < ndims; ++i) {
                        إذا (view_stride[i] != view_src_stride[i]) {
                            suffix_stride_سمون = خطأ;
                            اكسر;
                        }
                    }

                    إذا (suffix_stride_سمون && view_src_stride[ټوټه_dim] > 0 &&
                        relative_دننه_واټن % view_src_stride[ټوټه_dim] == 0) {
                        ثابت int64_t begin_val = static_cast<int64_t>(relative_دننه_واټن / view_src_stride[ټوټه_dim]);
                        ثابت int64_t پای_ټکی_val = begin_val + static_cast<int64_t>(view_ggml_shape[ټوټه_dim]);
                        ثابت int64_t dim_اندازه = static_cast<int64_t>(view_src_ggml_shape[ټوټه_dim]);

                        إذا (begin_val >= 0 && پای_ټکی_val <= dim_اندازه) {
                            خپلکاری sliced = std::make_shared<ov::op::v8::ټوټه>(
                                current, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {پای_ټکی_val}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(ټوټه_dim)}));
                            sliced->سېټ_friendly_name(view_name);
                            أرجع sliced;
                        }
                    }
                }
            }

            size_t view_elems = 1;
            size_t src_elems = 1;
            إذا (same_rank) {
                لكل (size_t i = 0; i < view_ggml_shape.size(); ++i) {
                    view_elems *= view_ggml_shape[i];
                    src_elems *= view_src_ggml_shape[i];
                }
            }

            منطقي same_عدد_elements = same_rank && view_elems == src_elems;

            إذا (same_rank && relative_دننه_واټن == 0 && same_عدد_elements) {
                خپلکاری reshape_نمونه = build_reshape_نمونه(view_ov_shape, view_ggml_shape);

                خپلکاری reshaped = std::make_shared<ov::op::v1::Reshape>(
                    current, ov::op::v0::Constant::create(ov::element::i64, {reshape_نمونه.size()}, reshape_نمونه),
                    خطأ);
                reshaped->سېټ_friendly_name(view_name);
                أرجع reshaped;
            }

            إذا (same_rank) {
                ثابت size_t ndims = view_ggml_shape.size();

                // Match views that can be expressed as a regular strided slice over the
                // already reconstructed source tensor, e.g. offset on one axis plus step > 1
                // on another axis.
                منطقي is_regular_ټوټه = view_src_ggml_shape.size() == ndims;
                std::vector<int64_t> begin(ndims, 0);
                std::vector<int64_t> end(ndims, 0);
                std::vector<int64_t> ګام(ndims, 1);
                std::vector<int64_t> axes(ndims, 0);
                size_t پاتې_شوی_دننه_واټن = relative_دننه_واټن;

                إذا (is_regular_ټوټه) {
                    لكل (size_t i = 0; i < ndims; ++i) {
                        axes[i] = static_cast<int64_t>(i);

                        إذا (view_src_stride[i] == 0 || view_stride[i] == 0 ||
                            view_stride[i] % view_src_stride[i] != 0) {
                            is_regular_ټوټه = خطأ;
                            اكسر;
                        }

                        ګام[i] = static_cast<int64_t>(view_stride[i] / view_src_stride[i]);
                        إذا (ګام[i] <= 0) {
                            is_regular_ټوټه = خطأ;
                            اكسر;
                        }

                        begin[i] = static_cast<int64_t>(پاتې_شوی_دننه_واټن / view_src_stride[i]);
                        پاتې_شوی_دننه_واټن %= view_src_stride[i];

                        إذا (view_ggml_shape[i] == 0) {
                            end[i] = begin[i];
                            استمر;
                        }

                        end[i] = begin[i] + ګام[i] * static_cast<int64_t>(view_ggml_shape[i] - 1) + 1;

                        إذا (begin[i] < 0 || end[i] > static_cast<int64_t>(view_src_ggml_shape[i])) {
                            is_regular_ټوټه = خطأ;
                            اكسر;
                        }
                    }
                }

                إذا (is_regular_ټوټه && پاتې_شوی_دننه_واټن == 0) {
                    خپلکاری sliced = std::make_shared<ov::op::v8::ټوټه>(
                        current, ov::op::v0::Constant::create(ov::element::i64, {ndims}, begin),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, end),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, ګام),
                        ov::op::v0::Constant::create(ov::element::i64, {ndims}, axes));

                    sliced->سېټ_friendly_name(view_name);
                    أرجع sliced;
                }

                ثابت size_t elem_stride = view_src_stride.back();
                ثابت منطقي aligned_دننه_واټن = elem_stride > 0 && relative_دننه_واټن % elem_stride == 0;

                إذا (aligned_دننه_واټن) {
                    size_t suffix_پیل = 0;
                    size_t expected_stride = elem_stride;
                    لكل (صحيح i = static_cast<صحيح>(ndims) - 1; i >= 0; --i) {
                        إذا (view_stride[i] != expected_stride) {
                            suffix_پیل = static_cast<size_t>(i + 1);
                            اكسر;
                        }
                        expected_stride *= view_ggml_shape[i];
                    }

                    size_t مختاړی_elems = 1;
                    size_t suffix_elems = 1;
                    لكل (size_t i = 0; i < suffix_پیل; ++i) {
                        مختاړی_elems *= view_ggml_shape[i];
                    }
                    لكل (size_t i = suffix_پیل; i < ndims; ++i) {
                        suffix_elems *= view_ggml_shape[i];
                    }

                    إذا (مختاړی_elems > 0 && src_elems % مختاړی_elems == 0) {
                        ثابت size_t src_لکۍ_elems = src_elems / مختاړی_elems;
                        ثابت int64_t لکۍ_begin = static_cast<int64_t>(relative_دننه_واټن / elem_stride);
                        ثابت int64_t لکۍ_پای_ټکی = لکۍ_begin + static_cast<int64_t>(suffix_elems);

                        إذا (لکۍ_begin >= 0 && لکۍ_پای_ټکی <= static_cast<int64_t>(src_لکۍ_elems)) {
                            خپلکاری مختاړی_لکۍ_نمونه = build_مختاړی_لکۍ_reshape_نمونه(
                                view_ov_shape, view_ggml_shape, suffix_پیل, static_cast<int64_t>(src_لکۍ_elems));

                            خپلکاری مختاړی_لکۍ = std::make_shared<ov::op::v1::Reshape>(
                                current,
                                ov::op::v0::Constant::create(ov::element::i64, {مختاړی_لکۍ_نمونه.size()},
                                                             مختاړی_لکۍ_نمونه),
                                خطأ);

                            ov::Output<ov::Node> selected = مختاړی_لکۍ;
                            إذا (لکۍ_begin != 0 || لکۍ_پای_ټکی != static_cast<int64_t>(src_لکۍ_elems)) {
                                selected = std::make_shared<ov::op::v8::ټوټه>(
                                    مختاړی_لکۍ, ov::op::v0::Constant::create(ov::element::i64, {1}, {لکۍ_begin}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {لکۍ_پای_ټکی}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {static_cast<int64_t>(suffix_پیل)}));
                            }

                            خپلکاری reshape_نمونه = build_reshape_نمونه(view_ov_shape, view_ggml_shape);
                            خپلکاری reshaped = std::make_shared<ov::op::v1::Reshape>(
                                selected,
                                ov::op::v0::Constant::create(ov::element::i64, {reshape_نمونه.size()},
                                                             reshape_نمونه),
                                خطأ);
                            reshaped->سېټ_friendly_name(view_name);
                            أرجع reshaped;
                        }
                    }
                }
            }

            أرجع current;
        }

        (فراغ) view_name;
        (فراغ) view_src_ov_shape;
        (فراغ) view_src_name;

        أرجع current;
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
        خپلکاری innermost_دننه_واټن = context.get_view_ننوت_دننه_واټن(ننوت_شاخص, 0);
        خپلکاری innermost_src_دننه_واټن = context.get_view_ننوت_src_دننه_واټن(ننوت_شاخص, 0);
        خپلکاری innermost_shape = context.get_view_ننوت_ggml_shape(ننوت_شاخص, 0);
        خپلکاری innermost_src_shape = context.get_view_ننوت_src_ggml_shape(ننوت_شاخص, 0);
        إذا (innermost_دننه_واټن != innermost_src_دننه_واټن && innermost_shape == innermost_src_shape) {
            size_t ریښه_view_idx = view_ننوت_اندازه - 1;
            خپلکاری ریښه_ggml_shape = context.get_view_ننوت_src_ggml_shape(ننوت_شاخص, ریښه_view_idx);
            خپلکاری ریښه_stride = context.get_view_ننوت_src_stride(ننوت_شاخص, ریښه_view_idx);
            خپلکاری ریښه_دننه_واټن = context.get_view_ننوت_src_دننه_واټن(ننوت_شاخص, ریښه_view_idx);
            خپلکاری ریښه_ov_shape = context.get_view_ننوت_src_ov_shape(ننوت_شاخص, ریښه_view_idx);
            خپلکاری ریښه_name = context.get_view_ننوت_src_name(ننوت_شاخص, ریښه_view_idx);
            خپلکاری innermost_stride = context.get_view_ننوت_stride(ننوت_شاخص, 0);
            خپلکاری innermost_ov_shape = context.get_view_ننوت_ov_shape(ننوت_شاخص, 0);
            خپلکاری innermost_name = context.get_view_ننوت_name(ننوت_شاخص, 0);

            أرجع process_single_view(ننوت, innermost_دننه_واټن, innermost_stride, innermost_shape, innermost_ov_shape,
                                       innermost_name, ریښه_دننه_واټن, ریښه_stride, ریښه_ggml_shape, ریښه_ov_shape,
                                       ریښه_name);
        }
    }

    // Process views from the base tensor (last) to the current view (first)
    // Start with the base tensor
    ov::Output<ov::Node> current = ننوت;

    // Process each view in reverse order (from base to current)
    لكل (صحيح view_idx = view_ننوت_اندازه - 1; view_idx >= 0; view_idx--) {
        خپلکاری view_دننه_واټن = context.get_view_ننوت_دننه_واټن(ننوت_شاخص, view_idx);
        خپلکاری view_stride = context.get_view_ننوت_stride(ننوت_شاخص, view_idx);
        خپلکاری view_ggml_shape = context.get_view_ننوت_ggml_shape(ننوت_شاخص, view_idx);
        خپلکاری view_ov_shape = context.get_view_ننوت_ov_shape(ننوت_شاخص, view_idx);
        خپلکاری view_name = context.get_view_ننوت_name(ننوت_شاخص, view_idx);

        // print view info
        // std::cout << "View " << view_idx << ": name = " << view_name << ", offset = " << view_offset << ", stride = ["
        //       << view_stride[0] << "," << view_stride[1] << "," << view_stride[2] << "," << view_stride[3]
        //       << "], ggml shape = [" << view_ggml_shape[0] << "," << view_ggml_shape[1] << ","
        //       << view_ggml_shape[2] << "," << view_ggml_shape[3] << "], ov shape = " << view_ov_shape << std::endl;

        خپلکاری view_src_دننه_واټن = context.get_view_ننوت_src_دننه_واټن(ننوت_شاخص, view_idx);
        خپلکاری view_src_stride = context.get_view_ننوت_src_stride(ننوت_شاخص, view_idx);
        خپلکاری view_src_ggml_shape = context.get_view_ننوت_src_ggml_shape(ننوت_شاخص, view_idx);
        خپلکاری view_src_ov_shape = context.get_view_ننوت_src_ov_shape(ننوت_شاخص, view_idx);
        خپلکاری view_src_name = context.get_view_ننوت_src_name(ننوت_شاخص, view_idx);
        // print source view info
        // std::cout << "View " << view_idx << ": source name = " << view_src_name
        //           << ", source offset = " << view_src_offset << ", source stride = [" << view_src_stride[0] << ","
        //           << view_src_stride[1] << "," << view_src_stride[2] << "," << view_src_stride[3]
        //           << "], source ggml shape = [" << view_src_ggml_shape[0] << "," << view_src_ggml_shape[1] << ","
        //           << view_src_ggml_shape[2] << "," << view_src_ggml_shape[3]
        //           << "], source ov shape = " << view_src_ov_shape << std::endl;

        current = process_single_view(current, view_دننه_واټن, view_stride, view_ggml_shape, view_ov_shape, view_name,
                                      view_src_دننه_واټن, view_src_stride, view_src_ggml_shape, view_src_ov_shape,
                                      view_src_name);
    }

    أرجع current;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
