#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/node.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/core/node_output.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/broadcast.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/split.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/subtract.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/variadic_split.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_rope(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 2, 3);

    ಪೂರ್ಣಾಂಕ op_case = context.get_op_case();

    ov::Output<Node> res;

    ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_ಗಂಟು = process_view_ಒಳಹರಿವು_new(context, 0).get_ಗಂಟು_shared_ptr();
    ಸ್ವಯಂಚಾಲಿತ output_shape = context.get_output_shape().ಗಮ್ಯ_shape();
    int32_t * op_params = context.get_output_op_params();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ mode = op_case;
    ಸ್ಥಿರಾಂಕ int64_t ತಲೆ_dim = static_cast<int64_t>(output_shape[3]);
    ಸ್ಥಿರಾಂಕ int64_t configured_n_dims = static_cast<int64_t>(op_params[1]);
    ಸ್ಥಿರಾಂಕ int64_t n_dims = configured_n_dims == 0 ? ತಲೆ_dim : configured_n_dims;
    ಸ್ಥಿರಾಂಕ int64_t n_offs = static_cast<int64_t>(op_params[15]);

    constexpr ಪೂರ್ಣಾಂಕ TYPE_ಸಾಮಾನ್ಯ = 0;
    constexpr ಪೂರ್ಣಾಂಕ TYPE_NEOX = 1;
    constexpr ಪೂರ್ಣಾಂಕ TYPE_IMROPE = 2;
    constexpr ಪೂರ್ಣಾಂಕ TYPE_VISION = 3;
    constexpr ಪೂರ್ಣಾಂಕ TYPE_MROPE = 4;

    Output<Node> cos_theta_ಗಂಟು;
    Output<Node> sin_theta_ಗಂಟು;
    ಆದರೆ (context.has_ಒಳಹರಿವು("rope_cos")) {
        cos_theta_ಗಂಟು = context.get_ಒಳಹರಿವು("rope_cos");
        sin_theta_ಗಂಟು = context.get_ಒಳಹರಿವು("rope_sin");
    } ಇಲ್ಲದಿದ್ದರೆ {
        std::string cache_ಕೀಲಿ = "rope_sin_cos";
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 15; i++) {
            cache_ಕೀಲಿ += "_" + std::to_string(op_params[i]);
        }
        ಆದರೆ (context.get_ಒಳಹರಿವು_ಗಾತ್ರ() == 3) {
            cache_ಕೀಲಿ += "_ff_" + context.get_ಒಳಹರಿವು_ಹೆಸರುಗಳು()[2];
        }
        ಆದರೆ (context.has_ಒಳಹರಿವು(cache_ಕೀಲಿ + "_cos")) {
            cos_theta_ಗಂಟು = context.get_ಒಳಹರಿವು(cache_ಕೀಲಿ + "_cos");
            sin_theta_ಗಂಟು = context.get_ಒಳಹರಿವು(cache_ಕೀಲಿ + "_sin");
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ವಯಂಚಾಲಿತ inp_pos = context.get_ಒಳಹರಿವು(1).get_ಗಂಟು_shared_ptr();
            std::shared_ptr<ov::Node> rope_freqs_ತೂಕ;
            ಆದರೆ (context.get_ಒಳಹರಿವು_ಗಾತ್ರ() == 3) {
                rope_freqs_ತೂಕ = context.get_ಒಳಹರಿವು(2).get_ಗಂಟು_shared_ptr();
            }
            ಸ್ವಯಂಚಾಲಿತ sin_cos = make_sin_cos(op_params, inp_pos, rope_freqs_ತೂಕ, mode, ಸುಳ್ಳು, ತಲೆ_dim);
            sin_theta_ಗಂಟು = sin_cos.ಮೊದಲ;
            cos_theta_ಗಂಟು = sin_cos.second;
            context.put_shared(cache_ಕೀಲಿ + "_cos", cos_theta_ಗಂಟು);
            context.put_shared(cache_ಕೀಲಿ + "_sin", sin_theta_ಗಂಟು);
        }
    }

    ಸ್ವಯಂಚಾಲಿತ output_type = context.get_output_type();
    ಆದರೆ (ದತ್ತಾಂಶ_ಗಂಟು->get_element_type() != ov::element::f32) {
        ದತ್ತಾಂಶ_ಗಂಟು = std::make_shared<ov::op::v0::Convert>(ದತ್ತಾಂಶ_ಗಂಟು, ov::element::f32);
    }

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_rope_dims = (mode == TYPE_VISION) ? (2 * n_dims) : n_dims;
    FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(n_offs >= 0 && (n_offs % 2 == 0),
                                  "ROPE expects non-negative even n_offs");
    FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(n_dims > 0 && ಒಟ್ಟು_rope_dims + n_offs <= ತಲೆ_dim && (n_dims % 2 == 0),
                                  "ROPE expects even n_dims with total_rope_dims + n_offs <= head_dim");

    // RoPEFusionFlux requires rank_equals(4) on x, t_cos and t_sin. The cos/sin
    // tables are already built rank-4 ([1, S, 1, head_size/2]) for both modes. In
    // stateful mode the data arrives rank-3 ([S, n_heads, head_size]), so lift it
    // to rank-4 ([1, S, n_heads, head_size]) here. Stateful RoPE already produced
    // rank-4 output, so downstream attention is unaffected.
    ಆದರೆ (context.is_stateful()) {
        ಸ್ಥಿರಾಂಕ int64_t batch = static_cast<int64_t>(output_shape[0]);
        ಸ್ವಯಂಚಾಲಿತ r4_shape = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{batch, -1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        ದತ್ತಾಂಶ_ಗಂಟು = std::make_shared<ov::op::v1::Reshape>(ದತ್ತಾಂಶ_ಗಂಟು, r4_shape, ಸುಳ್ಳು);
    }
    // For TYPE_NORMAL rope (both stateful and stateless) we emit the Flux-style
    // interleaved pattern below so the GPU plugin's RoPEFusionFlux matcher folds it
    // into ov::op::internal::RoPE.
    ಆದರೆ (mode == TYPE_ಸಾಮಾನ್ಯ) {
        ಸ್ವಯಂಚಾಲಿತ axis_ಕೊನೆಯ = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        ಸ್ವಯಂಚಾಲಿತ ಹೆಜ್ಜೆ_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        ಸ್ಥಿರಾಂಕ int64_t batch = static_cast<int64_t>(output_shape[0]);
        ಸ್ಥಿರಾಂಕ int64_t n_heads = static_cast<int64_t>(output_shape[2]);
        ಸ್ಥಿರಾಂಕ int64_t half = n_dims / 2;
        ಸ್ವಯಂಚಾಲಿತ rot_ಆರಂಭ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
        ಸ್ವಯಂಚಾಲಿತ rot_ಕೊನೆ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
        ಸ್ವಯಂಚಾಲಿತ rot_ದತ್ತಾಂಶ = std::make_shared<ov::op::v8::ತುಂಡು>(ದತ್ತಾಂಶ_ಗಂಟು, rot_ಆರಂಭ, rot_ಕೊನೆ, ಹೆಜ್ಜೆ_one, axis_ಕೊನೆಯ);

        ಸ್ವಯಂಚಾಲಿತ neg_one_f = ov::op::v0::Constant::create(ದತ್ತಾಂಶ_ಗಂಟು->get_element_type(), ov::Shape{}, {-1.0f});

        ಸ್ವಯಂಚಾಲಿತ paired_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, std::vector<int64_t>{batch, -1, n_heads, half, 2});
        ಸ್ವಯಂಚಾಲಿತ x_paired = std::make_shared<ov::op::v1::Reshape>(rot_ದತ್ತಾಂಶ, paired_shape, ಸುಳ್ಳು);

        ಸ್ವಯಂಚಾಲಿತ ವಿಭಜಿಸು_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_ವಿಭಜಿಸು = std::make_shared<ov::op::v1::ವಿಭಜಿಸು_2>(x_paired, ವಿಭಜಿಸು_axis, 2);
        Output<Node> x0 = ದತ್ತಾಂಶ_ವಿಭಜಿಸು->outputs()[0];
        Output<Node> x1 = ದತ್ತಾಂಶ_ವಿಭಜಿಸು->outputs()[1];

        ಸ್ವಯಂಚಾಲಿತ x1_neg = std::make_shared<ov::op::v1::Multiply>(x1, neg_one_f);
        ಸ್ವಯಂಚಾಲಿತ x_rotated_paired = std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{x1_neg, x0}, -1);

        ಸ್ವಯಂಚಾಲಿತ flat_shape =
            ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{batch, -1, n_heads, n_dims});
        ಸ್ವಯಂಚಾಲಿತ x_rotated =
            std::make_shared<ov::op::v1::Reshape>(x_rotated_paired, flat_shape, ಸುಳ್ಳು);

        // Expand cos/sin from [..., n_dims/2] to [..., n_dims] by repeating each
        // entry twice. Use special_zero on the final Reshape so the seq dim passes
        // through dynamically. Final rank is 4 to satisfy the matcher's predicate.
        ಸ್ವಯಂಚಾಲಿತ expand_cos_sin = [&](ಸ್ಥಿರಾಂಕ Output<Node>& cs) {
            ಸ್ವಯಂಚಾಲಿತ cs_unsq = std::make_shared<ov::op::v0::Unsqueeze>(
                cs, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}));
            ಸ್ವಯಂಚಾಲಿತ bcast_ಗುರಿ = ov::op::v0::Constant::create(
                ov::element::i64, {5}, std::vector<int64_t>{1, 1, 1, half, 2});
            ಸ್ವಯಂಚಾಲಿತ bcast = std::make_shared<ov::op::v3::ಪ್ರಸಾರ>(
                cs_unsq, bcast_ಗುರಿ, ov::op::ಪ್ರಸಾರ_Type::BIDIRECTIONAL);
            ಸ್ವಯಂಚಾಲಿತ flat = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, 0, n_dims});
            ಹಿಂತಿರುಗಿ std::make_shared<ov::op::v1::Reshape>(bcast, flat, ನಿಜ);
        };
        Output<Node> cos_ತುಂಬಿದ = expand_cos_sin(cos_theta_ಗಂಟು);
        Output<Node> sin_ತುಂಬಿದ = expand_cos_sin(sin_theta_ಗಂಟು);

        ಸ್ವಯಂಚಾಲಿತ y1 = std::make_shared<ov::op::v1::Multiply>(rot_ದತ್ತಾಂಶ, cos_ತುಂಬಿದ);
        ಸ್ವಯಂಚಾಲಿತ y2 = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin_ತುಂಬಿದ);
        ಸ್ವಯಂಚಾಲಿತ rotated = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(y1, y2);

        ov::Output_ಸದಿಶ concat_parts;
        ಆದರೆ (n_offs > 0) {
            ಸ್ವಯಂಚಾಲಿತ ತಲೆ_ಆರಂಭ = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            ಸ್ವಯಂಚಾಲಿತ ತಲೆ_ಕೊನೆ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            ಸ್ವಯಂಚಾಲಿತ ತಲೆ = std::make_shared<ov::op::v8::ತುಂಡು>(ದತ್ತಾಂಶ_ಗಂಟು, ತಲೆ_ಆರಂಭ, ತಲೆ_ಕೊನೆ, ಹೆಜ್ಜೆ_one, axis_ಕೊನೆಯ);
            concat_parts.push_back(ತಲೆ);
        }
        concat_parts.push_back(rotated);
        ಆದರೆ (n_offs + n_dims < ತಲೆ_dim) {
            ಸ್ವಯಂಚಾಲಿತ ಬಾಲ_ಆರಂಭ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + n_dims});
            ಸ್ವಯಂಚಾಲಿತ ಬಾಲ_ಕೊನೆ = ov::op::v0::Constant::create(ov::element::i64, {1}, {ತಲೆ_dim});
            ಸ್ವಯಂಚಾಲಿತ ಬಾಲ = std::make_shared<ov::op::v8::ತುಂಡು>(ದತ್ತಾಂಶ_ಗಂಟು, ಬಾಲ_ಆರಂಭ, ಬಾಲ_ಕೊನೆ, ಹೆಜ್ಜೆ_one, axis_ಕೊನೆಯ);
            concat_parts.push_back(ಬಾಲ);
        }
        ಆದರೆ (concat_parts.size() == 1) {
            res = rotated;
        } ಇಲ್ಲದಿದ್ದರೆ {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (mode == TYPE_NEOX || mode == TYPE_IMROPE || mode == TYPE_MROPE || mode == TYPE_VISION) {
        ಸ್ಥಿರಾಂಕ int64_t half = (mode == TYPE_VISION) ? n_dims : (n_dims / 2);
        ಸ್ಥಿರಾಂಕ int64_t rot_dims = 2 * half;

        ಆದರೆ (mode != TYPE_NEOX) {
            ಸ್ವಯಂಚಾಲಿತ cos_sin_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
                                                                        std::vector<int64_t>{1, -1, 1, half});
            cos_theta_ಗಂಟು = std::make_shared<ov::op::v1::Reshape>(cos_theta_ಗಂಟು, cos_sin_shape, ನಿಜ);
            sin_theta_ಗಂಟು = std::make_shared<ov::op::v1::Reshape>(sin_theta_ಗಂಟು, cos_sin_shape, ನಿಜ);
        }

        ಸ್ವಯಂಚಾಲಿತ axis_ಕೊನೆಯ = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
        ಸ್ವಯಂಚಾಲಿತ ಹೆಜ್ಜೆ_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        Output<Node> rot_ದತ್ತಾಂಶ = ದತ್ತಾಂಶ_ಗಂಟು;
        ಆದರೆ (n_offs > 0 || n_offs + rot_dims < ತಲೆ_dim) {
            ಸ್ವಯಂಚಾಲಿತ rot_ಆರಂಭ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            ಸ್ವಯಂಚಾಲಿತ rot_ಕೊನೆ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            rot_ದತ್ತಾಂಶ = std::make_shared<ov::op::v8::ತುಂಡು>(ದತ್ತಾಂಶ_ಗಂಟು, rot_ಆರಂಭ, rot_ಕೊನೆ, ಹೆಜ್ಜೆ_one, axis_ಕೊನೆಯ);
        }

        ಸ್ವಯಂಚಾಲಿತ neg_one_f = ov::op::v0::Constant::create(ದತ್ತಾಂಶ_ಗಂಟು->get_element_type(), ov::Shape{}, {-1.0f});

        ಸ್ವಯಂಚಾಲಿತ ವಿಭಜಿಸು_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        ಸ್ವಯಂಚಾಲಿತ ವಿಭಜಿಸು_lengths = ov::op::v0::Constant::create(ov::element::i64, {2}, {half, half});
        ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_ವಿಭಜಿಸು = std::make_shared<ov::op::v1::Variadic_ವಿಭಜಿಸು>(rot_ದತ್ತಾಂಶ, ವಿಭಜಿಸು_axis, ವಿಭಜಿಸು_lengths);
        Output<Node> x1 = ದತ್ತಾಂಶ_ವಿಭಜಿಸು->outputs()[0];
        Output<Node> x2 = ದತ್ತಾಂಶ_ವಿಭಜಿಸು->outputs()[1];

        ಸ್ವಯಂಚಾಲಿತ x2_neg = std::make_shared<ov::op::v1::Multiply>(x2, neg_one_f);
        ಸ್ವಯಂಚಾಲಿತ x_rotate_half = std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{x2_neg, x1}, -1);

        ಸ್ವಯಂಚಾಲಿತ cos_ತುಂಬಿದ = std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{cos_theta_ಗಂಟು, cos_theta_ಗಂಟು}, -1);
        ಸ್ವಯಂಚಾಲಿತ sin_ತುಂಬಿದ = std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{sin_theta_ಗಂಟು, sin_theta_ಗಂಟು}, -1);

        ಸ್ವಯಂಚಾಲಿತ y1 = std::make_shared<ov::op::v1::Multiply>(rot_ದತ್ತಾಂಶ, cos_ತುಂಬಿದ);
        ಸ್ವಯಂಚಾಲಿತ y2 = std::make_shared<ov::op::v1::Multiply>(x_rotate_half, sin_ತುಂಬಿದ);
        ಸ್ವಯಂಚಾಲಿತ rotated = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(y1, y2);

        ov::Output_ಸದಿಶ concat_parts;
        ಆದರೆ (n_offs > 0) {
            ಸ್ವಯಂಚಾಲಿತ ತಲೆ_ಆರಂಭ = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
            ಸ್ವಯಂಚಾಲಿತ ತಲೆ_ಕೊನೆ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs});
            ಸ್ವಯಂಚಾಲಿತ ತಲೆ = std::make_shared<ov::op::v8::ತುಂಡು>(ದತ್ತಾಂಶ_ಗಂಟು, ತಲೆ_ಆರಂಭ, ತಲೆ_ಕೊನೆ, ಹೆಜ್ಜೆ_one, axis_ಕೊನೆಯ);
            concat_parts.push_back(ತಲೆ);
        }
        concat_parts.push_back(rotated);
        ಆದರೆ (n_offs + rot_dims < ತಲೆ_dim) {
            ಸ್ವಯಂಚಾಲಿತ ಬಾಲ_ಆರಂಭ = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_offs + rot_dims});
            ಸ್ವಯಂಚಾಲಿತ ಬಾಲ_ಕೊನೆ = ov::op::v0::Constant::create(ov::element::i64, {1}, {ತಲೆ_dim});
            ಸ್ವಯಂಚಾಲಿತ ಬಾಲ = std::make_shared<ov::op::v8::ತುಂಡು>(ದತ್ತಾಂಶ_ಗಂಟು, ಬಾಲ_ಆರಂಭ, ಬಾಲ_ಕೊನೆ, ಹೆಜ್ಜೆ_one, axis_ಕೊನೆಯ);
            concat_parts.push_back(ಬಾಲ);
        }
        ಆದರೆ (concat_parts.size() == 1) {
            res = rotated;
        } ಇಲ್ಲದಿದ್ದರೆ {
            res = std::make_shared<ov::op::v0::Concat>(concat_parts, -1);
        }
    }

    ಆದರೆ (res.get_element_type() != output_type) {
        res = std::make_shared<ov::op::v0::Convert>(res, output_type);
    }

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
