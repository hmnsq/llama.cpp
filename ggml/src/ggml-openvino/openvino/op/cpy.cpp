#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <climits>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <numeric>
#ಅನ್ತರ್ಭಾವ <openvino/frontend/exception.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/convert.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/gather.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/negative.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/range.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/scatter_update.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/shape_of.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/squeeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/subtract.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_cpy(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಸ್ವಯಂಚಾಲಿತ op_case = context.get_op_case();
    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_shape = context.get_ಒಳಹರಿವು_shape(0);
    ಸ್ವಯಂಚಾಲಿತ output_shape = context.get_ಒಳಹರಿವು_shape(1);

    ಆದರೆ (op_case == 4) {
        ಸ್ವಯಂಚಾಲಿತ src = process_view_ಒಳಹರಿವು_new(context, 0);
        ಸ್ವಯಂಚಾಲಿತ ತಳ = context.get_ಒಳಹರಿವು(1);

        int64_t n_elems = 1;
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & dim : context.get_output_shape().ಗಮ್ಯ_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_stride = context.get_output_stride();
        ಸ್ಥಿರಾಂಕ size_t elem_ಗಾತ್ರ = output_stride.ಖಾಲಿಗಳು() ? context.get_output_type().size() : output_stride.back();
        FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(elem_ಗಾತ್ರ > 0, "CPY conv state view update has invalid element size");

        ಸ್ಥಿರಾಂಕ int64_t begin_val = static_cast<int64_t>(context.get_output_op_ಸ್ಥಾನಾಂತರ() / elem_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ int64_t ಕೊನೆ_val = begin_val + n_elems;

        ಸ್ವಯಂಚಾಲಿತ flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, ಸುಳ್ಳು);
        ಆದರೆ (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        ಸ್ವಯಂಚಾಲಿತ zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ಸ್ವಯಂಚಾಲಿತ begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        ಸ್ವಯಂಚಾಲಿತ end = ov::op::v0::Constant::create(ov::element::i64, {1}, {ಕೊನೆ_val});
        ಸ್ವಯಂಚಾಲಿತ int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {ಗರಿಷ್ಠ_ಪೂರ್ಣಾಂಕ});
        ಸ್ವಯಂಚಾಲಿತ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ಸ್ವಯಂಚಾಲಿತ axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        ಸ್ವಯಂಚಾಲಿತ ತಲೆ_part = std::make_shared<ov::op::v8::ತುಂಡು>(ತಳ, zero, begin, one, axis);
        ಸ್ವಯಂಚಾಲಿತ ಬಾಲ_part = std::make_shared<ov::op::v8::ತುಂಡು>(ತಳ, end, int_max, one, axis);
        ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{ತಲೆ_part, src, ಬಾಲ_part}, 3);
        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
    }

    // Recurrent state cache writeback into a slot block of the cache. Where the block starts is a
    // runtime input, so the cached model works for any kv head and active sequence count. The
    // result is the full updated cache.
    // op_case 1: gated-delta-net state, op_case 2: conv state, op_case 3: defrag remainder.
    ಆದರೆ (op_case == 3) {
        // With -np 1 (and generally whenever there is no defrag remainder) this GET_ROWS gathers
        // zero rows: nothing to write back, and the cache is unchanged. NPU rejects zero-size
        // tensors, so short-circuit instead of building a degenerate Slice/Concat chain.
        ತಾರ್ಕಿಕ is_ಖಾಲಿಗಳು = ಸುಳ್ಳು;
        ಆದರೆ (ಒಳಹರಿವು_shape.rank().is_static()) {
            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & d : ಒಳಹರಿವು_shape) {
                ಆದರೆ (d.is_static() && d.get_ಉದ್ದಳತೆ() == 0) {
                    is_ಖಾಲಿಗಳು = ನಿಜ;
                    ನಿಲ್ಲಿಸು;
                }
            }
        }
        ಆದರೆ (is_ಖಾಲಿಗಳು) {
            ಹಿಂತಿರುಗಿ {context.get_ಒಳಹರಿವು(1)};
        }
    }
    // op_case 7/8/9 are the single-slot variants; op_case 10 writes native GDN state into a
    // multi-slot cache without rollback snapshots.
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ single_ಸ್ಥಾನ_ನಿಯೋಜಿಸು = op_case >= 7 && op_case <= 9;
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ನೇರ_gdn_state = op_case == 7 || op_case == 10;
    ಪೂರ್ಣಾಂಕ writeback_case = op_case;
    ಆದರೆ (op_case == 10) {
        writeback_case = 1;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (single_ಸ್ಥಾನ_ನಿಯೋಜಿಸು) {
        writeback_case = op_case - 6;
    }
    ಸ್ಥಿರಾಂಕ std::string ಸ್ಥಾನ_begin_name = "rs_slot_begin_" + context.get_name();
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ತುಂಡು_ನಿಯೋಜಿಸು = writeback_case >= 1 && writeback_case <= 3 &&
                              (single_ಸ್ಥಾನ_ನಿಯೋಜಿಸು || context.has_ಒಳಹರಿವು(ಸ್ಥಾನ_begin_name));
    ಆದರೆ (ತುಂಡು_ನಿಯೋಜಿಸು) {
        ಆದರೆ (single_ಸ್ಥಾನ_ನಿಯೋಜಿಸು && writeback_case == 3) {
            ಹಿಂತಿರುಗಿ {context.get_ಒಳಹರಿವು(1)};
        }

        ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನ_axis = 2;
        ಸ್ವಯಂಚಾಲಿತ zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ಸ್ವಯಂಚಾಲಿತ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ಸ್ವಯಂಚಾಲಿತ int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {ಗರಿಷ್ಠ_ಪೂರ್ಣಾಂಕ});
        ಸ್ವಯಂಚಾಲಿತ axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {ಸ್ಥಾನ_axis});
        ಸ್ವಯಂಚಾಲಿತ feature = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                    std::vector<int64_t>{1, 1, -1, output_shape[3].get_ಉದ್ದಳತೆ()});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin;
        ಆದರೆ (!single_ಸ್ಥಾನ_ನಿಯೋಜಿಸು) {
            begin = context.get_ಒಳಹರಿವು(ಸ್ಥಾನ_begin_name);
        }
        ಸ್ವಯಂಚಾಲಿತ ತಳ = context.get_ಒಳಹರಿವು(1);
        ಆದರೆ (writeback_case == 1) {
            ಆದರೆ (ನೇರ_gdn_state) {
                // Non-rollback GDN publishes state directly as [active_slots, heads, value_dim,
                // key_dim]. Flatten each active slot before replacing or updating the cache.
                src = std::make_shared<ov::op::v1::Reshape>(context.get_ಒಳಹರಿವು(0), feature, ಸುಳ್ಳು);
            } ಇಲ್ಲದಿದ್ದರೆ {
                // Multi-slot rollback still consumes GGML's packed [attention | state snapshots]
                // layout. Slice the state block using the runtime source offset.
                ಸ್ವಯಂಚಾಲಿತ src_begin = context.get_ಒಳಹರಿವು("rs_src_begin_" + context.get_name());
                ಸ್ವಯಂಚಾಲಿತ state_part =
                    std::make_shared<ov::op::v8::ತುಂಡು>(context.get_ಒಳಹರಿವು(0), src_begin, int_max, one, axis);
                src = std::make_shared<ov::op::v1::Reshape>(state_part, feature, ಸುಳ್ಳು);
            }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (writeback_case == 2) {
            // conv_input is [previous conv state | new tokens]; the snapshot is the conv_kernel_size - 1
            // columns ending at the last *valid* token. Gather (rather than Slice) keeps the output
            // shape static even though the window start is a runtime value.
            ಸ್ವಯಂಚಾಲಿತ ಕಿಟಕಿ_ಗಾತ್ರ = (int64_t) ಒಳಹರಿವು_shape[3].get_ಉದ್ದಳತೆ();
            ov::Output<ov::Node> ಕಿಟಕಿ;
            ಸ್ವಯಂಚಾಲಿತ ಲಂಬಸಾಲು_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            ಸ್ಥಿರಾಂಕ std::string src_begin_name = "rs_src_begin_" + context.get_name();
            ಆದರೆ (context.has_ಒಳಹರಿವು(src_begin_name)) {
                ಸ್ವಯಂಚಾಲಿತ src_begin = context.get_ಒಳಹರಿವು(src_begin_name);
                ಸ್ವಯಂಚಾಲಿತ src_ಕೊನೆ = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(
                    src_begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {ಕಿಟಕಿ_ಗಾತ್ರ}));
                ಕಿಟಕಿ = std::make_shared<ov::op::v8::ತುಂಡು>(context.get_ಒಳಹರಿವು(0), src_begin, src_ಕೊನೆ, one, ಲಂಬಸಾಲು_axis);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (context.has_ಒಳಹರಿವು("chunk_valid_len")) {
                std::vector<int64_t> offsets(ಕಿಟಕಿ_ಗಾತ್ರ);
                std::iota(offsets.begin(), offsets.end(), 0);
                ಸ್ವಯಂಚಾಲಿತ indices = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(
                    ov::op::v0::Constant::create(ov::element::i64, {(size_t) ಕಿಟಕಿ_ಗಾತ್ರ}, offsets),
                    context.get_ಒಳಹರಿವು("chunk_valid_len"));
                ಕಿಟಕಿ = std::make_shared<ov::op::v8::Gather>(context.get_ಒಳಹರಿವು(0), indices, ಲಂಬಸಾಲು_axis);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ವಯಂಚಾಲಿತ ಕಿಟಕಿ_begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-ಕಿಟಕಿ_ಗಾತ್ರ});
                ಕಿಟಕಿ =
                    std::make_shared<ov::op::v8::ತುಂಡು>(context.get_ಒಳಹರಿವು(0), ಕಿಟಕಿ_begin, int_max, one, ಲಂಬಸಾಲು_axis);
            }
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ ತಳ_shape = ತಳ.get_partial_shape();
            FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(ತಳ_shape.rank().is_static() && ತಳ_shape.rank().get_ಉದ್ದಳತೆ() == 4,
                                          "CPY conv state cache update requires rank-4 base cache");
            FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(ತಳ_shape[3].is_static(),
                                          "CPY conv state cache update requires static feature size");
            FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(ಒಳಹರಿವು_shape.rank().is_static() && ಒಳಹರಿವು_shape.rank().get_ಉದ್ದಳತೆ() == 4 &&
                                              ಒಳಹರಿವು_shape[2].is_static() && ಒಳಹರಿವು_shape[3].is_static(),
                                          "CPY conv state cache update requires static source feature view");

            ಸ್ಥಿರಾಂಕ int64_t ತುಂಬಿದ_feature_ಗಾತ್ರ = ತಳ_shape[3].get_ಉದ್ದಳತೆ();
            ಸ್ಥಿರಾಂಕ int64_t update_feature_ಗಾತ್ರ = ಒಳಹರಿವು_shape[2].get_ಉದ್ದಳತೆ() * ಒಳಹರಿವು_shape[3].get_ಉದ್ದಳತೆ();
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ output_stride = context.get_output_stride();
            ಸ್ಥಿರಾಂಕ size_t elem_ಗಾತ್ರ = output_stride.ಖಾಲಿಗಳು() ? context.get_output_type().size() : output_stride.back();
            FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(elem_ಗಾತ್ರ > 0,
                                          "CPY conv state cache update has invalid element size");
            ಸ್ಥಿರಾಂಕ int64_t feature_begin = static_cast<int64_t>(context.get_output_op_ಸ್ಥಾನಾಂತರ() / elem_ಗಾತ್ರ) %
                                          ತುಂಬಿದ_feature_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ int64_t feature_ಕೊನೆ = feature_begin + update_feature_ಗಾತ್ರ;
            FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(feature_begin >= 0 && feature_ಕೊನೆ <= ತುಂಬಿದ_feature_ಗಾತ್ರ,
                                          "CPY conv state cache update feature range is out of bounds");

            ಸ್ವಯಂಚಾಲಿತ partial_feature = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, update_feature_ಗಾತ್ರ});
            src = std::make_shared<ov::op::v1::Reshape>(ಕಿಟಕಿ, partial_feature, ಸುಳ್ಳು);
            ಆದರೆ (src.get_element_type() != context.get_output_type()) {
                src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
            }

            ಆದರೆ (single_ಸ್ಥಾನ_ನಿಯೋಜಿಸು) {
                ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({src}, context.get_name());
            }

            ಸ್ವಯಂಚಾಲಿತ src_len = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::Shape_ದ>(src, ov::element::i64), axis,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            ಸ್ವಯಂಚಾಲಿತ ಸ್ಥಾನ_ಕೊನೆ = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(begin, src_len);
            ಸ್ವಯಂಚಾಲಿತ ಸಕ್ರಿಯ_slots = std::make_shared<ov::op::v8::ತುಂಡು>(ತಳ, begin, ಸ್ಥಾನ_ಕೊನೆ, one, axis);

            ಸ್ವಯಂಚಾಲಿತ feature_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            ಸ್ವಯಂಚಾಲಿತ feature_begin_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_begin});
            ಸ್ವಯಂಚಾಲಿತ feature_ಕೊನೆ_ಗಂಟು = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_ಕೊನೆ});
            ಸ್ವಯಂಚಾಲಿತ feature_ತಲೆ = std::make_shared<ov::op::v8::ತುಂಡು>(ಸಕ್ರಿಯ_slots, zero, feature_begin_ಗಂಟು, one,
                                                                    feature_axis);
            ಸ್ವಯಂಚಾಲಿತ feature_ಬಾಲ = std::make_shared<ov::op::v8::ತುಂಡು>(ಸಕ್ರಿಯ_slots, feature_ಕೊನೆ_ಗಂಟು, int_max, one,
                                                                    feature_axis);
            src = std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{feature_ತಲೆ, src, feature_ಬಾಲ}, 3);
        } ಇಲ್ಲದಿದ್ದರೆ {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_ಒಳಹರಿವು(0);
        }

        ಆದರೆ (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        ಆದರೆ (single_ಸ್ಥಾನ_ನಿಯೋಜಿಸು) {
            ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({src}, context.get_name());
        }

        ಸ್ವಯಂಚಾಲಿತ src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::Shape_ದ>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        ಸ್ವಯಂಚಾಲಿತ end = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(begin, src_len);
        ಸ್ವಯಂಚಾಲಿತ ತಲೆ_part = std::make_shared<ov::op::v8::ತುಂಡು>(ತಳ, zero, begin, one, axis);
        ಸ್ವಯಂಚಾಲಿತ ಬಾಲ_part = std::make_shared<ov::op::v8::ತುಂಡು>(ತಳ, end, int_max, one, axis);
        ಸ್ವಯಂಚಾಲಿತ res = std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{ತಲೆ_part, src, ಬಾಲ_part}, ಸ್ಥಾನ_axis);
        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
    }

    ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು = process_view_ಒಳಹರಿವು_new(context, 0);

    ಆದರೆ (op_case == 5 || op_case == 6) {
        ಸ್ವಯಂಚಾಲಿತ ಒಳಹರಿವು_shape = context.get_ಒಳಹರಿವು_shape(0);
        ಸ್ವಯಂಚಾಲಿತ output_shape = context.get_output_shape();
        ಸ್ವಯಂಚಾಲಿತ dst_ggml_shape = context.get_view_ಒಳಹರಿವು_ggml_shape(1, 0);
        ಸ್ವಯಂಚಾಲಿತ dst_stride = context.get_view_ಒಳಹರಿವು_stride(1, 0);
        size_t ಸ್ಥಾನಾಂತರ_ಬೈಟುಗಳು = context.get_view_ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ(1, 0);
        ಸ್ವಯಂಚಾಲಿತ n_state = (int64_t) context.get_ಒಳಹರಿವು_shape(0)[3].get_ಉದ್ದಳತೆ();
        ಸ್ವಯಂಚಾಲಿತ n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
        ಸ್ವಯಂಚಾಲಿತ kv_buf = context.get_ಒಳಹರಿವು(1);  // shape {1,1,1,N}

        Output<Node> token_len_ಪ್ರತಿ_seq;
        Output<Node> n_write_dyn;
        ಆದರೆ (context.has_ಒಳಹರಿವು("token_len_per_seq")) {
            token_len_ಪ್ರತಿ_seq = context.get_ಒಳಹರಿವು("token_len_per_seq");
            n_write_dyn = std::make_shared<ov::op::v1::Multiply>(token_len_ಪ್ರತಿ_seq, n_state_c);
        } ಇಲ್ಲದಿದ್ದರೆ {
            n_write_dyn = ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) dst_ggml_shape[3]});
        }
        size_t elem_ಗಾತ್ರ = dst_stride[3];
        FRONT_ಕೊನೆ_OP_CONVERSION_CHECK(elem_ಗಾತ್ರ > 0, "CPY KV cache view update has invalid element size");
        int64_t ಆರಂಭ_elem = (int64_t) (ಸ್ಥಾನಾಂತರ_ಬೈಟುಗಳು / elem_ಗಾತ್ರ);
        // op_case 5: decoder self-attention – write offset advances each step.
        // op_case 6: encoder self-attn or cross-attn – offset fixed at compile time.
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_decoder_self_attn = (op_case == 5);
        ಸ್ವಯಂಚಾಲಿತ ones_c = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, 1});
        ಸ್ವಯಂಚಾಲಿತ new_shape = std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{ones_c, n_write_dyn}, 0);

        ಸ್ವಯಂಚಾಲಿತ reshaped = std::make_shared<ov::op::v1::Reshape>(ಒಳಹರಿವು, new_shape, ಸುಳ್ಳು);
        ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ = std::make_shared<ov::op::v0::Convert>(reshaped, context.get_output_type());
        // Indices [start_elem .. start_elem + n_write) on axis 3 of {1,1,1,N}
        // For decoder self-attention the write offset advances each step, so compute it
        // dynamically from the model inputs: start = (attention_size - token_len_per_seq) * n_state.
        // For encoder self-attn and cross-attn the offset is fixed at graph-compile time.
        ov::Output<ov::Node> start;
        ಆದರೆ (is_decoder_self_attn && context.has_ಒಳಹರಿವು("attention_size") && context.has_ಒಳಹರಿವು("token_len_per_seq")) {
            ಸ್ವಯಂಚಾಲಿತ attention_ಗಾತ್ರ_ಒಳ = context.get_ಒಳಹರಿವು("attention_size");
            ಸ್ವಯಂಚಾಲಿತ token_len_ಒಳ = context.get_ಒಳಹರಿವು("token_len_per_seq");
            ಸ್ವಯಂಚಾಲಿತ past_ಟೋಕನುಗಳು = std::make_shared<ov::op::v1::ಕಳೆ>(attention_ಗಾತ್ರ_ಒಳ, token_len_ಒಳ);
            ಸ್ವಯಂಚಾಲಿತ new_ಆರಂಭ = std::make_shared<ov::op::v1::Multiply>(past_ಟೋಕನುಗಳು, n_state_c);
            start = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(
                new_ಆರಂಭ, ov::op::v0::Constant::create(ov::element::i64, {1}, {ಆರಂಭ_elem}));
        } ಇಲ್ಲದಿದ್ದರೆ {
            start = ov::op::v0::Constant::create(ov::element::i64, {1}, {ಆರಂಭ_elem});
        }
        ಸ್ವಯಂಚಾಲಿತ ಆರಂಭ_squeezed = std::make_shared<ov::op::v0::Squeeze>(start);
        ಸ್ವಯಂಚಾಲಿತ end = std::make_shared<ov::op::v1::ಕೂಡಿಸು>(ಆರಂಭ_squeezed, n_write_dyn);
        ಸ್ವಯಂಚಾಲಿತ ಕೊನೆ_squeezed = std::make_shared<ov::op::v0::Squeeze>(end);
        ಸ್ವಯಂಚಾಲಿತ ಹೆಜ್ಜೆ = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ಸ್ವಯಂಚಾಲಿತ ಹೆಜ್ಜೆ_squeezed = std::make_shared<ov::op::v0::Squeeze>(ಹೆಜ್ಜೆ);
        ಸ್ವಯಂಚಾಲಿತ indices =
            std::make_shared<ov::op::v4::Range>(ಆರಂಭ_squeezed, ಕೊನೆ_squeezed, ಹೆಜ್ಜೆ_squeezed, ov::element::i64);
        ಸ್ವಯಂಚಾಲಿತ axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        ಸ್ವಯಂಚಾಲಿತ kv_updated = std::make_shared<ov::op::v3::ScatterUpdate>(kv_buf, indices, ದತ್ತಾಂಶ, axis);
        ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({kv_updated}, context.get_name());
    }

    ಆದರೆ (ಒಳಹರಿವು_shape != output_shape) {
        ಸ್ವಯಂಚಾಲಿತ new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<size_t>(output_shape.rank().get_ಉದ್ದಳತೆ())}, output_shape.ಗಮ್ಯ_shape());
        ಒಳಹರಿವು = std::make_shared<ov::op::v1::Reshape>(ಒಳಹರಿವು, new_shape, ಸುಳ್ಳು);
    }

    ov::Output<Node> res;
    ಆದರೆ (context.get_ಒಳಹರಿವು_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(ಒಳಹರಿವು, context.get_output_type());
    } ಇಲ್ಲದಿದ್ದರೆ {
        res = ಒಳಹರಿವು;
    }

    ಆದರೆ (res.get_ಗಂಟು_shared_ptr() == context.get_ಒಳಹರಿವು(0).get_ಗಂಟು_shared_ptr()) {
        ಹಿಂತಿರುಗಿ {res};
    }

    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
