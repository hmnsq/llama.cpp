#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <climits>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <numeric>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/frontend/exception.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/convert.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/gather.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/multiply.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/negative.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/range.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/scatter_update.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/shape_of.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/squeeze.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/subtract.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_cpy(ਅਚਲ ਗੰਢ_Context & context) {
    ਸਵੈਚਾਲਿਤ op_case = context.get_op_case();
    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_shape = context.get_ਨਿਵੇਸ਼_shape(0);
    ਸਵੈਚਾਲਿਤ output_shape = context.get_ਨਿਵੇਸ਼_shape(1);

    ਜੇ (op_case == 4) {
        ਸਵੈਚਾਲਿਤ src = process_view_ਨਿਵੇਸ਼_new(context, 0);
        ਸਵੈਚਾਲਿਤ ਅਧਾਰ = context.get_ਨਿਵੇਸ਼(1);

        int64_t n_elems = 1;
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & dim : context.get_output_shape().ਮੰਜ਼ਿਲ_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        ਅਚਲ ਸਵੈਚਾਲਿਤ output_stride = context.get_output_stride();
        ਅਚਲ size_t elem_ਮਾਪ = output_stride.ਖਾਲੀ_ਥਾਵਾਂ() ? context.get_output_type().size() : output_stride.back();
        FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(elem_ਮਾਪ > 0, "CPY conv state view update has invalid element size");

        ਅਚਲ int64_t begin_val = static_cast<int64_t>(context.get_output_op_ਵਿੱਥ() / elem_ਮਾਪ);
        ਅਚਲ int64_t ਸਿਰਾ_val = begin_val + n_elems;

        ਸਵੈਚਾਲਿਤ flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, ਝੂਠ);
        ਜੇ (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        ਸਵੈਚਾਲਿਤ zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ਸਵੈਚਾਲਿਤ begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        ਸਵੈਚਾਲਿਤ end = ov::op::v0::Constant::create(ov::element::i64, {1}, {ਸਿਰਾ_val});
        ਸਵੈਚਾਲਿਤ int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {ਵੱਧ_ਤੋਂ_ਵੱਧ_ਅੰਕ});
        ਸਵੈਚਾਲਿਤ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ਸਵੈਚਾਲਿਤ axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        ਸਵੈਚਾਲਿਤ ਸਿਰ_part = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਅਧਾਰ, zero, begin, one, axis);
        ਸਵੈਚਾਲਿਤ ਪੂਛ_part = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਅਧਾਰ, end, int_max, one, axis);
        ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{ਸਿਰ_part, src, ਪੂਛ_part}, 3);
        ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
    }

    // Recurrent state cache writeback into a slot block of the cache. Where the block starts is a
    // runtime input, so the cached model works for any kv head and active sequence count. The
    // result is the full updated cache.
    // op_case 1: gated-delta-net state, op_case 2: conv state, op_case 3: defrag remainder.
    ਜੇ (op_case == 3) {
        // With -np 1 (and generally whenever there is no defrag remainder) this GET_ROWS gathers
        // zero rows: nothing to write back, and the cache is unchanged. NPU rejects zero-size
        // tensors, so short-circuit instead of building a degenerate Slice/Concat chain.
        ਤਾਰਕਿਕ is_ਖਾਲੀ_ਥਾਵਾਂ = ਝੂਠ;
        ਜੇ (ਨਿਵੇਸ਼_shape.rank().is_static()) {
            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & d : ਨਿਵੇਸ਼_shape) {
                ਜੇ (d.is_static() && d.get_ਲੰਮਾਈ() == 0) {
                    is_ਖਾਲੀ_ਥਾਵਾਂ = ਸੱਚ;
                    ਰੋਕੋ;
                }
            }
        }
        ਜੇ (is_ਖਾਲੀ_ਥਾਵਾਂ) {
            ਵਾਪਸ {context.get_ਨਿਵੇਸ਼(1)};
        }
    }
    // op_case 7/8/9 are the single-slot variants; op_case 10 writes native GDN state into a
    // multi-slot cache without rollback snapshots.
    ਅਚਲ ਤਾਰਕਿਕ single_ਥਾਂ_ਸੌਂਪੋ = op_case >= 7 && op_case <= 9;
    ਅਚਲ ਤਾਰਕਿਕ ਸਿੱਧਾ_gdn_state = op_case == 7 || op_case == 10;
    ਪੂਰਨ writeback_case = op_case;
    ਜੇ (op_case == 10) {
        writeback_case = 1;
    } ਨਹੀਂਤਾਂ ਜੇ (single_ਥਾਂ_ਸੌਂਪੋ) {
        writeback_case = op_case - 6;
    }
    ਅਚਲ std::string ਥਾਂ_begin_name = "rs_slot_begin_" + context.get_name();
    ਅਚਲ ਤਾਰਕਿਕ ਟੁਕੜਾ_ਸੌਂਪੋ = writeback_case >= 1 && writeback_case <= 3 &&
                              (single_ਥਾਂ_ਸੌਂਪੋ || context.has_ਨਿਵੇਸ਼(ਥਾਂ_begin_name));
    ਜੇ (ਟੁਕੜਾ_ਸੌਂਪੋ) {
        ਜੇ (single_ਥਾਂ_ਸੌਂਪੋ && writeback_case == 3) {
            ਵਾਪਸ {context.get_ਨਿਵੇਸ਼(1)};
        }

        ਅਚਲ int64_t ਥਾਂ_axis = 2;
        ਸਵੈਚਾਲਿਤ zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ਸਵੈਚਾਲਿਤ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ਸਵੈਚਾਲਿਤ int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {ਵੱਧ_ਤੋਂ_ਵੱਧ_ਅੰਕ});
        ਸਵੈਚਾਲਿਤ axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {ਥਾਂ_axis});
        ਸਵੈਚਾਲਿਤ feature = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                    std::vector<int64_t>{1, 1, -1, output_shape[3].get_ਲੰਮਾਈ()});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin;
        ਜੇ (!single_ਥਾਂ_ਸੌਂਪੋ) {
            begin = context.get_ਨਿਵੇਸ਼(ਥਾਂ_begin_name);
        }
        ਸਵੈਚਾਲਿਤ ਅਧਾਰ = context.get_ਨਿਵੇਸ਼(1);
        ਜੇ (writeback_case == 1) {
            ਜੇ (ਸਿੱਧਾ_gdn_state) {
                // Non-rollback GDN publishes state directly as [active_slots, heads, value_dim,
                // key_dim]. Flatten each active slot before replacing or updating the cache.
                src = std::make_shared<ov::op::v1::Reshape>(context.get_ਨਿਵੇਸ਼(0), feature, ਝੂਠ);
            } ਨਹੀਂਤਾਂ {
                // Multi-slot rollback still consumes GGML's packed [attention | state snapshots]
                // layout. Slice the state block using the runtime source offset.
                ਸਵੈਚਾਲਿਤ src_begin = context.get_ਨਿਵੇਸ਼("rs_src_begin_" + context.get_name());
                ਸਵੈਚਾਲਿਤ state_part =
                    std::make_shared<ov::op::v8::ਟੁਕੜਾ>(context.get_ਨਿਵੇਸ਼(0), src_begin, int_max, one, axis);
                src = std::make_shared<ov::op::v1::Reshape>(state_part, feature, ਝੂਠ);
            }
        } ਨਹੀਂਤਾਂ ਜੇ (writeback_case == 2) {
            // conv_input is [previous conv state | new tokens]; the snapshot is the conv_kernel_size - 1
            // columns ending at the last *valid* token. Gather (rather than Slice) keeps the output
            // shape static even though the window start is a runtime value.
            ਸਵੈਚਾਲਿਤ ਖਿੜਕੀ_ਮਾਪ = (int64_t) ਨਿਵੇਸ਼_shape[3].get_ਲੰਮਾਈ();
            ov::Output<ov::Node> ਖਿੜਕੀ;
            ਸਵੈਚਾਲਿਤ ਖਾਨਾ_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            ਅਚਲ std::string src_begin_name = "rs_src_begin_" + context.get_name();
            ਜੇ (context.has_ਨਿਵੇਸ਼(src_begin_name)) {
                ਸਵੈਚਾਲਿਤ src_begin = context.get_ਨਿਵੇਸ਼(src_begin_name);
                ਸਵੈਚਾਲਿਤ src_ਸਿਰਾ = std::make_shared<ov::op::v1::ਜੋੜੋ>(
                    src_begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {ਖਿੜਕੀ_ਮਾਪ}));
                ਖਿੜਕੀ = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(context.get_ਨਿਵੇਸ਼(0), src_begin, src_ਸਿਰਾ, one, ਖਾਨਾ_axis);
            } ਨਹੀਂਤਾਂ ਜੇ (context.has_ਨਿਵੇਸ਼("chunk_valid_len")) {
                std::vector<int64_t> offsets(ਖਿੜਕੀ_ਮਾਪ);
                std::iota(offsets.begin(), offsets.end(), 0);
                ਸਵੈਚਾਲਿਤ indices = std::make_shared<ov::op::v1::ਜੋੜੋ>(
                    ov::op::v0::Constant::create(ov::element::i64, {(size_t) ਖਿੜਕੀ_ਮਾਪ}, offsets),
                    context.get_ਨਿਵੇਸ਼("chunk_valid_len"));
                ਖਿੜਕੀ = std::make_shared<ov::op::v8::Gather>(context.get_ਨਿਵੇਸ਼(0), indices, ਖਾਨਾ_axis);
            } ਨਹੀਂਤਾਂ {
                ਸਵੈਚਾਲਿਤ ਖਿੜਕੀ_begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-ਖਿੜਕੀ_ਮਾਪ});
                ਖਿੜਕੀ =
                    std::make_shared<ov::op::v8::ਟੁਕੜਾ>(context.get_ਨਿਵੇਸ਼(0), ਖਿੜਕੀ_begin, int_max, one, ਖਾਨਾ_axis);
            }
            ਅਚਲ ਸਵੈਚਾਲਿਤ ਅਧਾਰ_shape = ਅਧਾਰ.get_partial_shape();
            FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(ਅਧਾਰ_shape.rank().is_static() && ਅਧਾਰ_shape.rank().get_ਲੰਮਾਈ() == 4,
                                          "CPY conv state cache update requires rank-4 base cache");
            FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(ਅਧਾਰ_shape[3].is_static(),
                                          "CPY conv state cache update requires static feature size");
            FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(ਨਿਵੇਸ਼_shape.rank().is_static() && ਨਿਵੇਸ਼_shape.rank().get_ਲੰਮਾਈ() == 4 &&
                                              ਨਿਵੇਸ਼_shape[2].is_static() && ਨਿਵੇਸ਼_shape[3].is_static(),
                                          "CPY conv state cache update requires static source feature view");

            ਅਚਲ int64_t ਭਰਿਆ_feature_ਮਾਪ = ਅਧਾਰ_shape[3].get_ਲੰਮਾਈ();
            ਅਚਲ int64_t update_feature_ਮਾਪ = ਨਿਵੇਸ਼_shape[2].get_ਲੰਮਾਈ() * ਨਿਵੇਸ਼_shape[3].get_ਲੰਮਾਈ();
            ਅਚਲ ਸਵੈਚਾਲਿਤ output_stride = context.get_output_stride();
            ਅਚਲ size_t elem_ਮਾਪ = output_stride.ਖਾਲੀ_ਥਾਵਾਂ() ? context.get_output_type().size() : output_stride.back();
            FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(elem_ਮਾਪ > 0,
                                          "CPY conv state cache update has invalid element size");
            ਅਚਲ int64_t feature_begin = static_cast<int64_t>(context.get_output_op_ਵਿੱਥ() / elem_ਮਾਪ) %
                                          ਭਰਿਆ_feature_ਮਾਪ;
            ਅਚਲ int64_t feature_ਸਿਰਾ = feature_begin + update_feature_ਮਾਪ;
            FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(feature_begin >= 0 && feature_ਸਿਰਾ <= ਭਰਿਆ_feature_ਮਾਪ,
                                          "CPY conv state cache update feature range is out of bounds");

            ਸਵੈਚਾਲਿਤ partial_feature = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, update_feature_ਮਾਪ});
            src = std::make_shared<ov::op::v1::Reshape>(ਖਿੜਕੀ, partial_feature, ਝੂਠ);
            ਜੇ (src.get_element_type() != context.get_output_type()) {
                src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
            }

            ਜੇ (single_ਥਾਂ_ਸੌਂਪੋ) {
                ਵਾਪਸ rename_outputs_with_suffix({src}, context.get_name());
            }

            ਸਵੈਚਾਲਿਤ src_len = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::Shape_ਦਾ>(src, ov::element::i64), axis,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            ਸਵੈਚਾਲਿਤ ਥਾਂ_ਸਿਰਾ = std::make_shared<ov::op::v1::ਜੋੜੋ>(begin, src_len);
            ਸਵੈਚਾਲਿਤ ਸਰਗਰਮ_slots = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਅਧਾਰ, begin, ਥਾਂ_ਸਿਰਾ, one, axis);

            ਸਵੈਚਾਲਿਤ feature_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            ਸਵੈਚਾਲਿਤ feature_begin_ਗੰਢ = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_begin});
            ਸਵੈਚਾਲਿਤ feature_ਸਿਰਾ_ਗੰਢ = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_ਸਿਰਾ});
            ਸਵੈਚਾਲਿਤ feature_ਸਿਰ = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਸਰਗਰਮ_slots, zero, feature_begin_ਗੰਢ, one,
                                                                    feature_axis);
            ਸਵੈਚਾਲਿਤ feature_ਪੂਛ = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਸਰਗਰਮ_slots, feature_ਸਿਰਾ_ਗੰਢ, int_max, one,
                                                                    feature_axis);
            src = std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{feature_ਸਿਰ, src, feature_ਪੂਛ}, 3);
        } ਨਹੀਂਤਾਂ {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_ਨਿਵੇਸ਼(0);
        }

        ਜੇ (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        ਜੇ (single_ਥਾਂ_ਸੌਂਪੋ) {
            ਵਾਪਸ rename_outputs_with_suffix({src}, context.get_name());
        }

        ਸਵੈਚਾਲਿਤ src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::Shape_ਦਾ>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        ਸਵੈਚਾਲਿਤ end = std::make_shared<ov::op::v1::ਜੋੜੋ>(begin, src_len);
        ਸਵੈਚਾਲਿਤ ਸਿਰ_part = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਅਧਾਰ, zero, begin, one, axis);
        ਸਵੈਚਾਲਿਤ ਪੂਛ_part = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(ਅਧਾਰ, end, int_max, one, axis);
        ਸਵੈਚਾਲਿਤ res = std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{ਸਿਰ_part, src, ਪੂਛ_part}, ਥਾਂ_axis);
        ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
    }

    ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼ = process_view_ਨਿਵੇਸ਼_new(context, 0);

    ਜੇ (op_case == 5 || op_case == 6) {
        ਸਵੈਚਾਲਿਤ ਨਿਵੇਸ਼_shape = context.get_ਨਿਵੇਸ਼_shape(0);
        ਸਵੈਚਾਲਿਤ output_shape = context.get_output_shape();
        ਸਵੈਚਾਲਿਤ dst_ggml_shape = context.get_view_ਨਿਵੇਸ਼_ggml_shape(1, 0);
        ਸਵੈਚਾਲਿਤ dst_stride = context.get_view_ਨਿਵੇਸ਼_stride(1, 0);
        size_t ਵਿੱਥ_ਬਾਈਟਾਂ = context.get_view_ਨਿਵੇਸ਼_ਵਿੱਥ(1, 0);
        ਸਵੈਚਾਲਿਤ n_state = (int64_t) context.get_ਨਿਵੇਸ਼_shape(0)[3].get_ਲੰਮਾਈ();
        ਸਵੈਚਾਲਿਤ n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
        ਸਵੈਚਾਲਿਤ kv_buf = context.get_ਨਿਵੇਸ਼(1);  // shape {1,1,1,N}

        Output<Node> token_len_ਪ੍ਰਤੀ_seq;
        Output<Node> n_write_dyn;
        ਜੇ (context.has_ਨਿਵੇਸ਼("token_len_per_seq")) {
            token_len_ਪ੍ਰਤੀ_seq = context.get_ਨਿਵੇਸ਼("token_len_per_seq");
            n_write_dyn = std::make_shared<ov::op::v1::Multiply>(token_len_ਪ੍ਰਤੀ_seq, n_state_c);
        } ਨਹੀਂਤਾਂ {
            n_write_dyn = ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) dst_ggml_shape[3]});
        }
        size_t elem_ਮਾਪ = dst_stride[3];
        FRONT_ਸਿਰਾ_OP_CONVERSION_CHECK(elem_ਮਾਪ > 0, "CPY KV cache view update has invalid element size");
        int64_t ਸ਼ੁਰੂ_elem = (int64_t) (ਵਿੱਥ_ਬਾਈਟਾਂ / elem_ਮਾਪ);
        // op_case 5: decoder self-attention – write offset advances each step.
        // op_case 6: encoder self-attn or cross-attn – offset fixed at compile time.
        ਅਚਲ ਤਾਰਕਿਕ is_decoder_self_attn = (op_case == 5);
        ਸਵੈਚਾਲਿਤ ones_c = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, 1});
        ਸਵੈਚਾਲਿਤ new_shape = std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{ones_c, n_write_dyn}, 0);

        ਸਵੈਚਾਲਿਤ reshaped = std::make_shared<ov::op::v1::Reshape>(ਨਿਵੇਸ਼, new_shape, ਝੂਠ);
        ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ = std::make_shared<ov::op::v0::Convert>(reshaped, context.get_output_type());
        // Indices [start_elem .. start_elem + n_write) on axis 3 of {1,1,1,N}
        // For decoder self-attention the write offset advances each step, so compute it
        // dynamically from the model inputs: start = (attention_size - token_len_per_seq) * n_state.
        // For encoder self-attn and cross-attn the offset is fixed at graph-compile time.
        ov::Output<ov::Node> start;
        ਜੇ (is_decoder_self_attn && context.has_ਨਿਵੇਸ਼("attention_size") && context.has_ਨਿਵੇਸ਼("token_len_per_seq")) {
            ਸਵੈਚਾਲਿਤ attention_ਮਾਪ_ਵਿੱਚ = context.get_ਨਿਵੇਸ਼("attention_size");
            ਸਵੈਚਾਲਿਤ token_len_ਵਿੱਚ = context.get_ਨਿਵੇਸ਼("token_len_per_seq");
            ਸਵੈਚਾਲਿਤ past_ਟੋਕਨ = std::make_shared<ov::op::v1::ਘਟਾਓ>(attention_ਮਾਪ_ਵਿੱਚ, token_len_ਵਿੱਚ);
            ਸਵੈਚਾਲਿਤ new_ਸ਼ੁਰੂ = std::make_shared<ov::op::v1::Multiply>(past_ਟੋਕਨ, n_state_c);
            start = std::make_shared<ov::op::v1::ਜੋੜੋ>(
                new_ਸ਼ੁਰੂ, ov::op::v0::Constant::create(ov::element::i64, {1}, {ਸ਼ੁਰੂ_elem}));
        } ਨਹੀਂਤਾਂ {
            start = ov::op::v0::Constant::create(ov::element::i64, {1}, {ਸ਼ੁਰੂ_elem});
        }
        ਸਵੈਚਾਲਿਤ ਸ਼ੁਰੂ_squeezed = std::make_shared<ov::op::v0::Squeeze>(start);
        ਸਵੈਚਾਲਿਤ end = std::make_shared<ov::op::v1::ਜੋੜੋ>(ਸ਼ੁਰੂ_squeezed, n_write_dyn);
        ਸਵੈਚਾਲਿਤ ਸਿਰਾ_squeezed = std::make_shared<ov::op::v0::Squeeze>(end);
        ਸਵੈਚਾਲਿਤ ਕਦਮ = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        ਸਵੈਚਾਲਿਤ ਕਦਮ_squeezed = std::make_shared<ov::op::v0::Squeeze>(ਕਦਮ);
        ਸਵੈਚਾਲਿਤ indices =
            std::make_shared<ov::op::v4::Range>(ਸ਼ੁਰੂ_squeezed, ਸਿਰਾ_squeezed, ਕਦਮ_squeezed, ov::element::i64);
        ਸਵੈਚਾਲਿਤ axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        ਸਵੈਚਾਲਿਤ kv_updated = std::make_shared<ov::op::v3::ScatterUpdate>(kv_buf, indices, ਅੰਕੜੇ, axis);
        ਵਾਪਸ rename_outputs_with_suffix({kv_updated}, context.get_name());
    }

    ਜੇ (ਨਿਵੇਸ਼_shape != output_shape) {
        ਸਵੈਚਾਲਿਤ new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<size_t>(output_shape.rank().get_ਲੰਮਾਈ())}, output_shape.ਮੰਜ਼ਿਲ_shape());
        ਨਿਵੇਸ਼ = std::make_shared<ov::op::v1::Reshape>(ਨਿਵੇਸ਼, new_shape, ਝੂਠ);
    }

    ov::Output<Node> res;
    ਜੇ (context.get_ਨਿਵੇਸ਼_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(ਨਿਵੇਸ਼, context.get_output_type());
    } ਨਹੀਂਤਾਂ {
        res = ਨਿਵੇਸ਼;
    }

    ਜੇ (res.get_ਗੰਢ_shared_ptr() == context.get_ਨਿਵੇਸ਼(0).get_ਗੰਢ_shared_ptr()) {
        ਵਾਪਸ {res};
    }

    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
