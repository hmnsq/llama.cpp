#অন্তর্ভাব "../node_context.h"
#অন্তর্ভাব "../op_table.h"
#অন্তর্ভাব "../utils.h"

#অন্তর্ভাব <climits>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <cstdio>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <numeric>
#অন্তর্ভাব <openvino/frontend/exception.hpp>
#অন্তর্ভাব <openvino/op/add.hpp>
#অন্তর্ভাব <openvino/op/concat.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/convert.hpp>
#অন্তর্ভাব <openvino/op/gather.hpp>
#অন্তর্ভাব <openvino/op/multiply.hpp>
#অন্তর্ভাব <openvino/op/negative.hpp>
#অন্তর্ভাব <openvino/op/range.hpp>
#অন্তর্ভাব <openvino/op/reshape.hpp>
#অন্তর্ভাব <openvino/op/scatter_update.hpp>
#অন্তর্ভাব <openvino/op/shape_of.hpp>
#অন্তর্ভাব <openvino/op/slice.hpp>
#অন্তর্ভাব <openvino/op/squeeze.hpp>
#অন্তর্ভাব <openvino/op/subtract.hpp>
#অন্তর্ভাব <vector>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ op {

OutputVector translate_cpy(ধ্রুবক গ্রন্থি_Context & context) {
    auto op_case = context.get_op_case();
    auto প্রবেশ_shape = context.get_প্রবেশ_shape(0);
    auto output_shape = context.get_প্রবেশ_shape(1);

    যদি (op_case == 4) {
        auto src = process_view_প্রবেশ_new(context, 0);
        auto base = context.get_প্রবেশ(1);

        int64_t n_elems = 1;
        জন্য (ধ্রুবক auto & dim : context.get_output_shape().to_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        ধ্রুবক auto output_stride = context.get_output_stride();
        ধ্রুবক size_t elem_আকার = output_stride.empty() ? context.get_output_type().size() : output_stride.back();
        FRONT_END_OP_CONVERSION_CHECK(elem_আকার > 0, "CPY conv state view update has invalid element size");

        ধ্রুবক int64_t begin_val = static_cast<int64_t>(context.get_output_op_offset() / elem_আকার);
        ধ্রুবক int64_t end_val = begin_val + n_elems;

        auto flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, মিথ্যা);
        যদি (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val});
        auto int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        auto শীর্ষ_part = std::make_shared<ov::op::v8::Slice>(base, zero, begin, one, axis);
        auto লেজ_part = std::make_shared<ov::op::v8::Slice>(base, end, int_max, one, axis);
        auto res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{শীর্ষ_part, src, লেজ_part}, 3);
        ফেরত rename_outputs_with_suffix({res}, context.get_name());
    }

    // Recurrent state cache writeback into a slot block of the cache. Where the block starts is a
    // runtime input, so the cached model works for any kv head and active sequence count. The
    // result is the full updated cache.
    // op_case 1: gated-delta-net state, op_case 2: conv state, op_case 3: defrag remainder.
    যদি (op_case == 3) {
        // With -np 1 (and generally whenever there is no defrag remainder) this GET_ROWS gathers
        // zero rows: nothing to write back, and the cache is unchanged. NPU rejects zero-size
        // tensors, so short-circuit instead of building a degenerate Slice/Concat chain.
        যৌক্তিক is_empty = মিথ্যা;
        যদি (প্রবেশ_shape.rank().is_static()) {
            জন্য (ধ্রুবক auto & d : প্রবেশ_shape) {
                যদি (d.is_static() && d.get_দৈর্ঘ্য() == 0) {
                    is_empty = সত্য;
                    বিরতি;
                }
            }
        }
        যদি (is_empty) {
            ফেরত {context.get_প্রবেশ(1)};
        }
    }
    // op_case 7/8/9 are the single-slot variants; op_case 10 writes native GDN state into a
    // multi-slot cache without rollback snapshots.
    ধ্রুবক যৌক্তিক single_slot_assign = op_case >= 7 && op_case <= 9;
    ধ্রুবক যৌক্তিক direct_gdn_state = op_case == 7 || op_case == 10;
    পূর্ণসংখ্যা writeback_case = op_case;
    যদি (op_case == 10) {
        writeback_case = 1;
    } অন্যথা যদি (single_slot_assign) {
        writeback_case = op_case - 6;
    }
    ধ্রুবক std::string slot_begin_name = "rs_slot_begin_" + context.get_name();
    ধ্রুবক যৌক্তিক slice_assign = writeback_case >= 1 && writeback_case <= 3 &&
                              (single_slot_assign || context.has_প্রবেশ(slot_begin_name));
    যদি (slice_assign) {
        যদি (single_slot_assign && writeback_case == 3) {
            ফেরত {context.get_প্রবেশ(1)};
        }

        ধ্রুবক int64_t slot_axis = 2;
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {slot_axis});
        auto feature = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                    std::vector<int64_t>{1, 1, -1, output_shape[3].get_দৈর্ঘ্য()});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin;
        যদি (!single_slot_assign) {
            begin = context.get_প্রবেশ(slot_begin_name);
        }
        auto base = context.get_প্রবেশ(1);
        যদি (writeback_case == 1) {
            যদি (direct_gdn_state) {
                // Non-rollback GDN publishes state directly as [active_slots, heads, value_dim,
                // key_dim]. Flatten each active slot before replacing or updating the cache.
                src = std::make_shared<ov::op::v1::Reshape>(context.get_প্রবেশ(0), feature, মিথ্যা);
            } অন্যথা {
                // Multi-slot rollback still consumes GGML's packed [attention | state snapshots]
                // layout. Slice the state block using the runtime source offset.
                auto src_begin = context.get_প্রবেশ("rs_src_begin_" + context.get_name());
                auto state_part =
                    std::make_shared<ov::op::v8::Slice>(context.get_প্রবেশ(0), src_begin, int_max, one, axis);
                src = std::make_shared<ov::op::v1::Reshape>(state_part, feature, মিথ্যা);
            }
        } অন্যথা যদি (writeback_case == 2) {
            // conv_input is [previous conv state | new tokens]; the snapshot is the conv_kernel_size - 1
            // columns ending at the last *valid* token. Gather (rather than Slice) keeps the output
            // shape static even though the window start is a runtime value.
            auto window_আকার = (int64_t) প্রবেশ_shape[3].get_দৈর্ঘ্য();
            ov::Output<ov::Node> window;
            auto col_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            ধ্রুবক std::string src_begin_name = "rs_src_begin_" + context.get_name();
            যদি (context.has_প্রবেশ(src_begin_name)) {
                auto src_begin = context.get_প্রবেশ(src_begin_name);
                auto src_end = std::make_shared<ov::op::v1::Add>(
                    src_begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {window_আকার}));
                window = std::make_shared<ov::op::v8::Slice>(context.get_প্রবেশ(0), src_begin, src_end, one, col_axis);
            } অন্যথা যদি (context.has_প্রবেশ("chunk_valid_len")) {
                std::vector<int64_t> offsets(window_আকার);
                std::iota(offsets.begin(), offsets.end(), 0);
                auto indices = std::make_shared<ov::op::v1::Add>(
                    ov::op::v0::Constant::create(ov::element::i64, {(size_t) window_আকার}, offsets),
                    context.get_প্রবেশ("chunk_valid_len"));
                window = std::make_shared<ov::op::v8::Gather>(context.get_প্রবেশ(0), indices, col_axis);
            } অন্যথা {
                auto window_begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-window_আকার});
                window =
                    std::make_shared<ov::op::v8::Slice>(context.get_প্রবেশ(0), window_begin, int_max, one, col_axis);
            }
            ধ্রুবক auto base_shape = base.get_partial_shape();
            FRONT_END_OP_CONVERSION_CHECK(base_shape.rank().is_static() && base_shape.rank().get_দৈর্ঘ্য() == 4,
                                          "CPY conv state cache update requires rank-4 base cache");
            FRONT_END_OP_CONVERSION_CHECK(base_shape[3].is_static(),
                                          "CPY conv state cache update requires static feature size");
            FRONT_END_OP_CONVERSION_CHECK(প্রবেশ_shape.rank().is_static() && প্রবেশ_shape.rank().get_দৈর্ঘ্য() == 4 &&
                                              প্রবেশ_shape[2].is_static() && প্রবেশ_shape[3].is_static(),
                                          "CPY conv state cache update requires static source feature view");

            ধ্রুবক int64_t full_feature_আকার = base_shape[3].get_দৈর্ঘ্য();
            ধ্রুবক int64_t update_feature_আকার = প্রবেশ_shape[2].get_দৈর্ঘ্য() * প্রবেশ_shape[3].get_দৈর্ঘ্য();
            ধ্রুবক auto output_stride = context.get_output_stride();
            ধ্রুবক size_t elem_আকার = output_stride.empty() ? context.get_output_type().size() : output_stride.back();
            FRONT_END_OP_CONVERSION_CHECK(elem_আকার > 0,
                                          "CPY conv state cache update has invalid element size");
            ধ্রুবক int64_t feature_begin = static_cast<int64_t>(context.get_output_op_offset() / elem_আকার) %
                                          full_feature_আকার;
            ধ্রুবক int64_t feature_end = feature_begin + update_feature_আকার;
            FRONT_END_OP_CONVERSION_CHECK(feature_begin >= 0 && feature_end <= full_feature_আকার,
                                          "CPY conv state cache update feature range is out of bounds");

            auto partial_feature = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, update_feature_আকার});
            src = std::make_shared<ov::op::v1::Reshape>(window, partial_feature, মিথ্যা);
            যদি (src.get_element_type() != context.get_output_type()) {
                src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
            }

            যদি (single_slot_assign) {
                ফেরত rename_outputs_with_suffix({src}, context.get_name());
            }

            auto src_len = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64), axis,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            auto slot_end = std::make_shared<ov::op::v1::Add>(begin, src_len);
            auto active_slots = std::make_shared<ov::op::v8::Slice>(base, begin, slot_end, one, axis);

            auto feature_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            auto feature_begin_গ্রন্থি = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_begin});
            auto feature_end_গ্রন্থি = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_end});
            auto feature_শীর্ষ = std::make_shared<ov::op::v8::Slice>(active_slots, zero, feature_begin_গ্রন্থি, one,
                                                                    feature_axis);
            auto feature_লেজ = std::make_shared<ov::op::v8::Slice>(active_slots, feature_end_গ্রন্থি, int_max, one,
                                                                    feature_axis);
            src = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{feature_শীর্ষ, src, feature_লেজ}, 3);
        } অন্যথা {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_প্রবেশ(0);
        }

        যদি (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        যদি (single_slot_assign) {
            ফেরত rename_outputs_with_suffix({src}, context.get_name());
        }

        auto src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        auto end = std::make_shared<ov::op::v1::Add>(begin, src_len);
        auto শীর্ষ_part = std::make_shared<ov::op::v8::Slice>(base, zero, begin, one, axis);
        auto লেজ_part = std::make_shared<ov::op::v8::Slice>(base, end, int_max, one, axis);
        auto res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{শীর্ষ_part, src, লেজ_part}, slot_axis);
        ফেরত rename_outputs_with_suffix({res}, context.get_name());
    }

    auto প্রবেশ = process_view_প্রবেশ_new(context, 0);

    যদি (op_case == 5 || op_case == 6) {
        auto প্রবেশ_shape = context.get_প্রবেশ_shape(0);
        auto output_shape = context.get_output_shape();
        auto dst_ggml_shape = context.get_view_প্রবেশ_ggml_shape(1, 0);
        auto dst_stride = context.get_view_প্রবেশ_stride(1, 0);
        size_t offset_bytes = context.get_view_প্রবেশ_offset(1, 0);
        auto n_state = (int64_t) context.get_প্রবেশ_shape(0)[3].get_দৈর্ঘ্য();
        auto n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
        auto kv_buf = context.get_প্রবেশ(1);  // shape {1,1,1,N}

        Output<Node> token_len_per_seq;
        Output<Node> n_write_dyn;
        যদি (context.has_প্রবেশ("token_len_per_seq")) {
            token_len_per_seq = context.get_প্রবেশ("token_len_per_seq");
            n_write_dyn = std::make_shared<ov::op::v1::Multiply>(token_len_per_seq, n_state_c);
        } অন্যথা {
            n_write_dyn = ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) dst_ggml_shape[3]});
        }
        size_t elem_আকার = dst_stride[3];
        FRONT_END_OP_CONVERSION_CHECK(elem_আকার > 0, "CPY KV cache view update has invalid element size");
        int64_t start_elem = (int64_t) (offset_bytes / elem_আকার);
        // op_case 5: decoder self-attention – write offset advances each step.
        // op_case 6: encoder self-attn or cross-attn – offset fixed at compile time.
        ধ্রুবক যৌক্তিক is_decoder_self_attn = (op_case == 5);
        auto ones_c = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, 1});
        auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{ones_c, n_write_dyn}, 0);

        auto reshaped = std::make_shared<ov::op::v1::Reshape>(প্রবেশ, new_shape, মিথ্যা);
        auto তথ্য = std::make_shared<ov::op::v0::Convert>(reshaped, context.get_output_type());
        // Indices [start_elem .. start_elem + n_write) on axis 3 of {1,1,1,N}
        // For decoder self-attention the write offset advances each step, so compute it
        // dynamically from the model inputs: start = (attention_size - token_len_per_seq) * n_state.
        // For encoder self-attn and cross-attn the offset is fixed at graph-compile time.
        ov::Output<ov::Node> start;
        যদি (is_decoder_self_attn && context.has_প্রবেশ("attention_size") && context.has_প্রবেশ("token_len_per_seq")) {
            auto attention_আকার_in = context.get_প্রবেশ("attention_size");
            auto token_len_in = context.get_প্রবেশ("token_len_per_seq");
            auto past_tokens = std::make_shared<ov::op::v1::Subtract>(attention_আকার_in, token_len_in);
            auto new_start = std::make_shared<ov::op::v1::Multiply>(past_tokens, n_state_c);
            start = std::make_shared<ov::op::v1::Add>(
                new_start, ov::op::v0::Constant::create(ov::element::i64, {1}, {start_elem}));
        } অন্যথা {
            start = ov::op::v0::Constant::create(ov::element::i64, {1}, {start_elem});
        }
        auto start_squeezed = std::make_shared<ov::op::v0::Squeeze>(start);
        auto end = std::make_shared<ov::op::v1::Add>(start_squeezed, n_write_dyn);
        auto end_squeezed = std::make_shared<ov::op::v0::Squeeze>(end);
        auto step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto step_squeezed = std::make_shared<ov::op::v0::Squeeze>(step);
        auto indices =
            std::make_shared<ov::op::v4::Range>(start_squeezed, end_squeezed, step_squeezed, ov::element::i64);
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        auto kv_updated = std::make_shared<ov::op::v3::ScatterUpdate>(kv_buf, indices, তথ্য, axis);
        ফেরত rename_outputs_with_suffix({kv_updated}, context.get_name());
    }

    যদি (প্রবেশ_shape != output_shape) {
        auto new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<size_t>(output_shape.rank().get_দৈর্ঘ্য())}, output_shape.to_shape());
        প্রবেশ = std::make_shared<ov::op::v1::Reshape>(প্রবেশ, new_shape, মিথ্যা);
    }

    ov::Output<Node> res;
    যদি (context.get_প্রবেশ_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(প্রবেশ, context.get_output_type());
    } অন্যথা {
        res = প্রবেশ;
    }

    যদি (res.get_গ্রন্থি_shared_ptr() == context.get_প্রবেশ(0).get_গ্রন্থি_shared_ptr()) {
        ফেরত {res};
    }

    ফেরত rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
