#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <climits>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <cstdio>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <numeric>
#અન્તર્ભાવ <openvino/frontend/exception.hpp>
#અન્તર્ભાવ <openvino/op/add.hpp>
#અન્તર્ભાવ <openvino/op/concat.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/convert.hpp>
#અન્તર્ભાવ <openvino/op/gather.hpp>
#અન્તર્ભાવ <openvino/op/multiply.hpp>
#અન્તર્ભાવ <openvino/op/negative.hpp>
#અન્તર્ભાવ <openvino/op/range.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/scatter_update.hpp>
#અન્તર્ભાવ <openvino/op/shape_of.hpp>
#અન્તર્ભાવ <openvino/op/slice.hpp>
#અન્તર્ભાવ <openvino/op/squeeze.hpp>
#અન્તર્ભાવ <openvino/op/subtract.hpp>
#અન્તર્ભાવ <vector>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_cpy(અચલ ગાંઠ_Context & context) {
    auto op_case = context.get_op_case();
    auto આગત_shape = context.get_આગત_shape(0);
    auto output_shape = context.get_આગત_shape(1);

    જો (op_case == 4) {
        auto src = process_view_આગત_new(context, 0);
        auto base = context.get_આગત(1);

        int64_t n_elems = 1;
        માટે (અચલ auto & dim : context.get_output_shape().to_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        અચલ auto output_stride = context.get_output_stride();
        અચલ size_t elem_size = output_stride.empty() ? context.get_output_type().size() : output_stride.back();
        FRONT_END_OP_CONVERSION_CHECK(elem_size > 0, "CPY conv state view update has invalid element size");

        અચલ int64_t begin_val = static_cast<int64_t>(context.get_output_op_offset() / elem_size);
        અચલ int64_t end_val = begin_val + n_elems;

        auto flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, ખોટું);
        જો (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        auto end = ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val});
        auto int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        auto head_part = std::make_shared<ov::op::v8::Slice>(base, zero, begin, one, axis);
        auto tail_part = std::make_shared<ov::op::v8::Slice>(base, end, int_max, one, axis);
        auto res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{head_part, src, tail_part}, 3);
        પાછા rename_outputs_with_suffix({res}, context.get_name());
    }

    // Recurrent state cache writeback into a slot block of the cache. Where the block starts is a
    // runtime input, so the cached model works for any kv head and active sequence count. The
    // result is the full updated cache.
    // op_case 1: gated-delta-net state, op_case 2: conv state, op_case 3: defrag remainder.
    જો (op_case == 3) {
        // With -np 1 (and generally whenever there is no defrag remainder) this GET_ROWS gathers
        // zero rows: nothing to write back, and the cache is unchanged. NPU rejects zero-size
        // tensors, so short-circuit instead of building a degenerate Slice/Concat chain.
        તાર્કિક is_empty = ખોટું;
        જો (આગત_shape.rank().is_static()) {
            માટે (અચલ auto & d : આગત_shape) {
                જો (d.is_static() && d.get_length() == 0) {
                    is_empty = સાચું;
                    તોડો;
                }
            }
        }
        જો (is_empty) {
            પાછા {context.get_આગત(1)};
        }
    }
    // op_case 7/8/9 are the single-slot variants; op_case 10 writes native GDN state into a
    // multi-slot cache without rollback snapshots.
    અચલ તાર્કિક single_slot_assign = op_case >= 7 && op_case <= 9;
    અચલ તાર્કિક direct_gdn_state = op_case == 7 || op_case == 10;
    પૂર્ણાંક writeback_case = op_case;
    જો (op_case == 10) {
        writeback_case = 1;
    } નહીંતર જો (single_slot_assign) {
        writeback_case = op_case - 6;
    }
    અચલ std::string slot_begin_name = "rs_slot_begin_" + context.get_name();
    અચલ તાર્કિક slice_assign = writeback_case >= 1 && writeback_case <= 3 &&
                              (single_slot_assign || context.has_આગત(slot_begin_name));
    જો (slice_assign) {
        જો (single_slot_assign && writeback_case == 3) {
            પાછા {context.get_આગત(1)};
        }

        અચલ int64_t slot_axis = 2;
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {slot_axis});
        auto feature = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                    std::vector<int64_t>{1, 1, -1, output_shape[3].get_length()});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin;
        જો (!single_slot_assign) {
            begin = context.get_આગત(slot_begin_name);
        }
        auto base = context.get_આગત(1);
        જો (writeback_case == 1) {
            જો (direct_gdn_state) {
                // Non-rollback GDN publishes state directly as [active_slots, heads, value_dim,
                // key_dim]. Flatten each active slot before replacing or updating the cache.
                src = std::make_shared<ov::op::v1::Reshape>(context.get_આગત(0), feature, ખોટું);
            } નહીંતર {
                // Multi-slot rollback still consumes GGML's packed [attention | state snapshots]
                // layout. Slice the state block using the runtime source offset.
                auto src_begin = context.get_આગત("rs_src_begin_" + context.get_name());
                auto state_part =
                    std::make_shared<ov::op::v8::Slice>(context.get_આગત(0), src_begin, int_max, one, axis);
                src = std::make_shared<ov::op::v1::Reshape>(state_part, feature, ખોટું);
            }
        } નહીંતર જો (writeback_case == 2) {
            // conv_input is [previous conv state | new tokens]; the snapshot is the conv_kernel_size - 1
            // columns ending at the last *valid* token. Gather (rather than Slice) keeps the output
            // shape static even though the window start is a runtime value.
            auto window_size = (int64_t) આગત_shape[3].get_length();
            ov::Output<ov::Node> window;
            auto col_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            અચલ std::string src_begin_name = "rs_src_begin_" + context.get_name();
            જો (context.has_આગત(src_begin_name)) {
                auto src_begin = context.get_આગત(src_begin_name);
                auto src_end = std::make_shared<ov::op::v1::Add>(
                    src_begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {window_size}));
                window = std::make_shared<ov::op::v8::Slice>(context.get_આગત(0), src_begin, src_end, one, col_axis);
            } નહીંતર જો (context.has_આગત("chunk_valid_len")) {
                std::vector<int64_t> offsets(window_size);
                std::iota(offsets.begin(), offsets.end(), 0);
                auto indices = std::make_shared<ov::op::v1::Add>(
                    ov::op::v0::Constant::create(ov::element::i64, {(size_t) window_size}, offsets),
                    context.get_આગત("chunk_valid_len"));
                window = std::make_shared<ov::op::v8::Gather>(context.get_આગત(0), indices, col_axis);
            } નહીંતર {
                auto window_begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-window_size});
                window =
                    std::make_shared<ov::op::v8::Slice>(context.get_આગત(0), window_begin, int_max, one, col_axis);
            }
            અચલ auto base_shape = base.get_partial_shape();
            FRONT_END_OP_CONVERSION_CHECK(base_shape.rank().is_static() && base_shape.rank().get_length() == 4,
                                          "CPY conv state cache update requires rank-4 base cache");
            FRONT_END_OP_CONVERSION_CHECK(base_shape[3].is_static(),
                                          "CPY conv state cache update requires static feature size");
            FRONT_END_OP_CONVERSION_CHECK(આગત_shape.rank().is_static() && આગત_shape.rank().get_length() == 4 &&
                                              આગત_shape[2].is_static() && આગત_shape[3].is_static(),
                                          "CPY conv state cache update requires static source feature view");

            અચલ int64_t full_feature_size = base_shape[3].get_length();
            અચલ int64_t update_feature_size = આગત_shape[2].get_length() * આગત_shape[3].get_length();
            અચલ auto output_stride = context.get_output_stride();
            અચલ size_t elem_size = output_stride.empty() ? context.get_output_type().size() : output_stride.back();
            FRONT_END_OP_CONVERSION_CHECK(elem_size > 0,
                                          "CPY conv state cache update has invalid element size");
            અચલ int64_t feature_begin = static_cast<int64_t>(context.get_output_op_offset() / elem_size) %
                                          full_feature_size;
            અચલ int64_t feature_end = feature_begin + update_feature_size;
            FRONT_END_OP_CONVERSION_CHECK(feature_begin >= 0 && feature_end <= full_feature_size,
                                          "CPY conv state cache update feature range is out of bounds");

            auto partial_feature = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, update_feature_size});
            src = std::make_shared<ov::op::v1::Reshape>(window, partial_feature, ખોટું);
            જો (src.get_element_type() != context.get_output_type()) {
                src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
            }

            જો (single_slot_assign) {
                પાછા rename_outputs_with_suffix({src}, context.get_name());
            }

            auto src_len = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64), axis,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            auto slot_end = std::make_shared<ov::op::v1::Add>(begin, src_len);
            auto active_slots = std::make_shared<ov::op::v8::Slice>(base, begin, slot_end, one, axis);

            auto feature_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            auto feature_begin_ગાંઠ = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_begin});
            auto feature_end_ગાંઠ = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_end});
            auto feature_head = std::make_shared<ov::op::v8::Slice>(active_slots, zero, feature_begin_ગાંઠ, one,
                                                                    feature_axis);
            auto feature_tail = std::make_shared<ov::op::v8::Slice>(active_slots, feature_end_ગાંઠ, int_max, one,
                                                                    feature_axis);
            src = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{feature_head, src, feature_tail}, 3);
        } નહીંતર {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_આગત(0);
        }

        જો (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        જો (single_slot_assign) {
            પાછા rename_outputs_with_suffix({src}, context.get_name());
        }

        auto src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        auto end = std::make_shared<ov::op::v1::Add>(begin, src_len);
        auto head_part = std::make_shared<ov::op::v8::Slice>(base, zero, begin, one, axis);
        auto tail_part = std::make_shared<ov::op::v8::Slice>(base, end, int_max, one, axis);
        auto res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{head_part, src, tail_part}, slot_axis);
        પાછા rename_outputs_with_suffix({res}, context.get_name());
    }

    auto આગત = process_view_આગત_new(context, 0);

    જો (op_case == 5 || op_case == 6) {
        auto આગત_shape = context.get_આગત_shape(0);
        auto output_shape = context.get_output_shape();
        auto dst_ggml_shape = context.get_view_આગત_ggml_shape(1, 0);
        auto dst_stride = context.get_view_આગત_stride(1, 0);
        size_t offset_bytes = context.get_view_આગત_offset(1, 0);
        auto n_state = (int64_t) context.get_આગત_shape(0)[3].get_length();
        auto n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
        auto kv_buf = context.get_આગત(1);  // shape {1,1,1,N}

        Output<Node> token_len_per_seq;
        Output<Node> n_write_dyn;
        જો (context.has_આગત("token_len_per_seq")) {
            token_len_per_seq = context.get_આગત("token_len_per_seq");
            n_write_dyn = std::make_shared<ov::op::v1::Multiply>(token_len_per_seq, n_state_c);
        } નહીંતર {
            n_write_dyn = ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) dst_ggml_shape[3]});
        }
        size_t elem_size = dst_stride[3];
        FRONT_END_OP_CONVERSION_CHECK(elem_size > 0, "CPY KV cache view update has invalid element size");
        int64_t start_elem = (int64_t) (offset_bytes / elem_size);
        // op_case 5: decoder self-attention – write offset advances each step.
        // op_case 6: encoder self-attn or cross-attn – offset fixed at compile time.
        અચલ તાર્કિક is_decoder_self_attn = (op_case == 5);
        auto ones_c = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, 1});
        auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{ones_c, n_write_dyn}, 0);

        auto reshaped = std::make_shared<ov::op::v1::Reshape>(આગત, new_shape, ખોટું);
        auto data = std::make_shared<ov::op::v0::Convert>(reshaped, context.get_output_type());
        // Indices [start_elem .. start_elem + n_write) on axis 3 of {1,1,1,N}
        // For decoder self-attention the write offset advances each step, so compute it
        // dynamically from the model inputs: start = (attention_size - token_len_per_seq) * n_state.
        // For encoder self-attn and cross-attn the offset is fixed at graph-compile time.
        ov::Output<ov::Node> start;
        જો (is_decoder_self_attn && context.has_આગત("attention_size") && context.has_આગત("token_len_per_seq")) {
            auto attention_size_in = context.get_આગત("attention_size");
            auto token_len_in = context.get_આગત("token_len_per_seq");
            auto past_tokens = std::make_shared<ov::op::v1::Subtract>(attention_size_in, token_len_in);
            auto new_start = std::make_shared<ov::op::v1::Multiply>(past_tokens, n_state_c);
            start = std::make_shared<ov::op::v1::Add>(
                new_start, ov::op::v0::Constant::create(ov::element::i64, {1}, {start_elem}));
        } નહીંતર {
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

        auto kv_updated = std::make_shared<ov::op::v3::ScatterUpdate>(kv_buf, indices, data, axis);
        પાછા rename_outputs_with_suffix({kv_updated}, context.get_name());
    }

    જો (આગત_shape != output_shape) {
        auto new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<size_t>(output_shape.rank().get_length())}, output_shape.to_shape());
        આગત = std::make_shared<ov::op::v1::Reshape>(આગત, new_shape, ખોટું);
    }

    ov::Output<Node> res;
    જો (context.get_આગત_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(આગત, context.get_output_type());
    } નહીંતર {
        res = આગત;
    }

    જો (res.get_ગાંઠ_shared_ptr() == context.get_આગત(0).get_ગાંઠ_shared_ptr()) {
        પાછા {res};
    }

    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
