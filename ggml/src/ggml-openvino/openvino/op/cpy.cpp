#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <climits>
#समावेश <cstdint>
#समावेश <cstdio>
#समावेश <memory>
#समावेश <numeric>
#समावेश <openvino/frontend/exception.hpp>
#समावेश <openvino/op/add.hpp>
#समावेश <openvino/op/concat.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/convert.hpp>
#समावेश <openvino/op/gather.hpp>
#समावेश <openvino/op/multiply.hpp>
#समावेश <openvino/op/negative.hpp>
#समावेश <openvino/op/range.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/scatter_update.hpp>
#समावेश <openvino/op/shape_of.hpp>
#समावेश <openvino/op/slice.hpp>
#समावेश <openvino/op/squeeze.hpp>
#समावेश <openvino/op/subtract.hpp>
#समावेश <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_cpy(स्थिर गाँठ_Context & context) {
    स्वतः op_case = context.get_op_case();
    स्वतः निवेश_shape = context.get_निवेश_shape(0);
    स्वतः output_shape = context.get_निवेश_shape(1);

    अगर (op_case == 4) {
        स्वतः src = process_view_निवेश_new(context, 0);
        स्वतः base = context.get_निवेश(1);

        int64_t n_elems = 1;
        क्रम (स्थिर स्वतः & dim : context.get_output_shape().to_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        स्थिर स्वतः output_stride = context.get_output_stride();
        स्थिर माप_प्रकार elem_आकार = output_stride.empty() ? context.get_output_type().size() : output_stride.back();
        FRONT_END_OP_CONVERSION_CHECK(elem_आकार > 0, "CPY conv state view update has invalid element size");

        स्थिर int64_t begin_val = static_cast<int64_t>(context.get_output_op_offset() / elem_आकार);
        स्थिर int64_t end_val = begin_val + n_elems;

        स्वतः flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, false);
        अगर (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        स्वतः zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        स्वतः begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        स्वतः end = ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val});
        स्वतः int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {पूर्णांक_उच्च});
        स्वतः one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वतः axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        स्वतः शीर्ष_part = std::make_shared<ov::op::v8::Slice>(base, zero, begin, one, axis);
        स्वतः पुच्छ_part = std::make_shared<ov::op::v8::Slice>(base, end, int_max, one, axis);
        स्वतः res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{शीर्ष_part, src, पुच्छ_part}, 3);
        वापस rename_outputs_with_suffix({res}, context.get_name());
    }

    // Recurrent state cache writeback into a slot block of the cache. Where the block starts is a
    // runtime input, so the cached model works for any kv head and active sequence count. The
    // result is the full updated cache.
    // op_case 1: gated-delta-net state, op_case 2: conv state, op_case 3: defrag remainder.
    अगर (op_case == 3) {
        // With -np 1 (and generally whenever there is no defrag remainder) this GET_ROWS gathers
        // zero rows: nothing to write back, and the cache is unchanged. NPU rejects zero-size
        // tensors, so short-circuit instead of building a degenerate Slice/Concat chain.
        bool is_empty = false;
        अगर (निवेश_shape.rank().is_static()) {
            क्रम (स्थिर स्वतः & d : निवेश_shape) {
                अगर (d.is_static() && d.get_लंबाई() == 0) {
                    is_empty = true;
                    अवरोध;
                }
            }
        }
        अगर (is_empty) {
            वापस {context.get_निवेश(1)};
        }
    }
    // op_case 7/8/9 are the single-slot variants; op_case 10 writes native GDN state into a
    // multi-slot cache without rollback snapshots.
    स्थिर bool single_slot_assign = op_case >= 7 && op_case <= 9;
    स्थिर bool direct_gdn_state = op_case == 7 || op_case == 10;
    पूर्णांक writeback_case = op_case;
    अगर (op_case == 10) {
        writeback_case = 1;
    } अन्यथा अगर (single_slot_assign) {
        writeback_case = op_case - 6;
    }
    स्थिर std::string slot_begin_name = "rs_slot_begin_" + context.get_name();
    स्थिर bool slice_assign = writeback_case >= 1 && writeback_case <= 3 &&
                              (single_slot_assign || context.has_निवेश(slot_begin_name));
    अगर (slice_assign) {
        अगर (single_slot_assign && writeback_case == 3) {
            वापस {context.get_निवेश(1)};
        }

        स्थिर int64_t slot_axis = 2;
        स्वतः zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        स्वतः one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वतः int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {पूर्णांक_उच्च});
        स्वतः axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {slot_axis});
        स्वतः feature = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                    std::vector<int64_t>{1, 1, -1, output_shape[3].get_लंबाई()});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin;
        अगर (!single_slot_assign) {
            begin = context.get_निवेश(slot_begin_name);
        }
        स्वतः base = context.get_निवेश(1);
        अगर (writeback_case == 1) {
            अगर (direct_gdn_state) {
                // Non-rollback GDN publishes state directly as [active_slots, heads, value_dim,
                // key_dim]. Flatten each active slot before replacing or updating the cache.
                src = std::make_shared<ov::op::v1::Reshape>(context.get_निवेश(0), feature, false);
            } अन्यथा {
                // Multi-slot rollback still consumes GGML's packed [attention | state snapshots]
                // layout. Slice the state block using the runtime source offset.
                स्वतः src_begin = context.get_निवेश("rs_src_begin_" + context.get_name());
                स्वतः state_part =
                    std::make_shared<ov::op::v8::Slice>(context.get_निवेश(0), src_begin, int_max, one, axis);
                src = std::make_shared<ov::op::v1::Reshape>(state_part, feature, false);
            }
        } अन्यथा अगर (writeback_case == 2) {
            // conv_input is [previous conv state | new tokens]; the snapshot is the conv_kernel_size - 1
            // columns ending at the last *valid* token. Gather (rather than Slice) keeps the output
            // shape static even though the window start is a runtime value.
            स्वतः window_आकार = (int64_t) निवेश_shape[3].get_लंबाई();
            ov::Output<ov::Node> window;
            स्वतः col_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            स्थिर std::string src_begin_name = "rs_src_begin_" + context.get_name();
            अगर (context.has_निवेश(src_begin_name)) {
                स्वतः src_begin = context.get_निवेश(src_begin_name);
                स्वतः src_end = std::make_shared<ov::op::v1::Add>(
                    src_begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {window_आकार}));
                window = std::make_shared<ov::op::v8::Slice>(context.get_निवेश(0), src_begin, src_end, one, col_axis);
            } अन्यथा अगर (context.has_निवेश("chunk_valid_len")) {
                std::vector<int64_t> offsets(window_आकार);
                std::iota(offsets.begin(), offsets.end(), 0);
                स्वतः indices = std::make_shared<ov::op::v1::Add>(
                    ov::op::v0::Constant::create(ov::element::i64, {(माप_प्रकार) window_आकार}, offsets),
                    context.get_निवेश("chunk_valid_len"));
                window = std::make_shared<ov::op::v8::Gather>(context.get_निवेश(0), indices, col_axis);
            } अन्यथा {
                स्वतः window_begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-window_आकार});
                window =
                    std::make_shared<ov::op::v8::Slice>(context.get_निवेश(0), window_begin, int_max, one, col_axis);
            }
            स्थिर स्वतः base_shape = base.get_partial_shape();
            FRONT_END_OP_CONVERSION_CHECK(base_shape.rank().is_static() && base_shape.rank().get_लंबाई() == 4,
                                          "CPY conv state cache update requires rank-4 base cache");
            FRONT_END_OP_CONVERSION_CHECK(base_shape[3].is_static(),
                                          "CPY conv state cache update requires static feature size");
            FRONT_END_OP_CONVERSION_CHECK(निवेश_shape.rank().is_static() && निवेश_shape.rank().get_लंबाई() == 4 &&
                                              निवेश_shape[2].is_static() && निवेश_shape[3].is_static(),
                                          "CPY conv state cache update requires static source feature view");

            स्थिर int64_t full_feature_आकार = base_shape[3].get_लंबाई();
            स्थिर int64_t update_feature_आकार = निवेश_shape[2].get_लंबाई() * निवेश_shape[3].get_लंबाई();
            स्थिर स्वतः output_stride = context.get_output_stride();
            स्थिर माप_प्रकार elem_आकार = output_stride.empty() ? context.get_output_type().size() : output_stride.back();
            FRONT_END_OP_CONVERSION_CHECK(elem_आकार > 0,
                                          "CPY conv state cache update has invalid element size");
            स्थिर int64_t feature_begin = static_cast<int64_t>(context.get_output_op_offset() / elem_आकार) %
                                          full_feature_आकार;
            स्थिर int64_t feature_end = feature_begin + update_feature_आकार;
            FRONT_END_OP_CONVERSION_CHECK(feature_begin >= 0 && feature_end <= full_feature_आकार,
                                          "CPY conv state cache update feature range is out of bounds");

            स्वतः partial_feature = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, update_feature_आकार});
            src = std::make_shared<ov::op::v1::Reshape>(window, partial_feature, false);
            अगर (src.get_element_type() != context.get_output_type()) {
                src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
            }

            अगर (single_slot_assign) {
                वापस rename_outputs_with_suffix({src}, context.get_name());
            }

            स्वतः src_len = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64), axis,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            स्वतः slot_end = std::make_shared<ov::op::v1::Add>(begin, src_len);
            स्वतः active_slots = std::make_shared<ov::op::v8::Slice>(base, begin, slot_end, one, axis);

            स्वतः feature_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            स्वतः feature_begin_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_begin});
            स्वतः feature_end_गाँठ = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_end});
            स्वतः feature_शीर्ष = std::make_shared<ov::op::v8::Slice>(active_slots, zero, feature_begin_गाँठ, one,
                                                                    feature_axis);
            स्वतः feature_पुच्छ = std::make_shared<ov::op::v8::Slice>(active_slots, feature_end_गाँठ, int_max, one,
                                                                    feature_axis);
            src = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{feature_शीर्ष, src, feature_पुच्छ}, 3);
        } अन्यथा {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_निवेश(0);
        }

        अगर (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        अगर (single_slot_assign) {
            वापस rename_outputs_with_suffix({src}, context.get_name());
        }

        स्वतः src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        स्वतः end = std::make_shared<ov::op::v1::Add>(begin, src_len);
        स्वतः शीर्ष_part = std::make_shared<ov::op::v8::Slice>(base, zero, begin, one, axis);
        स्वतः पुच्छ_part = std::make_shared<ov::op::v8::Slice>(base, end, int_max, one, axis);
        स्वतः res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{शीर्ष_part, src, पुच्छ_part}, slot_axis);
        वापस rename_outputs_with_suffix({res}, context.get_name());
    }

    स्वतः निवेश = process_view_निवेश_new(context, 0);

    अगर (op_case == 5 || op_case == 6) {
        स्वतः निवेश_shape = context.get_निवेश_shape(0);
        स्वतः output_shape = context.get_output_shape();
        स्वतः dst_ggml_shape = context.get_view_निवेश_ggml_shape(1, 0);
        स्वतः dst_stride = context.get_view_निवेश_stride(1, 0);
        माप_प्रकार offset_bytes = context.get_view_निवेश_offset(1, 0);
        स्वतः n_state = (int64_t) context.get_निवेश_shape(0)[3].get_लंबाई();
        स्वतः n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
        स्वतः kv_buf = context.get_निवेश(1);  // shape {1,1,1,N}

        Output<Node> token_len_per_seq;
        Output<Node> n_write_dyn;
        अगर (context.has_निवेश("token_len_per_seq")) {
            token_len_per_seq = context.get_निवेश("token_len_per_seq");
            n_write_dyn = std::make_shared<ov::op::v1::Multiply>(token_len_per_seq, n_state_c);
        } अन्यथा {
            n_write_dyn = ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) dst_ggml_shape[3]});
        }
        माप_प्रकार elem_आकार = dst_stride[3];
        FRONT_END_OP_CONVERSION_CHECK(elem_आकार > 0, "CPY KV cache view update has invalid element size");
        int64_t start_elem = (int64_t) (offset_bytes / elem_आकार);
        // op_case 5: decoder self-attention – write offset advances each step.
        // op_case 6: encoder self-attn or cross-attn – offset fixed at compile time.
        स्थिर bool is_decoder_self_attn = (op_case == 5);
        स्वतः ones_c = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, 1});
        स्वतः new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{ones_c, n_write_dyn}, 0);

        स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(निवेश, new_shape, false);
        स्वतः तथ्य = std::make_shared<ov::op::v0::Convert>(reshaped, context.get_output_type());
        // Indices [start_elem .. start_elem + n_write) on axis 3 of {1,1,1,N}
        // For decoder self-attention the write offset advances each step, so compute it
        // dynamically from the model inputs: start = (attention_size - token_len_per_seq) * n_state.
        // For encoder self-attn and cross-attn the offset is fixed at graph-compile time.
        ov::Output<ov::Node> start;
        अगर (is_decoder_self_attn && context.has_निवेश("attention_size") && context.has_निवेश("token_len_per_seq")) {
            स्वतः attention_आकार_in = context.get_निवेश("attention_size");
            स्वतः token_len_in = context.get_निवेश("token_len_per_seq");
            स्वतः past_tokens = std::make_shared<ov::op::v1::Subtract>(attention_आकार_in, token_len_in);
            स्वतः new_start = std::make_shared<ov::op::v1::Multiply>(past_tokens, n_state_c);
            start = std::make_shared<ov::op::v1::Add>(
                new_start, ov::op::v0::Constant::create(ov::element::i64, {1}, {start_elem}));
        } अन्यथा {
            start = ov::op::v0::Constant::create(ov::element::i64, {1}, {start_elem});
        }
        स्वतः start_squeezed = std::make_shared<ov::op::v0::Squeeze>(start);
        स्वतः end = std::make_shared<ov::op::v1::Add>(start_squeezed, n_write_dyn);
        स्वतः end_squeezed = std::make_shared<ov::op::v0::Squeeze>(end);
        स्वतः step = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वतः step_squeezed = std::make_shared<ov::op::v0::Squeeze>(step);
        स्वतः indices =
            std::make_shared<ov::op::v4::Range>(start_squeezed, end_squeezed, step_squeezed, ov::element::i64);
        स्वतः axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        स्वतः kv_updated = std::make_shared<ov::op::v3::ScatterUpdate>(kv_buf, indices, तथ्य, axis);
        वापस rename_outputs_with_suffix({kv_updated}, context.get_name());
    }

    अगर (निवेश_shape != output_shape) {
        स्वतः new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<माप_प्रकार>(output_shape.rank().get_लंबाई())}, output_shape.to_shape());
        निवेश = std::make_shared<ov::op::v1::Reshape>(निवेश, new_shape, false);
    }

    ov::Output<Node> res;
    अगर (context.get_निवेश_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(निवेश, context.get_output_type());
    } अन्यथा {
        res = निवेश;
    }

    अगर (res.get_गाँठ_shared_ptr() == context.get_निवेश(0).get_गाँठ_shared_ptr()) {
        वापस {res};
    }

    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
