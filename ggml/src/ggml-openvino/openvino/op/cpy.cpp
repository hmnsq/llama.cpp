#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <climits>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdio>
#अन्तर्भाव <memory>
#अन्तर्भाव <numeric>
#अन्तर्भाव <openvino/frontend/exception.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/convert.hpp>
#अन्तर्भाव <openvino/op/gather.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/negative.hpp>
#अन्तर्भाव <openvino/op/range.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/scatter_update.hpp>
#अन्तर्भाव <openvino/op/shape_of.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/squeeze.hpp>
#अन्तर्भाव <openvino/op/subtract.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_cpy(स्थिरांक गाठ_Context & context) {
    स्वतः op_case = context.get_op_case();
    स्वतः आदान_shape = context.get_आदान_shape(0);
    स्वतः output_shape = context.get_आदान_shape(1);

    जर (op_case == 4) {
        स्वतः src = process_view_आदान_new(context, 0);
        स्वतः base = context.get_आदान(1);

        int64_t n_elems = 1;
        साठी (स्थिरांक स्वतः & dim : context.get_output_shape().to_shape()) {
            n_elems *= static_cast<int64_t>(dim);
        }

        स्थिरांक स्वतः output_stride = context.get_output_stride();
        स्थिरांक माप_प्रकार elem_size = output_stride.empty() ? context.get_output_type().size() : output_stride.back();
        FRONT_END_OP_CONVERSION_CHECK(elem_size > 0, "CPY conv state view update has invalid element size");

        स्थिरांक int64_t begin_val = static_cast<int64_t>(context.get_output_op_offset() / elem_size);
        स्थिरांक int64_t end_val = begin_val + n_elems;

        स्वतः flat_shape = ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, 1, -1});
        src = std::make_shared<ov::op::v1::Reshape>(src, flat_shape, असत्य);
        जर (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        स्वतः zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        स्वतः begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val});
        स्वतः end = ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val});
        स्वतः int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {पूर्णांक_उच्च});
        स्वतः one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वतः axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});

        स्वतः head_part = std::make_shared<ov::op::v8::Slice>(base, zero, begin, one, axis);
        स्वतः tail_part = std::make_shared<ov::op::v8::Slice>(base, end, int_max, one, axis);
        स्वतः res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{head_part, src, tail_part}, 3);
        परत rename_outputs_with_suffix({res}, context.get_name());
    }

    // Recurrent state cache writeback into a slot block of the cache. Where the block starts is a
    // runtime input, so the cached model works for any kv head and active sequence count. The
    // result is the full updated cache.
    // op_case 1: gated-delta-net state, op_case 2: conv state, op_case 3: defrag remainder.
    जर (op_case == 3) {
        // With -np 1 (and generally whenever there is no defrag remainder) this GET_ROWS gathers
        // zero rows: nothing to write back, and the cache is unchanged. NPU rejects zero-size
        // tensors, so short-circuit instead of building a degenerate Slice/Concat chain.
        तार्किक is_empty = असत्य;
        जर (आदान_shape.rank().is_static()) {
            साठी (स्थिरांक स्वतः & d : आदान_shape) {
                जर (d.is_static() && d.get_length() == 0) {
                    is_empty = सत्य;
                    थांबा;
                }
            }
        }
        जर (is_empty) {
            परत {context.get_आदान(1)};
        }
    }
    // op_case 7/8/9 are the single-slot variants; op_case 10 writes native GDN state into a
    // multi-slot cache without rollback snapshots.
    स्थिरांक तार्किक single_slot_assign = op_case >= 7 && op_case <= 9;
    स्थिरांक तार्किक direct_gdn_state = op_case == 7 || op_case == 10;
    पूर्णांक writeback_case = op_case;
    जर (op_case == 10) {
        writeback_case = 1;
    } अन्यथा जर (single_slot_assign) {
        writeback_case = op_case - 6;
    }
    स्थिरांक std::string slot_begin_name = "rs_slot_begin_" + context.get_name();
    स्थिरांक तार्किक slice_assign = writeback_case >= 1 && writeback_case <= 3 &&
                              (single_slot_assign || context.has_आदान(slot_begin_name));
    जर (slice_assign) {
        जर (single_slot_assign && writeback_case == 3) {
            परत {context.get_आदान(1)};
        }

        स्थिरांक int64_t slot_axis = 2;
        स्वतः zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        स्वतः one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        स्वतः int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {पूर्णांक_उच्च});
        स्वतः axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {slot_axis});
        स्वतः feature = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                    std::vector<int64_t>{1, 1, -1, output_shape[3].get_length()});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin;
        जर (!single_slot_assign) {
            begin = context.get_आदान(slot_begin_name);
        }
        स्वतः base = context.get_आदान(1);
        जर (writeback_case == 1) {
            जर (direct_gdn_state) {
                // Non-rollback GDN publishes state directly as [active_slots, heads, value_dim,
                // key_dim]. Flatten each active slot before replacing or updating the cache.
                src = std::make_shared<ov::op::v1::Reshape>(context.get_आदान(0), feature, असत्य);
            } अन्यथा {
                // Multi-slot rollback still consumes GGML's packed [attention | state snapshots]
                // layout. Slice the state block using the runtime source offset.
                स्वतः src_begin = context.get_आदान("rs_src_begin_" + context.get_name());
                स्वतः state_part =
                    std::make_shared<ov::op::v8::Slice>(context.get_आदान(0), src_begin, int_max, one, axis);
                src = std::make_shared<ov::op::v1::Reshape>(state_part, feature, असत्य);
            }
        } अन्यथा जर (writeback_case == 2) {
            // conv_input is [previous conv state | new tokens]; the snapshot is the conv_kernel_size - 1
            // columns ending at the last *valid* token. Gather (rather than Slice) keeps the output
            // shape static even though the window start is a runtime value.
            स्वतः window_size = (int64_t) आदान_shape[3].get_length();
            ov::Output<ov::Node> window;
            स्वतः col_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            स्थिरांक std::string src_begin_name = "rs_src_begin_" + context.get_name();
            जर (context.has_आदान(src_begin_name)) {
                स्वतः src_begin = context.get_आदान(src_begin_name);
                स्वतः src_end = std::make_shared<ov::op::v1::Add>(
                    src_begin, ov::op::v0::Constant::create(ov::element::i64, {1}, {window_size}));
                window = std::make_shared<ov::op::v8::Slice>(context.get_आदान(0), src_begin, src_end, one, col_axis);
            } अन्यथा जर (context.has_आदान("chunk_valid_len")) {
                std::vector<int64_t> offsets(window_size);
                std::iota(offsets.begin(), offsets.end(), 0);
                स्वतः indices = std::make_shared<ov::op::v1::Add>(
                    ov::op::v0::Constant::create(ov::element::i64, {(माप_प्रकार) window_size}, offsets),
                    context.get_आदान("chunk_valid_len"));
                window = std::make_shared<ov::op::v8::Gather>(context.get_आदान(0), indices, col_axis);
            } अन्यथा {
                स्वतः window_begin = ov::op::v0::Constant::create(ov::element::i64, {1}, {-window_size});
                window =
                    std::make_shared<ov::op::v8::Slice>(context.get_आदान(0), window_begin, int_max, one, col_axis);
            }
            स्थिरांक स्वतः base_shape = base.get_partial_shape();
            FRONT_END_OP_CONVERSION_CHECK(base_shape.rank().is_static() && base_shape.rank().get_length() == 4,
                                          "CPY conv state cache update requires rank-4 base cache");
            FRONT_END_OP_CONVERSION_CHECK(base_shape[3].is_static(),
                                          "CPY conv state cache update requires static feature size");
            FRONT_END_OP_CONVERSION_CHECK(आदान_shape.rank().is_static() && आदान_shape.rank().get_length() == 4 &&
                                              आदान_shape[2].is_static() && आदान_shape[3].is_static(),
                                          "CPY conv state cache update requires static source feature view");

            स्थिरांक int64_t full_feature_size = base_shape[3].get_length();
            स्थिरांक int64_t update_feature_size = आदान_shape[2].get_length() * आदान_shape[3].get_length();
            स्थिरांक स्वतः output_stride = context.get_output_stride();
            स्थिरांक माप_प्रकार elem_size = output_stride.empty() ? context.get_output_type().size() : output_stride.back();
            FRONT_END_OP_CONVERSION_CHECK(elem_size > 0,
                                          "CPY conv state cache update has invalid element size");
            स्थिरांक int64_t feature_begin = static_cast<int64_t>(context.get_output_op_offset() / elem_size) %
                                          full_feature_size;
            स्थिरांक int64_t feature_end = feature_begin + update_feature_size;
            FRONT_END_OP_CONVERSION_CHECK(feature_begin >= 0 && feature_end <= full_feature_size,
                                          "CPY conv state cache update feature range is out of bounds");

            स्वतः partial_feature = ov::op::v0::Constant::create(
                ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, update_feature_size});
            src = std::make_shared<ov::op::v1::Reshape>(window, partial_feature, असत्य);
            जर (src.get_element_type() != context.get_output_type()) {
                src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
            }

            जर (single_slot_assign) {
                परत rename_outputs_with_suffix({src}, context.get_name());
            }

            स्वतः src_len = std::make_shared<ov::op::v8::Gather>(
                std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64), axis,
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            स्वतः slot_end = std::make_shared<ov::op::v1::Add>(begin, src_len);
            स्वतः active_slots = std::make_shared<ov::op::v8::Slice>(base, begin, slot_end, one, axis);

            स्वतः feature_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            स्वतः feature_begin_गाठ = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_begin});
            स्वतः feature_end_गाठ = ov::op::v0::Constant::create(ov::element::i64, {1}, {feature_end});
            स्वतः feature_head = std::make_shared<ov::op::v8::Slice>(active_slots, zero, feature_begin_गाठ, one,
                                                                    feature_axis);
            स्वतः feature_tail = std::make_shared<ov::op::v8::Slice>(active_slots, feature_end_गाठ, int_max, one,
                                                                    feature_axis);
            src = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{feature_head, src, feature_tail}, 3);
        } अन्यथा {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_आदान(0);
        }

        जर (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        जर (single_slot_assign) {
            परत rename_outputs_with_suffix({src}, context.get_name());
        }

        स्वतः src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        स्वतः end = std::make_shared<ov::op::v1::Add>(begin, src_len);
        स्वतः head_part = std::make_shared<ov::op::v8::Slice>(base, zero, begin, one, axis);
        स्वतः tail_part = std::make_shared<ov::op::v8::Slice>(base, end, int_max, one, axis);
        स्वतः res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{head_part, src, tail_part}, slot_axis);
        परत rename_outputs_with_suffix({res}, context.get_name());
    }

    स्वतः आदान = process_view_आदान_new(context, 0);

    जर (op_case == 5 || op_case == 6) {
        स्वतः आदान_shape = context.get_आदान_shape(0);
        स्वतः output_shape = context.get_output_shape();
        स्वतः dst_ggml_shape = context.get_view_आदान_ggml_shape(1, 0);
        स्वतः dst_stride = context.get_view_आदान_stride(1, 0);
        माप_प्रकार offset_bytes = context.get_view_आदान_offset(1, 0);
        स्वतः n_state = (int64_t) context.get_आदान_shape(0)[3].get_length();
        स्वतः n_state_c = ov::op::v0::Constant::create(ov::element::i64, {1}, {n_state});
        स्वतः kv_buf = context.get_आदान(1);  // shape {1,1,1,N}

        Output<Node> token_len_per_seq;
        Output<Node> n_write_dyn;
        जर (context.has_आदान("token_len_per_seq")) {
            token_len_per_seq = context.get_आदान("token_len_per_seq");
            n_write_dyn = std::make_shared<ov::op::v1::Multiply>(token_len_per_seq, n_state_c);
        } अन्यथा {
            n_write_dyn = ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) dst_ggml_shape[3]});
        }
        माप_प्रकार elem_size = dst_stride[3];
        FRONT_END_OP_CONVERSION_CHECK(elem_size > 0, "CPY KV cache view update has invalid element size");
        int64_t start_elem = (int64_t) (offset_bytes / elem_size);
        // op_case 5: decoder self-attention – write offset advances each step.
        // op_case 6: encoder self-attn or cross-attn – offset fixed at compile time.
        स्थिरांक तार्किक is_decoder_self_attn = (op_case == 5);
        स्वतः ones_c = ov::op::v0::Constant::create(ov::element::i64, {3}, std::vector<int64_t>{1, 1, 1});
        स्वतः new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{ones_c, n_write_dyn}, 0);

        स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(आदान, new_shape, असत्य);
        स्वतः data = std::make_shared<ov::op::v0::Convert>(reshaped, context.get_output_type());
        // Indices [start_elem .. start_elem + n_write) on axis 3 of {1,1,1,N}
        // For decoder self-attention the write offset advances each step, so compute it
        // dynamically from the model inputs: start = (attention_size - token_len_per_seq) * n_state.
        // For encoder self-attn and cross-attn the offset is fixed at graph-compile time.
        ov::Output<ov::Node> start;
        जर (is_decoder_self_attn && context.has_आदान("attention_size") && context.has_आदान("token_len_per_seq")) {
            स्वतः attention_size_in = context.get_आदान("attention_size");
            स्वतः token_len_in = context.get_आदान("token_len_per_seq");
            स्वतः past_tokens = std::make_shared<ov::op::v1::Subtract>(attention_size_in, token_len_in);
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

        स्वतः kv_updated = std::make_shared<ov::op::v3::ScatterUpdate>(kv_buf, indices, data, axis);
        परत rename_outputs_with_suffix({kv_updated}, context.get_name());
    }

    जर (आदान_shape != output_shape) {
        स्वतः new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<माप_प्रकार>(output_shape.rank().get_length())}, output_shape.to_shape());
        आदान = std::make_shared<ov::op::v1::Reshape>(आदान, new_shape, असत्य);
    }

    ov::Output<Node> res;
    जर (context.get_आदान_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(आदान, context.get_output_type());
    } अन्यथा {
        res = आदान;
    }

    जर (res.get_गाठ_shared_ptr() == context.get_आदान(0).get_गाठ_shared_ptr()) {
        परत {res};
    }

    परत rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
