#अन्तर्भाव "../node_context.h"
#अन्तर्भाव "../op_table.h"
#अन्तर्भाव "../utils.h"

#अन्तर्भाव <climits>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <memory>
#अन्तर्भाव <openvino/core/node.hpp>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/concat.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/reshape.hpp>
#अन्तर्भाव <openvino/op/slice.hpp>
#अन्तर्भाव <openvino/op/transpose.hpp>
#अन्तर्भाव <vector>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश op {

OutputVector translate_permute(अचर गाँठो_Context & context) {
    अङ्क_inputs_check(context, 1, 1);

    पूर्णाङ्क op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case != 0, "Unsupported PERMUTE case");
    // op_case 1 is trivial permute
    // op_case 2 is to permute Q. It has a preceding VIEW that reshapes Q to restore the sequqence dimension
    // op_case 3 4 it to permute KV cache in the default layout
    // op_case 5 6 is to permute V cache when `-fa off`, where v_trans=true

    ov::Output<Node> res;
    ov::Output<Node> src;
    यदि (op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6) {
        src = context.get_आगत(0);
    } अन्यथा {
        src = process_view_आगत_new(context, 0);
    }
    std::vector<int64_t> perm_मानहरू{0, 2, 1, 3};
    अचर int32_t * op_params = context.get_output_op_params();
    यदि (op_params != nullptr) {
        लागि (माप_प्रकार आगत_axis = 0; आगत_axis < perm_मानहरू.size(); ++आगत_axis) {
            अचर माप_प्रकार output_axis = static_cast<माप_प्रकार>(op_params[आगत_axis]);
            perm_मानहरू[perm_मानहरू.size() - 1 - output_axis] =
                static_cast<int64_t>(perm_मानहरू.size() - 1 - आगत_axis);
        }
    }
    यदि (op_case == 1 || context.is_stateful()) {
        // The stateful path carries hidden-state tensors in a rank-3 layout (the
        // leading batch dim is dropped, e.g. Gemma4's per-layer-embedding path). The
        // perm above is rank-4; when the actual input is rank-3, drop the batch axis
        // (perm[0], which is always the identity 0 here) and shift the rest down by 1
        // so the transpose order matches the input rank.
        std::vector<int64_t> perm_used = perm_मानहरू;
        अचर स्वतः & src_ps = src.get_partial_shape();
        यदि (src_ps.rank().is_static() && src_ps.rank().get_length() == 3 && perm_मानहरू.size() == 4 &&
            perm_मानहरू[0] == 0) {
            perm_used = {perm_मानहरू[1] - 1, perm_मानहरू[2] - 1, perm_मानहरू[3] - 1};
        }
        स्वतः perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{perm_used.size()}, perm_used);
        res = std::make_shared<ov::op::v1::Transpose>(src, perm);
    } अन्यथा यदि (op_case == 2) {
        स्वतः perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_मानहरू);
        स्वतः output_shape = context.get_output_shape().to_shape();
        स्वतः n_heads = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[1]});
        स्वतः head_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        स्वतः n_seq_active = context.has_आगत("n_seq_active") ?
                                context.get_आगत("n_seq_active") :
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[0]});
        स्वतः neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        स्वतः new_shape =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, n_heads, head_size}, 0);

        // // Alternative
        // auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        // auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, zero, zero}, 0);

        स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(src, new_shape, सत्य);
        res = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    } अन्यथा {
        स्वतः perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_मानहरू);
        स्वतः cache_shape = src.get_partial_shape();
        स्वतः output_shape = context.get_output_shape().to_shape();
        int64_t head_size = output_shape[3];
        int64_t n_heads = output_shape[1];
        यदि (op_case == 5 || op_case == 6) {
            head_size = output_shape[2];
            n_heads = output_shape[1];
        }
        int64_t ctx_per_seq = cache_shape[2].is_static() ? cache_shape[2].get_length() : -1;
        int64_t n_seq = cache_shape[1].get_length();

        Output<Node> attention_size;
        यदि (!context.has_आगत("attention_size")) {
            attention_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[2]});
        } अन्यथा यदि (op_case == 3 || op_case == 5) {
            attention_size = context.get_आगत("attention_size");
        } अन्यथा {
            attention_size = context.get_आगत("attention_size_swa");
        }

        Output<Node> seq_active_start;
        Output<Node> seq_active_end;
        यदि (context.has_आगत("seq_active_start")) {
            seq_active_start = context.get_आगत("seq_active_start");
            seq_active_end = context.get_आगत("seq_active_end");
        } अन्यथा {
            int64_t n_seq_active = output_shape[0];
            माप_प्रकार offset = *((माप_प्रकार *) context.get_आगत_op_params(0));
            int64_t seq_active_start_val = offset / context.get_आगत_stride(0)[0];
            int64_t seq_active_end_val = seq_active_start_val + n_seq_active;
            seq_active_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_active_start_val});
            seq_active_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_active_end_val});
        }

        // 1. reshape to [n_seq, ctx_per_seq, n_heads, head_size] (for `-fa off` [n_seq, n_heads, head_size, ctx_per_seq])
        // 2. slice out the active sequences
        // 3. slice out the attention part in each sequence
        // 4. permute (skip for `-fa off`)
        स्वतः zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        स्वतः one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        यदि (op_case == 3 || op_case == 4) {
            स्वतः src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, ctx_per_seq, n_heads, head_size}),
                असत्य);
            ov::Output<ov::Node> after_seq_slice;
            यदि (n_seq == 1) {
                after_seq_slice = src_reshaped;
            } अन्यथा {
                after_seq_slice =
                    std::make_shared<ov::op::v8::Slice>(src_reshaped, seq_active_start, seq_active_end, one, zero);
            }
            स्वतः slice2 = std::make_shared<ov::op::v8::Slice>(after_seq_slice, zero, attention_size, one, one);
            res = std::make_shared<ov::op::v1::Transpose>(slice2, perm);
        } अन्यथा {
            स्वतः three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            स्वतः src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, n_heads, head_size, ctx_per_seq}),
                असत्य);
            ov::Output<ov::Node> after_seq_slice;
            यदि (n_seq == 1) {
                after_seq_slice = src_reshaped;
            } अन्यथा {
                after_seq_slice =
                    std::make_shared<ov::op::v8::Slice>(src_reshaped, seq_active_start, seq_active_end, one, zero);
            }
            स्वतः slice2 = std::make_shared<ov::op::v8::Slice>(after_seq_slice, zero, attention_size, one, three);
            res = slice2;
        }
    }
    फर्काऊ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
