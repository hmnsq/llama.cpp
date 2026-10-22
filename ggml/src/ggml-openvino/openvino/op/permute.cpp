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

OutputVector translate_permute(ध्रुव गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    पूर्णाङ्क op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case != 0, "Unsupported PERMUTE case");
    // op_case 1 is trivial permute
    // op_case 2 is to permute Q. It has a preceding VIEW that reshapes Q to restore the sequqence dimension
    // op_case 3 4 it to permute KV cache in the default layout
    // op_case 5 6 is to permute V cache when `-fa off`, where v_trans=true

    ov::Output<Node> res;
    ov::Output<Node> src;
    यदि (op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6) {
        src = context.get_निवेश(0);
    } अन्यथा {
        src = process_view_निवेश_new(context, 0);
    }
    std::vector<int64_t> perm_मूल्यावली{0, 2, 1, 3};
    ध्रुव int32_t * op_params = context.get_output_op_params();
    यदि (op_params != nullptr) {
        प्रति (size_t निवेश_axis = 0; निवेश_axis < perm_मूल्यावली.size(); ++निवेश_axis) {
            ध्रुव size_t output_axis = static_cast<size_t>(op_params[निवेश_axis]);
            perm_मूल्यावली[perm_मूल्यावली.size() - 1 - output_axis] =
                static_cast<int64_t>(perm_मूल्यावली.size() - 1 - निवेश_axis);
        }
    }
    यदि (op_case == 1 || context.is_stateful()) {
        // The stateful path carries hidden-state tensors in a rank-3 layout (the
        // leading batch dim is dropped, e.g. Gemma4's per-layer-embedding path). The
        // perm above is rank-4; when the actual input is rank-3, drop the batch axis
        // (perm[0], which is always the identity 0 here) and shift the rest down by 1
        // so the transpose order matches the input rank.
        std::vector<int64_t> perm_used = perm_मूल्यावली;
        ध्रुव auto & src_ps = src.get_partial_shape();
        यदि (src_ps.rank().is_static() && src_ps.rank().get_length() == 3 && perm_मूल्यावली.size() == 4 &&
            perm_मूल्यावली[0] == 0) {
            perm_used = {perm_मूल्यावली[1] - 1, perm_मूल्यावली[2] - 1, perm_मूल्यावली[3] - 1};
        }
        auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{perm_used.size()}, perm_used);
        res = std::make_shared<ov::op::v1::Transpose>(src, perm);
    } अन्यथा यदि (op_case == 2) {
        auto perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_मूल्यावली);
        auto output_shape = context.get_output_shape().to_shape();
        auto n_heads = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[1]});
        auto head_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        auto n_seq_active = context.has_निवेश("n_seq_active") ?
                                context.get_निवेश("n_seq_active") :
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[0]});
        auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        auto new_shape =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, n_heads, head_size}, 0);

        // // Alternative
        // auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        // auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, zero, zero}, 0);

        auto reshaped = std::make_shared<ov::op::v1::Reshape>(src, new_shape, सत्य);
        res = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    } अन्यथा {
        auto perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_मूल्यावली);
        auto cache_shape = src.get_partial_shape();
        auto output_shape = context.get_output_shape().to_shape();
        int64_t head_size = output_shape[3];
        int64_t n_heads = output_shape[1];
        यदि (op_case == 5 || op_case == 6) {
            head_size = output_shape[2];
            n_heads = output_shape[1];
        }
        int64_t ctx_per_seq = cache_shape[2].is_static() ? cache_shape[2].get_length() : -1;
        int64_t n_seq = cache_shape[1].get_length();

        Output<Node> attention_size;
        यदि (!context.has_निवेश("attention_size")) {
            attention_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[2]});
        } अन्यथा यदि (op_case == 3 || op_case == 5) {
            attention_size = context.get_निवेश("attention_size");
        } अन्यथा {
            attention_size = context.get_निवेश("attention_size_swa");
        }

        Output<Node> seq_active_start;
        Output<Node> seq_active_end;
        यदि (context.has_निवेश("seq_active_start")) {
            seq_active_start = context.get_निवेश("seq_active_start");
            seq_active_end = context.get_निवेश("seq_active_end");
        } अन्यथा {
            int64_t n_seq_active = output_shape[0];
            size_t offset = *((size_t *) context.get_निवेश_op_params(0));
            int64_t seq_active_start_val = offset / context.get_निवेश_stride(0)[0];
            int64_t seq_active_end_val = seq_active_start_val + n_seq_active;
            seq_active_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_active_start_val});
            seq_active_end = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_active_end_val});
        }

        // 1. reshape to [n_seq, ctx_per_seq, n_heads, head_size] (for `-fa off` [n_seq, n_heads, head_size, ctx_per_seq])
        // 2. slice out the active sequences
        // 3. slice out the attention part in each sequence
        // 4. permute (skip for `-fa off`)
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        यदि (op_case == 3 || op_case == 4) {
            auto src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, ctx_per_seq, n_heads, head_size}),
                असत्य);
            ov::Output<ov::Node> after_seq_slice;
            यदि (n_seq == 1) {
                after_seq_slice = src_reshaped;
            } अन्यथा {
                after_seq_slice =
                    std::make_shared<ov::op::v8::Slice>(src_reshaped, seq_active_start, seq_active_end, one, zero);
            }
            auto slice2 = std::make_shared<ov::op::v8::Slice>(after_seq_slice, zero, attention_size, one, one);
            res = std::make_shared<ov::op::v1::Transpose>(slice2, perm);
        } अन्यथा {
            auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            auto src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, n_heads, head_size, ctx_per_seq}),
                असत्य);
            ov::Output<ov::Node> after_seq_slice;
            यदि (n_seq == 1) {
                after_seq_slice = src_reshaped;
            } अन्यथा {
                after_seq_slice =
                    std::make_shared<ov::op::v8::Slice>(src_reshaped, seq_active_start, seq_active_end, one, zero);
            }
            auto slice2 = std::make_shared<ov::op::v8::Slice>(after_seq_slice, zero, attention_size, one, three);
            res = slice2;
        }
    }
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
