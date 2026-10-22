#समावेश "../node_context.h"
#समावेश "../op_table.h"
#समावेश "../utils.h"

#समावेश <climits>
#समावेश <cstdint>
#समावेश <memory>
#समावेश <openvino/core/node.hpp>
#समावेश <openvino/op/add.hpp>
#समावेश <openvino/op/concat.hpp>
#समावेश <openvino/op/constant.hpp>
#समावेश <openvino/op/reshape.hpp>
#समावेश <openvino/op/slice.hpp>
#समावेश <openvino/op/transpose.hpp>
#समावेश <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_permute(स्थिर गाँठ_Context & context) {
    अंक_inputs_check(context, 1, 1);

    पूर्णांक op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case != 0, "Unsupported PERMUTE case");
    // op_case 1 is trivial permute
    // op_case 2 is to permute Q. It has a preceding VIEW that reshapes Q to restore the sequqence dimension
    // op_case 3 4 it to permute KV cache in the default layout
    // op_case 5 6 is to permute V cache when `-fa off`, where v_trans=true

    ov::Output<Node> res;
    ov::Output<Node> src;
    अगर (op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6) {
        src = context.get_निवेश(0);
    } अन्यथा {
        src = process_view_निवेश_new(context, 0);
    }
    std::vector<int64_t> perm_मूल्यावली{0, 2, 1, 3};
    स्थिर int32_t * op_params = context.get_output_op_params();
    अगर (op_params != nullptr) {
        क्रम (माप_प्रकार निवेश_axis = 0; निवेश_axis < perm_मूल्यावली.size(); ++निवेश_axis) {
            स्थिर माप_प्रकार output_axis = static_cast<माप_प्रकार>(op_params[निवेश_axis]);
            perm_मूल्यावली[perm_मूल्यावली.size() - 1 - output_axis] =
                static_cast<int64_t>(perm_मूल्यावली.size() - 1 - निवेश_axis);
        }
    }
    अगर (op_case == 1 || context.is_stateful()) {
        // The stateful path carries hidden-state tensors in a rank-3 layout (the
        // leading batch dim is dropped, e.g. Gemma4's per-layer-embedding path). The
        // perm above is rank-4; when the actual input is rank-3, drop the batch axis
        // (perm[0], which is always the identity 0 here) and shift the rest down by 1
        // so the transpose order matches the input rank.
        std::vector<int64_t> perm_used = perm_मूल्यावली;
        स्थिर स्वतः & src_ps = src.get_partial_shape();
        अगर (src_ps.rank().is_static() && src_ps.rank().get_लंबाई() == 3 && perm_मूल्यावली.size() == 4 &&
            perm_मूल्यावली[0] == 0) {
            perm_used = {perm_मूल्यावली[1] - 1, perm_मूल्यावली[2] - 1, perm_मूल्यावली[3] - 1};
        }
        स्वतः perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{perm_used.size()}, perm_used);
        res = std::make_shared<ov::op::v1::Transpose>(src, perm);
    } अन्यथा अगर (op_case == 2) {
        स्वतः perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_मूल्यावली);
        स्वतः output_shape = context.get_output_shape().to_shape();
        स्वतः n_heads = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[1]});
        स्वतः शीर्ष_आकार = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        स्वतः n_seq_active = context.has_निवेश("n_seq_active") ?
                                context.get_निवेश("n_seq_active") :
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[0]});
        स्वतः neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        स्वतः new_shape =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, n_heads, शीर्ष_आकार}, 0);

        // // Alternative
        // auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        // auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, zero, zero}, 0);

        स्वतः reshaped = std::make_shared<ov::op::v1::Reshape>(src, new_shape, true);
        res = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    } अन्यथा {
        स्वतः perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_मूल्यावली);
        स्वतः cache_shape = src.get_partial_shape();
        स्वतः output_shape = context.get_output_shape().to_shape();
        int64_t शीर्ष_आकार = output_shape[3];
        int64_t n_heads = output_shape[1];
        अगर (op_case == 5 || op_case == 6) {
            शीर्ष_आकार = output_shape[2];
            n_heads = output_shape[1];
        }
        int64_t ctx_per_seq = cache_shape[2].is_static() ? cache_shape[2].get_लंबाई() : -1;
        int64_t n_seq = cache_shape[1].get_लंबाई();

        Output<Node> attention_आकार;
        अगर (!context.has_निवेश("attention_size")) {
            attention_आकार = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[2]});
        } अन्यथा अगर (op_case == 3 || op_case == 5) {
            attention_आकार = context.get_निवेश("attention_size");
        } अन्यथा {
            attention_आकार = context.get_निवेश("attention_size_swa");
        }

        Output<Node> seq_active_start;
        Output<Node> seq_active_end;
        अगर (context.has_निवेश("seq_active_start")) {
            seq_active_start = context.get_निवेश("seq_active_start");
            seq_active_end = context.get_निवेश("seq_active_end");
        } अन्यथा {
            int64_t n_seq_active = output_shape[0];
            माप_प्रकार offset = *((माप_प्रकार *) context.get_निवेश_op_params(0));
            int64_t seq_active_start_val = offset / context.get_निवेश_stride(0)[0];
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

        अगर (op_case == 3 || op_case == 4) {
            स्वतः src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, ctx_per_seq, n_heads, शीर्ष_आकार}),
                false);
            ov::Output<ov::Node> after_seq_slice;
            अगर (n_seq == 1) {
                after_seq_slice = src_reshaped;
            } अन्यथा {
                after_seq_slice =
                    std::make_shared<ov::op::v8::Slice>(src_reshaped, seq_active_start, seq_active_end, one, zero);
            }
            स्वतः slice2 = std::make_shared<ov::op::v8::Slice>(after_seq_slice, zero, attention_आकार, one, one);
            res = std::make_shared<ov::op::v1::Transpose>(slice2, perm);
        } अन्यथा {
            स्वतः three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            स्वतः src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, n_heads, शीर्ष_आकार, ctx_per_seq}),
                false);
            ov::Output<ov::Node> after_seq_slice;
            अगर (n_seq == 1) {
                after_seq_slice = src_reshaped;
            } अन्यथा {
                after_seq_slice =
                    std::make_shared<ov::op::v8::Slice>(src_reshaped, seq_active_start, seq_active_end, one, zero);
            }
            स्वतः slice2 = std::make_shared<ov::op::v8::Slice>(after_seq_slice, zero, attention_आकार, one, three);
            res = slice2;
        }
    }
    वापस rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
