#شامل "../node_context.h"
#شامل "../op_table.h"
#شامل "../utils.h"

#شامل <climits>
#شامل <cstdint>
#شامل <memory>
#شامل <openvino/core/node.hpp>
#شامل <openvino/op/add.hpp>
#شامل <openvino/op/concat.hpp>
#شامل <openvino/op/constant.hpp>
#شامل <openvino/op/reshape.hpp>
#شامل <openvino/op/slice.hpp>
#شامل <openvino/op/transpose.hpp>
#شامل <vector>

نامخلا ov {
نامخلا frontend {
نامخلا ggml {
نامخلا op {

Output_سمتیہ translate_ترتیب_بدلو(ثابت گرہ_Context & context) {
    عدد_inputs_check(context, 1, 1);

    صحیح op_case = context.get_op_case();
    FRONT_انتہا_CHECK_IMPLEMENTED(op_case != 0, "Unsupported PERMUTE case");
    // op_case 1 is trivial permute
    // op_case 2 is to permute Q. It has a preceding VIEW that reshapes Q to restore the sequqence dimension
    // op_case 3 4 it to permute KV cache in the default layout
    // op_case 5 6 is to permute V cache when `-fa off`, where v_trans=true

    ov::Output<Node> res;
    ov::Output<Node> src;
    اگر (op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6) {
        src = context.get_داخلہ(0);
    } ورنہ {
        src = process_view_داخلہ_new(context, 0);
    }
    std::vector<int64_t> perm_اقدار{0, 2, 1, 3};
    ثابت int32_t * op_params = context.get_output_op_params();
    اگر (op_params != خالی_اشاریہ) {
        برائے (size_t داخلہ_axis = 0; داخلہ_axis < perm_اقدار.size(); ++داخلہ_axis) {
            ثابت size_t output_axis = static_cast<size_t>(op_params[داخلہ_axis]);
            perm_اقدار[perm_اقدار.size() - 1 - output_axis] =
                static_cast<int64_t>(perm_اقدار.size() - 1 - داخلہ_axis);
        }
    }
    اگر (op_case == 1 || context.is_stateful()) {
        // The stateful path carries hidden-state tensors in a rank-3 layout (the
        // leading batch dim is dropped, e.g. Gemma4's per-layer-embedding path). The
        // perm above is rank-4; when the actual input is rank-3, drop the batch axis
        // (perm[0], which is always the identity 0 here) and shift the rest down by 1
        // so the transpose order matches the input rank.
        std::vector<int64_t> perm_مستعمل = perm_اقدار;
        ثابت خودکار & src_ps = src.get_partial_shape();
        اگر (src_ps.rank().is_static() && src_ps.rank().get_لمبائی() == 3 && perm_اقدار.size() == 4 &&
            perm_اقدار[0] == 0) {
            perm_مستعمل = {perm_اقدار[1] - 1, perm_اقدار[2] - 1, perm_اقدار[3] - 1};
        }
        خودکار perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{perm_مستعمل.size()}, perm_مستعمل);
        res = std::make_shared<ov::op::v1::Transpose>(src, perm);
    } ورنہ اگر (op_case == 2) {
        خودکار perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_اقدار);
        خودکار output_shape = context.get_output_shape().مقصد_shape();
        خودکار n_heads = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[1]});
        خودکار سر_سائز = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        خودکار n_seq_فعال = context.has_داخلہ("n_seq_active") ?
                                context.get_داخلہ("n_seq_active") :
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[0]});
        خودکار neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        خودکار new_shape =
            std::make_shared<ov::op::v0::Concat>(ov::Output_سمتیہ{n_seq_فعال, neg_one, n_heads, سر_سائز}, 0);

        // // Alternative
        // auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        // auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, zero, zero}, 0);

        خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(src, new_shape, سچ);
        res = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    } ورنہ {
        خودکار perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_اقدار);
        خودکار cache_shape = src.get_partial_shape();
        خودکار output_shape = context.get_output_shape().مقصد_shape();
        int64_t سر_سائز = output_shape[3];
        int64_t n_heads = output_shape[1];
        اگر (op_case == 5 || op_case == 6) {
            سر_سائز = output_shape[2];
            n_heads = output_shape[1];
        }
        int64_t ctx_فی_seq = cache_shape[2].is_static() ? cache_shape[2].get_لمبائی() : -1;
        int64_t n_seq = cache_shape[1].get_لمبائی();

        Output<Node> attention_سائز;
        اگر (!context.has_داخلہ("attention_size")) {
            attention_سائز = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[2]});
        } ورنہ اگر (op_case == 3 || op_case == 5) {
            attention_سائز = context.get_داخلہ("attention_size");
        } ورنہ {
            attention_سائز = context.get_داخلہ("attention_size_swa");
        }

        Output<Node> seq_فعال_آغاز;
        Output<Node> seq_فعال_انتہا;
        اگر (context.has_داخلہ("seq_active_start")) {
            seq_فعال_آغاز = context.get_داخلہ("seq_active_start");
            seq_فعال_انتہا = context.get_داخلہ("seq_active_end");
        } ورنہ {
            int64_t n_seq_فعال = output_shape[0];
            size_t انحراف = *((size_t *) context.get_داخلہ_op_params(0));
            int64_t seq_فعال_آغاز_val = انحراف / context.get_داخلہ_stride(0)[0];
            int64_t seq_فعال_انتہا_val = seq_فعال_آغاز_val + n_seq_فعال;
            seq_فعال_آغاز = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_فعال_آغاز_val});
            seq_فعال_انتہا = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_فعال_انتہا_val});
        }

        // 1. reshape to [n_seq, ctx_per_seq, n_heads, head_size] (for `-fa off` [n_seq, n_heads, head_size, ctx_per_seq])
        // 2. slice out the active sequences
        // 3. slice out the attention part in each sequence
        // 4. permute (skip for `-fa off`)
        خودکار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        اگر (op_case == 3 || op_case == 4) {
            خودکار src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, ctx_فی_seq, n_heads, سر_سائز}),
                جھوٹ);
            ov::Output<ov::Node> after_seq_قاش;
            اگر (n_seq == 1) {
                after_seq_قاش = src_reshaped;
            } ورنہ {
                after_seq_قاش =
                    std::make_shared<ov::op::v8::قاش>(src_reshaped, seq_فعال_آغاز, seq_فعال_انتہا, one, zero);
            }
            خودکار slice2 = std::make_shared<ov::op::v8::قاش>(after_seq_قاش, zero, attention_سائز, one, one);
            res = std::make_shared<ov::op::v1::Transpose>(slice2, perm);
        } ورنہ {
            خودکار three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            خودکار src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, n_heads, سر_سائز, ctx_فی_seq}),
                جھوٹ);
            ov::Output<ov::Node> after_seq_قاش;
            اگر (n_seq == 1) {
                after_seq_قاش = src_reshaped;
            } ورنہ {
                after_seq_قاش =
                    std::make_shared<ov::op::v8::قاش>(src_reshaped, seq_فعال_آغاز, seq_فعال_انتہا, one, zero);
            }
            خودکار slice2 = std::make_shared<ov::op::v8::قاش>(after_seq_قاش, zero, attention_سائز, one, three);
            res = slice2;
        }
    }
    واپس rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
