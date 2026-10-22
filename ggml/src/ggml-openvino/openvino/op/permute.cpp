#ਅਨ੍ਤਰ੍ਭਾਵ "../node_context.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../op_table.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "../utils.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <climits>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/core/node.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/add.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/concat.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/constant.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/reshape.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/slice.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <openvino/op/transpose.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

ਨਾਮਾਕਾਸ਼ ov {
ਨਾਮਾਕਾਸ਼ frontend {
ਨਾਮਾਕਾਸ਼ ggml {
ਨਾਮਾਕਾਸ਼ op {

Output_ਸਦਿਸ਼ translate_ਕ੍ਰਮ_ਬਦਲੋ(ਅਚਲ ਗੰਢ_Context & context) {
    ਅੰਕ_inputs_check(context, 1, 1);

    ਪੂਰਨ op_case = context.get_op_case();
    FRONT_ਸਿਰਾ_CHECK_IMPLEMENTED(op_case != 0, "Unsupported PERMUTE case");
    // op_case 1 is trivial permute
    // op_case 2 is to permute Q. It has a preceding VIEW that reshapes Q to restore the sequqence dimension
    // op_case 3 4 it to permute KV cache in the default layout
    // op_case 5 6 is to permute V cache when `-fa off`, where v_trans=true

    ov::Output<Node> res;
    ov::Output<Node> src;
    ਜੇ (op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6) {
        src = context.get_ਨਿਵੇਸ਼(0);
    } ਨਹੀਂਤਾਂ {
        src = process_view_ਨਿਵੇਸ਼_new(context, 0);
    }
    std::vector<int64_t> perm_ਮੁੱਲਾਂ{0, 2, 1, 3};
    ਅਚਲ int32_t * op_params = context.get_output_op_params();
    ਜੇ (op_params != ਖਾਲੀ_ਸੂਚਕ) {
        ਲਈ (size_t ਨਿਵੇਸ਼_axis = 0; ਨਿਵੇਸ਼_axis < perm_ਮੁੱਲਾਂ.size(); ++ਨਿਵੇਸ਼_axis) {
            ਅਚਲ size_t output_axis = static_cast<size_t>(op_params[ਨਿਵੇਸ਼_axis]);
            perm_ਮੁੱਲਾਂ[perm_ਮੁੱਲਾਂ.size() - 1 - output_axis] =
                static_cast<int64_t>(perm_ਮੁੱਲਾਂ.size() - 1 - ਨਿਵੇਸ਼_axis);
        }
    }
    ਜੇ (op_case == 1 || context.is_stateful()) {
        // The stateful path carries hidden-state tensors in a rank-3 layout (the
        // leading batch dim is dropped, e.g. Gemma4's per-layer-embedding path). The
        // perm above is rank-4; when the actual input is rank-3, drop the batch axis
        // (perm[0], which is always the identity 0 here) and shift the rest down by 1
        // so the transpose order matches the input rank.
        std::vector<int64_t> perm_ਵਰਤਿਆ = perm_ਮੁੱਲਾਂ;
        ਅਚਲ ਸਵੈਚਾਲਿਤ & src_ps = src.get_partial_shape();
        ਜੇ (src_ps.rank().is_static() && src_ps.rank().get_ਲੰਮਾਈ() == 3 && perm_ਮੁੱਲਾਂ.size() == 4 &&
            perm_ਮੁੱਲਾਂ[0] == 0) {
            perm_ਵਰਤਿਆ = {perm_ਮੁੱਲਾਂ[1] - 1, perm_ਮੁੱਲਾਂ[2] - 1, perm_ਮੁੱਲਾਂ[3] - 1};
        }
        ਸਵੈਚਾਲਿਤ perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{perm_ਵਰਤਿਆ.size()}, perm_ਵਰਤਿਆ);
        res = std::make_shared<ov::op::v1::Transpose>(src, perm);
    } ਨਹੀਂਤਾਂ ਜੇ (op_case == 2) {
        ਸਵੈਚਾਲਿਤ perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_ਮੁੱਲਾਂ);
        ਸਵੈਚਾਲਿਤ output_shape = context.get_output_shape().ਮੰਜ਼ਿਲ_shape();
        ਸਵੈਚਾਲਿਤ n_heads = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[1]});
        ਸਵੈਚਾਲਿਤ ਸਿਰ_ਮਾਪ = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        ਸਵੈਚਾਲਿਤ n_seq_ਸਰਗਰਮ = context.has_ਨਿਵੇਸ਼("n_seq_active") ?
                                context.get_ਨਿਵੇਸ਼("n_seq_active") :
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[0]});
        ਸਵੈਚਾਲਿਤ neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        ਸਵੈਚਾਲਿਤ new_shape =
            std::make_shared<ov::op::v0::Concat>(ov::Output_ਸਦਿਸ਼{n_seq_ਸਰਗਰਮ, neg_one, n_heads, ਸਿਰ_ਮਾਪ}, 0);

        // // Alternative
        // auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        // auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, zero, zero}, 0);

        ਸਵੈਚਾਲਿਤ reshaped = std::make_shared<ov::op::v1::Reshape>(src, new_shape, ਸੱਚ);
        res = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    } ਨਹੀਂਤਾਂ {
        ਸਵੈਚਾਲਿਤ perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_ਮੁੱਲਾਂ);
        ਸਵੈਚਾਲਿਤ cache_shape = src.get_partial_shape();
        ਸਵੈਚਾਲਿਤ output_shape = context.get_output_shape().ਮੰਜ਼ਿਲ_shape();
        int64_t ਸਿਰ_ਮਾਪ = output_shape[3];
        int64_t n_heads = output_shape[1];
        ਜੇ (op_case == 5 || op_case == 6) {
            ਸਿਰ_ਮਾਪ = output_shape[2];
            n_heads = output_shape[1];
        }
        int64_t ctx_ਪ੍ਰਤੀ_seq = cache_shape[2].is_static() ? cache_shape[2].get_ਲੰਮਾਈ() : -1;
        int64_t n_seq = cache_shape[1].get_ਲੰਮਾਈ();

        Output<Node> attention_ਮਾਪ;
        ਜੇ (!context.has_ਨਿਵੇਸ਼("attention_size")) {
            attention_ਮਾਪ = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[2]});
        } ਨਹੀਂਤਾਂ ਜੇ (op_case == 3 || op_case == 5) {
            attention_ਮਾਪ = context.get_ਨਿਵੇਸ਼("attention_size");
        } ਨਹੀਂਤਾਂ {
            attention_ਮਾਪ = context.get_ਨਿਵੇਸ਼("attention_size_swa");
        }

        Output<Node> seq_ਸਰਗਰਮ_ਸ਼ੁਰੂ;
        Output<Node> seq_ਸਰਗਰਮ_ਸਿਰਾ;
        ਜੇ (context.has_ਨਿਵੇਸ਼("seq_active_start")) {
            seq_ਸਰਗਰਮ_ਸ਼ੁਰੂ = context.get_ਨਿਵੇਸ਼("seq_active_start");
            seq_ਸਰਗਰਮ_ਸਿਰਾ = context.get_ਨਿਵੇਸ਼("seq_active_end");
        } ਨਹੀਂਤਾਂ {
            int64_t n_seq_ਸਰਗਰਮ = output_shape[0];
            size_t ਵਿੱਥ = *((size_t *) context.get_ਨਿਵੇਸ਼_op_params(0));
            int64_t seq_ਸਰਗਰਮ_ਸ਼ੁਰੂ_val = ਵਿੱਥ / context.get_ਨਿਵੇਸ਼_stride(0)[0];
            int64_t seq_ਸਰਗਰਮ_ਸਿਰਾ_val = seq_ਸਰਗਰਮ_ਸ਼ੁਰੂ_val + n_seq_ਸਰਗਰਮ;
            seq_ਸਰਗਰਮ_ਸ਼ੁਰੂ = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_ਸਰਗਰਮ_ਸ਼ੁਰੂ_val});
            seq_ਸਰਗਰਮ_ਸਿਰਾ = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_ਸਰਗਰਮ_ਸਿਰਾ_val});
        }

        // 1. reshape to [n_seq, ctx_per_seq, n_heads, head_size] (for `-fa off` [n_seq, n_heads, head_size, ctx_per_seq])
        // 2. slice out the active sequences
        // 3. slice out the attention part in each sequence
        // 4. permute (skip for `-fa off`)
        ਸਵੈਚਾਲਿਤ zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ਸਵੈਚਾਲਿਤ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        ਜੇ (op_case == 3 || op_case == 4) {
            ਸਵੈਚਾਲਿਤ src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, ctx_ਪ੍ਰਤੀ_seq, n_heads, ਸਿਰ_ਮਾਪ}),
                ਝੂਠ);
            ov::Output<ov::Node> after_seq_ਟੁਕੜਾ;
            ਜੇ (n_seq == 1) {
                after_seq_ਟੁਕੜਾ = src_reshaped;
            } ਨਹੀਂਤਾਂ {
                after_seq_ਟੁਕੜਾ =
                    std::make_shared<ov::op::v8::ਟੁਕੜਾ>(src_reshaped, seq_ਸਰਗਰਮ_ਸ਼ੁਰੂ, seq_ਸਰਗਰਮ_ਸਿਰਾ, one, zero);
            }
            ਸਵੈਚਾਲਿਤ slice2 = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(after_seq_ਟੁਕੜਾ, zero, attention_ਮਾਪ, one, one);
            res = std::make_shared<ov::op::v1::Transpose>(slice2, perm);
        } ਨਹੀਂਤਾਂ {
            ਸਵੈਚਾਲਿਤ three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            ਸਵੈਚਾਲਿਤ src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, n_heads, ਸਿਰ_ਮਾਪ, ctx_ਪ੍ਰਤੀ_seq}),
                ਝੂਠ);
            ov::Output<ov::Node> after_seq_ਟੁਕੜਾ;
            ਜੇ (n_seq == 1) {
                after_seq_ਟੁਕੜਾ = src_reshaped;
            } ਨਹੀਂਤਾਂ {
                after_seq_ਟੁਕੜਾ =
                    std::make_shared<ov::op::v8::ਟੁਕੜਾ>(src_reshaped, seq_ਸਰਗਰਮ_ਸ਼ੁਰੂ, seq_ਸਰਗਰਮ_ਸਿਰਾ, one, zero);
            }
            ਸਵੈਚਾਲਿਤ slice2 = std::make_shared<ov::op::v8::ਟੁਕੜਾ>(after_seq_ਟੁਕੜਾ, zero, attention_ਮਾਪ, one, three);
            res = slice2;
        }
    }
    ਵਾਪਸ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
