#અન્તર્ભાવ "../node_context.h"
#અન્તર્ભાવ "../op_table.h"
#અન્તર્ભાવ "../utils.h"

#અન્તર્ભાવ <climits>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <openvino/core/node.hpp>
#અન્તર્ભાવ <openvino/op/add.hpp>
#અન્તર્ભાવ <openvino/op/concat.hpp>
#અન્તર્ભાવ <openvino/op/constant.hpp>
#અન્તર્ભાવ <openvino/op/reshape.hpp>
#અન્તર્ભાવ <openvino/op/slice.hpp>
#અન્તર્ભાવ <openvino/op/transpose.hpp>
#અન્તર્ભાવ <vector>

નામાકાશ ov {
નામાકાશ frontend {
નામાકાશ ggml {
નામાકાશ op {

OutputVector translate_permute(અચલ ગાંઠ_Context & context) {
    અંક_inputs_check(context, 1, 1);

    પૂર્ણાંક op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case != 0, "Unsupported PERMUTE case");
    // op_case 1 is trivial permute
    // op_case 2 is to permute Q. It has a preceding VIEW that reshapes Q to restore the sequqence dimension
    // op_case 3 4 it to permute KV cache in the default layout
    // op_case 5 6 is to permute V cache when `-fa off`, where v_trans=true

    ov::Output<Node> res;
    ov::Output<Node> src;
    જો (op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6) {
        src = context.get_આગત(0);
    } નહીંતર {
        src = process_view_આગત_new(context, 0);
    }
    std::vector<int64_t> perm_કિંમતો{0, 2, 1, 3};
    અચલ int32_t * op_params = context.get_output_op_params();
    જો (op_params != nullptr) {
        માટે (size_t આગત_axis = 0; આગત_axis < perm_કિંમતો.size(); ++આગત_axis) {
            અચલ size_t output_axis = static_cast<size_t>(op_params[આગત_axis]);
            perm_કિંમતો[perm_કિંમતો.size() - 1 - output_axis] =
                static_cast<int64_t>(perm_કિંમતો.size() - 1 - આગત_axis);
        }
    }
    જો (op_case == 1 || context.is_stateful()) {
        // The stateful path carries hidden-state tensors in a rank-3 layout (the
        // leading batch dim is dropped, e.g. Gemma4's per-layer-embedding path). The
        // perm above is rank-4; when the actual input is rank-3, drop the batch axis
        // (perm[0], which is always the identity 0 here) and shift the rest down by 1
        // so the transpose order matches the input rank.
        std::vector<int64_t> perm_used = perm_કિંમતો;
        અચલ auto & src_ps = src.get_partial_shape();
        જો (src_ps.rank().is_static() && src_ps.rank().get_length() == 3 && perm_કિંમતો.size() == 4 &&
            perm_કિંમતો[0] == 0) {
            perm_used = {perm_કિંમતો[1] - 1, perm_કિંમતો[2] - 1, perm_કિંમતો[3] - 1};
        }
        auto perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{perm_used.size()}, perm_used);
        res = std::make_shared<ov::op::v1::Transpose>(src, perm);
    } નહીંતર જો (op_case == 2) {
        auto perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_કિંમતો);
        auto output_shape = context.get_output_shape().to_shape();
        auto n_heads = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[1]});
        auto head_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        auto n_seq_active = context.has_આગત("n_seq_active") ?
                                context.get_આગત("n_seq_active") :
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[0]});
        auto neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        auto new_shape =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, n_heads, head_size}, 0);

        // // Alternative
        // auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        // auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, zero, zero}, 0);

        auto reshaped = std::make_shared<ov::op::v1::Reshape>(src, new_shape, સાચું);
        res = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    } નહીંતર {
        auto perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_કિંમતો);
        auto cache_shape = src.get_partial_shape();
        auto output_shape = context.get_output_shape().to_shape();
        int64_t head_size = output_shape[3];
        int64_t n_heads = output_shape[1];
        જો (op_case == 5 || op_case == 6) {
            head_size = output_shape[2];
            n_heads = output_shape[1];
        }
        int64_t ctx_per_seq = cache_shape[2].is_static() ? cache_shape[2].get_length() : -1;
        int64_t n_seq = cache_shape[1].get_length();

        Output<Node> attention_size;
        જો (!context.has_આગત("attention_size")) {
            attention_size = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[2]});
        } નહીંતર જો (op_case == 3 || op_case == 5) {
            attention_size = context.get_આગત("attention_size");
        } નહીંતર {
            attention_size = context.get_આગત("attention_size_swa");
        }

        Output<Node> seq_active_start;
        Output<Node> seq_active_end;
        જો (context.has_આગત("seq_active_start")) {
            seq_active_start = context.get_આગત("seq_active_start");
            seq_active_end = context.get_આગત("seq_active_end");
        } નહીંતર {
            int64_t n_seq_active = output_shape[0];
            size_t offset = *((size_t *) context.get_આગત_op_params(0));
            int64_t seq_active_start_val = offset / context.get_આગત_stride(0)[0];
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

        જો (op_case == 3 || op_case == 4) {
            auto src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, ctx_per_seq, n_heads, head_size}),
                ખોટું);
            ov::Output<ov::Node> after_seq_slice;
            જો (n_seq == 1) {
                after_seq_slice = src_reshaped;
            } નહીંતર {
                after_seq_slice =
                    std::make_shared<ov::op::v8::Slice>(src_reshaped, seq_active_start, seq_active_end, one, zero);
            }
            auto slice2 = std::make_shared<ov::op::v8::Slice>(after_seq_slice, zero, attention_size, one, one);
            res = std::make_shared<ov::op::v1::Transpose>(slice2, perm);
        } નહીંતર {
            auto three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            auto src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, n_heads, head_size, ctx_per_seq}),
                ખોટું);
            ov::Output<ov::Node> after_seq_slice;
            જો (n_seq == 1) {
                after_seq_slice = src_reshaped;
            } નહીંતર {
                after_seq_slice =
                    std::make_shared<ov::op::v8::Slice>(src_reshaped, seq_active_start, seq_active_end, one, zero);
            }
            auto slice2 = std::make_shared<ov::op::v8::Slice>(after_seq_slice, zero, attention_size, one, three);
            res = slice2;
        }
    }
    પાછા rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
