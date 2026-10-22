#ಅನ್ತರ್ಭಾವ "../node_context.h"
#ಅನ್ತರ್ಭಾವ "../op_table.h"
#ಅನ್ತರ್ಭಾವ "../utils.h"

#ಅನ್ತರ್ಭಾವ <climits>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <openvino/core/node.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/concat.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/reshape.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/slice.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/transpose.hpp>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ op {

Output_ಸದಿಶ translate_ಕ್ರಮಬದಲಾವಣೆ(ಸ್ಥಿರಾಂಕ ಗಂಟು_Context & context) {
    ಅಂಕೆ_inputs_check(context, 1, 1);

    ಪೂರ್ಣಾಂಕ op_case = context.get_op_case();
    FRONT_ಕೊನೆ_CHECK_IMPLEMENTED(op_case != 0, "Unsupported PERMUTE case");
    // op_case 1 is trivial permute
    // op_case 2 is to permute Q. It has a preceding VIEW that reshapes Q to restore the sequqence dimension
    // op_case 3 4 it to permute KV cache in the default layout
    // op_case 5 6 is to permute V cache when `-fa off`, where v_trans=true

    ov::Output<Node> res;
    ov::Output<Node> src;
    ಆದರೆ (op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6) {
        src = context.get_ಒಳಹರಿವು(0);
    } ಇಲ್ಲದಿದ್ದರೆ {
        src = process_view_ಒಳಹರಿವು_new(context, 0);
    }
    std::vector<int64_t> perm_ಮೌಲ್ಯಗಳು{0, 2, 1, 3};
    ಸ್ಥಿರಾಂಕ int32_t * op_params = context.get_output_op_params();
    ಆದರೆ (op_params != ಶೂನ್ಯಸೂಚಕ) {
        ಗಾಗಿ (size_t ಒಳಹರಿವು_axis = 0; ಒಳಹರಿವು_axis < perm_ಮೌಲ್ಯಗಳು.size(); ++ಒಳಹರಿವು_axis) {
            ಸ್ಥಿರಾಂಕ size_t output_axis = static_cast<size_t>(op_params[ಒಳಹರಿವು_axis]);
            perm_ಮೌಲ್ಯಗಳು[perm_ಮೌಲ್ಯಗಳು.size() - 1 - output_axis] =
                static_cast<int64_t>(perm_ಮೌಲ್ಯಗಳು.size() - 1 - ಒಳಹರಿವು_axis);
        }
    }
    ಆದರೆ (op_case == 1 || context.is_stateful()) {
        // The stateful path carries hidden-state tensors in a rank-3 layout (the
        // leading batch dim is dropped, e.g. Gemma4's per-layer-embedding path). The
        // perm above is rank-4; when the actual input is rank-3, drop the batch axis
        // (perm[0], which is always the identity 0 here) and shift the rest down by 1
        // so the transpose order matches the input rank.
        std::vector<int64_t> perm_ಬಳಸಿದ = perm_ಮೌಲ್ಯಗಳು;
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & src_ps = src.get_partial_shape();
        ಆದರೆ (src_ps.rank().is_static() && src_ps.rank().get_ಉದ್ದಳತೆ() == 3 && perm_ಮೌಲ್ಯಗಳು.size() == 4 &&
            perm_ಮೌಲ್ಯಗಳು[0] == 0) {
            perm_ಬಳಸಿದ = {perm_ಮೌಲ್ಯಗಳು[1] - 1, perm_ಮೌಲ್ಯಗಳು[2] - 1, perm_ಮೌಲ್ಯಗಳು[3] - 1};
        }
        ಸ್ವಯಂಚಾಲಿತ perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{perm_ಬಳಸಿದ.size()}, perm_ಬಳಸಿದ);
        res = std::make_shared<ov::op::v1::Transpose>(src, perm);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op_case == 2) {
        ಸ್ವಯಂಚಾಲಿತ perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_ಮೌಲ್ಯಗಳು);
        ಸ್ವಯಂಚಾಲಿತ output_shape = context.get_output_shape().ಗಮ್ಯ_shape();
        ಸ್ವಯಂಚಾಲಿತ n_heads = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[1]});
        ಸ್ವಯಂಚಾಲಿತ ತಲೆ_ಗಾತ್ರ = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        ಸ್ವಯಂಚಾಲಿತ n_seq_ಸಕ್ರಿಯ = context.has_ಒಳಹರಿವು("n_seq_active") ?
                                context.get_ಒಳಹರಿವು("n_seq_active") :
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[0]});
        ಸ್ವಯಂಚಾಲಿತ neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        ಸ್ವಯಂಚಾಲಿತ new_shape =
            std::make_shared<ov::op::v0::Concat>(ov::Output_ಸದಿಶ{n_seq_ಸಕ್ರಿಯ, neg_one, n_heads, ತಲೆ_ಗಾತ್ರ}, 0);

        // // Alternative
        // auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        // auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, zero, zero}, 0);

        ಸ್ವಯಂಚಾಲಿತ reshaped = std::make_shared<ov::op::v1::Reshape>(src, new_shape, ನಿಜ);
        res = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ವಯಂಚಾಲಿತ perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_ಮೌಲ್ಯಗಳು);
        ಸ್ವಯಂಚಾಲಿತ cache_shape = src.get_partial_shape();
        ಸ್ವಯಂಚಾಲಿತ output_shape = context.get_output_shape().ಗಮ್ಯ_shape();
        int64_t ತಲೆ_ಗಾತ್ರ = output_shape[3];
        int64_t n_heads = output_shape[1];
        ಆದರೆ (op_case == 5 || op_case == 6) {
            ತಲೆ_ಗಾತ್ರ = output_shape[2];
            n_heads = output_shape[1];
        }
        int64_t ctx_ಪ್ರತಿ_seq = cache_shape[2].is_static() ? cache_shape[2].get_ಉದ್ದಳತೆ() : -1;
        int64_t n_seq = cache_shape[1].get_ಉದ್ದಳತೆ();

        Output<Node> attention_ಗಾತ್ರ;
        ಆದರೆ (!context.has_ಒಳಹರಿವು("attention_size")) {
            attention_ಗಾತ್ರ = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[2]});
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op_case == 3 || op_case == 5) {
            attention_ಗಾತ್ರ = context.get_ಒಳಹರಿವು("attention_size");
        } ಇಲ್ಲದಿದ್ದರೆ {
            attention_ಗಾತ್ರ = context.get_ಒಳಹರಿವು("attention_size_swa");
        }

        Output<Node> seq_ಸಕ್ರಿಯ_ಆರಂಭ;
        Output<Node> seq_ಸಕ್ರಿಯ_ಕೊನೆ;
        ಆದರೆ (context.has_ಒಳಹರಿವು("seq_active_start")) {
            seq_ಸಕ್ರಿಯ_ಆರಂಭ = context.get_ಒಳಹರಿವು("seq_active_start");
            seq_ಸಕ್ರಿಯ_ಕೊನೆ = context.get_ಒಳಹರಿವು("seq_active_end");
        } ಇಲ್ಲದಿದ್ದರೆ {
            int64_t n_seq_ಸಕ್ರಿಯ = output_shape[0];
            size_t ಸ್ಥಾನಾಂತರ = *((size_t *) context.get_ಒಳಹರಿವು_op_params(0));
            int64_t seq_ಸಕ್ರಿಯ_ಆರಂಭ_val = ಸ್ಥಾನಾಂತರ / context.get_ಒಳಹರಿವು_stride(0)[0];
            int64_t seq_ಸಕ್ರಿಯ_ಕೊನೆ_val = seq_ಸಕ್ರಿಯ_ಆರಂಭ_val + n_seq_ಸಕ್ರಿಯ;
            seq_ಸಕ್ರಿಯ_ಆರಂಭ = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_ಸಕ್ರಿಯ_ಆರಂಭ_val});
            seq_ಸಕ್ರಿಯ_ಕೊನೆ = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_ಸಕ್ರಿಯ_ಕೊನೆ_val});
        }

        // 1. reshape to [n_seq, ctx_per_seq, n_heads, head_size] (for `-fa off` [n_seq, n_heads, head_size, ctx_per_seq])
        // 2. slice out the active sequences
        // 3. slice out the attention part in each sequence
        // 4. permute (skip for `-fa off`)
        ಸ್ವಯಂಚಾಲಿತ zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        ಸ್ವಯಂಚಾಲಿತ one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        ಆದರೆ (op_case == 3 || op_case == 4) {
            ಸ್ವಯಂಚಾಲಿತ src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, ctx_ಪ್ರತಿ_seq, n_heads, ತಲೆ_ಗಾತ್ರ}),
                ಸುಳ್ಳು);
            ov::Output<ov::Node> after_seq_ತುಂಡು;
            ಆದರೆ (n_seq == 1) {
                after_seq_ತುಂಡು = src_reshaped;
            } ಇಲ್ಲದಿದ್ದರೆ {
                after_seq_ತುಂಡು =
                    std::make_shared<ov::op::v8::ತುಂಡು>(src_reshaped, seq_ಸಕ್ರಿಯ_ಆರಂಭ, seq_ಸಕ್ರಿಯ_ಕೊನೆ, one, zero);
            }
            ಸ್ವಯಂಚಾಲಿತ slice2 = std::make_shared<ov::op::v8::ತುಂಡು>(after_seq_ತುಂಡು, zero, attention_ಗಾತ್ರ, one, one);
            res = std::make_shared<ov::op::v1::Transpose>(slice2, perm);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ವಯಂಚಾಲಿತ three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            ಸ್ವಯಂಚಾಲಿತ src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, n_heads, ತಲೆ_ಗಾತ್ರ, ctx_ಪ್ರತಿ_seq}),
                ಸುಳ್ಳು);
            ov::Output<ov::Node> after_seq_ತುಂಡು;
            ಆದರೆ (n_seq == 1) {
                after_seq_ತುಂಡು = src_reshaped;
            } ಇಲ್ಲದಿದ್ದರೆ {
                after_seq_ತುಂಡು =
                    std::make_shared<ov::op::v8::ತುಂಡು>(src_reshaped, seq_ಸಕ್ರಿಯ_ಆರಂಭ, seq_ಸಕ್ರಿಯ_ಕೊನೆ, one, zero);
            }
            ಸ್ವಯಂಚಾಲಿತ slice2 = std::make_shared<ov::op::v8::ತುಂಡು>(after_seq_ತುಂಡು, zero, attention_ಗಾತ್ರ, one, three);
            res = slice2;
        }
    }
    ಹಿಂತಿರುಗಿ rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
