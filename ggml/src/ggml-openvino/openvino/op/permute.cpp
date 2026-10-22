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

Output_सदिशम् translate_permute(ध्रुव ग्रन्थिः_Context & context) {
    अङ्कः_inputs_check(context, 1, 1);

    पूर्णाङ्क op_case = context.get_op_case();
    FRONT_अन्तः_CHECK_IMPLEMENTED(op_case != 0, "Unsupported PERMUTE case");
    // op_case 1 is trivial permute
    // op_case 2 is to permute Q. It has a preceding VIEW that reshapes Q to restore the sequqence dimension
    // op_case 3 4 it to permute KV cache in the default layout
    // op_case 5 6 is to permute V cache when `-fa off`, where v_trans=true

    ov::Output<Node> res;
    ov::Output<Node> src;
    यदि (op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6) {
        src = context.get_निवेशः(0);
    } अन्यथा {
        src = process_view_निवेशः_new(context, 0);
    }
    std::vector<int64_t> perm_मानानि{0, 2, 1, 3};
    ध्रुव int32_t * op_params = context.get_output_op_params();
    यदि (op_params != शून्यसूचकः) {
        प्रति (size_t निवेशः_axis = 0; निवेशः_axis < perm_मानानि.size(); ++निवेशः_axis) {
            ध्रुव size_t output_axis = static_cast<size_t>(op_params[निवेशः_axis]);
            perm_मानानि[perm_मानानि.size() - 1 - output_axis] =
                static_cast<int64_t>(perm_मानानि.size() - 1 - निवेशः_axis);
        }
    }
    यदि (op_case == 1 || context.is_stateful()) {
        // The stateful path carries hidden-state tensors in a rank-3 layout (the
        // leading batch dim is dropped, e.g. Gemma4's per-layer-embedding path). The
        // perm above is rank-4; when the actual input is rank-3, drop the batch axis
        // (perm[0], which is always the identity 0 here) and shift the rest down by 1
        // so the transpose order matches the input rank.
        std::vector<int64_t> perm_प्रयुक्तम् = perm_मानानि;
        ध्रुव स्वचालित & src_ps = src.get_partial_shape();
        यदि (src_ps.rank().is_static() && src_ps.rank().get_दीर्घता() == 3 && perm_मानानि.size() == 4 &&
            perm_मानानि[0] == 0) {
            perm_प्रयुक्तम् = {perm_मानानि[1] - 1, perm_मानानि[2] - 1, perm_मानानि[3] - 1};
        }
        स्वचालित perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{perm_प्रयुक्तम्.size()}, perm_प्रयुक्तम्);
        res = std::make_shared<ov::op::v1::Transpose>(src, perm);
    } अन्यथा यदि (op_case == 2) {
        स्वचालित perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_मानानि);
        स्वचालित output_shape = context.get_output_shape().गन्तव्यम्_shape();
        स्वचालित n_heads = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[1]});
        स्वचालित शीर्ष_आकार = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        स्वचालित n_seq_सक्रिय = context.has_निवेशः("n_seq_active") ?
                                context.get_निवेशः("n_seq_active") :
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[0]});
        स्वचालित neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        स्वचालित new_shape =
            std::make_shared<ov::op::v0::Concat>(ov::Output_सदिशम्{n_seq_सक्रिय, neg_one, n_heads, शीर्ष_आकार}, 0);

        // // Alternative
        // auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        // auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, zero, zero}, 0);

        स्वचालित reshaped = std::make_shared<ov::op::v1::Reshape>(src, new_shape, सत्य);
        res = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    } अन्यथा {
        स्वचालित perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_मानानि);
        स्वचालित cache_shape = src.get_partial_shape();
        स्वचालित output_shape = context.get_output_shape().गन्तव्यम्_shape();
        int64_t शीर्ष_आकार = output_shape[3];
        int64_t n_heads = output_shape[1];
        यदि (op_case == 5 || op_case == 6) {
            शीर्ष_आकार = output_shape[2];
            n_heads = output_shape[1];
        }
        int64_t ctx_प्रत्येकम्_seq = cache_shape[2].is_static() ? cache_shape[2].get_दीर्घता() : -1;
        int64_t n_seq = cache_shape[1].get_दीर्घता();

        Output<Node> attention_आकार;
        यदि (!context.has_निवेशः("attention_size")) {
            attention_आकार = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[2]});
        } अन्यथा यदि (op_case == 3 || op_case == 5) {
            attention_आकार = context.get_निवेशः("attention_size");
        } अन्यथा {
            attention_आकार = context.get_निवेशः("attention_size_swa");
        }

        Output<Node> seq_सक्रिय_start;
        Output<Node> seq_सक्रिय_अन्तः;
        यदि (context.has_निवेशः("seq_active_start")) {
            seq_सक्रिय_start = context.get_निवेशः("seq_active_start");
            seq_सक्रिय_अन्तः = context.get_निवेशः("seq_active_end");
        } अन्यथा {
            int64_t n_seq_सक्रिय = output_shape[0];
            size_t विस्थापनम् = *((size_t *) context.get_निवेशः_op_params(0));
            int64_t seq_सक्रिय_start_val = विस्थापनम् / context.get_निवेशः_stride(0)[0];
            int64_t seq_सक्रिय_अन्तः_val = seq_सक्रिय_start_val + n_seq_सक्रिय;
            seq_सक्रिय_start = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_सक्रिय_start_val});
            seq_सक्रिय_अन्तः = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_सक्रिय_अन्तः_val});
        }

        // 1. reshape to [n_seq, ctx_per_seq, n_heads, head_size] (for `-fa off` [n_seq, n_heads, head_size, ctx_per_seq])
        // 2. slice out the active sequences
        // 3. slice out the attention part in each sequence
        // 4. permute (skip for `-fa off`)
        स्वचालित zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        स्वचालित one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        यदि (op_case == 3 || op_case == 4) {
            स्वचालित src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, ctx_प्रत्येकम्_seq, n_heads, शीर्ष_आकार}),
                असत्य);
            ov::Output<ov::Node> after_seq_खण्डिका;
            यदि (n_seq == 1) {
                after_seq_खण्डिका = src_reshaped;
            } अन्यथा {
                after_seq_खण्डिका =
                    std::make_shared<ov::op::v8::खण्डिका>(src_reshaped, seq_सक्रिय_start, seq_सक्रिय_अन्तः, one, zero);
            }
            स्वचालित slice2 = std::make_shared<ov::op::v8::खण्डिका>(after_seq_खण्डिका, zero, attention_आकार, one, one);
            res = std::make_shared<ov::op::v1::Transpose>(slice2, perm);
        } अन्यथा {
            स्वचालित three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            स्वचालित src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, n_heads, शीर्ष_आकार, ctx_प्रत्येकम्_seq}),
                असत्य);
            ov::Output<ov::Node> after_seq_खण्डिका;
            यदि (n_seq == 1) {
                after_seq_खण्डिका = src_reshaped;
            } अन्यथा {
                after_seq_खण्डिका =
                    std::make_shared<ov::op::v8::खण्डिका>(src_reshaped, seq_सक्रिय_start, seq_सक्रिय_अन्तः, one, zero);
            }
            स्वचालित slice2 = std::make_shared<ov::op::v8::खण्डिका>(after_seq_खण्डिका, zero, attention_आकार, one, three);
            res = slice2;
        }
    }
    प्रत्यागम rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
