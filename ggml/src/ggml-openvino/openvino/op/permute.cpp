#ضمن "../node_context.h"
#ضمن "../op_table.h"
#ضمن "../utils.h"

#ضمن <climits>
#ضمن <cstdint>
#ضمن <memory>
#ضمن <openvino/core/node.hpp>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/concat.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/reshape.hpp>
#ضمن <openvino/op/slice.hpp>
#ضمن <openvino/op/transpose.hpp>
#ضمن <vector>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق op {

Output_وکتور translate_جایگشت(ثابت گره_Context & context) {
    رقم_inputs_check(context, 1, 1);

    صحيح op_case = context.get_op_case();
    FRONT_انتها_CHECK_IMPLEMENTED(op_case != 0, "Unsupported PERMUTE case");
    // op_case 1 is trivial permute
    // op_case 2 is to permute Q. It has a preceding VIEW that reshapes Q to restore the sequqence dimension
    // op_case 3 4 it to permute KV cache in the default layout
    // op_case 5 6 is to permute V cache when `-fa off`, where v_trans=true

    ov::Output<Node> res;
    ov::Output<Node> src;
    إذا (op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6) {
        src = context.get_درون_داد(0);
    } وإلا {
        src = process_view_درون_داد_new(context, 0);
    }
    std::vector<int64_t> perm_مقادیر{0, 2, 1, 3};
    ثابت int32_t * op_params = context.get_output_op_params();
    إذا (op_params != اشاره_تهی) {
        لكل (size_t درون_داد_axis = 0; درون_داد_axis < perm_مقادیر.size(); ++درون_داد_axis) {
            ثابت size_t output_axis = static_cast<size_t>(op_params[درون_داد_axis]);
            perm_مقادیر[perm_مقادیر.size() - 1 - output_axis] =
                static_cast<int64_t>(perm_مقادیر.size() - 1 - درون_داد_axis);
        }
    }
    إذا (op_case == 1 || context.is_stateful()) {
        // The stateful path carries hidden-state tensors in a rank-3 layout (the
        // leading batch dim is dropped, e.g. Gemma4's per-layer-embedding path). The
        // perm above is rank-4; when the actual input is rank-3, drop the batch axis
        // (perm[0], which is always the identity 0 here) and shift the rest down by 1
        // so the transpose order matches the input rank.
        std::vector<int64_t> perm_استفاده_شده = perm_مقادیر;
        ثابت خودکار & src_ps = src.get_partial_shape();
        إذا (src_ps.rank().is_static() && src_ps.rank().get_درازی() == 3 && perm_مقادیر.size() == 4 &&
            perm_مقادیر[0] == 0) {
            perm_استفاده_شده = {perm_مقادیر[1] - 1, perm_مقادیر[2] - 1, perm_مقادیر[3] - 1};
        }
        خودکار perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{perm_استفاده_شده.size()}, perm_استفاده_شده);
        res = std::make_shared<ov::op::v1::Transpose>(src, perm);
    } وإلا إذا (op_case == 2) {
        خودکار perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_مقادیر);
        خودکار output_shape = context.get_output_shape().مقصد_shape();
        خودکار n_heads = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[1]});
        خودکار سر_اندازه = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        خودکار n_seq_فعال = context.has_درون_داد("n_seq_active") ?
                                context.get_درون_داد("n_seq_active") :
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[0]});
        خودکار neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        خودکار new_shape =
            std::make_shared<ov::op::v0::Concat>(ov::Output_وکتور{n_seq_فعال, neg_one, n_heads, سر_اندازه}, 0);

        // // Alternative
        // auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        // auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, zero, zero}, 0);

        خودکار reshaped = std::make_shared<ov::op::v1::Reshape>(src, new_shape, درست);
        res = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    } وإلا {
        خودکار perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_مقادیر);
        خودکار cache_shape = src.get_partial_shape();
        خودکار output_shape = context.get_output_shape().مقصد_shape();
        int64_t سر_اندازه = output_shape[3];
        int64_t n_heads = output_shape[1];
        إذا (op_case == 5 || op_case == 6) {
            سر_اندازه = output_shape[2];
            n_heads = output_shape[1];
        }
        int64_t ctx_در_هر_seq = cache_shape[2].is_static() ? cache_shape[2].get_درازی() : -1;
        int64_t n_seq = cache_shape[1].get_درازی();

        Output<Node> attention_اندازه;
        إذا (!context.has_درون_داد("attention_size")) {
            attention_اندازه = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[2]});
        } وإلا إذا (op_case == 3 || op_case == 5) {
            attention_اندازه = context.get_درون_داد("attention_size");
        } وإلا {
            attention_اندازه = context.get_درون_داد("attention_size_swa");
        }

        Output<Node> seq_فعال_شروع;
        Output<Node> seq_فعال_انتها;
        إذا (context.has_درون_داد("seq_active_start")) {
            seq_فعال_شروع = context.get_درون_داد("seq_active_start");
            seq_فعال_انتها = context.get_درون_داد("seq_active_end");
        } وإلا {
            int64_t n_seq_فعال = output_shape[0];
            size_t جابه_جایی = *((size_t *) context.get_درون_داد_op_params(0));
            int64_t seq_فعال_شروع_val = جابه_جایی / context.get_درون_داد_stride(0)[0];
            int64_t seq_فعال_انتها_val = seq_فعال_شروع_val + n_seq_فعال;
            seq_فعال_شروع = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_فعال_شروع_val});
            seq_فعال_انتها = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_فعال_انتها_val});
        }

        // 1. reshape to [n_seq, ctx_per_seq, n_heads, head_size] (for `-fa off` [n_seq, n_heads, head_size, ctx_per_seq])
        // 2. slice out the active sequences
        // 3. slice out the attention part in each sequence
        // 4. permute (skip for `-fa off`)
        خودکار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودکار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        إذا (op_case == 3 || op_case == 4) {
            خودکار src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, ctx_در_هر_seq, n_heads, سر_اندازه}),
                خطأ);
            ov::Output<ov::Node> after_seq_برش;
            إذا (n_seq == 1) {
                after_seq_برش = src_reshaped;
            } وإلا {
                after_seq_برش =
                    std::make_shared<ov::op::v8::برش>(src_reshaped, seq_فعال_شروع, seq_فعال_انتها, one, zero);
            }
            خودکار slice2 = std::make_shared<ov::op::v8::برش>(after_seq_برش, zero, attention_اندازه, one, one);
            res = std::make_shared<ov::op::v1::Transpose>(slice2, perm);
        } وإلا {
            خودکار three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            خودکار src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, n_heads, سر_اندازه, ctx_در_هر_seq}),
                خطأ);
            ov::Output<ov::Node> after_seq_برش;
            إذا (n_seq == 1) {
                after_seq_برش = src_reshaped;
            } وإلا {
                after_seq_برش =
                    std::make_shared<ov::op::v8::برش>(src_reshaped, seq_فعال_شروع, seq_فعال_انتها, one, zero);
            }
            خودکار slice2 = std::make_shared<ov::op::v8::برش>(after_seq_برش, zero, attention_اندازه, one, three);
            res = slice2;
        }
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
