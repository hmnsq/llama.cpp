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

Output_ويڪٽر translate_ترتيب_بدلايو(ثابت ڳنڍ_Context & context) {
    انگ_inputs_check(context, 1, 1);

    صحيح op_case = context.get_op_case();
    FRONT_پڇاڙي_CHECK_IMPLEMENTED(op_case != 0, "Unsupported PERMUTE case");
    // op_case 1 is trivial permute
    // op_case 2 is to permute Q. It has a preceding VIEW that reshapes Q to restore the sequqence dimension
    // op_case 3 4 it to permute KV cache in the default layout
    // op_case 5 6 is to permute V cache when `-fa off`, where v_trans=true

    ov::Output<Node> res;
    ov::Output<Node> src;
    إذا (op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6) {
        src = context.get_داخلا(0);
    } وإلا {
        src = process_view_داخلا_new(context, 0);
    }
    std::vector<int64_t> perm_قدرون{0, 2, 1, 3};
    ثابت int32_t * op_params = context.get_output_op_params();
    إذا (op_params != خالي_اشارو) {
        لكل (size_t داخلا_axis = 0; داخلا_axis < perm_قدرون.size(); ++داخلا_axis) {
            ثابت size_t output_axis = static_cast<size_t>(op_params[داخلا_axis]);
            perm_قدرون[perm_قدرون.size() - 1 - output_axis] =
                static_cast<int64_t>(perm_قدرون.size() - 1 - داخلا_axis);
        }
    }
    إذا (op_case == 1 || context.is_stateful()) {
        // The stateful path carries hidden-state tensors in a rank-3 layout (the
        // leading batch dim is dropped, e.g. Gemma4's per-layer-embedding path). The
        // perm above is rank-4; when the actual input is rank-3, drop the batch axis
        // (perm[0], which is always the identity 0 here) and shift the rest down by 1
        // so the transpose order matches the input rank.
        std::vector<int64_t> perm_استعمال_ٿيل = perm_قدرون;
        ثابت خودڪار & src_ps = src.get_partial_shape();
        إذا (src_ps.rank().is_static() && src_ps.rank().get_ڊيگهائي() == 3 && perm_قدرون.size() == 4 &&
            perm_قدرون[0] == 0) {
            perm_استعمال_ٿيل = {perm_قدرون[1] - 1, perm_قدرون[2] - 1, perm_قدرون[3] - 1};
        }
        خودڪار perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{perm_استعمال_ٿيل.size()}, perm_استعمال_ٿيل);
        res = std::make_shared<ov::op::v1::Transpose>(src, perm);
    } وإلا إذا (op_case == 2) {
        خودڪار perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_قدرون);
        خودڪار output_shape = context.get_output_shape().منزل_shape();
        خودڪار n_heads = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[1]});
        خودڪار سر_سائيز = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        خودڪار n_seq_سرگرم = context.has_داخلا("n_seq_active") ?
                                context.get_داخلا("n_seq_active") :
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[0]});
        خودڪار neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        خودڪار new_shape =
            std::make_shared<ov::op::v0::Concat>(ov::Output_ويڪٽر{n_seq_سرگرم, neg_one, n_heads, سر_سائيز}, 0);

        // // Alternative
        // auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        // auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, zero, zero}, 0);

        خودڪار reshaped = std::make_shared<ov::op::v1::Reshape>(src, new_shape, سچ);
        res = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    } وإلا {
        خودڪار perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_قدرون);
        خودڪار cache_shape = src.get_partial_shape();
        خودڪار output_shape = context.get_output_shape().منزل_shape();
        int64_t سر_سائيز = output_shape[3];
        int64_t n_heads = output_shape[1];
        إذا (op_case == 5 || op_case == 6) {
            سر_سائيز = output_shape[2];
            n_heads = output_shape[1];
        }
        int64_t ctx_في_seq = cache_shape[2].is_static() ? cache_shape[2].get_ڊيگهائي() : -1;
        int64_t n_seq = cache_shape[1].get_ڊيگهائي();

        Output<Node> attention_سائيز;
        إذا (!context.has_داخلا("attention_size")) {
            attention_سائيز = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[2]});
        } وإلا إذا (op_case == 3 || op_case == 5) {
            attention_سائيز = context.get_داخلا("attention_size");
        } وإلا {
            attention_سائيز = context.get_داخلا("attention_size_swa");
        }

        Output<Node> seq_سرگرم_شروعات;
        Output<Node> seq_سرگرم_پڇاڙي;
        إذا (context.has_داخلا("seq_active_start")) {
            seq_سرگرم_شروعات = context.get_داخلا("seq_active_start");
            seq_سرگرم_پڇاڙي = context.get_داخلا("seq_active_end");
        } وإلا {
            int64_t n_seq_سرگرم = output_shape[0];
            size_t وٿي = *((size_t *) context.get_داخلا_op_params(0));
            int64_t seq_سرگرم_شروعات_val = وٿي / context.get_داخلا_stride(0)[0];
            int64_t seq_سرگرم_پڇاڙي_val = seq_سرگرم_شروعات_val + n_seq_سرگرم;
            seq_سرگرم_شروعات = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_سرگرم_شروعات_val});
            seq_سرگرم_پڇاڙي = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_سرگرم_پڇاڙي_val});
        }

        // 1. reshape to [n_seq, ctx_per_seq, n_heads, head_size] (for `-fa off` [n_seq, n_heads, head_size, ctx_per_seq])
        // 2. slice out the active sequences
        // 3. slice out the attention part in each sequence
        // 4. permute (skip for `-fa off`)
        خودڪار zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        خودڪار one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        إذا (op_case == 3 || op_case == 4) {
            خودڪار src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, ctx_في_seq, n_heads, سر_سائيز}),
                خطأ);
            ov::Output<ov::Node> after_seq_ٽڪرو;
            إذا (n_seq == 1) {
                after_seq_ٽڪرو = src_reshaped;
            } وإلا {
                after_seq_ٽڪرو =
                    std::make_shared<ov::op::v8::ٽڪرو>(src_reshaped, seq_سرگرم_شروعات, seq_سرگرم_پڇاڙي, one, zero);
            }
            خودڪار slice2 = std::make_shared<ov::op::v8::ٽڪرو>(after_seq_ٽڪرو, zero, attention_سائيز, one, one);
            res = std::make_shared<ov::op::v1::Transpose>(slice2, perm);
        } وإلا {
            خودڪار three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            خودڪار src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, n_heads, سر_سائيز, ctx_في_seq}),
                خطأ);
            ov::Output<ov::Node> after_seq_ٽڪرو;
            إذا (n_seq == 1) {
                after_seq_ٽڪرو = src_reshaped;
            } وإلا {
                after_seq_ٽڪرو =
                    std::make_shared<ov::op::v8::ٽڪرو>(src_reshaped, seq_سرگرم_شروعات, seq_سرگرم_پڇاڙي, one, zero);
            }
            خودڪار slice2 = std::make_shared<ov::op::v8::ٽڪرو>(after_seq_ٽڪرو, zero, attention_سائيز, one, three);
            res = slice2;
        }
    }
    أرجع rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
