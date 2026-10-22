#అన్తర్భావ "../node_context.h"
#అన్తర్భావ "../op_table.h"
#అన్తర్భావ "../utils.h"

#అన్తర్భావ <climits>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <memory>
#అన్తర్భావ <openvino/core/node.hpp>
#అన్తర్భావ <openvino/op/add.hpp>
#అన్తర్భావ <openvino/op/concat.hpp>
#అన్తర్భావ <openvino/op/constant.hpp>
#అన్తర్భావ <openvino/op/reshape.hpp>
#అన్తర్భావ <openvino/op/slice.hpp>
#అన్తర్భావ <openvino/op/transpose.hpp>
#అన్తర్భావ <vector>

నామాకాశ ov {
నామాకాశ frontend {
నామాకాశ ggml {
నామాకాశ op {

Output_సదిశ translate_క్రమమార్పు(స్థిరరాశి కణుపు_Context & context) {
    అంకె_inputs_check(context, 1, 1);

    పూర్ణాంకం op_case = context.get_op_case();
    FRONT_చివర_CHECK_IMPLEMENTED(op_case != 0, "Unsupported PERMUTE case");
    // op_case 1 is trivial permute
    // op_case 2 is to permute Q. It has a preceding VIEW that reshapes Q to restore the sequqence dimension
    // op_case 3 4 it to permute KV cache in the default layout
    // op_case 5 6 is to permute V cache when `-fa off`, where v_trans=true

    ov::Output<Node> res;
    ov::Output<Node> src;
    ఒకవేళ (op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6) {
        src = context.get_ఆదానం(0);
    } లేకపోతే {
        src = process_view_ఆదానం_new(context, 0);
    }
    std::vector<int64_t> perm_విలువలు{0, 2, 1, 3};
    స్థిరరాశి int32_t * op_params = context.get_output_op_params();
    ఒకవేళ (op_params != శూన్యసూచిక) {
        కొరకు (size_t ఆదానం_axis = 0; ఆదానం_axis < perm_విలువలు.size(); ++ఆదానం_axis) {
            స్థిరరాశి size_t output_axis = static_cast<size_t>(op_params[ఆదానం_axis]);
            perm_విలువలు[perm_విలువలు.size() - 1 - output_axis] =
                static_cast<int64_t>(perm_విలువలు.size() - 1 - ఆదానం_axis);
        }
    }
    ఒకవేళ (op_case == 1 || context.is_stateful()) {
        // The stateful path carries hidden-state tensors in a rank-3 layout (the
        // leading batch dim is dropped, e.g. Gemma4's per-layer-embedding path). The
        // perm above is rank-4; when the actual input is rank-3, drop the batch axis
        // (perm[0], which is always the identity 0 here) and shift the rest down by 1
        // so the transpose order matches the input rank.
        std::vector<int64_t> perm_వాడిన = perm_విలువలు;
        స్థిరరాశి స్వయంచాలక & src_ps = src.get_partial_shape();
        ఒకవేళ (src_ps.rank().is_static() && src_ps.rank().get_నిడివి() == 3 && perm_విలువలు.size() == 4 &&
            perm_విలువలు[0] == 0) {
            perm_వాడిన = {perm_విలువలు[1] - 1, perm_విలువలు[2] - 1, perm_విలువలు[3] - 1};
        }
        స్వయంచాలక perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{perm_వాడిన.size()}, perm_వాడిన);
        res = std::make_shared<ov::op::v1::Transpose>(src, perm);
    } లేకపోతే ఒకవేళ (op_case == 2) {
        స్వయంచాలక perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_విలువలు);
        స్వయంచాలక output_shape = context.get_output_shape().గమ్యం_shape();
        స్వయంచాలక n_heads = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[1]});
        స్వయంచాలక తల_కొలత = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[3]});
        స్వయంచాలక n_seq_చురుకైన = context.has_ఆదానం("n_seq_active") ?
                                context.get_ఆదానం("n_seq_active") :
                                ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[0]});
        స్వయంచాలక neg_one = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});

        స్వయంచాలక new_shape =
            std::make_shared<ov::op::v0::Concat>(ov::Output_సదిశ{n_seq_చురుకైన, neg_one, n_heads, తల_కొలత}, 0);

        // // Alternative
        // auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        // auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{n_seq_active, neg_one, zero, zero}, 0);

        స్వయంచాలక reshaped = std::make_shared<ov::op::v1::Reshape>(src, new_shape, నిజం);
        res = std::make_shared<ov::op::v1::Transpose>(reshaped, perm);
    } లేకపోతే {
        స్వయంచాలక perm = ov::op::v0::Constant::create(ov::element::i64, {4}, perm_విలువలు);
        స్వయంచాలక cache_shape = src.get_partial_shape();
        స్వయంచాలక output_shape = context.get_output_shape().గమ్యం_shape();
        int64_t తల_కొలత = output_shape[3];
        int64_t n_heads = output_shape[1];
        ఒకవేళ (op_case == 5 || op_case == 6) {
            తల_కొలత = output_shape[2];
            n_heads = output_shape[1];
        }
        int64_t ctx_చొప్పున_seq = cache_shape[2].is_static() ? cache_shape[2].get_నిడివి() : -1;
        int64_t n_seq = cache_shape[1].get_నిడివి();

        Output<Node> attention_కొలత;
        ఒకవేళ (!context.has_ఆదానం("attention_size")) {
            attention_కొలత = ov::op::v0::Constant::create(ov::element::i64, {1}, {output_shape[2]});
        } లేకపోతే ఒకవేళ (op_case == 3 || op_case == 5) {
            attention_కొలత = context.get_ఆదానం("attention_size");
        } లేకపోతే {
            attention_కొలత = context.get_ఆదానం("attention_size_swa");
        }

        Output<Node> seq_చురుకైన_ప్రారంభం;
        Output<Node> seq_చురుకైన_చివర;
        ఒకవేళ (context.has_ఆదానం("seq_active_start")) {
            seq_చురుకైన_ప్రారంభం = context.get_ఆదానం("seq_active_start");
            seq_చురుకైన_చివర = context.get_ఆదానం("seq_active_end");
        } లేకపోతే {
            int64_t n_seq_చురుకైన = output_shape[0];
            size_t స్థానభ్రంశం = *((size_t *) context.get_ఆదానం_op_params(0));
            int64_t seq_చురుకైన_ప్రారంభం_val = స్థానభ్రంశం / context.get_ఆదానం_stride(0)[0];
            int64_t seq_చురుకైన_చివర_val = seq_చురుకైన_ప్రారంభం_val + n_seq_చురుకైన;
            seq_చురుకైన_ప్రారంభం = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_చురుకైన_ప్రారంభం_val});
            seq_చురుకైన_చివర = ov::op::v0::Constant::create(ov::element::i64, {1}, {seq_చురుకైన_చివర_val});
        }

        // 1. reshape to [n_seq, ctx_per_seq, n_heads, head_size] (for `-fa off` [n_seq, n_heads, head_size, ctx_per_seq])
        // 2. slice out the active sequences
        // 3. slice out the attention part in each sequence
        // 4. permute (skip for `-fa off`)
        స్వయంచాలక zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        స్వయంచాలక one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});

        ఒకవేళ (op_case == 3 || op_case == 4) {
            స్వయంచాలక src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, ctx_చొప్పున_seq, n_heads, తల_కొలత}),
                అబద్ధం);
            ov::Output<ov::Node> after_seq_ముక్క;
            ఒకవేళ (n_seq == 1) {
                after_seq_ముక్క = src_reshaped;
            } లేకపోతే {
                after_seq_ముక్క =
                    std::make_shared<ov::op::v8::ముక్క>(src_reshaped, seq_చురుకైన_ప్రారంభం, seq_చురుకైన_చివర, one, zero);
            }
            స్వయంచాలక slice2 = std::make_shared<ov::op::v8::ముక్క>(after_seq_ముక్క, zero, attention_కొలత, one, one);
            res = std::make_shared<ov::op::v1::Transpose>(slice2, perm);
        } లేకపోతే {
            స్వయంచాలక three = ov::op::v0::Constant::create(ov::element::i64, {1}, {3});
            స్వయంచాలక src_reshaped = std::make_shared<ov::op::v1::Reshape>(
                src, ov::op::v0::Constant::create(ov::element::i64, {4}, {n_seq, n_heads, తల_కొలత, ctx_చొప్పున_seq}),
                అబద్ధం);
            ov::Output<ov::Node> after_seq_ముక్క;
            ఒకవేళ (n_seq == 1) {
                after_seq_ముక్క = src_reshaped;
            } లేకపోతే {
                after_seq_ముక్క =
                    std::make_shared<ov::op::v8::ముక్క>(src_reshaped, seq_చురుకైన_ప్రారంభం, seq_చురుకైన_చివర, one, zero);
            }
            స్వయంచాలక slice2 = std::make_shared<ov::op::v8::ముక్క>(after_seq_ముక్క, zero, attention_కొలత, one, three);
            res = slice2;
        }
    }
    తిరిగి rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
