#ضمن "align_eltwise_ranks.h"

#ضمن <numeric>
#ضمن <openvino/op/add.hpp>
#ضمن <openvino/op/constant.hpp>
#ضمن <openvino/op/divide.hpp>
#ضمن <openvino/op/multiply.hpp>
#ضمن <openvino/op/sqrt.hpp>
#ضمن <openvino/op/subtract.hpp>
#ضمن <openvino/op/unsqueeze.hpp>
#ضمن <openvino/pass/pattern/op/wrap_type.hpp>

نطاق ov {
نطاق frontend {
نطاق ggml {
نطاق pass {

نطاق {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
منطقي is_rms_norm_output(ثابت ov::Output<ov::Node> & value, صحيح depth = 1) {
    ثابت auto * node = value.get_node();
    ܐܢ (!ov::is_type<ov::op::v1::Multiply>(node)) {
        ܦܢܐ ܕܓܠܐ;
    }
    لكل (ثابت auto & input : node->input_values()) {
        ثابت auto * src = input.get_node();
        ܐܢ (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_input_node_ptr(1))) {
            ܦܢܐ ܫܪܪܐ;
        }
        ܐܢ (depth > 0 && is_rms_norm_output(input, depth - 1)) {
            ܦܢܐ ܫܪܪܐ;
        }
    }
    ܦܢܐ ܕܓܠܐ;
}

}  // namespace

AlignEltwiseOperandRanks::AlignEltwiseOperandRanks() {
    auto eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::Add, ov::op::v1::Multiply, ov::op::v1::Subtract>();

    ثابت auto callback = [هذا](ov::pass::pattern::Matcher & m) {
        auto node = m.get_match_root();
        ܐܢ (node->get_input_size() != 2) {
            ܦܢܐ ܕܓܠܐ;
        }

        auto lhs = node->input_value(0);
        auto rhs = node->input_value(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        ܐܢ (ov::is_type<ov::op::v0::Constant>(lhs.get_node()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_node())) {
            ܦܢܐ ܕܓܠܐ;
        }

        ثابت auto lhs_rank = lhs.get_partial_shape().rank();
        ثابت auto rhs_rank = rhs.get_partial_shape().rank();
        ܐܢ (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            ܦܢܐ ܕܓܠܐ;
        }

        ثابت size_t shorter_idx = lhs_rank.get_length() < rhs_rank.get_length() ? 0 : 1;
        ثابت auto & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        ܐܢ (is_rms_norm_output(shorter)) {
            ܦܢܐ ܕܓܠܐ;
        }
        ثابت int64_t diff = std::abs(lhs_rank.get_length() - rhs_rank.get_length());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        node->input(shorter_idx).replace_source_output(unsqueeze->output(0));
        register_new_node(unsqueeze);
        ܦܢܐ ܫܪܪܐ;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
