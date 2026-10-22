#அந்தர்பாவ "align_eltwise_ranks.h"

#அந்தர்பாவ <numeric>
#அந்தர்பாவ <openvino/op/add.hpp>
#அந்தர்பாவ <openvino/op/constant.hpp>
#அந்தர்பாவ <openvino/op/divide.hpp>
#அந்தர்பாவ <openvino/op/multiply.hpp>
#அந்தர்பாவ <openvino/op/sqrt.hpp>
#அந்தர்பாவ <openvino/op/subtract.hpp>
#அந்தர்பாவ <openvino/op/unsqueeze.hpp>
#அந்தர்பாவ <openvino/pass/pattern/op/wrap_type.hpp>

நாமாகாஶ ov {
நாமாகாஶ frontend {
நாமாகாஶ ggml {
நாமாகாஶ pass {

நாமாகாஶ {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
தருக்க is_rms_norm_output(மாறிலி ov::Output<ov::Node> & value, முழு depth = 1) {
    மாறிலி auto * முனை = value.get_முனை();
    எனில் (!ov::is_type<ov::op::v1::Multiply>(முனை)) {
        திரும்பு பொய்;
    }
    க்கு (மாறிலி auto & உள்ளீடு : முனை->உள்ளீடு_மதிப்புகள்()) {
        மாறிலி auto * src = உள்ளீடு.get_முனை();
        எனில் (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_உள்ளீடு_முனை_ptr(1))) {
            திரும்பு உண்மை;
        }
        எனில் (depth > 0 && is_rms_norm_output(உள்ளீடு, depth - 1)) {
            திரும்பு உண்மை;
        }
    }
    திரும்பு பொய்;
}

}  // namespace

AlignEltwiseOperandRanks::AlignEltwiseOperandRanks() {
    auto eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::Add, ov::op::v1::Multiply, ov::op::v1::Subtract>();

    மாறிலி auto callback = [இது](ov::pass::pattern::Matcher & m) {
        auto முனை = m.get_match_வேர்();
        எனில் (முனை->get_உள்ளீடு_size() != 2) {
            திரும்பு பொய்;
        }

        auto lhs = முனை->உள்ளீடு_மதிப்பு(0);
        auto rhs = முனை->உள்ளீடு_மதிப்பு(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        எனில் (ov::is_type<ov::op::v0::Constant>(lhs.get_முனை()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_முனை())) {
            திரும்பு பொய்;
        }

        மாறிலி auto lhs_rank = lhs.get_partial_shape().rank();
        மாறிலி auto rhs_rank = rhs.get_partial_shape().rank();
        எனில் (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            திரும்பு பொய்;
        }

        மாறிலி size_t shorter_idx = lhs_rank.get_length() < rhs_rank.get_length() ? 0 : 1;
        மாறிலி auto & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        எனில் (is_rms_norm_output(shorter)) {
            திரும்பு பொய்;
        }
        மாறிலி int64_t diff = std::abs(lhs_rank.get_length() - rhs_rank.get_length());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        முனை->உள்ளீடு(shorter_idx).replace_source_output(unsqueeze->output(0));
        register_new_முனை(unsqueeze);
        திரும்பு உண்மை;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
