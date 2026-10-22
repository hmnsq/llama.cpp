#অন্তর্ভাব "align_eltwise_ranks.h"

#অন্তর্ভাব <numeric>
#অন্তর্ভাব <openvino/op/add.hpp>
#অন্তর্ভাব <openvino/op/constant.hpp>
#অন্তর্ভাব <openvino/op/divide.hpp>
#অন্তর্ভাব <openvino/op/multiply.hpp>
#অন্তর্ভাব <openvino/op/sqrt.hpp>
#অন্তর্ভাব <openvino/op/subtract.hpp>
#অন্তর্ভাব <openvino/op/unsqueeze.hpp>
#অন্তর্ভাব <openvino/pass/pattern/op/wrap_type.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ pass {

নামাকাশ {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
যৌক্তিক is_rms_norm_output(ধ্রুবক ov::Output<ov::Node> & value, পূর্ণসংখ্যা depth = 1) {
    ধ্রুবক auto * গ্রন্থি = value.get_গ্রন্থি();
    যদি (!ov::is_type<ov::op::v1::Multiply>(গ্রন্থি)) {
        ফেরত মিথ্যা;
    }
    জন্য (ধ্রুবক auto & প্রবেশ : গ্রন্থি->প্রবেশ_মানসমূহ()) {
        ধ্রুবক auto * src = প্রবেশ.get_গ্রন্থি();
        যদি (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_প্রবেশ_গ্রন্থি_ptr(1))) {
            ফেরত সত্য;
        }
        যদি (depth > 0 && is_rms_norm_output(প্রবেশ, depth - 1)) {
            ফেরত সত্য;
        }
    }
    ফেরত মিথ্যা;
}

}  // namespace

AlignEltwiseOperandRanks::AlignEltwiseOperandRanks() {
    auto eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::Add, ov::op::v1::Multiply, ov::op::v1::Subtract>();

    ধ্রুবক auto callback = [এই](ov::pass::pattern::Matcher & m) {
        auto গ্রন্থি = m.get_match_মূল();
        যদি (গ্রন্থি->get_প্রবেশ_আকার() != 2) {
            ফেরত মিথ্যা;
        }

        auto lhs = গ্রন্থি->প্রবেশ_মান(0);
        auto rhs = গ্রন্থি->প্রবেশ_মান(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        যদি (ov::is_type<ov::op::v0::Constant>(lhs.get_গ্রন্থি()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_গ্রন্থি())) {
            ফেরত মিথ্যা;
        }

        ধ্রুবক auto lhs_rank = lhs.get_partial_shape().rank();
        ধ্রুবক auto rhs_rank = rhs.get_partial_shape().rank();
        যদি (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            ফেরত মিথ্যা;
        }

        ধ্রুবক size_t shorter_idx = lhs_rank.get_দৈর্ঘ্য() < rhs_rank.get_দৈর্ঘ্য() ? 0 : 1;
        ধ্রুবক auto & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        যদি (is_rms_norm_output(shorter)) {
            ফেরত মিথ্যা;
        }
        ধ্রুবক int64_t diff = std::abs(lhs_rank.get_দৈর্ঘ্য() - rhs_rank.get_দৈর্ঘ্য());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        গ্রন্থি->প্রবেশ(shorter_idx).replace_source_output(unsqueeze->output(0));
        register_new_গ্রন্থি(unsqueeze);
        ফেরত সত্য;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
