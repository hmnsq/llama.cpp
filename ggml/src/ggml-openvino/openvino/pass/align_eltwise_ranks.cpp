#অন্তর্ভাৱ "align_eltwise_ranks.h"

#অন্তর্ভাৱ <numeric>
#অন্তর্ভাৱ <openvino/op/add.hpp>
#অন্তর্ভাৱ <openvino/op/constant.hpp>
#অন্তর্ভাৱ <openvino/op/divide.hpp>
#অন্তর্ভাৱ <openvino/op/multiply.hpp>
#অন্তর্ভাৱ <openvino/op/sqrt.hpp>
#অন্তর্ভাৱ <openvino/op/subtract.hpp>
#অন্তর্ভাৱ <openvino/op/unsqueeze.hpp>
#অন্তর্ভাৱ <openvino/pass/pattern/op/wrap_type.hpp>

নামাকাশ ov {
নামাকাশ frontend {
নামাকাশ ggml {
নামাকাশ pass {

নামাকাশ {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
তার্কিক is_rms_norm_output(ধ্রুৱ ov::Output<ov::Node> & value, পূর্ণাঙ্ক depth = 1) {
    ধ্রুৱ auto * গ্ৰন্থি = value.get_গ্ৰন্থি();
    যদি (!ov::is_type<ov::op::v1::Multiply>(গ্ৰন্থি)) {
        ঘূৰাই মিছা;
    }
    বাবে (ধ্রুৱ auto & প্ৰবেশ : গ্ৰন্থি->প্ৰবেশ_মানসমূহ()) {
        ধ্রুৱ auto * src = প্ৰবেশ.get_গ্ৰন্থি();
        যদি (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_প্ৰবেশ_গ্ৰন্থি_ptr(1))) {
            ঘূৰাই সঁচা;
        }
        যদি (depth > 0 && is_rms_norm_output(প্ৰবেশ, depth - 1)) {
            ঘূৰাই সঁচা;
        }
    }
    ঘূৰাই মিছা;
}

}  // namespace

AlignEltwiseOperandRanks::AlignEltwiseOperandRanks() {
    auto eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::Add, ov::op::v1::Multiply, ov::op::v1::Subtract>();

    ধ্রুৱ auto callback = [এতত্](ov::pass::pattern::Matcher & m) {
        auto গ্ৰন্থি = m.get_match_মূল();
        যদি (গ্ৰন্থি->get_প্ৰবেশ_size() != 2) {
            ঘূৰাই মিছা;
        }

        auto lhs = গ্ৰন্থি->প্ৰবেশ_মান(0);
        auto rhs = গ্ৰন্থি->প্ৰবেশ_মান(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        যদি (ov::is_type<ov::op::v0::Constant>(lhs.get_গ্ৰন্থি()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_গ্ৰন্থি())) {
            ঘূৰাই মিছা;
        }

        ধ্রুৱ auto lhs_rank = lhs.get_partial_shape().rank();
        ধ্রুৱ auto rhs_rank = rhs.get_partial_shape().rank();
        যদি (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            ঘূৰাই মিছা;
        }

        ধ্রুৱ size_t shorter_idx = lhs_rank.get_length() < rhs_rank.get_length() ? 0 : 1;
        ধ্রুৱ auto & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        যদি (is_rms_norm_output(shorter)) {
            ঘূৰাই মিছা;
        }
        ধ্রুৱ int64_t diff = std::abs(lhs_rank.get_length() - rhs_rank.get_length());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        গ্ৰন্থি->প্ৰবেশ(shorter_idx).replace_source_output(unsqueeze->output(0));
        register_new_গ্ৰন্থি(unsqueeze);
        ঘূৰাই সঁচা;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
