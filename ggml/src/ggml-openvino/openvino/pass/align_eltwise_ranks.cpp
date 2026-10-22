#ಅನ್ತರ್ಭಾವ "align_eltwise_ranks.h"

#ಅನ್ತರ್ಭಾವ <numeric>
#ಅನ್ತರ್ಭಾವ <openvino/op/add.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/constant.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/divide.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/multiply.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/sqrt.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/subtract.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/op/unsqueeze.hpp>
#ಅನ್ತರ್ಭಾವ <openvino/pass/pattern/op/wrap_type.hpp>

ನಾಮಾಕಾಶ ov {
ನಾಮಾಕಾಶ frontend {
ನಾಮಾಕಾಶ ggml {
ನಾಮಾಕಾಶ pass {

ನಾಮಾಕಾಶ {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
ತಾರ್ಕಿಕ is_rms_norm_output(ಧ್ರುವ ov::Output<ov::Node> & value, ಪೂರ್ಣಾಙ್ಕ depth = 1) {
    ಧ್ರುವ auto * ಗಂಟು = value.get_ಗಂಟು();
    ಯದಿ (!ov::is_type<ov::op::v1::Multiply>(ಗಂಟು)) {
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
    ಪ್ರತಿ (ಧ್ರುವ auto & ಒಳಹರಿವು : ಗಂಟು->ಒಳಹರಿವು_ಮೌಲ್ಯಗಳು()) {
        ಧ್ರುವ auto * src = ಒಳಹರಿವು.get_ಗಂಟು();
        ಯದಿ (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_ಒಳಹರಿವು_ಗಂಟು_ptr(1))) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        ಯದಿ (depth > 0 && is_rms_norm_output(ಒಳಹರಿವು, depth - 1)) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
    }
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

}  // namespace

AlignEltwiseOperandRanks::AlignEltwiseOperandRanks() {
    auto eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::Add, ov::op::v1::Multiply, ov::op::v1::Subtract>();

    ಧ್ರುವ auto callback = [ಏತತ್](ov::pass::pattern::Matcher & m) {
        auto ಗಂಟು = m.get_match_ಬೇರು();
        ಯದಿ (ಗಂಟು->get_ಒಳಹರಿವು_size() != 2) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        auto lhs = ಗಂಟು->ಒಳಹರಿವು_ಮೌಲ್ಯ(0);
        auto rhs = ಗಂಟು->ಒಳಹರಿವು_ಮೌಲ್ಯ(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        ಯದಿ (ov::is_type<ov::op::v0::Constant>(lhs.get_ಗಂಟು()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_ಗಂಟು())) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ auto lhs_rank = lhs.get_partial_shape().rank();
        ಧ್ರುವ auto rhs_rank = rhs.get_partial_shape().rank();
        ಯದಿ (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಧ್ರುವ size_t shorter_idx = lhs_rank.get_length() < rhs_rank.get_length() ? 0 : 1;
        ಧ್ರುವ auto & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        ಯದಿ (is_rms_norm_output(shorter)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಧ್ರುವ int64_t diff = std::abs(lhs_rank.get_length() - rhs_rank.get_length());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        ಗಂಟು->ಒಳಹರಿವು(shorter_idx).replace_source_output(unsqueeze->output(0));
        register_new_ಗಂಟು(unsqueeze);
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
