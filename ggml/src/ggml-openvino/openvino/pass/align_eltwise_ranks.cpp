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
ತಾರ್ಕಿಕ is_rms_norm_output(ಸ್ಥಿರಾಂಕ ov::Output<ov::Node> & value, ಪೂರ್ಣಾಂಕ ಆಳ = 1) {
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * ಗಂಟು = value.get_ಗಂಟು();
    ಆದರೆ (!ov::is_type<ov::op::v1::Multiply>(ಗಂಟು)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಒಳಹರಿವು : ಗಂಟು->ಒಳಹರಿವು_ಮೌಲ್ಯಗಳು()) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * src = ಒಳಹರಿವು.get_ಗಂಟು();
        ಆದರೆ (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_ಒಳಹರಿವು_ಗಂಟು_ptr(1))) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಆದರೆ (ಆಳ > 0 && is_rms_norm_output(ಒಳಹರಿವು, ಆಳ - 1)) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

}  // namespace

Align_Eltwise_ಕಾರ್ಯಾಂಶ_Ranks::Align_Eltwise_ಕಾರ್ಯಾಂಶ_Ranks() {
    ಸ್ವಯಂಚಾಲಿತ eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::ಕೂಡಿಸು, ov::op::v1::Multiply, ov::op::v1::ಕಳೆ>();

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ callback = [ಇದು](ov::pass::pattern::Matcher & m) {
        ಸ್ವಯಂಚಾಲಿತ ಗಂಟು = m.get_ಹೊಂದಿಕೆ_ಬೇರು();
        ಆದರೆ (ಗಂಟು->get_ಒಳಹರಿವು_ಗಾತ್ರ() != 2) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ವಯಂಚಾಲಿತ lhs = ಗಂಟು->ಒಳಹರಿವು_ಮೌಲ್ಯ(0);
        ಸ್ವಯಂಚಾಲಿತ rhs = ಗಂಟು->ಒಳಹರಿವು_ಮೌಲ್ಯ(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        ಆದರೆ (ov::is_type<ov::op::v0::Constant>(lhs.get_ಗಂಟು()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_ಗಂಟು())) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ lhs_rank = lhs.get_partial_shape().rank();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ rhs_rank = rhs.get_partial_shape().rank();
        ಆದರೆ (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ size_t shorter_idx = lhs_rank.get_ಉದ್ದಳತೆ() < rhs_rank.get_ಉದ್ದಳತೆ() ? 0 : 1;
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        ಆದರೆ (is_rms_norm_output(shorter)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಸ್ಥಿರಾಂಕ int64_t diff = std::ನಿರಪೇಕ್ಷ(lhs_rank.get_ಉದ್ದಳತೆ() - rhs_rank.get_ಉದ್ದಳತೆ());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        ಸ್ವಯಂಚಾಲಿತ unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        ಗಂಟು->ಒಳಹರಿವು(shorter_idx).replace_ಮೂಲ_output(unsqueeze->output(0));
        ರಿಜಿಸ್ಟರ್_new_ಗಂಟು(unsqueeze);
        ಹಿಂತಿರುಗಿ ನಿಜ;
    };

    ರಿಜಿಸ್ಟರ್_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
