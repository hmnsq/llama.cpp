#अन्तर्भाव "align_eltwise_ranks.h"

#अन्तर्भाव <numeric>
#अन्तर्भाव <openvino/op/add.hpp>
#अन्तर्भाव <openvino/op/constant.hpp>
#अन्तर्भाव <openvino/op/divide.hpp>
#अन्तर्भाव <openvino/op/multiply.hpp>
#अन्तर्भाव <openvino/op/sqrt.hpp>
#अन्तर्भाव <openvino/op/subtract.hpp>
#अन्तर्भाव <openvino/op/unsqueeze.hpp>
#अन्तर्भाव <openvino/pass/pattern/op/wrap_type.hpp>

नामाकाश ov {
नामाकाश frontend {
नामाकाश ggml {
नामाकाश pass {

नामाकाश {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
तार्किक is_rms_norm_output(ध्रुव ov::Output<ov::Node> & value, पूर्णाङ्क depth = 1) {
    ध्रुव auto * गाठ = value.get_गाठ();
    यदि (!ov::is_type<ov::op::v1::Multiply>(गाठ)) {
        प्रत्यागम असत्य;
    }
    प्रति (ध्रुव auto & आदान : गाठ->आदान_मूल्ये()) {
        ध्रुव auto * src = आदान.get_गाठ();
        यदि (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_आदान_गाठ_ptr(1))) {
            प्रत्यागम सत्य;
        }
        यदि (depth > 0 && is_rms_norm_output(आदान, depth - 1)) {
            प्रत्यागम सत्य;
        }
    }
    प्रत्यागम असत्य;
}

}  // namespace

AlignEltwiseOperandRanks::AlignEltwiseOperandRanks() {
    auto eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::Add, ov::op::v1::Multiply, ov::op::v1::Subtract>();

    ध्रुव auto callback = [एतत्](ov::pass::pattern::Matcher & m) {
        auto गाठ = m.get_match_मूळ();
        यदि (गाठ->get_आदान_size() != 2) {
            प्रत्यागम असत्य;
        }

        auto lhs = गाठ->आदान_मूल्य(0);
        auto rhs = गाठ->आदान_मूल्य(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        यदि (ov::is_type<ov::op::v0::Constant>(lhs.get_गाठ()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_गाठ())) {
            प्रत्यागम असत्य;
        }

        ध्रुव auto lhs_rank = lhs.get_partial_shape().rank();
        ध्रुव auto rhs_rank = rhs.get_partial_shape().rank();
        यदि (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            प्रत्यागम असत्य;
        }

        ध्रुव size_t shorter_idx = lhs_rank.get_length() < rhs_rank.get_length() ? 0 : 1;
        ध्रुव auto & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        यदि (is_rms_norm_output(shorter)) {
            प्रत्यागम असत्य;
        }
        ध्रुव int64_t diff = std::abs(lhs_rank.get_length() - rhs_rank.get_length());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        गाठ->आदान(shorter_idx).replace_source_output(unsqueeze->output(0));
        register_new_गाठ(unsqueeze);
        प्रत्यागम सत्य;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
