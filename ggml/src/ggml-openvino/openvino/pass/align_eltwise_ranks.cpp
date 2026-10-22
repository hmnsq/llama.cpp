#അന്തര്ഭാവ "align_eltwise_ranks.h"

#അന്തര്ഭാവ <numeric>
#അന്തര്ഭാവ <openvino/op/add.hpp>
#അന്തര്ഭാവ <openvino/op/constant.hpp>
#അന്തര്ഭാവ <openvino/op/divide.hpp>
#അന്തര്ഭാവ <openvino/op/multiply.hpp>
#അന്തര്ഭാവ <openvino/op/sqrt.hpp>
#അന്തര്ഭാവ <openvino/op/subtract.hpp>
#അന്തര്ഭാവ <openvino/op/unsqueeze.hpp>
#അന്തര്ഭാവ <openvino/pass/pattern/op/wrap_type.hpp>

നാമാകാശ ov {
നാമാകാശ frontend {
നാമാകാശ ggml {
നാമാകാശ pass {

നാമാകാശ {

// True for an RMS-norm output, x * (1 / sqrt(mean(x^2) + eps)) as translate_rms_norm builds it, optionally
// scaled by the norm weight.
തർക്കം is_rms_norm_output(സ്ഥിരം ov::Output<ov::Node> & value, പൂർണ്ണസംഖ്യ depth = 1) {
    സ്ഥിരം auto * കണ്ണി = value.get_കണ്ണി();
    എങ്കിൽ (!ov::is_type<ov::op::v1::Multiply>(കണ്ണി)) {
        തിരികെ അസത്യം;
    }
    വേണ്ടി (സ്ഥിരം auto & നിവേശം : കണ്ണി->നിവേശം_മൂല്യങ്ങൾ()) {
        സ്ഥിരം auto * src = നിവേശം.get_കണ്ണി();
        എങ്കിൽ (ov::is_type<ov::op::v1::Divide>(src) && ov::is_type<ov::op::v0::Sqrt>(src->get_നിവേശം_കണ്ണി_ptr(1))) {
            തിരികെ സത്യം;
        }
        എങ്കിൽ (depth > 0 && is_rms_norm_output(നിവേശം, depth - 1)) {
            തിരികെ സത്യം;
        }
    }
    തിരികെ അസത്യം;
}

}  // namespace

AlignEltwiseOperandRanks::AlignEltwiseOperandRanks() {
    auto eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::Add, ov::op::v1::Multiply, ov::op::v1::Subtract>();

    സ്ഥിരം auto callback = [ഇത്](ov::pass::pattern::Matcher & m) {
        auto കണ്ണി = m.get_match_വേര്();
        എങ്കിൽ (കണ്ണി->get_നിവേശം_size() != 2) {
            തിരികെ അസത്യം;
        }

        auto lhs = കണ്ണി->നിവേശം_മൂല്യം(0);
        auto rhs = കണ്ണി->നിവേശം_മൂല്യം(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        എങ്കിൽ (ov::is_type<ov::op::v0::Constant>(lhs.get_കണ്ണി()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_കണ്ണി())) {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം auto lhs_rank = lhs.get_partial_shape().rank();
        സ്ഥിരം auto rhs_rank = rhs.get_partial_shape().rank();
        എങ്കിൽ (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            തിരികെ അസത്യം;
        }

        സ്ഥിരം size_t shorter_idx = lhs_rank.get_length() < rhs_rank.get_length() ? 0 : 1;
        സ്ഥിരം auto & shorter = shorter_idx == 0 ? lhs : rhs;

        // The defect is with the norm output as the higher-rank operand. When the norm output is the
        // lower-rank one (gemma-3 adds it to a rank-4 residual), unsqueezing it puts the Unsqueeze between
        // `rms` and its post-op, and the GPU plugin then computes the layer wrongly.
        എങ്കിൽ (is_rms_norm_output(shorter)) {
            തിരികെ അസത്യം;
        }
        സ്ഥിരം int64_t diff = std::abs(lhs_rank.get_length() - rhs_rank.get_length());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        കണ്ണി->നിവേശം(shorter_idx).replace_source_output(unsqueeze->output(0));
        register_new_കണ്ണി(unsqueeze);
        തിരികെ സത്യം;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
