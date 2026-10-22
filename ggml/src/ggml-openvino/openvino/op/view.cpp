#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_view(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    ប្រសិនបើ (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        ត្រឡប់ {context.get_input(0)};
    }

    ប្រសិនបើ (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        const std::string & vname = context.get_name();
        ប្រសិនបើ (vname.find("ffn_moe_weighted") != std::string::npos) {
            auto src_ps = context.get_input_shape(0);
            auto dst_ps = context.get_output_shape();
            ប្រសិនបើ (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                auto sst = context.get_input_stride(0);
                auto dst = context.get_output_stride();
                size_t voff = context.get_output_op_offset();
                auto ss = src_ps.to_shape();
                auto dd = dst_ps.to_shape();
                const size_t nd = ss.size();
                // Stateful models drop the leading size-1 batch dim, so the real OV tensor can
                // be one rank lower than the ggml shape metadata above. Axis indices derived
                // from that metadata must be shifted down by the difference before they are
                // used as OV axes. Bail out if the dims we would drop are not all 1.
                const auto in_rank = context.get_input(0).get_partial_shape().rank();
                const ចំនួនគត់ axis_shift =
                    in_rank.is_static() ? (ចំនួនគត់) nd - (ចំនួនគត់) in_rank.get_length() : 0;
                bool shift_ok = axis_shift >= 0 && (size_t) axis_shift < nd;
                សម្រាប់ (ចំនួនគត់ a = 0; a < axis_shift && shift_ok; ++a) {
                    shift_ok = (ss[a] == 1 && dd[a] == 1);
                }
                ប្រសិនបើ (shift_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<bool> used(nd, false);
                    std::vector<ចំនួនគត់> dst_to_src(nd, -1);
                    bool ok = true;
                    សម្រាប់ (size_t d = 0; d < nd; ++d) {
                        ប្រសិនបើ (dd[d] == 1) {
                            continue;
                        }
                        ចំនួនគត់ found = -1;
                        សម្រាប់ (size_t s = 0; s < nd; ++s) {
                            ប្រសិនបើ (!used[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                found = (ចំនួនគត់) s;
                                bha_nga;
                            }
                        }
                        ប្រសិនបើ (found < 0) {
                            ok = false;
                            bha_nga;
                        }
                        used[found] = true;
                        dst_to_src[d] = found;
                    }
                    ចំនួនគត់ dropped = -1;
                    ប្រសិនបើ (ok) {
                        សម្រាប់ (size_t s = 0; s < nd; ++s) {
                            ប្រសិនបើ (!used[s] && ss[s] > 1) {
                                ប្រសិនបើ (dropped >= 0) {
                                    ok = false;
                                    bha_nga;
                                }
                                dropped = (ចំនួនគត់) s;
                            }
                        }
                    }
                    ប្រសិនបើ (ok && dropped >= 0) {
                        const size_t dstr = sst[dropped];
                        const int64_t dsz = (int64_t) ss[dropped];
                        ប្រសិនបើ (dstr > 0 && voff % dstr == 0) {
                            const int64_t sel = (int64_t) (voff / dstr);
                            ប្រសិនបើ (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::Slice>(
                                    context.get_input(0),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel + 1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                 {dropped - axis_shift}));
                                // Build the reshape target from the (concrete) dst shape, but
                                // keep the dynamic token axis dynamic instead of freezing it
                                // to the captured n_tokens. Without this the constant dst
                                // shape bakes in the prefill token count and the static value
                                // flows downstream, turning every later decoder layer static
                                // (the GPU in-place-concat KV-cache bug). The token axis is
                                // PERMUTED between the sliced input and the dst (e.g. input
                                // [1,tok,expert,emb] -> dst [1,1,tok,emb]), so special_zero
                                // (which copies the same-position dim) is not enough: pull the
                                // dynamic dim from the correct SOURCE axis via ShapeOf+Gather
                                // and place it at the dst token position.
                                const int32_t dyn = context.get_op_dynamic_dim();  // output ggml axis, -1 if none
                                // still in ggml metadata axis space; get_shape() reverses ggml order
                                ចំនួនគត់ dst_ov_axis = (dyn != -1) ? ((ចំនួនគត់) nd - 1 - (ចំនួនគត់) dyn) : -1;
                                ចំនួនគត់ src_ov_axis = (dst_ov_axis >= axis_shift && dst_ov_axis < (ចំនួនគត់) nd)
                                                      ? dst_to_src[dst_ov_axis]
                                                      : -1;
                                ប្រសិនបើ (dst_ov_axis >= 0 && src_ov_axis >= axis_shift) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    auto sl_shape = std::make_shared<ov::op::v3::ShapeOf>(sl, ov::element::i64);
                                    auto tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_shift}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::OutputVector parts;
                                    សម្រាប់ (ចំនួនគត់ a = axis_shift; a < (ចំនួនគត់) nd; ++a) {
                                        ប្រសិនបើ (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } បើមិនដូច្នេះ {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    auto dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, false);
                                    ត្រឡប់ rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_shift, dd.end());
                                auto dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, false);
                                ត្រឡប់ rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        ត្រឡប់ {context.get_input(0)};
    }

    auto input = context.get_input(0);
    auto src_shape = context.get_input_shape(0);
    auto dst_shape = context.get_output_shape();

    ប្រសិនបើ (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        ត្រឡប់ {input};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    សម្រាប់ (int64_t i = 0; i < src_shape.rank().get_length(); ++i) {
        ប្រសិនបើ (src_shape[i].is_dynamic()) {
            ត្រឡប់ {input};
        }
        src_elems *= src_shape[i].get_length();
    }
    សម្រាប់ (int64_t i = 0; i < dst_shape.rank().get_length(); ++i) {
        ប្រសិនបើ (dst_shape[i].is_dynamic()) {
            ត្រឡប់ {input};
        }
        dst_elems *= dst_shape[i].get_length();
    }

    ប្រសិនបើ (dst_elems >= src_elems) {
        ត្រឡប់ {input};
    }

    auto src_stride = context.get_input_stride(0);
    auto dst_stride = context.get_output_stride();
    size_t view_offset = context.get_output_op_offset();

    bool same_stride = (src_stride.size() == dst_stride.size());
    ប្រសិនបើ (same_stride) {
        សម្រាប់ (size_t i = 0; i < src_stride.size(); ++i) {
            ប្រសិនបើ (src_stride[i] != dst_stride[i]) {
                same_stride = false;
                bha_nga;
            }
        }
    }

    ប្រសិនបើ (!same_stride) {
        ត្រឡប់ {input};
    }

    auto src_ov_shape = src_shape.to_shape();
    auto dst_ov_shape = dst_shape.to_shape();
    size_t ndims = src_ov_shape.size();
    ប្រសិនបើ (dst_ov_shape.size() != ndims) {
        ត្រឡប់ {input};
    }

    std::vector<ចំនួនគត់> diff_dims;
    សម្រាប់ (size_t i = 0; i < ndims; ++i) {
        ប្រសិនបើ (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<ចំនួនគត់>(i));
        }
    }

    ប្រសិនបើ (diff_dims.size() != 1) {
        ត្រឡប់ {input};
    }

    ចំនួនគត់ slice_dim = diff_dims[0];
    int64_t dim_size = static_cast<int64_t>(src_ov_shape[slice_dim]);

    size_t ov_stride_for_dim = 1;
    សម្រាប់ (size_t i = slice_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_size = src_stride.back();
    ប្រសិនបើ (elem_size == 0) {
        elem_size = 1;
    }

    int64_t begin_val = 0;
    ប្រសិនបើ (ov_stride_for_dim > 0 && elem_size > 0) {
        begin_val = static_cast<int64_t>((view_offset / elem_size) / ov_stride_for_dim);
    }
    int64_t end_val = begin_val + static_cast<int64_t>(dst_ov_shape[slice_dim]);

    ប្រសិនបើ (begin_val < 0 || end_val > dim_size) {
        ត្រឡប់ {input};
    }

    auto sliced =
        std::make_shared<ov::op::v8::Slice>(input, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

    sliced->set_friendly_name(context.get_output_name());
    ត្រឡប់ {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
