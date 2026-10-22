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

    만약 (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        반환 {context.get_input(0)};
    }

    만약 (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        const std::string & vname = context.get_name();
        만약 (vname.find("ffn_moe_weighted") != std::string::npos) {
            auto src_ps = context.get_input_shape(0);
            auto dst_ps = context.get_output_shape();
            만약 (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
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
                const 정수 axis_shift =
                    in_rank.is_static() ? (정수) nd - (정수) in_rank.get_length() : 0;
                bool shift_ok = axis_shift >= 0 && (size_t) axis_shift < nd;
                위해 (정수 a = 0; a < axis_shift && shift_ok; ++a) {
                    shift_ok = (ss[a] == 1 && dd[a] == 1);
                }
                만약 (shift_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<bool> used(nd, false);
                    std::vector<정수> dst_to_src(nd, -1);
                    bool ok = true;
                    위해 (size_t d = 0; d < nd; ++d) {
                        만약 (dd[d] == 1) {
                            continue;
                        }
                        정수 found = -1;
                        위해 (size_t s = 0; s < nd; ++s) {
                            만약 (!used[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                found = (정수) s;
                                break;
                            }
                        }
                        만약 (found < 0) {
                            ok = false;
                            break;
                        }
                        used[found] = true;
                        dst_to_src[d] = found;
                    }
                    정수 dropped = -1;
                    만약 (ok) {
                        위해 (size_t s = 0; s < nd; ++s) {
                            만약 (!used[s] && ss[s] > 1) {
                                만약 (dropped >= 0) {
                                    ok = false;
                                    break;
                                }
                                dropped = (정수) s;
                            }
                        }
                    }
                    만약 (ok && dropped >= 0) {
                        const size_t dstr = sst[dropped];
                        const int64_t dsz = (int64_t) ss[dropped];
                        만약 (dstr > 0 && voff % dstr == 0) {
                            const int64_t sel = (int64_t) (voff / dstr);
                            만약 (sel >= 0 && sel < dsz) {
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
                                정수 dst_ov_axis = (dyn != -1) ? ((정수) nd - 1 - (정수) dyn) : -1;
                                정수 src_ov_axis = (dst_ov_axis >= axis_shift && dst_ov_axis < (정수) nd)
                                                      ? dst_to_src[dst_ov_axis]
                                                      : -1;
                                만약 (dst_ov_axis >= 0 && src_ov_axis >= axis_shift) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    auto sl_shape = std::make_shared<ov::op::v3::ShapeOf>(sl, ov::element::i64);
                                    auto tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_shift}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::OutputVector parts;
                                    위해 (정수 a = axis_shift; a < (정수) nd; ++a) {
                                        만약 (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } 아니면 {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    auto dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, false);
                                    반환 rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_shift, dd.end());
                                auto dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, false);
                                반환 rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        반환 {context.get_input(0)};
    }

    auto input = context.get_input(0);
    auto src_shape = context.get_input_shape(0);
    auto dst_shape = context.get_output_shape();

    만약 (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        반환 {input};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    위해 (int64_t i = 0; i < src_shape.rank().get_length(); ++i) {
        만약 (src_shape[i].is_dynamic()) {
            반환 {input};
        }
        src_elems *= src_shape[i].get_length();
    }
    위해 (int64_t i = 0; i < dst_shape.rank().get_length(); ++i) {
        만약 (dst_shape[i].is_dynamic()) {
            반환 {input};
        }
        dst_elems *= dst_shape[i].get_length();
    }

    만약 (dst_elems >= src_elems) {
        반환 {input};
    }

    auto src_stride = context.get_input_stride(0);
    auto dst_stride = context.get_output_stride();
    size_t view_offset = context.get_output_op_offset();

    bool same_stride = (src_stride.size() == dst_stride.size());
    만약 (same_stride) {
        위해 (size_t i = 0; i < src_stride.size(); ++i) {
            만약 (src_stride[i] != dst_stride[i]) {
                same_stride = false;
                break;
            }
        }
    }

    만약 (!same_stride) {
        반환 {input};
    }

    auto src_ov_shape = src_shape.to_shape();
    auto dst_ov_shape = dst_shape.to_shape();
    size_t ndims = src_ov_shape.size();
    만약 (dst_ov_shape.size() != ndims) {
        반환 {input};
    }

    std::vector<정수> diff_dims;
    위해 (size_t i = 0; i < ndims; ++i) {
        만약 (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<정수>(i));
        }
    }

    만약 (diff_dims.size() != 1) {
        반환 {input};
    }

    정수 slice_dim = diff_dims[0];
    int64_t dim_size = static_cast<int64_t>(src_ov_shape[slice_dim]);

    size_t ov_stride_for_dim = 1;
    위해 (size_t i = slice_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_size = src_stride.back();
    만약 (elem_size == 0) {
        elem_size = 1;
    }

    int64_t begin_val = 0;
    만약 (ov_stride_for_dim > 0 && elem_size > 0) {
        begin_val = static_cast<int64_t>((view_offset / elem_size) / ov_stride_for_dim);
    }
    int64_t end_val = begin_val + static_cast<int64_t>(dst_ov_shape[slice_dim]);

    만약 (begin_val < 0 || end_val > dim_size) {
        반환 {input};
    }

    auto sliced =
        std::make_shared<ov::op::v8::Slice>(input, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

    sliced->set_friendly_name(context.get_output_name());
    반환 {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
