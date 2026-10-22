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

    если (context.get_op_case() == 1) {
        // Static-mode identity pass-through for VIEWs over a GATED_DELTA_NET combined output or
        // the conv_input CONCAT; the consuming op (CPY/RMS_NORM) does its own runtime-correct
        // slicing on the full tensor (see ggml-decoder.cpp compute_op_case, GGML_OP_VIEW).
        вернуть {context.get_input(0)};
    }

    если (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        const std::string & vname = context.get_name();
        если (vname.find("ffn_moe_weighted") != std::string::npos) {
            auto src_ps = context.get_input_shape(0);
            auto dst_ps = context.get_output_shape();
            если (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
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
                const целое axis_shift =
                    in_rank.is_static() ? (целое) nd - (целое) in_rank.get_length() : 0;
                bool shift_ok = axis_shift >= 0 && (size_t) axis_shift < nd;
                для (целое a = 0; a < axis_shift && shift_ok; ++a) {
                    shift_ok = (ss[a] == 1 && dd[a] == 1);
                }
                если (shift_ok && sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    // dst_to_src[d] records which src axis each dst axis came from, so we can
                    // later pull the dynamic (token) dim from the right source axis at runtime.
                    std::vector<bool> used(nd, false);
                    std::vector<целое> dst_to_src(nd, -1);
                    bool ok = true;
                    для (size_t d = 0; d < nd; ++d) {
                        если (dd[d] == 1) {
                            continue;
                        }
                        целое found = -1;
                        для (size_t s = 0; s < nd; ++s) {
                            если (!used[s] && ss[s] == dd[d] && sst[s] == dst[d]) {
                                found = (целое) s;
                                break;
                            }
                        }
                        если (found < 0) {
                            ok = false;
                            break;
                        }
                        used[found] = true;
                        dst_to_src[d] = found;
                    }
                    целое dropped = -1;
                    если (ok) {
                        для (size_t s = 0; s < nd; ++s) {
                            если (!used[s] && ss[s] > 1) {
                                если (dropped >= 0) {
                                    ok = false;
                                    break;
                                }
                                dropped = (целое) s;
                            }
                        }
                    }
                    если (ok && dropped >= 0) {
                        const size_t dstr = sst[dropped];
                        const int64_t dsz = (int64_t) ss[dropped];
                        если (dstr > 0 && voff % dstr == 0) {
                            const int64_t sel = (int64_t) (voff / dstr);
                            если (sel >= 0 && sel < dsz) {
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
                                целое dst_ov_axis = (dyn != -1) ? ((целое) nd - 1 - (целое) dyn) : -1;
                                целое src_ov_axis = (dst_ov_axis >= axis_shift && dst_ov_axis < (целое) nd)
                                                      ? dst_to_src[dst_ov_axis]
                                                      : -1;
                                если (dst_ov_axis >= 0 && src_ov_axis >= axis_shift) {
                                    // target = concat of per-axis scalars; the token axis is a
                                    // runtime Gather of the slice's shape, the rest are constants.
                                    auto sl_shape = std::make_shared<ov::op::v3::ShapeOf>(sl, ov::element::i64);
                                    auto tok_dim = std::make_shared<ov::op::v8::Gather>(
                                        sl_shape,
                                        ov::op::v0::Constant::create(ov::element::i64, {1},
                                                                     {src_ov_axis - axis_shift}),
                                        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
                                    ov::OutputVector parts;
                                    для (целое a = axis_shift; a < (целое) nd; ++a) {
                                        если (a == dst_ov_axis) {
                                            parts.push_back(tok_dim);
                                        } иначе {
                                            parts.push_back(ov::op::v0::Constant::create(
                                                ov::element::i64, {1}, {(int64_t) dd[a]}));
                                        }
                                    }
                                    auto dc = std::make_shared<ov::op::v0::Concat>(parts, 0);
                                    auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, false);
                                    вернуть rename_outputs_with_suffix({rs}, context.get_name());
                                }
                                std::vector<int64_t> dd_ov(dd.begin() + axis_shift, dd.end());
                                auto dc = ov::op::v0::Constant::create(ov::element::i64, {dd_ov.size()}, dd_ov);
                                auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, false);
                                вернуть rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        вернуть {context.get_input(0)};
    }

    auto input = context.get_input(0);
    auto src_shape = context.get_input_shape(0);
    auto dst_shape = context.get_output_shape();

    если (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        вернуть {input};
    }

    int64_t src_elems = 1;
    int64_t dst_elems = 1;
    для (int64_t i = 0; i < src_shape.rank().get_length(); ++i) {
        если (src_shape[i].is_dynamic()) {
            вернуть {input};
        }
        src_elems *= src_shape[i].get_length();
    }
    для (int64_t i = 0; i < dst_shape.rank().get_length(); ++i) {
        если (dst_shape[i].is_dynamic()) {
            вернуть {input};
        }
        dst_elems *= dst_shape[i].get_length();
    }

    если (dst_elems >= src_elems) {
        вернуть {input};
    }

    auto src_stride = context.get_input_stride(0);
    auto dst_stride = context.get_output_stride();
    size_t view_offset = context.get_output_op_offset();

    bool same_stride = (src_stride.size() == dst_stride.size());
    если (same_stride) {
        для (size_t i = 0; i < src_stride.size(); ++i) {
            если (src_stride[i] != dst_stride[i]) {
                same_stride = false;
                break;
            }
        }
    }

    если (!same_stride) {
        вернуть {input};
    }

    auto src_ov_shape = src_shape.to_shape();
    auto dst_ov_shape = dst_shape.to_shape();
    size_t ndims = src_ov_shape.size();
    если (dst_ov_shape.size() != ndims) {
        вернуть {input};
    }

    std::vector<целое> diff_dims;
    для (size_t i = 0; i < ndims; ++i) {
        если (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<целое>(i));
        }
    }

    если (diff_dims.size() != 1) {
        вернуть {input};
    }

    целое slice_dim = diff_dims[0];
    int64_t dim_size = static_cast<int64_t>(src_ov_shape[slice_dim]);

    size_t ov_stride_for_dim = 1;
    для (size_t i = slice_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_size = src_stride.back();
    если (elem_size == 0) {
        elem_size = 1;
    }

    int64_t begin_val = 0;
    если (ov_stride_for_dim > 0 && elem_size > 0) {
        begin_val = static_cast<int64_t>((view_offset / elem_size) / ov_stride_for_dim);
    }
    int64_t end_val = begin_val + static_cast<int64_t>(dst_ov_shape[slice_dim]);

    если (begin_val < 0 || end_val > dim_size) {
        вернуть {input};
    }

    auto sliced =
        std::make_shared<ov::op::v8::Slice>(input, ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

    sliced->set_friendly_name(context.get_output_name());
    вернуть {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
