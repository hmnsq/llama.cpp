#pragma once

#شامل "openvino/op/op.hpp"

فضاینام ov::op::internal {
/// \note GatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing Gated Delta Net computation
/// \ingroup ov_ops_cpp_api
رده OPENVINO_API GatedDeltaNet : عمومی ov::op::Op {
عمومی:
    OPENVINO_OP("GatedDeltaNet")

    GatedDeltaNet() = پیشفرض;
    /// \brief Constructs a GatedDeltaNet operation.
    ///
    /// \param query Query tensor input.
    /// \param key Key tensor input.
    /// \param value Value tensor input.
    /// \param recurrent_state Initial recurrent state tensor.
    /// \param gate Gate tensor controlling state decay/update.
    /// \param beta Beta tensor scaling the delta update.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(ثابت Output<Node>& query,
                  ثابت Output<Node>& کلید,
                  ثابت Output<Node>& value,
                  ثابت Output<Node>& recurrent_state,
                  ثابت Output<Node>& gate,
                  ثابت Output<Node>& beta,
                  ثابت منطقی fuse_qk_l2norm = نادرست,
                  ثابت اعشاری q_l2_norm_eps = 1e-6F,
                  ثابت اعشاری k_l2_norm_eps = 1e-6F);

    /// \brief Constructs a GatedDeltaNet operation from input vector.
    ///
    /// \param args Input tensor vector in order: query, key, value, recurrent_state, gate, beta.
    /// \param fuse_qk_l2norm Enables fusing q/k L2-normalization into this op.
    /// \param q_l2_norm_eps Epsilon used for query L2-normalization when fusion is enabled.
    /// \param k_l2_norm_eps Epsilon used for key L2-normalization when fusion is enabled.
    GatedDeltaNet(ثابت ov::OutputVector& args,
                  ثابت منطقی fuse_qk_l2norm = نادرست,
                  ثابت اعشاری q_l2_norm_eps = 1e-6F,
                  ثابت اعشاری k_l2_norm_eps = 1e-6F);
    تهی validate_and_infer_types() override;
    منطقی visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(ثابت ov::OutputVector& new_args) ثابت override;
    منطقی get_fuse_qk_l2norm() ثابت {
        بازگشت m_fuse_qk_l2norm;
    }
    اعشاری get_q_l2_norm_eps() ثابت {
        بازگشت m_q_l2_norm_eps;
    }
    اعشاری get_k_l2_norm_eps() ثابت {
        بازگشت m_k_l2_norm_eps;
    }

خصوصی:
    منطقی m_fuse_qk_l2norm = نادرست;
    اعشاری m_q_l2_norm_eps = 1e-6F;
    اعشاری m_k_l2_norm_eps = 1e-6F;
};

}  // namespace ov::op::internal
