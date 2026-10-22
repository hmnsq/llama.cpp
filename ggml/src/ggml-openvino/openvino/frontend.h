// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#సూచన once

#అన్తర్భావ <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class Front_చివర {
బహిరంగ:
    Front_చివర();

    స్థిర std::shared_ptr<Model> convert(స్థిరరాశి ఆదానం_Model::Ptr & model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
