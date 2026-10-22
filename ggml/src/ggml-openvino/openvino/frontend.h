// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#شامل <openvino/frontend/frontend.hpp>

namespace ov {
namespace frontend {
namespace ggml {

class FrontEnd {
عمومی:
    FrontEnd();

    ایستا std::shared_ptr<Model> convert(ثابت درون_داد_Model::Ptr & model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
