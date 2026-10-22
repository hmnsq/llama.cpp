#అన్తర్భావ "ggml-vulkan-common.h"

#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <mutex>
#అన్తర్భావ <shared_mutex>

స్థిర శూన్య * portable_aligned_malloc(size_t align, size_t size) {
#నిర్వచితమైతే _WIN32
    తిరిగి _aligned_malloc(size, align);
#లేకపోతే
    శూన్య * ptr = శూన్యసూచిక;
    ఒకవేళ (posix_memalign(&ptr, align, size) != 0) {
        తిరిగి శూన్యసూచిక;
    }
    తిరిగి ptr;
#అయితే_ముగింపు
}

స్థిర శూన్య portable_aligned_free(శూన్య * ptr) {
#నిర్వచితమైతే _WIN32
    _aligned_free(ptr);
#లేకపోతే
    std::విడుదల(ptr);
#అయితే_ముగింపు
}

ggml_backend_బఫరు_type_i ggml_backend_vk_బఫరు_type_interface = {
    /* .get_name            = */ ggml_backend_vk_బఫరు_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_బఫరు_type_alloc_బఫరు,
    /* .alloc_buffer_n      = */ ఖాళీ,
    /* .get_alignment       = */ ggml_backend_vk_బఫరు_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_బఫరు_type_get_max_కొలత,
    /* .get_alloc_size      = */ ggml_backend_vk_బఫరు_type_get_alloc_కొలత,
    /* .get_alloc_size_n    = */ ఖాళీ,
    /* .is_host             = */ ఖాళీ,
};

స్థిర std::vector<uint32_t> ggml_vk_కనుగొను_జ్ఞాపకం_properties(స్థిరరాశి vk::Physical_Device_జ్ఞాపకం_Properties* mem_props, vk::జ్ఞాపకం_Requirements* mem_req, vk::జ్ఞాపకం_Property_Flags flags) {
    std::vector<uint32_t> indices;

    కొరకు (uint32_t i = 0; i < mem_props->జ్ఞాపకం_Type_లెక్క; ++i) {
        vk::జ్ఞాపకం_Type జ్ఞాపకం_type = mem_props->జ్ఞాపకం_Types[i];
        ఒకవేళ ((mem_req->జ్ఞాపకం_Type_బిట్లు & ((uint64_t)1 << i)) &&
            (flags & జ్ఞాపకం_type.propertyFlags) == flags &&
            mem_props->జ్ఞాపకం_Heaps[జ్ఞాపకం_type.కుప్ప_సూచీ].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    తిరిగి indices;
}

స్థిర vk_బఫరు ggml_vk_create_బఫరు(vk_device& device, size_t size, స్థిరరాశి std::initializer_list<vk::జ్ఞాపకం_Property_Flags> & req_flags_list,
                                       శూన్య *import_ptr = శూన్యసూచిక) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    ఒకవేళ (size > device->max_బఫరు_కొలత) {
        విసురు vk::బహిర్గతం_యొక్క_Device_జ్ఞాపకం_Error("Requested buffer size exceeds device buffer size limit");
    }

    vk_బఫరు buf = std::make_shared<vk_బఫరు_struct>();

    ఒకవేళ (size == 0) {
        buf->size = 0;
        తిరిగి buf;
    }

    vk::బఫరు_Usage_Flags usage_flags = vk::బఫరు_Usage_Flag_బిట్లు::e_Storage_బఫరు | vk::బఫరు_Usage_Flag_బిట్లు::eTransferSrc | vk::బఫరు_Usage_Flag_బిట్లు::eTransferDst;
    vk::జ్ఞాపకం_Allocate_Flags mem_flags {};
    ఒకవేళ (device->బఫరు_device_చిరునామా) {
        usage_flags |= vk::బఫరు_Usage_Flag_బిట్లు::e_Shader_Device_చిరునామా;
        mem_flags |= vk::జ్ఞాపకం_Allocate_Flag_బిట్లు::e_Device_చిరునామా;
    }

    vk::బఫరు_Create_Info బఫరు_create_info{
        vk::బఫరు_Create_Flags(),
        size,
        usage_flags,
        vk::Sharing_బహుళకం::eExclusive,
        0,
        శూన్యసూచిక,
    };

    vk::External_జ్ఞాపకం_బఫరు_Create_Info external_జ్ఞాపకం_bci;
    ఒకవేళ (import_ptr) {
        external_జ్ఞాపకం_bci.handleTypes = vk::External_జ్ఞాపకం_Handle_Type_Flag_బిట్లు::e_Host_కేటాయింపు_EXT;
        బఫరు_create_info.సమితి_PNext(&external_జ్ఞాపకం_bci);
    }

    buf->బఫరు = device->device.create_బఫరు(బఫరు_create_info);

    vk::జ్ఞాపకం_Requirements mem_req = device->device.get_బఫరు_జ్ఞాపకం_Requirements(buf->బఫరు);

    vk::Physical_Device_జ్ఞాపకం_Properties mem_props = device->physical_device.get_జ్ఞాపకం_Properties();

    స్థిరరాశి vk::జ్ఞాపకం_ప్రాధాన్యత_Allocate_Info_EXT mem_ప్రాధాన్యత_info { 1.0f };

    vk::జ్ఞాపకం_Allocate_Flags_Info mem_flags_info { mem_flags };

    ఒకవేళ (device->జ్ఞాపకం_ప్రాధాన్యత) {
        mem_flags_info.సమితి_PNext(&mem_ప్రాధాన్యత_info);
    }

    ఒకవేళ (import_ptr) {
        vk::జ్ఞాపకం_Host_సూచిక_Properties_EXT host_సూచిక_props;
        ప్రయత్నించు {
            host_సూచిక_props = device->device.get_జ్ఞాపకం_Host_సూచిక_Properties_EXT(vk::External_జ్ఞాపకం_Handle_Type_Flag_బిట్లు::e_Host_కేటాయింపు_EXT, import_ptr);
        } పట్టుకో (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroy_బఫరు(buf->బఫరు);
            తిరిగి {};
        }
        vk::Physical_Device_జ్ఞాపకం_Properties mem_props = device->physical_device.get_జ్ఞాపకం_Properties();

        uint32_t జ్ఞాపకం_type_idx;
        vk::జ్ఞాపకం_Property_Flags property_flags = *req_flags_list.begin();
        కొరకు (జ్ఞాపకం_type_idx = 0; జ్ఞాపకం_type_idx < 32; ++జ్ఞాపకం_type_idx) {
            ఒకవేళ (!(host_సూచిక_props.జ్ఞాపకం_Type_బిట్లు & (1u << జ్ఞాపకం_type_idx))) {
                కొనసాగు;
            }
            ఒకవేళ (!(mem_req.జ్ఞాపకం_Type_బిట్లు & (1u << జ్ఞాపకం_type_idx))) {
                కొనసాగు;
            }

            vk::జ్ఞాపకం_Type జ్ఞాపకం_type = mem_props.జ్ఞాపకం_Types[జ్ఞాపకం_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            ఒకవేళ ((జ్ఞాపకం_type.propertyFlags & property_flags) == property_flags) {
                property_flags = జ్ఞాపకం_type.propertyFlags;
                ఆపు;
            }
        }
        ఒకవేళ (జ్ఞాపకం_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroy_బఫరు(buf->బఫరు);
            తిరిగి {};
        }

        buf->జ్ఞాపకం_property_flags = mem_props.జ్ఞాపకం_Types[జ్ఞాపకం_type_idx].propertyFlags;
        ప్రయత్నించు {
            vk::Import_జ్ఞాపకం_Host_సూచిక_Info_EXT import_info;
            import_info.handleType = vk::External_జ్ఞాపకం_Handle_Type_Flag_బిట్లు::e_Host_కేటాయింపు_EXT;
            import_info.p_Host_సూచిక = import_ptr;
            import_info.సమితి_PNext(&mem_flags_info);
            buf->device_జ్ఞాపకం = device->device.allocate_జ్ఞాపకం({ size, జ్ఞాపకం_type_idx, &import_info });
        } పట్టుకో (స్థిరరాశి vk::SystemError& e) {
        }
    } లేకపోతే {
        కొరకు (స్వయంచాలక it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            స్థిరరాశి స్వయంచాలక & req_flags = *it;

            స్థిరరాశి std::vector<uint32_t> జ్ఞాపకం_type_indices = ggml_vk_కనుగొను_జ్ఞాపకం_properties(&mem_props, &mem_req, req_flags);

            ఒకవేళ (జ్ఞాపకం_type_indices.ఖాళీలు()) {
                కొనసాగు;
            }

            తార్కిక పూర్తి = అబద్ధం;

            కొరకు (స్వయంచాలక mtype_it = జ్ఞాపకం_type_indices.begin(); mtype_it != జ్ఞాపకం_type_indices.end(); mtype_it++) {
                ప్రయత్నించు {
                    buf->device_జ్ఞాపకం = device->device.allocate_జ్ఞాపకం({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->జ్ఞాపకం_property_flags = mem_props.జ్ఞాపకం_Types[*mtype_it].propertyFlags;
                    పూర్తి = నిజం;
                    ఆపు;
                } పట్టుకో (స్థిరరాశి vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    ఒకవేళ (it + 1 == req_flags_list.end() && mtype_it + 1 == జ్ఞాపకం_type_indices.end()) {
                        device->device.destroy_బఫరు(buf->బఫరు);
                        విసురు e;
                    }
                }
            }

            ఒకవేళ (పూర్తి) {
                ఆపు;
            }
        }
    }

    ఒకవేళ (!buf->device_జ్ఞాపకం) {
        device->device.destroy_బఫరు(buf->బఫరు);
        విసురు vk::బహిర్గతం_యొక్క_Device_జ్ఞాపకం_Error("No suitable memory type found");
    }

    buf->ptr = శూన్యసూచిక;

    ఒకవేళ (import_ptr) {
        buf->ptr = import_ptr;
    } లేకపోతే {
        ఒకవేళ (buf->జ్ఞాపకం_property_flags & vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible) {
            buf->ptr = device->device.map_జ్ఞాపకం(buf->device_జ్ఞాపకం, 0, VK_సంపూర్ణ_కొలత);
        }
    }

    device->device.bind_బఫరు_జ్ఞాపకం(buf->బఫరు, buf->device_జ్ఞాపకం, 0);

    buf->device = device;
    buf->size = size;

    ఒకవేళ (device->బఫరు_device_చిరునామా) {
        స్థిరరాశి vk::బఫరు_Device_చిరునామా_Info చిరునామా_Info(buf->బఫరు);
        buf->bda_addr = device->device.get_బఫరు_చిరునామా(చిరునామా_Info);
    }

    device->జ్ఞాపకం_logger->log_కేటాయింపు(buf, size);

    తిరిగి buf;
}

vk_బఫరు ggml_vk_create_బఫరు_check(vk_device& device, size_t size, vk::జ్ఞాపకం_Property_Flags req_flags, vk::జ్ఞాపకం_Property_Flags fallback_flags) {
    ప్రయత్నించు {
        తిరిగి ggml_vk_create_బఫరు(device, size, {req_flags, fallback_flags});
    } పట్టుకో (స్థిరరాశి vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::పంక్తి_ముగింపు;
        std::cerr << "ggml_vulkan: " << e.what() << std::పంక్తి_ముగింపు;
        విసురు e;
    }
}

vk_బఫరు ggml_vk_create_బఫరు_device(vk_device& device, size_t size) {
    vk_బఫరు buf;
    ప్రయత్నించు {
        ఒకవేళ (device->prefer_host_జ్ఞాపకం) {
            buf = ggml_vk_create_బఫరు(device, size, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent,
                                                       vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal});
        } లేకపోతే ఒకవేళ (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_బఫరు(device, size, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent,
                                                       vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal,
                                                       vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent});
        } లేకపోతే ఒకవేళ (device->disable_host_visible_vidmem) {
            ఒకవేళ (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_బఫరు(device, size, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal,
                                                           vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent});
            } లేకపోతే {
                buf = ggml_vk_create_బఫరు(device, size, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal});
            }
        } లేకపోతే {
            // use rebar if available, otherwise fallback to device only visible memory
            ఒకవేళ (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_బఫరు(device, size, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent,
                                                           vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal,
                                                           vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent});
            } లేకపోతే {
                buf = ggml_vk_create_బఫరు(device, size, {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent,
                                                           vk::జ్ఞాపకం_Property_Flag_బిట్లు::eDeviceLocal});
            }
        }
    } పట్టుకో (స్థిరరాశి vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::పంక్తి_ముగింపు;
        std::cerr << "ggml_vulkan: " << e.what() << std::పంక్తి_ముగింపు;
        విసురు e;
    }

    తిరిగి buf;
}

శూన్య ggml_vk_destroy_బఫరు(vk_బఫరు& buf) {
    ఒకవేళ (buf == శూన్యసూచిక) {
        తిరిగి;
    }

    ఒకవేళ (buf->device != శూన్యసూచిక) {
        buf->device->జ్ఞాపకం_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
స్థిర size_t ggml_vk_host_కేటాయింపు_కనుగొను(స్థిరరాశి శూన్య * ptr) {
    కొరకు (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        స్థిరరాశి uint8_t * పునాది = (స్థిరరాశి uint8_t *) vk_instance.host_allocations[i].పునాది;
        ఒకవేళ (ptr >= పునాది && ptr < పునాది + vk_instance.host_allocations[i].size) {
            తిరిగి i;
        }
    }
    తిరిగి SIZE_MAX;
}

శూన్య * ggml_vk_host_malloc(size_t size) {
    VK_LOG_జ్ఞాపకం("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    ఒకవేళ (vk_instance.multi_device) {
        స్థిరరాశి size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        స్థిరరాశి size_t alloc_కొలత = (size + align - 1) / align * align;

        శూన్య * ptr = portable_aligned_malloc(align, alloc_కొలత);
        ఒకవేళ (ptr == శూన్యసూచిక) {
            తిరిగి శూన్యసూచిక;
        }
        vk_host_కేటాయింపు నమోదు_2;
        నమోదు_2.పునాది = ptr;
        నమోదు_2.size = alloc_కొలత;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(నమోదు_2));
        తిరిగి ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_బఫరు buf;
    ప్రయత్నించు {
        buf = ggml_vk_create_బఫరు(dev0, size,
            {vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCached,
             vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent});
    } పట్టుకో (vk::SystemError&) {
        తిరిగి శూన్యసూచిక;
    }

    ఒకవేళ (!(buf->జ్ఞాపకం_property_flags & vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_బఫరు(buf);
        తిరిగి శూన్యసూచిక;
    }

    vk_host_కేటాయింపు నమోదు_2;
    నమోదు_2.పునాది             = buf->ptr;
    నమోదు_2.size             = size;
    నమోదు_2.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(నమోదు_2));
    తిరిగి buf->ptr;
}

శూన్య ggml_vk_host_free(శూన్య * ptr) {
    ఒకవేళ (ptr == శూన్యసూచిక) {
        తిరిగి;
    }
    VK_LOG_జ్ఞాపకం("ggml_vk_host_free(" << ptr << ")");

    vk_host_కేటాయింపు నమోదు_2;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        స్థిరరాశి size_t సూచీ = ggml_vk_host_కేటాయింపు_కనుగొను(ptr);
        ఒకవేళ (సూచీ == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            తిరిగి;
        }
        నమోదు_2 = std::move(vk_instance.host_allocations[సూచీ]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + సూచీ);
    }

    కొరకు (స్వయంచాలక & view : నమోదు_2.views) {
        ggml_vk_destroy_బఫరు(view);
    }
    ఒకవేళ (vk_instance.multi_device) {
        portable_aligned_free(నమోదు_2.పునాది);
    }
}

శూన్య ggml_vk_host_get(స్థిరరాశి vk_device& device, స్థిరరాశి శూన్య * ptr, vk_బఫరు& buf, size_t& buf_స్థానభ్రంశం) {
    buf = శూన్యసూచిక;
    buf_స్థానభ్రంశం = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    స్థిరరాశి size_t సూచీ = ggml_vk_host_కేటాయింపు_కనుగొను(ptr);
    ఒకవేళ (సూచీ == SIZE_MAX) {
        తిరిగి;
    }
    vk_host_కేటాయింపు & నమోదు_2 = vk_instance.host_allocations[సూచీ];

    ఒకవేళ (నమోదు_2.views[device->idx] == శూన్యసూచిక) {
        స్థిరరాశి uint32_t బిట్టు = 1u << device->idx;
        ఒకవేళ (!vk_instance.multi_device || !vk_instance.device_host_బఫరు_import[device->idx] || (vk_instance.host_import_failed & బిట్టు)) {
            తిరిగి;
        }
        GGML_ASSERT(device->external_జ్ఞాపకం_host);
        // const only propagates from the const context; the pointee is mutable
        నమోదు_2.views[device->idx] = ggml_vk_బఫరు_from_host_ptr(const_cast<vk_device&>(device), నమోదు_2.పునాది, నమోదు_2.size);
        ఒకవేళ (నమోదు_2.views[device->idx] == శూన్యసూచిక) {
            vk_instance.host_import_failed |= బిట్టు;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            తిరిగి;
        }
    }

    buf = నమోదు_2.views[device->idx];
    buf_స్థానభ్రంశం = (స్థిరరాశి uint8_t *) ptr - (స్థిరరాశి uint8_t *) నమోదు_2.పునాది;
}

శూన్య ggml_vk_ensure_sync_staging_బఫరు(vk_device& device, size_t size) {
    ఒకవేళ (device->sync_staging == శూన్యసూచిక || device->sync_staging->size < size) {
        VK_LOG_జ్ఞాపకం("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_బఫరు(device->sync_staging);
        device->sync_staging = ggml_vk_create_బఫరు_check(device, size,
            vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCached,
            vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent);
    }
}

శూన్య ggml_vk_ensure_sync_staging_బఫరు(ggml_backend_vk_context * ctx, size_t size) {
    ఒకవేళ (ctx->sync_staging == శూన్యసూచిక || ctx->sync_staging->size < size) {
        VK_LOG_జ్ఞాపకం("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_బఫరు(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_బఫరు_check(ctx->device, size,
            vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCached,
            vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent);
    }
}

స్థిర శూన్య ggml_vk_బఫరు_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_బఫరు& dst, size_t స్థానభ్రంశం, స్థిరరాశి ggml_tensor * tensor, తార్కిక sync_staging = అబద్ధం) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    ఒకవేళ(dst->జ్ఞాపకం_property_flags & vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::పంక్తి_ముగింపు;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_బఫరు buf = శూన్యసూచిక;
    size_t buf_స్థానభ్రంశం = 0;
    ggml_vk_host_get(ctx->device, tensor->దత్తాంశం, buf, buf_స్థానభ్రంశం);

    స్థిరరాశి uint64_t ne0 = tensor->ne[0];
    స్థిరరాశి uint64_t ne1 = tensor->ne[1];
    స్థిరరాశి uint64_t ne2 = tensor->ne[2];
    స్థిరరాశి uint64_t ne3 = tensor->ne[3];
    స్థిరరాశి uint64_t nb0 = tensor->nb[0];
    స్థిరరాశి uint64_t nb1 = tensor->nb[1];
    స్థిరరాశి uint64_t nb2 = tensor->nb[2];
    స్థిరరాశి uint64_t nb3 = tensor->nb[3];
    స్థిరరాశి ggml_type type = tensor->type;
    స్థిరరాశి uint64_t ts = ggml_type_కొలత(type);
    స్థిరరాశి uint64_t bs = ggml_blck_కొలత(type);

    స్థిరరాశి uint64_t dstnb0 = ts;
    స్థిరరాశి uint64_t dstnb1 = dstnb0*(ne0/bs);
    స్థిరరాశి uint64_t dstnb2 = dstnb1*ne1;
    స్థిరరాశి uint64_t dstnb3 = dstnb2*ne2;

    స్థిరరాశి uint64_t ne = ggml_nelements(tensor);

    ఒకవేళ (buf != శూన్యసూచిక) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::బఫరు_Copy> slices;

        కొరకు (uint64_t i3 = 0; i3 < ne3; i3++) {
            కొరకు (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                ఒకవేళ (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_స్థానభ్రంశం + i3*nb3 + i2*nb2, స్థానభ్రంశం + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } లేకపోతే {
                    కొరకు (uint64_t i1 = 0; i1 < ne1; i1++) {
                        ఒకవేళ (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_స్థానభ్రంశం + i3*nb3 + i2*nb2 + i1*nb1, స్థానభ్రంశం + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } లేకపోతే {
                            స్థిరరాశి uint64_t s_off = buf_స్థానభ్రంశం + i3*nb3 + i2*nb2 + i1*nb1;
                            స్థిరరాశి uint64_t d_off = స్థానభ్రంశం + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            కొరకు (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->బఫరు->buf.copy_బఫరు(buf->బఫరు, dst->బఫరు, slices);
        తిరిగి;
    }

    ఒకవేళ (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_బఫరు& staging = ctx->device->sync_staging;
    స్థిరరాశి uint64_t copy_కొలత = ts*ne/bs;
    ggml_vk_ensure_sync_staging_బఫరు(ctx->device, copy_కొలత);
    Vk_బఫరు_Copy buf_copy{ 0, స్థానభ్రంశం, copy_కొలత };

    ggml_vk_sync_buffers(ctx, subctx);
    vk_Cmd_Copy_బఫరు(subctx->s->బఫరు->buf, (Vk_బఫరు)staging->బఫరు, (Vk_బఫరు)dst->బఫరు, 1, &buf_copy);

    కొరకు (uint64_t i3 = 0; i3 < ne3; i3++) {
        కొరకు (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            ఒకవేళ (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (స్థిరరాశి uint8_t *) tensor->దత్తాంశం + buf_స్థానభ్రంశం + i3*nb3 + i2*nb2, dstnb2, &subctx->లోపల_memcpys);
            } లేకపోతే {
                కొరకు (uint64_t i1 = 0; i1 < ne1; i1++) {
                    ఒకవేళ (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (స్థిరరాశి uint8_t *) tensor->దత్తాంశం + buf_స్థానభ్రంశం + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->లోపల_memcpys);
                    } లేకపోతే {
                        స్థిరరాశి uint64_t s_off = buf_స్థానభ్రంశం + i3*nb3 + i2*nb2 + i1*nb1;
                        స్థిరరాశి uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        కొరకు (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (స్థిరరాశి uint8_t *) tensor->దత్తాంశం + s_off + i0*nb0, dstnb0, &subctx->లోపల_memcpys);
                        }
                    }
                }
            }
        }
    }
}

తార్కిక ggml_vk_బఫరు_write_2d_async(vk_context subctx, vk_బఫరు& dst, size_t స్థానభ్రంశం, స్థిరరాశి శూన్య * src, size_t spitch, size_t dpitch, size_t వెడల్పు, size_t ఎత్తు, తార్కిక sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << వెడల్పు << ", " << ఎత్తు << ")");
    // Check if src is pinned memory
    vk_బఫరు buf = శూన్యసూచిక;
    size_t buf_స్థానభ్రంశం = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_స్థానభ్రంశం);

    ఒకవేళ (buf != శూన్యసూచిక) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::బఫరు_Copy> slices(1);
        ఒకవేళ (వెడల్పు == spitch && వెడల్పు == dpitch) {
            // Only do single write if stride is equal
            slices[0].src_స్థానభ్రంశం_2 = buf_స్థానభ్రంశం;
            slices[0].dst_స్థానభ్రంశం_2 = స్థానభ్రంశం;
            slices[0].size = వెడల్పు * ఎత్తు;
        } లేకపోతే {
            slices.resize(ఎత్తు);
            కొరకు (size_t i = 0; i < ఎత్తు; i++) {
                slices[i].src_స్థానభ్రంశం_2 = buf_స్థానభ్రంశం + i * spitch;
                slices[i].dst_స్థానభ్రంశం_2 = స్థానభ్రంశం + i * dpitch;
                slices[i].size = వెడల్పు;
            }
        }

        ggml_vk_sync_buffers(శూన్యసూచిక, subctx);
        subctx->s->బఫరు->buf.copy_బఫరు(buf->బఫరు, dst->బఫరు, slices);
        తిరిగి నిజం;
    }
    VK_LOG_DEBUG("STAGING");

    ఒకవేళ (!sync_staging) {
        // copy was not handled caller needs to fall back
        తిరిగి అబద్ధం;
    }

    // Staging buffer required
    స్థిరరాశి size_t staging_కొలత = వెడల్పు * ఎత్తు;
    ggml_vk_ensure_sync_staging_బఫరు(dst->device, staging_కొలత);

    vk_బఫరు& staging_బఫరు = dst->device->sync_staging;

    std::vector<vk::బఫరు_Copy> slices(1);
    ఒకవేళ (వెడల్పు == dpitch) {
        slices[0].src_స్థానభ్రంశం_2 = 0;
        slices[0].dst_స్థానభ్రంశం_2 = స్థానభ్రంశం;
        slices[0].size = staging_కొలత;
    } లేకపోతే {
        slices.resize(ఎత్తు);
        కొరకు (size_t i = 0; i < ఎత్తు; i++) {
            slices[i].src_స్థానభ్రంశం_2 = i * వెడల్పు;
            slices[i].dst_స్థానభ్రంశం_2 = స్థానభ్రంశం + i * dpitch;
            slices[i].size = వెడల్పు;
        }
    }

    ggml_vk_sync_buffers(శూన్యసూచిక, subctx);
    subctx->s->బఫరు->buf.copy_బఫరు(staging_బఫరు->బఫరు, dst->బఫరు, slices);

    ఒకవేళ (వెడల్పు == spitch) {
        deferred_memcpy((uint8_t *)staging_బఫరు->ptr, src, staging_కొలత, &subctx->లోపల_memcpys);
    } లేకపోతే {
        కొరకు (size_t i = 0; i < ఎత్తు; i++) {
            deferred_memcpy((uint8_t *)staging_బఫరు->ptr + i * వెడల్పు, (స్థిరరాశి uint8_t *) src + i * spitch, వెడల్పు, &subctx->లోపల_memcpys);
        }
    }
    తిరిగి నిజం;
}

తార్కిక ggml_vk_బఫరు_write_async(vk_context subctx, vk_బఫరు& dst, size_t స్థానభ్రంశం, స్థిరరాశి శూన్య * src, size_t size, తార్కిక sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    తిరిగి ggml_vk_బఫరు_write_2d_async(subctx, dst, స్థానభ్రంశం, src, size, size, size, 1, sync_staging);
}

శూన్య ggml_vk_బఫరు_write_2d(vk_బఫరు& dst, size_t స్థానభ్రంశం, స్థిరరాశి శూన్య * src, size_t spitch, size_t dpitch, size_t వెడల్పు, size_t ఎత్తు) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << వెడల్పు << ", " << ఎత్తు << ")");
    // Buffer is already mapped
    ఒకవేళ(dst->జ్ఞాపకం_property_flags & vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible) {
        GGML_ASSERT(dst->జ్ఞాపకం_property_flags & vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent);

        ఒకవేళ (వెడల్పు == spitch && వెడల్పు == dpitch) {
            జ్ఞాపకం_నకలు((uint8_t *)dst->ptr + స్థానభ్రంశం, src, వెడల్పు * ఎత్తు);
        } లేకపోతే {
            కొరకు (size_t i = 0; i < ఎత్తు; i++) {
                జ్ఞాపకం_నకలు((uint8_t *)dst->ptr + స్థానభ్రంశం + i * dpitch, (స్థిరరాశి uint8_t *) src + i * spitch, వెడల్పు);
            }
        }
    } లేకపోతే {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_వరుస->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        తార్కిక ret = ggml_vk_బఫరు_write_2d_async(subctx, dst, స్థానభ్రంశం, src, spitch, dpitch, వెడల్పు, ఎత్తు, నిజం);
        GGML_ASSERT(ret);
        ggml_vk_ctx_చివర(subctx);

        కొరకు (స్వయంచాలక& cpy : subctx->లోపల_memcpys) {
            జ్ఞాపకం_నకలు(cpy.dst, cpy.src, cpy.n);
        }

        కొరకు (స్వయంచాలక& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, నిజం, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_వరుస_command_pools_cleanup(dst->device);
    }
}

శూన్య ggml_vk_బఫరు_write(vk_బఫరు& dst, size_t స్థానభ్రంశం, స్థిరరాశి శూన్య * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_బఫరు_write_2d(dst, స్థానభ్రంశం, src, size, size, size, 1);
}

తార్కిక ggml_vk_బఫరు_read_2d_async(vk_context subctx, vk_బఫరు& src, size_t స్థానభ్రంశం, శూన్య * dst, size_t spitch, size_t dpitch, size_t వెడల్పు, size_t ఎత్తు, తార్కిక sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << స్థానభ్రంశం << ", width=" << వెడల్పు << ", height=" << ఎత్తు << ")");
    GGML_ASSERT(వెడల్పు > 0);
    GGML_ASSERT(ఎత్తు > 0);
    GGML_ASSERT(src != శూన్యసూచిక);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_బఫరు buf = శూన్యసూచిక;
    size_t buf_స్థానభ్రంశం = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_స్థానభ్రంశం);

    std::vector<vk::బఫరు_Copy> slices(1);
    ఒకవేళ (వెడల్పు == spitch && వెడల్పు == dpitch) {
        // Only do single write if stride is equal
        slices[0].src_స్థానభ్రంశం_2 = స్థానభ్రంశం;
        slices[0].dst_స్థానభ్రంశం_2 = buf_స్థానభ్రంశం;
        slices[0].size = వెడల్పు * ఎత్తు;
    } లేకపోతే {
        slices.resize(ఎత్తు);
        కొరకు (size_t i = 0; i < ఎత్తు; i++) {
            slices[i].src_స్థానభ్రంశం_2 = స్థానభ్రంశం + i * spitch;
            slices[i].dst_స్థానభ్రంశం_2 = buf_స్థానభ్రంశం + i * dpitch;
            slices[i].size = వెడల్పు;
        }
    }

    ఒకవేళ (buf != శూన్యసూచిక) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(శూన్యసూచిక, subctx);
        subctx->s->బఫరు->buf.copy_బఫరు(src->బఫరు, buf->బఫరు, slices);

        తిరిగి నిజం;
    }
    VK_LOG_DEBUG("STAGING");

    ఒకవేళ (!sync_staging) {
        // copy was not handled caller needs to fall back
        తిరిగి అబద్ధం;
    }

    // Fall back to staging buffer
    స్థిరరాశి size_t staging_కొలత = వెడల్పు * ఎత్తు;
    ggml_vk_ensure_sync_staging_బఫరు(src->device, staging_కొలత);

    vk_బఫరు& staging_బఫరు = src->device->sync_staging;

    std::vector<vk::బఫరు_Copy> staging_slices(1);
    ఒకవేళ (వెడల్పు == spitch) {
        staging_slices[0].src_స్థానభ్రంశం_2 = స్థానభ్రంశం;
        staging_slices[0].dst_స్థానభ్రంశం_2 = 0;
        staging_slices[0].size = staging_కొలత;
    } లేకపోతే {
        staging_slices.resize(ఎత్తు);
        కొరకు (size_t i = 0; i < ఎత్తు; i++) {
            staging_slices[i].src_స్థానభ్రంశం_2 = స్థానభ్రంశం + i * spitch;
            staging_slices[i].dst_స్థానభ్రంశం_2 = i * వెడల్పు;
            staging_slices[i].size = వెడల్పు;
        }
    }

    ggml_vk_sync_buffers(శూన్యసూచిక, subctx);
    subctx->s->బఫరు->buf.copy_బఫరు(src->బఫరు, staging_బఫరు->బఫరు, staging_slices);

    ఒకవేళ (వెడల్పు == dpitch) {
        deferred_memcpy(dst, staging_బఫరు->ptr, staging_కొలత, &subctx->బహిర్గతం_memcpys);
    } లేకపోతే {
        కొరకు (size_t i = 0; i < ఎత్తు; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (స్థిరరాశి uint8_t *) staging_బఫరు->ptr + i * వెడల్పు, వెడల్పు, &subctx->బహిర్గతం_memcpys);
        }
    }
    తిరిగి నిజం;
}

స్థిర తార్కిక ggml_vk_బఫరు_read_async(vk_context subctx, vk_బఫరు& src, size_t స్థానభ్రంశం, శూన్య * dst, size_t size, తార్కిక sync_staging = అబద్ధం) {
    తిరిగి ggml_vk_బఫరు_read_2d_async(subctx, src, స్థానభ్రంశం, dst, size, size, size, 1, sync_staging);
}

శూన్య ggml_vk_బఫరు_read_2d(vk_బఫరు& src, size_t స్థానభ్రంశం, శూన్య * dst, size_t spitch, size_t dpitch, size_t వెడల్పు, size_t ఎత్తు) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->బఫరు << ", " << స్థానభ్రంశం << ", " << వెడల్పు << ", " << ఎత్తు << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    స్థిరరాశి తార్కిక slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->జ్ఞాపకం_property_flags & vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCached);
    ఒకవేళ(src->జ్ఞాపకం_property_flags & vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible && src->device->uma && !slow_host_read) {
        GGML_ASSERT(src->జ్ఞాపకం_property_flags & vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_వరుస->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->బఫరు->buf.pipelineBarrier(
            vk::Pipeline_Stage_Flag_బిట్లు::eComputeShader | vk::Pipeline_Stage_Flag_బిట్లు::eTransfer,
            vk::Pipeline_Stage_Flag_బిట్లు::eHost,
            {},
            { { vk::Access_Flag_బిట్లు::eShaderWrite | vk::Access_Flag_బిట్లు::eTransferWrite,
                vk::Access_Flag_బిట్లు::eHostRead } },
            {}, {});
        ggml_vk_ctx_చివర(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, నిజం, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_వరుస_command_pools_cleanup(src->device);

        ఒకవేళ (వెడల్పు == spitch && వెడల్పు == dpitch) {
            జ్ఞాపకం_నకలు(dst, (స్థిరరాశి uint8_t *) src->ptr + స్థానభ్రంశం, వెడల్పు * ఎత్తు);
        } లేకపోతే {
            కొరకు (size_t i = 0; i < ఎత్తు; i++) {
                జ్ఞాపకం_నకలు((uint8_t *) dst + i * dpitch, (స్థిరరాశి uint8_t *) src->ptr + స్థానభ్రంశం + i * spitch, వెడల్పు);
            }
        }
    } లేకపోతే {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_వరుస->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        తార్కిక ret = ggml_vk_బఫరు_read_2d_async(subctx, src, స్థానభ్రంశం, dst, spitch, dpitch, వెడల్పు, ఎత్తు, నిజం);
        GGML_ASSERT(ret);
        ggml_vk_ctx_చివర(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, నిజం, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_వరుస_command_pools_cleanup(src->device);

        కొరకు (స్వయంచాలక& cpy : subctx->బహిర్గతం_memcpys) {
            జ్ఞాపకం_నకలు(cpy.dst, cpy.src, cpy.n);
        }
    }
}

శూన్య ggml_vk_బఫరు_read(vk_బఫరు& src, size_t స్థానభ్రంశం, శూన్య * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->బఫరు << ", " << స్థానభ్రంశం << ", " << size << ")");
    ggml_vk_బఫరు_read_2d(src, స్థానభ్రంశం, dst, size, size, size, 1);
}

శూన్య ggml_vk_బఫరు_copy_async(vk_context& ctx, vk_బఫరు& dst, size_t dst_స్థానభ్రంశం, vk_బఫరు& src, size_t src_స్థానభ్రంశం, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    Vk_బఫరు_Copy bc{ src_స్థానభ్రంశం, dst_స్థానభ్రంశం, size };

    vk_Cmd_Copy_బఫరు(ctx->s->బఫరు->buf, (Vk_బఫరు)src->బఫరు, (Vk_బఫరు)dst->బఫరు, 1, &bc);
}

శూన్య ggml_vk_బఫరు_copy(vk_బఫరు& dst, size_t dst_స్థానభ్రంశం, vk_బఫరు& src, size_t src_స్థానభ్రంశం, size_t size) {
    ఒకవేళ (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_వరుస->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_బఫరు_copy_async(subctx, dst, dst_స్థానభ్రంశం, src, src_స్థానభ్రంశం, size);
        ggml_vk_ctx_చివర(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, నిజం, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_వరుస_command_pools_cleanup(src->device);
    } లేకపోతే {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_బఫరు(src->device, size);

        // Copy to src staging buffer
        ggml_vk_బఫరు_copy(src->device->sync_staging, 0, src, src_స్థానభ్రంశం, size);
        // Copy to dst buffer
        ggml_vk_బఫరు_write(dst, dst_స్థానభ్రంశం, src->device->sync_staging->ptr, size);
    }
}

శూన్య ggml_vk_బఫరు_memset_async(vk_context& ctx, vk_బఫరు& dst, size_t స్థానభ్రంశం, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << స్థానభ్రంశం << ", " << c << ", " << size << ")");

    ఒకవేళ (dst->జ్ఞాపకం_property_flags & vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + స్థానభ్రంశం, c, size, &ctx->memsets);
        తిరిగి;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->బఫరు->buf.fill_బఫరు(dst->బఫరు, స్థానభ్రంశం, size, c);
}

శూన్య ggml_vk_బఫరు_memset(vk_బఫరు& dst, size_t స్థానభ్రంశం, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << స్థానభ్రంశం << ", " << c << ", " << size << ")");

    ఒకవేళ (dst->జ్ఞాపకం_property_flags & vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + స్థానభ్రంశం, c, size);
        తిరిగి;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_వరుస->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->బఫరు->buf.fill_బఫరు(dst->బఫరు, స్థానభ్రంశం, size, c);
    ggml_vk_ctx_చివర(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, నిజం, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_వరుస_command_pools_cleanup(dst->device);
}

ggml_backend_బఫరు_i ggml_backend_vk_బఫరు_interface = {
    /* .free_buffer     = */ ggml_backend_vk_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_vk_బఫరు_get_పునాది,
    /* .init_tensor     = */ ggml_backend_vk_బఫరు_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_బఫరు_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ ggml_backend_vk_బఫరు_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_బఫరు_సమితి_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_బఫరు_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_బఫరు_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_బఫరు_clear,
    /* .reset           = */ ఖాళీ,
};

vk_బఫరు ggml_vk_బఫరు_from_host_ptr(vk_device & device, శూన్య * ptr, size_t size) {
    ఒకవేళ (!device->external_జ్ఞాపకం_host) {
        తిరిగి {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    ఒకవేళ (uptr & (device->min_imported_host_సూచిక_alignment - 1)) {
        తిరిగి {};
    }
    ఒకవేళ (size & (device->min_imported_host_సూచిక_alignment - 1)) {
        తిరిగి {};
    }

    స్థిరరాశి vk::జ్ఞాపకం_Property_Flags property_flags = vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostVisible | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCoherent | vk::జ్ఞాపకం_Property_Flag_బిట్లు::eHostCached;

    vk_బఫరు buf {};
    ప్రయత్నించు {
        buf = ggml_vk_create_బఫరు(device, size, { property_flags }, ptr);
    } పట్టుకో (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    తిరిగి buf;
}

