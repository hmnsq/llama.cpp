#অন্তর্ভাব <filesystem>
#অন্তর্ভাব <set>
#অন্তর্ভাব <sstream>
#অন্তর্ভাব <string>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#অন্যথা
#   include <dlfcn.h>
#   include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "htp-drv.h"
#অন্তর্ভাব "libdl.h"

#অন্তর্ভাব <domain.h>

//
// Driver API types
//

প্রকারলেখ শূন্য * (*rpcmem_alloc_pfn_t)(পূর্ণসংখ্যা heapid, uint32_t flags, পূর্ণসংখ্যা size);
প্রকারলেখ শূন্য * (*rpcmem_alloc2_pfn_t)(পূর্ণসংখ্যা heapid, uint32_t flags, size_t size);
প্রকারলেখ শূন্য   (*rpcmem_free_pfn_t)(শূন্য * po);
প্রকারলেখ পূর্ণসংখ্যা    (*rpcmem_to_fd_pfn_t)(শূন্য * po);

প্রকারলেখ AEEResult (*dspqueue_create_pfn_t)(পূর্ণসংখ্যা                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_queue_আকার,
                                           uint32_t            resp_queue_আকার,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           শূন্য *              callback_context,
                                           dspqueue_t *        queue);
প্রকারলেখ AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
প্রকারলেখ AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *queue_id);
প্রকারলেখ AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t অঙ্ক_buffers,
                                          কাঠামো dspqueue_বাফার *buffers,
                                          uint32_t message_দৈর্ঘ্য,
                                          ধ্রুবক uint8_t *message,
                                          uint32_t timeout_us);
প্রকারলেখ AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *অঙ্ক_buffers,
                                         কাঠামো dspqueue_বাফার *buffers,
                                         uint32_t max_message_দৈর্ঘ্য,
                                         uint32_t *message_দৈর্ঘ্য, uint8_t *message,
                                         uint32_t timeout_us);
প্রকারলেখ AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *অঙ্ক_buffers,
                                                 কাঠামো dspqueue_বাফার *buffers,
                                                 uint32_t max_message_দৈর্ঘ্য,
                                                 uint32_t *message_দৈর্ঘ্য, uint8_t *message);
প্রকারলেখ পূর্ণসংখ্যা (*fastrpc_mmap_pfn_t)(পূর্ণসংখ্যা domain, পূর্ণসংখ্যা fd, শূন্য *addr, পূর্ণসংখ্যা offset, size_t দৈর্ঘ্য, গণনা fastrpc_map_flags flags);
প্রকারলেখ পূর্ণসংখ্যা (*fastrpc_munmap_pfn_t)(পূর্ণসংখ্যা domain, পূর্ণসংখ্যা fd, শূন্য *addr, size_t দৈর্ঘ্য);

প্রকারলেখ পূর্ণসংখ্যা (*remote_handle64_open_pfn_t)(ধ্রুবক অক্ষর* name, remote_handle64 *ph);
প্রকারলেখ পূর্ণসংখ্যা (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
প্রকারলেখ পূর্ণসংখ্যা (*remote_handle64_close_pfn_t)(remote_handle h);
প্রকারলেখ পূর্ণসংখ্যা (*remote_handle_control_pfn_t)(uint32_t req, শূন্য* তথ্য, uint32_t datalen);
প্রকারলেখ পূর্ণসংখ্যা (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, শূন্য* তথ্য, uint32_t datalen);
প্রকারলেখ পূর্ণসংখ্যা (*remote_session_control_pfn_t)(uint32_t req, শূন্য *তথ্য, uint32_t datalen);
প্রকারলেখ পূর্ণসংখ্যা (*remote_system_request_pfn_t)(system_req_payload * req);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = nullptr;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = nullptr;
rpcmem_free_pfn_t   rpcmem_free_pfn   = nullptr;
rpcmem_to_fd_pfn_t  rpcmem_to_fd_pfn  = nullptr;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = nullptr;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = nullptr;

dspqueue_create_pfn_t       dspqueue_create_pfn       = nullptr;
dspqueue_close_pfn_t        dspqueue_close_pfn        = nullptr;
dspqueue_export_pfn_t       dspqueue_export_pfn       = nullptr;
dspqueue_write_pfn_t        dspqueue_write_pfn        = nullptr;
dspqueue_read_pfn_t         dspqueue_read_pfn         = nullptr;
dspqueue_read_noblock_pfn_t dspqueue_read_noblock_pfn = nullptr;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = nullptr;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = nullptr;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = nullptr;
remote_handle_control_pfn_t   remote_handle_control_pfn   = nullptr;
remote_handle64_control_pfn_t remote_handle64_control_pfn = nullptr;
remote_session_control_pfn_t  remote_session_control_pfn  = nullptr;
remote_system_request_pfn_t   remote_system_request_pfn   = nullptr;

//
// Driver API
//

শূন্য * rpcmem_alloc(পূর্ণসংখ্যা heapid, uint32_t flags, পূর্ণসংখ্যা size) {
    ফেরত rpcmem_alloc_pfn(heapid, flags, size);
}

শূন্য * rpcmem_alloc2(পূর্ণসংখ্যা heapid, uint32_t flags, size_t size) {
    যদি (rpcmem_alloc2_pfn) {
        ফেরত rpcmem_alloc2_pfn(heapid, flags, size);
    } অন্যথা {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        ফেরত rpcmem_alloc_pfn(heapid, flags, size);
    }
}

শূন্য rpcmem_free(শূন্য * po) {
    ফেরত rpcmem_free_pfn(po);
}

পূর্ণসংখ্যা rpcmem_to_fd(শূন্য * po) {
    ফেরত rpcmem_to_fd_pfn(po);
}

HTPDRV_API পূর্ণসংখ্যা fastrpc_mmap(পূর্ণসংখ্যা domain, পূর্ণসংখ্যা fd, শূন্য * addr, পূর্ণসংখ্যা offset, size_t দৈর্ঘ্য, গণনা fastrpc_map_flags flags) {
    ফেরত fastrpc_mmap_pfn(domain, fd, addr, offset, দৈর্ঘ্য, flags);
}

HTPDRV_API পূর্ণসংখ্যা fastrpc_munmap(পূর্ণসংখ্যা domain, পূর্ণসংখ্যা fd, শূন্য * addr, size_t দৈর্ঘ্য) {
    ফেরত fastrpc_munmap_pfn(domain, fd, addr, দৈর্ঘ্য);
}

AEEResult dspqueue_create(পূর্ণসংখ্যা                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_আকার,
                          uint32_t            resp_queue_আকার,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          শূন্য *              callback_context,
                          dspqueue_t *        queue) {
    ফেরত dspqueue_create_pfn(domain, flags, req_queue_আকার, resp_queue_আকার, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    ফেরত dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    ফেরত dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 অঙ্ক_buffers,
                         কাঠামো dspqueue_বাফার * buffers,
                         uint32_t                 message_দৈর্ঘ্য,
                         ধ্রুবক uint8_t *          message,
                         uint32_t                 timeout_us) {
    ফেরত dspqueue_write_pfn(queue, flags, অঙ্ক_buffers, buffers, message_দৈর্ঘ্য, message, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               অঙ্ক_buffers,
                        কাঠামো dspqueue_বাফার * buffers,
                        uint32_t                 max_message_দৈর্ঘ্য,
                        uint32_t *               message_দৈর্ঘ্য,
                        uint8_t *                message,
                        uint32_t                 timeout_us) {
#ifdef _WIN32
    যদি (timeout_us == 0) {
        ফেরত dspqueue_read_noblock_pfn(queue, flags, max_buffers, অঙ্ক_buffers, buffers, max_message_দৈর্ঘ্য,
                                     message_দৈর্ঘ্য, message);
    }
#endif
    ফেরত dspqueue_read_pfn(queue, flags, max_buffers, অঙ্ক_buffers, buffers, max_message_দৈর্ঘ্য, message_দৈর্ঘ্য,
                             message, timeout_us);
}

HTPDRV_API পূর্ণসংখ্যা remote_handle64_open(ধ্রুবক অক্ষর * name, remote_handle64 * ph) {
    ফেরত remote_handle64_open_pfn(name, ph);
}

HTPDRV_API পূর্ণসংখ্যা remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    ফেরত remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API পূর্ণসংখ্যা remote_handle64_close(remote_handle64 h) {
    ফেরত remote_handle64_close_pfn(h);
}

HTPDRV_API পূর্ণসংখ্যা remote_handle_control(uint32_t req, শূন্য * তথ্য, uint32_t datalen) {
    ফেরত remote_handle_control_pfn(req, তথ্য, datalen);
}

HTPDRV_API পূর্ণসংখ্যা remote_handle64_control(remote_handle64 h, uint32_t req, শূন্য * তথ্য, uint32_t datalen) {
    ফেরত remote_handle64_control_pfn(h, req, তথ্য, datalen);
}

HTPDRV_API পূর্ণসংখ্যা remote_session_control(uint32_t req, শূন্য * তথ্য, uint32_t datalen) {
    ফেরত remote_session_control_pfn(req, তথ্য, datalen);
}

HTPDRV_API পূর্ণসংখ্যা remote_system_request(system_req_payload * req) {
    যদি (!remote_system_request_pfn) {
        ফেরত AEE_EUNSUPPORTEDAPI;
    }
    ফেরত remote_system_request_pfn(req);
}

#ifdef _WIN32

স্থির std::string wstr_to_str(std::wstring_view wstr) {
    std::string ফলাফল;
    যদি (wstr.empty()) {
        ফেরত ফলাফল;
    }
    auto bytes_needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.তথ্য(), (পূর্ণসংখ্যা) wstr.size(),
                                            nullptr, 0, nullptr, nullptr);
    যদি (bytes_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        ছোড়ো std::runtime_error("Invalid wstring input");
    }

    ফলাফল.resize(bytes_needed, '\0');
    পূর্ণসংখ্যা bytes_written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.তথ্য(), (পূর্ণসংখ্যা) wstr.size(),
                                            ফলাফল.তথ্য(), bytes_needed,
                                            nullptr, nullptr);
    যদি (bytes_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        ছোড়ো std::runtime_error("Wstring conversion failed");
    }
    ফেরত ফলাফল;
}

স্থির std::string get_driver_path() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string ফলাফল;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(ফাঁকা, ফাঁকা, STANDARD_RIGHTS_READ);
    যদি (nullptr == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", GetLastError());
        ফেরত ফলাফল;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    যদি (nullptr == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", GetLastError());
        CloseServiceHandle(schSCManager);
        ফেরত ফলাফল;
    }

    // Store the size of buffer used as an output.
    DWORD বাফার_আকার_2;
    যদি (!QueryServiceConfigW(schService, ফাঁকা, 0, &বাফার_আকার_2) &&
        (GetLastError() != ERROR_INSUFFICIENT_বাফার)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ফেরত ফলাফল;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, বাফার_আকার_2));
    যদি (!QueryServiceConfigW(schService, serviceConfig, বাফার_আকার_2, &বাফার_আকার_2)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        ফেরত ফলাফল;
    }

    // Read the driver file path get its parent directory
    std::wstring driverPath = std::wstring(serviceConfig->lp_দ্বিমিক_Path_Name);
    driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    ধ্রুবক std::wstring system_মূল_Placeholder = L"\\SystemRoot";
    যদি (0 != driverPath.compare(0, system_মূল_Placeholder.দৈর্ঘ্য(), system_মূল_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        ফেরত ফলাফল;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    ধ্রুবক std::wstring system_মূল_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD অঙ্ক_শব্দগুলি = GetEnvironmentVariableW(system_মূল_Env.c_str(), ফাঁকা, 0);
    যদি (অঙ্ক_শব্দগুলি == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        ফেরত ফলাফল;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> system_মূল(অঙ্ক_শব্দগুলি + 1);
    অঙ্ক_শব্দগুলি = GetEnvironmentVariableW(system_মূল_Env.c_str(), system_মূল.তথ্য(), অঙ্ক_শব্দগুলি + 1);
    যদি (অঙ্ক_শব্দগুলি == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        ফেরত ফলাফল;
    }
    driverPath.replace(0, system_মূল_Placeholder.দৈর্ঘ্য(), std::wstring(system_মূল.তথ্য()));

    ফেরত wstr_to_str(driverPath);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

পূর্ণসংখ্যা htpdrv_init() {
    স্থির dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    স্থির যৌক্তিক initialized = মিথ্যা;
#ifdef _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#অন্যথা
    std::string drv_path = "libcdsprpc.so";
#endif
    যদি (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        ফেরত AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    যদি (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        ফেরত AEE_EUNABLETOLOAD;
    }

#define dlsym(drv, type, pfn, প্রতীক, ignore)                               \
    করো {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        যদি (!ignore && nullptr == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            ফেরত AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } যখন (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, মিথ্যা);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, সত্য);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, মিথ্যা);
    dlsym(handle.get(), rpcmem_to_fd_pfn_t, rpcmem_to_fd_pfn, rpcmem_to_fd, মিথ্যা);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, মিথ্যা);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, মিথ্যা);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, মিথ্যা);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, মিথ্যা);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, মিথ্যা);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, মিথ্যা);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, মিথ্যা);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, মিথ্যা);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, মিথ্যা);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, মিথ্যা);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, মিথ্যা);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, মিথ্যা);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, মিথ্যা);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, মিথ্যা);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, সত্য);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = সত্য;

    ফেরত AEE_SUCCESS;
}

domain * htpdrv_get_domain(পূর্ণসংখ্যা domain_id) {
    পূর্ণসংখ্যা i    = 0;
    পূর্ণসংখ্যা size = আকার(supported_domains) / আকার(domain);

    জন্য (i = 0; i < size; i++) {
        যদি (supported_domains[i].id == domain_id) {
            ফেরত &supported_domains[i];
        }
    }

    ফেরত ফাঁকা;
}

পূর্ণসংখ্যা htpdrv_get_arch(পূর্ণসংখ্যা domain, পূর্ণসংখ্যা * arch) {
    যদি (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        ফেরত AEE_EUNSUPPORTEDAPI;
    }

    কাঠামো remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    পূর্ণসংখ্যা err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, আকার(arch_ver));
    যদি ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        ফেরত AEE_EUNSUPPORTEDAPI;
    }

    যদি (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        ফেরত err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (পূর্ণসংখ্যা) ((val >> 4) * 10 + (val & 0x0f));
    ফেরত 0;
}
