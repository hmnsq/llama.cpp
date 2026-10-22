#અન્તર્ભાવ <filesystem>
#અન્તર્ભાવ <set>
#અન્તર્ભાવ <sstream>
#અન્તર્ભાવ <string>

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#નહીંતર
#   include <dlfcn.h>
#   include <unistd.h>
#endif

#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wsign-compare"
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wmicrosoft-enum-value"
#pragma clang diagnostic ignored "-Wnested-anon-types"

#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "htp-drv.h"
#અન્તર્ભાવ "libdl.h"

#અન્તર્ભાવ <domain.h>

//
// Driver API types
//

પ્રકારવ્યાખ્યા શૂન્ય * (*rpcmem_alloc_pfn_t)(પૂર્ણાંક heapid, uint32_t flags, પૂર્ણાંક size);
પ્રકારવ્યાખ્યા શૂન્ય * (*rpcmem_alloc2_pfn_t)(પૂર્ણાંક heapid, uint32_t flags, size_t size);
પ્રકારવ્યાખ્યા શૂન્ય   (*rpcmem_free_pfn_t)(શૂન્ય * po);
પ્રકારવ્યાખ્યા પૂર્ણાંક    (*rpcmem_to_fd_pfn_t)(શૂન્ય * po);

પ્રકારવ્યાખ્યા AEEResult (*dspqueue_create_pfn_t)(પૂર્ણાંક                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_queue_size,
                                           uint32_t            resp_queue_size,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           શૂન્ય *              callback_context,
                                           dspqueue_t *        queue);
પ્રકારવ્યાખ્યા AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
પ્રકારવ્યાખ્યા AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *queue_id);
પ્રકારવ્યાખ્યા AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t અંક_buffers,
                                          સંરચના dspqueue_buffer *buffers,
                                          uint32_t message_length,
                                          અચલ uint8_t *message,
                                          uint32_t timeout_us);
પ્રકારવ્યાખ્યા AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *અંક_buffers,
                                         સંરચના dspqueue_buffer *buffers,
                                         uint32_t max_message_length,
                                         uint32_t *message_length, uint8_t *message,
                                         uint32_t timeout_us);
પ્રકારવ્યાખ્યા AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *અંક_buffers,
                                                 સંરચના dspqueue_buffer *buffers,
                                                 uint32_t max_message_length,
                                                 uint32_t *message_length, uint8_t *message);
પ્રકારવ્યાખ્યા પૂર્ણાંક (*fastrpc_mmap_pfn_t)(પૂર્ણાંક domain, પૂર્ણાંક fd, શૂન્ય *addr, પૂર્ણાંક offset, size_t length, ગણના fastrpc_map_flags flags);
પ્રકારવ્યાખ્યા પૂર્ણાંક (*fastrpc_munmap_pfn_t)(પૂર્ણાંક domain, પૂર્ણાંક fd, શૂન્ય *addr, size_t length);

પ્રકારવ્યાખ્યા પૂર્ણાંક (*remote_handle64_open_pfn_t)(અચલ અક્ષર* name, remote_handle64 *ph);
પ્રકારવ્યાખ્યા પૂર્ણાંક (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
પ્રકારવ્યાખ્યા પૂર્ણાંક (*remote_handle64_close_pfn_t)(remote_handle h);
પ્રકારવ્યાખ્યા પૂર્ણાંક (*remote_handle_control_pfn_t)(uint32_t req, શૂન્ય* data, uint32_t datalen);
પ્રકારવ્યાખ્યા પૂર્ણાંક (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, શૂન્ય* data, uint32_t datalen);
પ્રકારવ્યાખ્યા પૂર્ણાંક (*remote_session_control_pfn_t)(uint32_t req, શૂન્ય *data, uint32_t datalen);
પ્રકારવ્યાખ્યા પૂર્ણાંક (*remote_system_request_pfn_t)(system_req_payload * req);

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

શૂન્ય * rpcmem_alloc(પૂર્ણાંક heapid, uint32_t flags, પૂર્ણાંક size) {
    પાછા rpcmem_alloc_pfn(heapid, flags, size);
}

શૂન્ય * rpcmem_alloc2(પૂર્ણાંક heapid, uint32_t flags, size_t size) {
    જો (rpcmem_alloc2_pfn) {
        પાછા rpcmem_alloc2_pfn(heapid, flags, size);
    } નહીંતર {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        પાછા rpcmem_alloc_pfn(heapid, flags, size);
    }
}

શૂન્ય rpcmem_free(શૂન્ય * po) {
    પાછા rpcmem_free_pfn(po);
}

પૂર્ણાંક rpcmem_to_fd(શૂન્ય * po) {
    પાછા rpcmem_to_fd_pfn(po);
}

HTPDRV_API પૂર્ણાંક fastrpc_mmap(પૂર્ણાંક domain, પૂર્ણાંક fd, શૂન્ય * addr, પૂર્ણાંક offset, size_t length, ગણના fastrpc_map_flags flags) {
    પાછા fastrpc_mmap_pfn(domain, fd, addr, offset, length, flags);
}

HTPDRV_API પૂર્ણાંક fastrpc_munmap(પૂર્ણાંક domain, પૂર્ણાંક fd, શૂન્ય * addr, size_t length) {
    પાછા fastrpc_munmap_pfn(domain, fd, addr, length);
}

AEEResult dspqueue_create(પૂર્ણાંક                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_size,
                          uint32_t            resp_queue_size,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          શૂન્ય *              callback_context,
                          dspqueue_t *        queue) {
    પાછા dspqueue_create_pfn(domain, flags, req_queue_size, resp_queue_size, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    પાછા dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    પાછા dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 અંક_buffers,
                         સંરચના dspqueue_buffer * buffers,
                         uint32_t                 message_length,
                         અચલ uint8_t *          message,
                         uint32_t                 timeout_us) {
    પાછા dspqueue_write_pfn(queue, flags, અંક_buffers, buffers, message_length, message, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               અંક_buffers,
                        સંરચના dspqueue_buffer * buffers,
                        uint32_t                 max_message_length,
                        uint32_t *               message_length,
                        uint8_t *                message,
                        uint32_t                 timeout_us) {
#ifdef _WIN32
    જો (timeout_us == 0) {
        પાછા dspqueue_read_noblock_pfn(queue, flags, max_buffers, અંક_buffers, buffers, max_message_length,
                                     message_length, message);
    }
#endif
    પાછા dspqueue_read_pfn(queue, flags, max_buffers, અંક_buffers, buffers, max_message_length, message_length,
                             message, timeout_us);
}

HTPDRV_API પૂર્ણાંક remote_handle64_open(અચલ અક્ષર * name, remote_handle64 * ph) {
    પાછા remote_handle64_open_pfn(name, ph);
}

HTPDRV_API પૂર્ણાંક remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    પાછા remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API પૂર્ણાંક remote_handle64_close(remote_handle64 h) {
    પાછા remote_handle64_close_pfn(h);
}

HTPDRV_API પૂર્ણાંક remote_handle_control(uint32_t req, શૂન્ય * data, uint32_t datalen) {
    પાછા remote_handle_control_pfn(req, data, datalen);
}

HTPDRV_API પૂર્ણાંક remote_handle64_control(remote_handle64 h, uint32_t req, શૂન્ય * data, uint32_t datalen) {
    પાછા remote_handle64_control_pfn(h, req, data, datalen);
}

HTPDRV_API પૂર્ણાંક remote_session_control(uint32_t req, શૂન્ય * data, uint32_t datalen) {
    પાછા remote_session_control_pfn(req, data, datalen);
}

HTPDRV_API પૂર્ણાંક remote_system_request(system_req_payload * req) {
    જો (!remote_system_request_pfn) {
        પાછા AEE_EUNSUPPORTEDAPI;
    }
    પાછા remote_system_request_pfn(req);
}

#ifdef _WIN32

સ્થિર std::string wstr_to_str(std::wstring_view wstr) {
    std::string result;
    જો (wstr.empty()) {
        પાછા result;
    }
    auto bytes_needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (પૂર્ણાંક) wstr.size(),
                                            nullptr, 0, nullptr, nullptr);
    જો (bytes_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        ફેંકો std::runtime_error("Invalid wstring input");
    }

    result.resize(bytes_needed, '\0');
    પૂર્ણાંક bytes_written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.data(), (પૂર્ણાંક) wstr.size(),
                                            result.data(), bytes_needed,
                                            nullptr, nullptr);
    જો (bytes_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        ફેંકો std::runtime_error("Wstring conversion failed");
    }
    પાછા result;
}

સ્થિર std::string get_driver_path() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string result;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(ખાલી, ખાલી, STANDARD_RIGHTS_READ);
    જો (nullptr == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", GetLastError());
        પાછા result;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    જો (nullptr == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", GetLastError());
        CloseServiceHandle(schSCManager);
        પાછા result;
    }

    // Store the size of buffer used as an output.
    DWORD bufferSize;
    જો (!QueryServiceConfigW(schService, ખાલી, 0, &bufferSize) &&
        (GetLastError() != ERROR_INSUFFICIENT_BUFFER)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        પાછા result;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, bufferSize));
    જો (!QueryServiceConfigW(schService, serviceConfig, bufferSize, &bufferSize)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        પાછા result;
    }

    // Read the driver file path get its parent directory
    std::wstring driverPath = std::wstring(serviceConfig->lp_દ્વિઅંકી_Path_Name);
    driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    અચલ std::wstring system_મૂળ_Placeholder = L"\\SystemRoot";
    જો (0 != driverPath.compare(0, system_મૂળ_Placeholder.length(), system_મૂળ_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        પાછા result;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    અચલ std::wstring system_મૂળ_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD અંક_શબ્દો = GetEnvironmentVariableW(system_મૂળ_Env.c_str(), ખાલી, 0);
    જો (અંક_શબ્દો == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        પાછા result;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> system_મૂળ(અંક_શબ્દો + 1);
    અંક_શબ્દો = GetEnvironmentVariableW(system_મૂળ_Env.c_str(), system_મૂળ.data(), અંક_શબ્દો + 1);
    જો (અંક_શબ્દો == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        પાછા result;
    }
    driverPath.replace(0, system_મૂળ_Placeholder.length(), std::wstring(system_મૂળ.data()));

    પાછા wstr_to_str(driverPath);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

પૂર્ણાંક htpdrv_init() {
    સ્થિર dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    સ્થિર તાર્કિક initialized = ખોટું;
#ifdef _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#નહીંતર
    std::string drv_path = "libcdsprpc.so";
#endif
    જો (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        પાછા AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    જો (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        પાછા AEE_EUNABLETOLOAD;
    }

#define dlsym(drv, type, pfn, પ્રતીક, ignore)                               \
    કરો {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        જો (!ignore && nullptr == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            પાછા AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } જ્યારે (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, ખોટું);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, સાચું);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, ખોટું);
    dlsym(handle.get(), rpcmem_to_fd_pfn_t, rpcmem_to_fd_pfn, rpcmem_to_fd, ખોટું);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, ખોટું);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, ખોટું);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, ખોટું);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, ખોટું);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, ખોટું);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, ખોટું);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, ખોટું);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, ખોટું);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, ખોટું);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, ખોટું);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, ખોટું);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, ખોટું);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, ખોટું);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, ખોટું);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, સાચું);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = સાચું;

    પાછા AEE_SUCCESS;
}

domain * htpdrv_get_domain(પૂર્ણાંક domain_id) {
    પૂર્ણાંક i    = 0;
    પૂર્ણાંક size = કદ(supported_domains) / કદ(domain);

    માટે (i = 0; i < size; i++) {
        જો (supported_domains[i].id == domain_id) {
            પાછા &supported_domains[i];
        }
    }

    પાછા ખાલી;
}

પૂર્ણાંક htpdrv_get_arch(પૂર્ણાંક domain, પૂર્ણાંક * arch) {
    જો (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        પાછા AEE_EUNSUPPORTEDAPI;
    }

    સંરચના remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    પૂર્ણાંક err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, કદ(arch_ver));
    જો ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        પાછા AEE_EUNSUPPORTEDAPI;
    }

    જો (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        પાછા err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (પૂર્ણાંક) ((val >> 4) * 10 + (val & 0x0f));
    પાછા 0;
}
