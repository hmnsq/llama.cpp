#!/usr/bin/env python3

ਆਯਾਤ argparse
ਆਯਾਤ logging
ਆਯਾਤ os
ਆਯਾਤ shutil
ਆਯਾਤ struct
ਆਯਾਤ sys
ਆਯਾਤ tempfile
ਆਯਾਤ zlib

ELF_MAGIC = b'\x7fELF'
ELF_CLASS_ਸੂਚਕਾਂਕ = 4
ELFCLASS32 = 1
ELFCLASS64 = 2
ELF_ਸੰਸਕਰਣ_ਸੂਚਕਾਂਕ = 6
EV_CURRENT = 1
MAX_PHDR_ਗਿਣਤੀ = 100

ਲੱਦੋ_TYPE = 0x1

SHT_NOBITS = 8

SWIV_MAGIC = b'SWIV'
SWIV_SEG_ਮਾਪ_MIN = 16
# SHT_LOUSER plus ASCII "SWIV".
SWIV_SECTION_TYPE = 0xD3574956

logger = logging.getLogger("htp-swiv")


ਵਰਗ Elf_Ehdr_common:
    s = struct.Struct('16sHHI')

    ਪਰਿਭਾਸ਼ਾ __init__(self, ਅੰਕੜੇ):
        unpacked_ਅੰਕੜੇ = (Elf_Ehdr_common.s).unpack(ਅੰਕੜੇ)
        self.e_ident = unpacked_ਅੰਕੜੇ[0]
        self.e_type = unpacked_ਅੰਕੜੇ[1]
        self.e_machine = unpacked_ਅੰਕੜੇ[2]
        self.e_ਸੰਸਕਰਣ = unpacked_ਅੰਕੜੇ[3]


ਵਰਗ Elf32_Ehdr:
    s = struct.Struct('16sHHIIIIIHHHHHH')

    ਪਰਿਭਾਸ਼ਾ __init__(self, ਅੰਕੜੇ):
        unpacked_ਅੰਕੜੇ = (Elf32_Ehdr.s).unpack(ਅੰਕੜੇ)
        self.e_ident = unpacked_ਅੰਕੜੇ[0]
        self.e_type = unpacked_ਅੰਕੜੇ[1]
        self.e_machine = unpacked_ਅੰਕੜੇ[2]
        self.e_ਸੰਸਕਰਣ = unpacked_ਅੰਕੜੇ[3]
        self.e_ਇੰਦਰਾਜ = unpacked_ਅੰਕੜੇ[4]
        self.e_phoff = unpacked_ਅੰਕੜੇ[5]
        self.e_shoff = unpacked_ਅੰਕੜੇ[6]
        self.e_flags = unpacked_ਅੰਕੜੇ[7]
        self.e_ehsize = unpacked_ਅੰਕੜੇ[8]
        self.e_phentsize = unpacked_ਅੰਕੜੇ[9]
        self.e_phnum = unpacked_ਅੰਕੜੇ[10]
        self.e_shentsize = unpacked_ਅੰਕੜੇ[11]
        self.e_shnum = unpacked_ਅੰਕੜੇ[12]
        self.e_shstrndx = unpacked_ਅੰਕੜੇ[13]

    ਪਰਿਭਾਸ਼ਾ get_packed_ਅੰਕੜੇ(self):
        ਮੁੱਲਾਂ = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_ਸੰਸਕਰਣ,
                  self.e_ਇੰਦਰਾਜ,
                  self.e_phoff,
                  self.e_shoff,
                  self.e_flags,
                  self.e_ehsize,
                  self.e_phentsize,
                  self.e_phnum,
                  self.e_shentsize,
                  self.e_shnum,
                  self.e_shstrndx
                  ]

        ਵਾਪਸ (Elf32_Ehdr.s).pack(*ਮੁੱਲਾਂ)


ਵਰਗ Elf32_Phdr:
    s = struct.Struct('I' * 8)

    ਪਰਿਭਾਸ਼ਾ __init__(self, ਅੰਕੜੇ):
        unpacked_ਅੰਕੜੇ = (Elf32_Phdr.s).unpack(ਅੰਕੜੇ)
        self.p_type = unpacked_ਅੰਕੜੇ[0]
        self.p_ਵਿੱਥ = unpacked_ਅੰਕੜੇ[1]
        self.p_vaddr = unpacked_ਅੰਕੜੇ[2]
        self.p_paddr = unpacked_ਅੰਕੜੇ[3]
        self.p_filesz = unpacked_ਅੰਕੜੇ[4]
        self.p_memsz = unpacked_ਅੰਕੜੇ[5]
        self.p_flags = unpacked_ਅੰਕੜੇ[6]
        self.p_align = unpacked_ਅੰਕੜੇ[7]

    ਪਰਿਭਾਸ਼ਾ get_packed_ਅੰਕੜੇ(self):
        ਮੁੱਲਾਂ = [self.p_type,
                  self.p_ਵਿੱਥ,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_flags,
                  self.p_align
                  ]

        ਵਾਪਸ (Elf32_Phdr.s).pack(*ਮੁੱਲਾਂ)


ਵਰਗ Elf32_Shdr:
    s = struct.Struct('I' * 10)

    ਪਰਿਭਾਸ਼ਾ __init__(self, ਅੰਕੜੇ):
        unpacked_ਅੰਕੜੇ = (Elf32_Shdr.s).unpack(ਅੰਕੜੇ)
        self.sh_name = unpacked_ਅੰਕੜੇ[0]
        self.sh_type = unpacked_ਅੰਕੜੇ[1]
        self.sh_flags = unpacked_ਅੰਕੜੇ[2]
        self.sh_addr = unpacked_ਅੰਕੜੇ[3]
        self.sh_ਵਿੱਥ = unpacked_ਅੰਕੜੇ[4]
        self.sh_ਮਾਪ = unpacked_ਅੰਕੜੇ[5]
        self.sh_link = unpacked_ਅੰਕੜੇ[6]
        self.sh_info = unpacked_ਅੰਕੜੇ[7]
        self.sh_addralign = unpacked_ਅੰਕੜੇ[8]
        self.sh_entsize = unpacked_ਅੰਕੜੇ[9]
        self.sh_name_str = ""

    ਪਰਿਭਾਸ਼ਾ get_packed_ਅੰਕੜੇ(self):
        ਮੁੱਲਾਂ = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_ਵਿੱਥ,
                  self.sh_ਮਾਪ,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        ਵਾਪਸ (Elf32_Shdr.s).pack(*ਮੁੱਲਾਂ)


ਵਰਗ Elf64_Ehdr:
    s = struct.Struct('16sHHIQQQIHHHHHH')

    ਪਰਿਭਾਸ਼ਾ __init__(self, ਅੰਕੜੇ):
        unpacked_ਅੰਕੜੇ = (Elf64_Ehdr.s).unpack(ਅੰਕੜੇ)
        self.e_ident = unpacked_ਅੰਕੜੇ[0]
        self.e_type = unpacked_ਅੰਕੜੇ[1]
        self.e_machine = unpacked_ਅੰਕੜੇ[2]
        self.e_ਸੰਸਕਰਣ = unpacked_ਅੰਕੜੇ[3]
        self.e_ਇੰਦਰਾਜ = unpacked_ਅੰਕੜੇ[4]
        self.e_phoff = unpacked_ਅੰਕੜੇ[5]
        self.e_shoff = unpacked_ਅੰਕੜੇ[6]
        self.e_flags = unpacked_ਅੰਕੜੇ[7]
        self.e_ehsize = unpacked_ਅੰਕੜੇ[8]
        self.e_phentsize = unpacked_ਅੰਕੜੇ[9]
        self.e_phnum = unpacked_ਅੰਕੜੇ[10]
        self.e_shentsize = unpacked_ਅੰਕੜੇ[11]
        self.e_shnum = unpacked_ਅੰਕੜੇ[12]
        self.e_shstrndx = unpacked_ਅੰਕੜੇ[13]

    ਪਰਿਭਾਸ਼ਾ get_packed_ਅੰਕੜੇ(self):
        ਮੁੱਲਾਂ = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_ਸੰਸਕਰਣ,
                  self.e_ਇੰਦਰਾਜ,
                  self.e_phoff,
                  self.e_shoff,
                  self.e_flags,
                  self.e_ehsize,
                  self.e_phentsize,
                  self.e_phnum,
                  self.e_shentsize,
                  self.e_shnum,
                  self.e_shstrndx
                  ]

        ਵਾਪਸ (Elf64_Ehdr.s).pack(*ਮੁੱਲਾਂ)


ਵਰਗ Elf64_Phdr:
    s = struct.Struct('IIQQQQQQ')

    ਪਰਿਭਾਸ਼ਾ __init__(self, ਅੰਕੜੇ):
        unpacked_ਅੰਕੜੇ = (Elf64_Phdr.s).unpack(ਅੰਕੜੇ)
        self.p_type = unpacked_ਅੰਕੜੇ[0]
        self.p_flags = unpacked_ਅੰਕੜੇ[1]
        self.p_ਵਿੱਥ = unpacked_ਅੰਕੜੇ[2]
        self.p_vaddr = unpacked_ਅੰਕੜੇ[3]
        self.p_paddr = unpacked_ਅੰਕੜੇ[4]
        self.p_filesz = unpacked_ਅੰਕੜੇ[5]
        self.p_memsz = unpacked_ਅੰਕੜੇ[6]
        self.p_align = unpacked_ਅੰਕੜੇ[7]

    ਪਰਿਭਾਸ਼ਾ get_packed_ਅੰਕੜੇ(self):
        ਮੁੱਲਾਂ = [self.p_type,
                  self.p_flags,
                  self.p_ਵਿੱਥ,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_align
                  ]

        ਵਾਪਸ (Elf64_Phdr.s).pack(*ਮੁੱਲਾਂ)


ਵਰਗ Elf64_Shdr:
    s = struct.Struct('IIQQQQIIQQ')

    ਪਰਿਭਾਸ਼ਾ __init__(self, ਅੰਕੜੇ):
        unpacked_ਅੰਕੜੇ = (Elf64_Shdr.s).unpack(ਅੰਕੜੇ)
        self.sh_name = unpacked_ਅੰਕੜੇ[0]
        self.sh_type = unpacked_ਅੰਕੜੇ[1]
        self.sh_flags = unpacked_ਅੰਕੜੇ[2]
        self.sh_addr = unpacked_ਅੰਕੜੇ[3]
        self.sh_ਵਿੱਥ = unpacked_ਅੰਕੜੇ[4]
        self.sh_ਮਾਪ = unpacked_ਅੰਕੜੇ[5]
        self.sh_link = unpacked_ਅੰਕੜੇ[6]
        self.sh_info = unpacked_ਅੰਕੜੇ[7]
        self.sh_addralign = unpacked_ਅੰਕੜੇ[8]
        self.sh_entsize = unpacked_ਅੰਕੜੇ[9]
        self.sh_name_str = ""

    ਪਰਿਭਾਸ਼ਾ get_packed_ਅੰਕੜੇ(self):
        ਮੁੱਲਾਂ = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_ਵਿੱਥ,
                  self.sh_ਮਾਪ,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        ਵਾਪਸ (Elf64_Shdr.s).pack(*ਮੁੱਲਾਂ)


ਵਰਗ SWIV_Seg:
    s = struct.Struct('4sIQ')

    ਪਰਿਭਾਸ਼ਾ __init__(self):
        self.magic = SWIV_MAGIC
        self.ਚੱਕਰੀ_ਜਾਂਚ = 0
        self.padding = 0

    ਪਰਿਭਾਸ਼ਾ get_packed_ਅੰਕੜੇ(self):
        ਮੁੱਲਾਂ = [self.magic,
                  self.ਚੱਕਰੀ_ਜਾਂਚ,
                  self.padding
                  ]

        ਵਾਪਸ (SWIV_Seg.s).pack(*ਮੁੱਲਾਂ)


ਪਰਿਭਾਸ਼ਾ is_elf64(elf_ਸਿਰਲੇਖ):
    ਵਾਪਸ elf_ਸਿਰਲੇਖ.e_ident[ELF_CLASS_ਸੂਚਕਾਂਕ] == ELFCLASS64


ਪਰਿਭਾਸ਼ਾ verify_elf_ਸਿਰਲੇਖ(elf_ਸਿਰਲੇਖ):
    ident = elf_ਸਿਰਲੇਖ.e_ident
    ਵਾਪਸ (ident.startswith(ELF_MAGIC)
            ਅਤੇ ident[ELF_CLASS_ਸੂਚਕਾਂਕ] ਵਿੱਚ (ELFCLASS32, ELFCLASS64)
            ਅਤੇ ident[ELF_ਸੰਸਕਰਣ_ਸੂਚਕਾਂਕ] == EV_CURRENT)


ਪਰਿਭਾਸ਼ਾ file_copy_ਵਿੱਥ(ਵਿੱਚ_fp, ਵਿੱਚ_off, ਬਾਹਰਲਾ_fp, ਬਾਹਰਲਾ_off, ਅੰਕ_ਬਾਈਟਾਂ):
    ਵਿੱਚ_fp.seek(ਵਿੱਚ_off)
    read_ਵਿੱਚ = ਵਿੱਚ_fp.ਪਠ(ਅੰਕ_ਬਾਈਟਾਂ)
    ਬਾਹਰਲਾ_fp.seek(ਬਾਹਰਲਾ_off)
    ਬਾਹਰਲਾ_fp.ਲਿਖ(read_ਵਿੱਚ)

    ਵਾਪਸ ਅੰਕ_ਬਾਈਟਾਂ


ਪਰਿਭਾਸ਼ਾ preprocess_elf_file(elf_file_name):
    ਨਾਲ open(elf_file_name, 'rb') ਇਵ elf_fp:
        elf_ਸਿਰਲੇਖ = Elf_Ehdr_common(elf_fp.ਪਠ(Elf_Ehdr_common.s.ਮਾਪ))

        ਜੇ verify_elf_ਸਿਰਲੇਖ(elf_ਸਿਰਲੇਖ) ਹੈ ਝੂਠ:
            ਉਠਾਓ RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        ਜੇ is_elf64(elf_ਸਿਰਲੇਖ):
            elf_ਸਿਰਲੇਖ = Elf64_Ehdr(elf_fp.ਪਠ(Elf64_Ehdr.s.ਮਾਪ))
        ਨਹੀਂਤਾਂ:
            elf_ਸਿਰਲੇਖ = Elf32_Ehdr(elf_fp.ਪਠ(Elf32_Ehdr.s.ਮਾਪ))

        phdr_ਸਾਰਣੀ = []

        phdr_ਮਾਪ = elf_ਸਿਰਲੇਖ.e_phentsize

        file_ਵਿੱਥ = elf_ਸਿਰਲੇਖ.e_phoff
        elf_fp.seek(file_ਵਿੱਥ)

        ਲਈ _ ਵਿੱਚ ਸੀਮਾ(elf_ਸਿਰਲੇਖ.e_phnum):
            ਜੇ is_elf64(elf_ਸਿਰਲੇਖ):
                phdr_ਸਾਰਣੀ.append(Elf64_Phdr(elf_fp.ਪਠ(phdr_ਮਾਪ)))
            ਨਹੀਂਤਾਂ:
                phdr_ਸਾਰਣੀ.append(Elf32_Phdr(elf_fp.ਪਠ(phdr_ਮਾਪ)))

    ਵਾਪਸ [elf_ਸਿਰਲੇਖ, phdr_ਸਾਰਣੀ]


ਪਰਿਭਾਸ਼ਾ process_section_ਸਿਰਲੇਖ(elf_file_name):
    ਨਾਲ open(elf_file_name, 'rb') ਇਵ elf_fp:
        elf_ਸਿਰਲੇਖ = Elf_Ehdr_common(elf_fp.ਪਠ(Elf_Ehdr_common.s.ਮਾਪ))

        ਜੇ verify_elf_ਸਿਰਲੇਖ(elf_ਸਿਰਲੇਖ) ਹੈ ਝੂਠ:
            ਉਠਾਓ RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        ਜੇ is_elf64(elf_ਸਿਰਲੇਖ):
            elf_ਸਿਰਲੇਖ = Elf64_Ehdr(elf_fp.ਪਠ(Elf64_Ehdr.s.ਮਾਪ))
        ਨਹੀਂਤਾਂ:
            elf_ਸਿਰਲੇਖ = Elf32_Ehdr(elf_fp.ਪਠ(Elf32_Ehdr.s.ਮਾਪ))

        shdr_ਸਾਰਣੀ = []

        shdr_ਮਾਪ = elf_ਸਿਰਲੇਖ.e_shentsize
        elf_fp.seek(elf_ਸਿਰਲੇਖ.e_shoff)

        ਲਈ _ ਵਿੱਚ ਸੀਮਾ(elf_ਸਿਰਲੇਖ.e_shnum):
            ਜੇ is_elf64(elf_ਸਿਰਲੇਖ):
                shdr_ਸਾਰਣੀ.append(Elf64_Shdr(elf_fp.ਪਠ(shdr_ਮਾਪ)))
            ਨਹੀਂਤਾਂ:
                shdr_ਸਾਰਣੀ.append(Elf32_Shdr(elf_fp.ਪਠ(shdr_ਮਾਪ)))

    ਵਾਪਸ shdr_ਸਾਰਣੀ


ਪਰਿਭਾਸ਼ਾ ਚੱਕਰੀ_ਜਾਂਚ_32_update_file(elf_fp, ਵਿੱਥ, ਮਾਪ, ਚੱਕਰੀ_ਜਾਂਚ):
    elf_fp.seek(ਵਿੱਥ)
    ਬਚਿਆ = ਮਾਪ
    ਜਦੋਂ ਬਚਿਆ > 0:
        chunk = elf_fp.ਪਠ(min(1024 * 1024, ਬਚਿਆ))
        ਜੇ ਨਹੀਂ chunk:
            ਉਠਾਓ RuntimeError("Unexpected EOF while calculating CRC")
        ਚੱਕਰੀ_ਜਾਂਚ = zlib.ਚੱਕਰੀ_ਜਾਂਚ_32(chunk, ਚੱਕਰੀ_ਜਾਂਚ)
        ਬਚਿਆ -= ਲੰਬਾਈ(chunk)
    ਵਾਪਸ ਚੱਕਰੀ_ਜਾਂਚ


ਪਰਿਭਾਸ਼ਾ ਚੱਕਰੀ_ਜਾਂਚ_32_update_zeros(ਮਾਪ, ਚੱਕਰੀ_ਜਾਂਚ):
    zero_chunk = b'\x00' * min(1024 * 1024, ਮਾਪ)
    ਬਚਿਆ = ਮਾਪ
    ਜਦੋਂ ਬਚਿਆ > 0:
        chunk = zero_chunk[:min(ਲੰਬਾਈ(zero_chunk), ਬਚਿਆ)]
        ਚੱਕਰੀ_ਜਾਂਚ = zlib.ਚੱਕਰੀ_ਜਾਂਚ_32(chunk, ਚੱਕਰੀ_ਜਾਂਚ)
        ਬਚਿਆ -= ਲੰਬਾਈ(chunk)
    ਵਾਪਸ ਚੱਕਰੀ_ਜਾਂਚ


ਪਰਿਭਾਸ਼ਾ ਚੱਕਰੀ_ਜਾਂਚ_32_calculate(elf_fp, phdr_ਸਾਰਣੀ):
    ਚੱਕਰੀ_ਜਾਂਚ = 0

    # SWIV validates loadable segment contents, including BSS as zeros.
    ਲਈ curr_phdr ਵਿੱਚ phdr_ਸਾਰਣੀ:
        ਜੇ curr_phdr.p_type != ਲੱਦੋ_TYPE:
            ਜਾਰੀ

        ਚੱਕਰੀ_ਜਾਂਚ = ਚੱਕਰੀ_ਜਾਂਚ_32_update_file(elf_fp, curr_phdr.p_ਵਿੱਥ, curr_phdr.p_filesz, ਚੱਕਰੀ_ਜਾਂਚ)

        zero_ਮਾਪ = curr_phdr.p_memsz - curr_phdr.p_filesz
        ਜੇ zero_ਮਾਪ > 0:
            ਚੱਕਰੀ_ਜਾਂਚ = ਚੱਕਰੀ_ਜਾਂਚ_32_update_zeros(zero_ਮਾਪ, ਚੱਕਰੀ_ਜਾਂਚ)

    ਚੱਕਰੀ_ਜਾਂਚ &= 0xffffffff
    logger.info("calculated CRC: %#x", ਚੱਕਰੀ_ਜਾਂਚ)

    ਵਾਪਸ ਚੱਕਰੀ_ਜਾਂਚ


ਪਰਿਭਾਸ਼ਾ verify_swiv_seg(elf_ਵਿੱਚ_fp, ਵਿੱਥ):

    elf_ਵਿੱਚ_fp.seek(ਵਿੱਥ)
    swiv_seg = SWIV_Seg()
    unpacked_ਅੰਕੜੇ = (SWIV_Seg.s).unpack(elf_ਵਿੱਚ_fp.ਪਠ(SWIV_SEG_ਮਾਪ_MIN))
    swiv_seg.magic = unpacked_ਅੰਕੜੇ[0]

    ਵਾਪਸ swiv_seg.magic == SWIV_MAGIC


ਪਰਿਭਾਸ਼ਾ same_output_file(ਨਿਵੇਸ਼_elf, output_elf):
    ਕੋਸ਼ਿਸ਼:
        ਵਾਪਸ os.ਰਸਤਾ.exists(output_elf) ਅਤੇ os.ਰਸਤਾ.samefile(ਨਿਵੇਸ਼_elf, output_elf)
    ਅਪਵਾਦ File_Not_ਮਿਲਿਆ_Error:
        ਵਾਪਸ ਝੂਠ


ਪਰਿਭਾਸ਼ਾ make_output_ਨਿਸ਼ਾਨਾ(ਨਿਵੇਸ਼_elf, output_elf):
    ਜੇ ਨਹੀਂ same_output_file(ਨਿਵੇਸ਼_elf, output_elf):
        ਵਾਪਸ output_elf, ਕੁਝ_ਨਹੀਂ

    output_dir = os.ਰਸਤਾ.dirname(os.ਰਸਤਾ.abspath(output_elf))
    output_ਅਧਾਰ = os.ਰਸਤਾ.basename(output_elf)
    fd, ਆਰਜ਼ੀ_output = tempfile.mkstemp(ਅਗੇਤਰ=output_ਅਧਾਰ + ".", suffix=".tmp", dir=output_dir)
    os.close(fd)
    shutil.copymode(ਨਿਵੇਸ਼_elf, ਆਰਜ਼ੀ_output)
    ਵਾਪਸ ਆਰਜ਼ੀ_output, ਆਰਜ਼ੀ_output


ਪਰਿਭਾਸ਼ਾ swiv_generate(ਨਿਵੇਸ਼_elf, output_elf):
    output_ਨਿਸ਼ਾਨਾ, ਆਰਜ਼ੀ_output = make_output_ਨਿਸ਼ਾਨਾ(ਨਿਵੇਸ਼_elf, output_elf)

    ਕੋਸ਼ਿਸ਼:
        ਨਾਲ open(ਨਿਵੇਸ਼_elf, "rb") ਇਵ elf_ਵਿੱਚ_fp, open(output_ਨਿਸ਼ਾਨਾ, "wb+") ਇਵ elf_ਬਾਹਰਲਾ_fp:
            [elf_ਸਿਰਲੇਖ, phdr_ਸਾਰਣੀ] = preprocess_elf_file(ਨਿਵੇਸ਼_elf)

            ਅੰਕ_phdrs = elf_ਸਿਰਲੇਖ.e_phnum
            ਅੰਕ_shdrs = elf_ਸਿਰਲੇਖ.e_shnum
            phdr_ਮਾਪ = elf_ਸਿਰਲੇਖ.e_phentsize

            ਜੇ ਅੰਕ_phdrs == 0 ਜਾਂ ਅੰਕ_phdrs > MAX_PHDR_ਗਿਣਤੀ:
                ਉਠਾਓ RuntimeError("Input ELF has invalid number of program headers.")

            ਜੇ elf_ਸਿਰਲੇਖ.e_shoff == 0 ਜਾਂ ਅੰਕ_shdrs == 0:
                ਉਠਾਓ RuntimeError("Input ELF does not contain a section header table.")

            shdr_ਸਾਰਣੀ_ਨਿਵੇਸ਼ = process_section_ਸਿਰਲੇਖ(ਨਿਵੇਸ਼_elf)
            shdr_ਸਾਰਣੀ_output = process_section_ਸਿਰਲੇਖ(ਨਿਵੇਸ਼_elf)

            ਆਖਰੀ_phdr = phdr_ਸਾਰਣੀ[ਅੰਕ_phdrs - 1]
            ਜੇ ਆਖਰੀ_phdr.p_memsz == SWIV_SEG_ਮਾਪ_MIN ਅਤੇ verify_swiv_seg(elf_ਵਿੱਚ_fp, ਆਖਰੀ_phdr.p_ਵਿੱਥ):
                ਉਠਾਓ RuntimeError("A SWIV segment is detected in the input ELF.")

            ਆਖਰੀ_shdr = shdr_ਸਾਰਣੀ_ਨਿਵੇਸ਼[elf_ਸਿਰਲੇਖ.e_shnum - 1]
            ਜੇ ਆਖਰੀ_shdr.sh_ਮਾਪ == SWIV_SEG_ਮਾਪ_MIN ਅਤੇ verify_swiv_seg(elf_ਵਿੱਚ_fp, ਆਖਰੀ_shdr.sh_ਵਿੱਥ):
                ਉਠਾਓ RuntimeError("A SWIV section is detected in the input ELF.")

            ਜੇ is_elf64(elf_ਸਿਰਲੇਖ):
                swiv_shdr = Elf64_Shdr(b'\0' * Elf64_Shdr.s.ਮਾਪ)
            ਨਹੀਂਤਾਂ:
                swiv_shdr = Elf32_Shdr(b'\0' * Elf32_Shdr.s.ਮਾਪ)

            swiv_shdr.sh_name = 0
            swiv_shdr.sh_type = SWIV_SECTION_TYPE
            swiv_shdr.sh_flags = 0
            swiv_shdr.sh_addr = 0
            swiv_shdr.sh_ਮਾਪ = SWIV_SEG_ਮਾਪ_MIN
            swiv_shdr.sh_ਵਿੱਥ = elf_ਸਿਰਲੇਖ.e_shoff + (elf_ਸਿਰਲੇਖ.e_shnum + 1) * elf_ਸਿਰਲੇਖ.e_shentsize
            shdr_ਸਾਰਣੀ_output.append(swiv_shdr)
            elf_ਸਿਰਲੇਖ.e_shnum += 1

            ਜੇ elf_ਸਿਰਲੇਖ.e_phoff != 0:
                ਲਈ phdr ਵਿੱਚ phdr_ਸਾਰਣੀ:
                    file_copy_ਵਿੱਥ(elf_ਵਿੱਚ_fp, phdr.p_ਵਿੱਥ, elf_ਬਾਹਰਲਾ_fp, phdr.p_ਵਿੱਥ, phdr.p_filesz)

            elf_ਬਾਹਰਲਾ_fp.seek(0)
            elf_ਬਾਹਰਲਾ_fp.ਲਿਖ(elf_ਸਿਰਲੇਖ.get_packed_ਅੰਕੜੇ())

            phdr_ਸ਼ੁਰੂ = elf_ਸਿਰਲੇਖ.e_phoff
            ਲਈ phdr ਵਿੱਚ phdr_ਸਾਰਣੀ:
                elf_ਬਾਹਰਲਾ_fp.seek(phdr_ਸ਼ੁਰੂ)
                elf_ਬਾਹਰਲਾ_fp.ਲਿਖ(phdr.get_packed_ਅੰਕੜੇ())
                phdr_ਸ਼ੁਰੂ += phdr_ਮਾਪ

            ਲਈ shdr ਵਿੱਚ shdr_ਸਾਰਣੀ_ਨਿਵੇਸ਼[:ਅੰਕ_shdrs]:
                ਜੇ shdr.sh_type == SHT_NOBITS ਜਾਂ shdr.sh_ਮਾਪ == 0:
                    ਜਾਰੀ
                file_copy_ਵਿੱਥ(elf_ਵਿੱਚ_fp, shdr.sh_ਵਿੱਥ, elf_ਬਾਹਰਲਾ_fp, shdr.sh_ਵਿੱਥ, shdr.sh_ਮਾਪ)

            shdr_ਸ਼ੁਰੂ = elf_ਸਿਰਲੇਖ.e_shoff
            ਲਈ shdr ਵਿੱਚ shdr_ਸਾਰਣੀ_output:
                elf_ਬਾਹਰਲਾ_fp.seek(shdr_ਸ਼ੁਰੂ)
                elf_ਬਾਹਰਲਾ_fp.ਲਿਖ(shdr.get_packed_ਅੰਕੜੇ())
                shdr_ਸ਼ੁਰੂ += elf_ਸਿਰਲੇਖ.e_shentsize

            elf_ਬਾਹਰਲਾ_fp.flush()

            swiv_seg = SWIV_Seg()
            swiv_seg.ਚੱਕਰੀ_ਜਾਂਚ = ਚੱਕਰੀ_ਜਾਂਚ_32_calculate(elf_ਬਾਹਰਲਾ_fp, phdr_ਸਾਰਣੀ)
            elf_ਬਾਹਰਲਾ_fp.seek(swiv_shdr.sh_ਵਿੱਥ)
            elf_ਬਾਹਰਲਾ_fp.ਲਿਖ(swiv_seg.get_packed_ਅੰਕੜੇ())

        ਜੇ ਆਰਜ਼ੀ_output ਹੈ ਨਹੀਂ ਕੁਝ_ਨਹੀਂ:
            os.replace(ਆਰਜ਼ੀ_output, output_elf)
            ਆਰਜ਼ੀ_output = ਕੁਝ_ਨਹੀਂ

        ਵਾਪਸ 0
    ਅਖੀਰ:
        ਜੇ ਆਰਜ਼ੀ_output ਹੈ ਨਹੀਂ ਕੁਝ_ਨਹੀਂ ਅਤੇ os.ਰਸਤਾ.exists(ਆਰਜ਼ੀ_output):
            os.unlink(ਆਰਜ਼ੀ_output)


ਪਰਿਭਾਸ਼ਾ ਮੁੱਖ(argv):
    parser = argparse.ਦਲੀਲ_Parser(description="Add a Software Image Validation (SWIV) section to an HTP ELF file.")
    parser.ਜੋੜੋ_ਦਲੀਲ("-i", "--input", dest="input_file", metavar="INPUT", required=ਸੱਚ)
    parser.ਜੋੜੋ_ਦਲੀਲ("-o", "--output", dest="output_file", metavar="OUTPUT", required=ਸੱਚ)
    parser.ਜੋੜੋ_ਦਲੀਲ("-v", "--verbose", action="store_true", help="show calculated CRC")
    ਦਲੀਲਾਂ_2 = parser.parse_ਦਲੀਲਾਂ(argv)

    logging.basicConfig(
        level=logging.INFO ਜੇ ਦਲੀਲਾਂ_2.verbose ਨਹੀਂਤਾਂ logging.WARNING,
        format="%(message)s",
    )

    ਕੋਸ਼ਿਸ਼:
        ਵਾਪਸ swiv_generate(ਦਲੀਲਾਂ_2.ਨਿਵੇਸ਼_file, ਦਲੀਲਾਂ_2.output_file)
    ਅਪਵਾਦ RuntimeError ਇਵ exc:
        logger.error("ERROR: %s", exc)
        ਵਾਪਸ 2


ਜੇ __name__ == "__main__":
    sys.exit(ਮੁੱਖ(sys.argv[1:]))
