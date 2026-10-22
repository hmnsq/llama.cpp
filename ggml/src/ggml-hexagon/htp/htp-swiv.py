#!/usr/bin/env python3

ಆಮದು argparse
ಆಮದು logging
ಆಮದು os
ಆಮದು shutil
ಆಮದು struct
ಆಮದು sys
ಆಮದು tempfile
ಆಮದು zlib

ELF_MAGIC = b'\x7fELF'
ELF_CLASS_ಸೂಚ್ಯಂಕ = 4
ELFCLASS32 = 1
ELFCLASS64 = 2
ELF_ಆವೃತ್ತಿ_ಸೂಚ್ಯಂಕ = 6
EV_CURRENT = 1
MAX_PHDR_ಎಣಿಕೆ = 100

ತುಂಬು_TYPE = 0x1

SHT_NOBITS = 8

SWIV_MAGIC = b'SWIV'
SWIV_SEG_ಗಾತ್ರ_MIN = 16
# SHT_LOUSER plus ASCII "SWIV".
SWIV_SECTION_TYPE = 0xD3574956

logger = logging.getLogger("htp-swiv")


ವರ್ಗ Elf_Ehdr_common:
    s = struct.Struct('16sHHI')

    ವ್ಯಾಖ್ಯಾನ __init__(self, ದತ್ತಾಂಶ):
        unpacked_ದತ್ತಾಂಶ = (Elf_Ehdr_common.s).unpack(ದತ್ತಾಂಶ)
        self.e_ident = unpacked_ದತ್ತಾಂಶ[0]
        self.e_type = unpacked_ದತ್ತಾಂಶ[1]
        self.e_machine = unpacked_ದತ್ತಾಂಶ[2]
        self.e_ಆವೃತ್ತಿ = unpacked_ದತ್ತಾಂಶ[3]


ವರ್ಗ Elf32_Ehdr:
    s = struct.Struct('16sHHIIIIIHHHHHH')

    ವ್ಯಾಖ್ಯಾನ __init__(self, ದತ್ತಾಂಶ):
        unpacked_ದತ್ತಾಂಶ = (Elf32_Ehdr.s).unpack(ದತ್ತಾಂಶ)
        self.e_ident = unpacked_ದತ್ತಾಂಶ[0]
        self.e_type = unpacked_ದತ್ತಾಂಶ[1]
        self.e_machine = unpacked_ದತ್ತಾಂಶ[2]
        self.e_ಆವೃತ್ತಿ = unpacked_ದತ್ತಾಂಶ[3]
        self.e_ನಮೂದು = unpacked_ದತ್ತಾಂಶ[4]
        self.e_phoff = unpacked_ದತ್ತಾಂಶ[5]
        self.e_shoff = unpacked_ದತ್ತಾಂಶ[6]
        self.e_flags = unpacked_ದತ್ತಾಂಶ[7]
        self.e_ehsize = unpacked_ದತ್ತಾಂಶ[8]
        self.e_phentsize = unpacked_ದತ್ತಾಂಶ[9]
        self.e_phnum = unpacked_ದತ್ತಾಂಶ[10]
        self.e_shentsize = unpacked_ದತ್ತಾಂಶ[11]
        self.e_shnum = unpacked_ದತ್ತಾಂಶ[12]
        self.e_shstrndx = unpacked_ದತ್ತಾಂಶ[13]

    ವ್ಯಾಖ್ಯಾನ get_packed_ದತ್ತಾಂಶ(self):
        ಮೌಲ್ಯಗಳು = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_ಆವೃತ್ತಿ,
                  self.e_ನಮೂದು,
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

        ಹಿಂತಿರುಗಿ (Elf32_Ehdr.s).pack(*ಮೌಲ್ಯಗಳು)


ವರ್ಗ Elf32_Phdr:
    s = struct.Struct('I' * 8)

    ವ್ಯಾಖ್ಯಾನ __init__(self, ದತ್ತಾಂಶ):
        unpacked_ದತ್ತಾಂಶ = (Elf32_Phdr.s).unpack(ದತ್ತಾಂಶ)
        self.p_type = unpacked_ದತ್ತಾಂಶ[0]
        self.p_ಸ್ಥಾನಾಂತರ = unpacked_ದತ್ತಾಂಶ[1]
        self.p_vaddr = unpacked_ದತ್ತಾಂಶ[2]
        self.p_paddr = unpacked_ದತ್ತಾಂಶ[3]
        self.p_filesz = unpacked_ದತ್ತಾಂಶ[4]
        self.p_memsz = unpacked_ದತ್ತಾಂಶ[5]
        self.p_flags = unpacked_ದತ್ತಾಂಶ[6]
        self.p_align = unpacked_ದತ್ತಾಂಶ[7]

    ವ್ಯಾಖ್ಯಾನ get_packed_ದತ್ತಾಂಶ(self):
        ಮೌಲ್ಯಗಳು = [self.p_type,
                  self.p_ಸ್ಥಾನಾಂತರ,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_flags,
                  self.p_align
                  ]

        ಹಿಂತಿರುಗಿ (Elf32_Phdr.s).pack(*ಮೌಲ್ಯಗಳು)


ವರ್ಗ Elf32_Shdr:
    s = struct.Struct('I' * 10)

    ವ್ಯಾಖ್ಯಾನ __init__(self, ದತ್ತಾಂಶ):
        unpacked_ದತ್ತಾಂಶ = (Elf32_Shdr.s).unpack(ದತ್ತಾಂಶ)
        self.sh_name = unpacked_ದತ್ತಾಂಶ[0]
        self.sh_type = unpacked_ದತ್ತಾಂಶ[1]
        self.sh_flags = unpacked_ದತ್ತಾಂಶ[2]
        self.sh_addr = unpacked_ದತ್ತಾಂಶ[3]
        self.sh_ಸ್ಥಾನಾಂತರ = unpacked_ದತ್ತಾಂಶ[4]
        self.sh_ಗಾತ್ರ = unpacked_ದತ್ತಾಂಶ[5]
        self.sh_link = unpacked_ದತ್ತಾಂಶ[6]
        self.sh_info = unpacked_ದತ್ತಾಂಶ[7]
        self.sh_addralign = unpacked_ದತ್ತಾಂಶ[8]
        self.sh_entsize = unpacked_ದತ್ತಾಂಶ[9]
        self.sh_name_str = ""

    ವ್ಯಾಖ್ಯಾನ get_packed_ದತ್ತಾಂಶ(self):
        ಮೌಲ್ಯಗಳು = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_ಸ್ಥಾನಾಂತರ,
                  self.sh_ಗಾತ್ರ,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        ಹಿಂತಿರುಗಿ (Elf32_Shdr.s).pack(*ಮೌಲ್ಯಗಳು)


ವರ್ಗ Elf64_Ehdr:
    s = struct.Struct('16sHHIQQQIHHHHHH')

    ವ್ಯಾಖ್ಯಾನ __init__(self, ದತ್ತಾಂಶ):
        unpacked_ದತ್ತಾಂಶ = (Elf64_Ehdr.s).unpack(ದತ್ತಾಂಶ)
        self.e_ident = unpacked_ದತ್ತಾಂಶ[0]
        self.e_type = unpacked_ದತ್ತಾಂಶ[1]
        self.e_machine = unpacked_ದತ್ತಾಂಶ[2]
        self.e_ಆವೃತ್ತಿ = unpacked_ದತ್ತಾಂಶ[3]
        self.e_ನಮೂದು = unpacked_ದತ್ತಾಂಶ[4]
        self.e_phoff = unpacked_ದತ್ತಾಂಶ[5]
        self.e_shoff = unpacked_ದತ್ತಾಂಶ[6]
        self.e_flags = unpacked_ದತ್ತಾಂಶ[7]
        self.e_ehsize = unpacked_ದತ್ತಾಂಶ[8]
        self.e_phentsize = unpacked_ದತ್ತಾಂಶ[9]
        self.e_phnum = unpacked_ದತ್ತಾಂಶ[10]
        self.e_shentsize = unpacked_ದತ್ತಾಂಶ[11]
        self.e_shnum = unpacked_ದತ್ತಾಂಶ[12]
        self.e_shstrndx = unpacked_ದತ್ತಾಂಶ[13]

    ವ್ಯಾಖ್ಯಾನ get_packed_ದತ್ತಾಂಶ(self):
        ಮೌಲ್ಯಗಳು = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_ಆವೃತ್ತಿ,
                  self.e_ನಮೂದು,
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

        ಹಿಂತಿರುಗಿ (Elf64_Ehdr.s).pack(*ಮೌಲ್ಯಗಳು)


ವರ್ಗ Elf64_Phdr:
    s = struct.Struct('IIQQQQQQ')

    ವ್ಯಾಖ್ಯಾನ __init__(self, ದತ್ತಾಂಶ):
        unpacked_ದತ್ತಾಂಶ = (Elf64_Phdr.s).unpack(ದತ್ತಾಂಶ)
        self.p_type = unpacked_ದತ್ತಾಂಶ[0]
        self.p_flags = unpacked_ದತ್ತಾಂಶ[1]
        self.p_ಸ್ಥಾನಾಂತರ = unpacked_ದತ್ತಾಂಶ[2]
        self.p_vaddr = unpacked_ದತ್ತಾಂಶ[3]
        self.p_paddr = unpacked_ದತ್ತಾಂಶ[4]
        self.p_filesz = unpacked_ದತ್ತಾಂಶ[5]
        self.p_memsz = unpacked_ದತ್ತಾಂಶ[6]
        self.p_align = unpacked_ದತ್ತಾಂಶ[7]

    ವ್ಯಾಖ್ಯಾನ get_packed_ದತ್ತಾಂಶ(self):
        ಮೌಲ್ಯಗಳು = [self.p_type,
                  self.p_flags,
                  self.p_ಸ್ಥಾನಾಂತರ,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_align
                  ]

        ಹಿಂತಿರುಗಿ (Elf64_Phdr.s).pack(*ಮೌಲ್ಯಗಳು)


ವರ್ಗ Elf64_Shdr:
    s = struct.Struct('IIQQQQIIQQ')

    ವ್ಯಾಖ್ಯಾನ __init__(self, ದತ್ತಾಂಶ):
        unpacked_ದತ್ತಾಂಶ = (Elf64_Shdr.s).unpack(ದತ್ತಾಂಶ)
        self.sh_name = unpacked_ದತ್ತಾಂಶ[0]
        self.sh_type = unpacked_ದತ್ತಾಂಶ[1]
        self.sh_flags = unpacked_ದತ್ತಾಂಶ[2]
        self.sh_addr = unpacked_ದತ್ತಾಂಶ[3]
        self.sh_ಸ್ಥಾನಾಂತರ = unpacked_ದತ್ತಾಂಶ[4]
        self.sh_ಗಾತ್ರ = unpacked_ದತ್ತಾಂಶ[5]
        self.sh_link = unpacked_ದತ್ತಾಂಶ[6]
        self.sh_info = unpacked_ದತ್ತಾಂಶ[7]
        self.sh_addralign = unpacked_ದತ್ತಾಂಶ[8]
        self.sh_entsize = unpacked_ದತ್ತಾಂಶ[9]
        self.sh_name_str = ""

    ವ್ಯಾಖ್ಯಾನ get_packed_ದತ್ತಾಂಶ(self):
        ಮೌಲ್ಯಗಳು = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_ಸ್ಥಾನಾಂತರ,
                  self.sh_ಗಾತ್ರ,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        ಹಿಂತಿರುಗಿ (Elf64_Shdr.s).pack(*ಮೌಲ್ಯಗಳು)


ವರ್ಗ SWIV_Seg:
    s = struct.Struct('4sIQ')

    ವ್ಯಾಖ್ಯಾನ __init__(self):
        self.magic = SWIV_MAGIC
        self.ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ = 0
        self.padding = 0

    ವ್ಯಾಖ್ಯಾನ get_packed_ದತ್ತಾಂಶ(self):
        ಮೌಲ್ಯಗಳು = [self.magic,
                  self.ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ,
                  self.padding
                  ]

        ಹಿಂತಿರುಗಿ (SWIV_Seg.s).pack(*ಮೌಲ್ಯಗಳು)


ವ್ಯಾಖ್ಯಾನ is_elf64(elf_ಶೀರ್ಷಿಕೆ):
    ಹಿಂತಿರುಗಿ elf_ಶೀರ್ಷಿಕೆ.e_ident[ELF_CLASS_ಸೂಚ್ಯಂಕ] == ELFCLASS64


ವ್ಯಾಖ್ಯಾನ verify_elf_ಶೀರ್ಷಿಕೆ(elf_ಶೀರ್ಷಿಕೆ):
    ident = elf_ಶೀರ್ಷಿಕೆ.e_ident
    ಹಿಂತಿರುಗಿ (ident.startswith(ELF_MAGIC)
            ಮತ್ತು ident[ELF_CLASS_ಸೂಚ್ಯಂಕ] ಒಳಗೆ (ELFCLASS32, ELFCLASS64)
            ಮತ್ತು ident[ELF_ಆವೃತ್ತಿ_ಸೂಚ್ಯಂಕ] == EV_CURRENT)


ವ್ಯಾಖ್ಯಾನ file_copy_ಸ್ಥಾನಾಂತರ(ಒಳ_fp, ಒಳ_off, ಹೊರಹೋಗು_fp, ಹೊರಹೋಗು_off, ಅಂಕೆ_ಬೈಟುಗಳು):
    ಒಳ_fp.seek(ಒಳ_off)
    read_ಒಳ = ಒಳ_fp.ಪಠ(ಅಂಕೆ_ಬೈಟುಗಳು)
    ಹೊರಹೋಗು_fp.seek(ಹೊರಹೋಗು_off)
    ಹೊರಹೋಗು_fp.ಲಿಖ(read_ಒಳ)

    ಹಿಂತಿರುಗಿ ಅಂಕೆ_ಬೈಟುಗಳು


ವ್ಯಾಖ್ಯಾನ preprocess_elf_file(elf_file_name):
    ಜೊತೆ open(elf_file_name, 'rb') ಇವ elf_fp:
        elf_ಶೀರ್ಷಿಕೆ = Elf_Ehdr_common(elf_fp.ಪಠ(Elf_Ehdr_common.s.ಗಾತ್ರ_2))

        ಆದರೆ verify_elf_ಶೀರ್ಷಿಕೆ(elf_ಶೀರ್ಷಿಕೆ) ಆಗಿರುವ ಸುಳ್ಳು:
            ಎತ್ತು RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        ಆದರೆ is_elf64(elf_ಶೀರ್ಷಿಕೆ):
            elf_ಶೀರ್ಷಿಕೆ = Elf64_Ehdr(elf_fp.ಪಠ(Elf64_Ehdr.s.ಗಾತ್ರ_2))
        ಇಲ್ಲದಿದ್ದರೆ:
            elf_ಶೀರ್ಷಿಕೆ = Elf32_Ehdr(elf_fp.ಪಠ(Elf32_Ehdr.s.ಗಾತ್ರ_2))

        phdr_ಕೋಷ್ಟಕ = []

        phdr_ಗಾತ್ರ = elf_ಶೀರ್ಷಿಕೆ.e_phentsize

        file_ಸ್ಥಾನಾಂತರ = elf_ಶೀರ್ಷಿಕೆ.e_phoff
        elf_fp.seek(file_ಸ್ಥಾನಾಂತರ)

        ಗಾಗಿ _ ಒಳಗೆ ಶ್ರೇಣಿ(elf_ಶೀರ್ಷಿಕೆ.e_phnum):
            ಆದರೆ is_elf64(elf_ಶೀರ್ಷಿಕೆ):
                phdr_ಕೋಷ್ಟಕ.append(Elf64_Phdr(elf_fp.ಪಠ(phdr_ಗಾತ್ರ)))
            ಇಲ್ಲದಿದ್ದರೆ:
                phdr_ಕೋಷ್ಟಕ.append(Elf32_Phdr(elf_fp.ಪಠ(phdr_ಗಾತ್ರ)))

    ಹಿಂತಿರುಗಿ [elf_ಶೀರ್ಷಿಕೆ, phdr_ಕೋಷ್ಟಕ]


ವ್ಯಾಖ್ಯಾನ process_section_ಶೀರ್ಷಿಕೆ(elf_file_name):
    ಜೊತೆ open(elf_file_name, 'rb') ಇವ elf_fp:
        elf_ಶೀರ್ಷಿಕೆ = Elf_Ehdr_common(elf_fp.ಪಠ(Elf_Ehdr_common.s.ಗಾತ್ರ_2))

        ಆದರೆ verify_elf_ಶೀರ್ಷಿಕೆ(elf_ಶೀರ್ಷಿಕೆ) ಆಗಿರುವ ಸುಳ್ಳು:
            ಎತ್ತು RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        ಆದರೆ is_elf64(elf_ಶೀರ್ಷಿಕೆ):
            elf_ಶೀರ್ಷಿಕೆ = Elf64_Ehdr(elf_fp.ಪಠ(Elf64_Ehdr.s.ಗಾತ್ರ_2))
        ಇಲ್ಲದಿದ್ದರೆ:
            elf_ಶೀರ್ಷಿಕೆ = Elf32_Ehdr(elf_fp.ಪಠ(Elf32_Ehdr.s.ಗಾತ್ರ_2))

        shdr_ಕೋಷ್ಟಕ = []

        shdr_ಗಾತ್ರ = elf_ಶೀರ್ಷಿಕೆ.e_shentsize
        elf_fp.seek(elf_ಶೀರ್ಷಿಕೆ.e_shoff)

        ಗಾಗಿ _ ಒಳಗೆ ಶ್ರೇಣಿ(elf_ಶೀರ್ಷಿಕೆ.e_shnum):
            ಆದರೆ is_elf64(elf_ಶೀರ್ಷಿಕೆ):
                shdr_ಕೋಷ್ಟಕ.append(Elf64_Shdr(elf_fp.ಪಠ(shdr_ಗಾತ್ರ)))
            ಇಲ್ಲದಿದ್ದರೆ:
                shdr_ಕೋಷ್ಟಕ.append(Elf32_Shdr(elf_fp.ಪಠ(shdr_ಗಾತ್ರ)))

    ಹಿಂತಿರುಗಿ shdr_ಕೋಷ್ಟಕ


ವ್ಯಾಖ್ಯಾನ ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ_32_update_file(elf_fp, ಸ್ಥಾನಾಂತರ, ಗಾತ್ರ_2, ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ):
    elf_fp.seek(ಸ್ಥಾನಾಂತರ)
    ಉಳಿದ = ಗಾತ್ರ_2
    ಆಗ ಉಳಿದ > 0:
        chunk = elf_fp.ಪಠ(min(1024 * 1024, ಉಳಿದ))
        ಆದರೆ ಅಲ್ಲ chunk:
            ಎತ್ತು RuntimeError("Unexpected EOF while calculating CRC")
        ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ = zlib.ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ_32(chunk, ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ)
        ಉಳಿದ -= ಉದ್ದ(chunk)
    ಹಿಂತಿರುಗಿ ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ


ವ್ಯಾಖ್ಯಾನ ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ_32_update_zeros(ಗಾತ್ರ_2, ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ):
    zero_chunk = b'\x00' * min(1024 * 1024, ಗಾತ್ರ_2)
    ಉಳಿದ = ಗಾತ್ರ_2
    ಆಗ ಉಳಿದ > 0:
        chunk = zero_chunk[:min(ಉದ್ದ(zero_chunk), ಉಳಿದ)]
        ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ = zlib.ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ_32(chunk, ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ)
        ಉಳಿದ -= ಉದ್ದ(chunk)
    ಹಿಂತಿರುಗಿ ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ


ವ್ಯಾಖ್ಯಾನ ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ_32_calculate(elf_fp, phdr_ಕೋಷ್ಟಕ):
    ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ = 0

    # SWIV validates loadable segment contents, including BSS as zeros.
    ಗಾಗಿ curr_phdr ಒಳಗೆ phdr_ಕೋಷ್ಟಕ:
        ಆದರೆ curr_phdr.p_type != ತುಂಬು_TYPE:
            ಮುಂದುವರಿ

        ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ = ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ_32_update_file(elf_fp, curr_phdr.p_ಸ್ಥಾನಾಂತರ, curr_phdr.p_filesz, ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ)

        zero_ಗಾತ್ರ = curr_phdr.p_memsz - curr_phdr.p_filesz
        ಆದರೆ zero_ಗಾತ್ರ > 0:
            ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ = ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ_32_update_zeros(zero_ಗಾತ್ರ, ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ)

    ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ &= 0xffffffff
    logger.info("calculated CRC: %#x", ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ)

    ಹಿಂತಿರುಗಿ ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ


ವ್ಯಾಖ್ಯಾನ verify_swiv_seg(elf_ಒಳ_fp, ಸ್ಥಾನಾಂತರ):

    elf_ಒಳ_fp.seek(ಸ್ಥಾನಾಂತರ)
    swiv_seg = SWIV_Seg()
    unpacked_ದತ್ತಾಂಶ = (SWIV_Seg.s).unpack(elf_ಒಳ_fp.ಪಠ(SWIV_SEG_ಗಾತ್ರ_MIN))
    swiv_seg.magic = unpacked_ದತ್ತಾಂಶ[0]

    ಹಿಂತಿರುಗಿ swiv_seg.magic == SWIV_MAGIC


ವ್ಯಾಖ್ಯಾನ same_output_file(ಒಳಹರಿವು_elf, output_elf):
    ಪ್ರಯತ್ನಿಸು:
        ಹಿಂತಿರುಗಿ os.ಮಾರ್ಗ.exists(output_elf) ಮತ್ತು os.ಮಾರ್ಗ.samefile(ಒಳಹರಿವು_elf, output_elf)
    ಅಪವಾದ File_Not_ಸಿಕ್ಕಿತು_Error:
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು


ವ್ಯಾಖ್ಯಾನ make_output_ಗುರಿ(ಒಳಹರಿವು_elf, output_elf):
    ಆದರೆ ಅಲ್ಲ same_output_file(ಒಳಹರಿವು_elf, output_elf):
        ಹಿಂತಿರುಗಿ output_elf, ಏನೂಇಲ್ಲ

    output_dir = os.ಮಾರ್ಗ.dirname(os.ಮಾರ್ಗ.abspath(output_elf))
    output_ತಳ = os.ಮಾರ್ಗ.basename(output_elf)
    fd, ತಾತ್ಕಾಲಿಕ_output = tempfile.mkstemp(ಪೂರ್ವಪ್ರತ್ಯಯ=output_ತಳ + ".", suffix=".tmp", dir=output_dir)
    os.close(fd)
    shutil.copymode(ಒಳಹರಿವು_elf, ತಾತ್ಕಾಲಿಕ_output)
    ಹಿಂತಿರುಗಿ ತಾತ್ಕಾಲಿಕ_output, ತಾತ್ಕಾಲಿಕ_output


ವ್ಯಾಖ್ಯಾನ swiv_generate(ಒಳಹರಿವು_elf, output_elf):
    output_ಗುರಿ, ತಾತ್ಕಾಲಿಕ_output = make_output_ಗುರಿ(ಒಳಹರಿವು_elf, output_elf)

    ಪ್ರಯತ್ನಿಸು:
        ಜೊತೆ open(ಒಳಹರಿವು_elf, "rb") ಇವ elf_ಒಳ_fp, open(output_ಗುರಿ, "wb+") ಇವ elf_ಹೊರಹೋಗು_fp:
            [elf_ಶೀರ್ಷಿಕೆ, phdr_ಕೋಷ್ಟಕ] = preprocess_elf_file(ಒಳಹರಿವು_elf)

            ಅಂಕೆ_phdrs = elf_ಶೀರ್ಷಿಕೆ.e_phnum
            ಅಂಕೆ_shdrs = elf_ಶೀರ್ಷಿಕೆ.e_shnum
            phdr_ಗಾತ್ರ = elf_ಶೀರ್ಷಿಕೆ.e_phentsize

            ಆದರೆ ಅಂಕೆ_phdrs == 0 ಅಥವಾ ಅಂಕೆ_phdrs > MAX_PHDR_ಎಣಿಕೆ:
                ಎತ್ತು RuntimeError("Input ELF has invalid number of program headers.")

            ಆದರೆ elf_ಶೀರ್ಷಿಕೆ.e_shoff == 0 ಅಥವಾ ಅಂಕೆ_shdrs == 0:
                ಎತ್ತು RuntimeError("Input ELF does not contain a section header table.")

            shdr_ಕೋಷ್ಟಕ_ಒಳಹರಿವು = process_section_ಶೀರ್ಷಿಕೆ(ಒಳಹರಿವು_elf)
            shdr_ಕೋಷ್ಟಕ_output = process_section_ಶೀರ್ಷಿಕೆ(ಒಳಹರಿವು_elf)

            ಕೊನೆಯ_phdr = phdr_ಕೋಷ್ಟಕ[ಅಂಕೆ_phdrs - 1]
            ಆದರೆ ಕೊನೆಯ_phdr.p_memsz == SWIV_SEG_ಗಾತ್ರ_MIN ಮತ್ತು verify_swiv_seg(elf_ಒಳ_fp, ಕೊನೆಯ_phdr.p_ಸ್ಥಾನಾಂತರ):
                ಎತ್ತು RuntimeError("A SWIV segment is detected in the input ELF.")

            ಕೊನೆಯ_shdr = shdr_ಕೋಷ್ಟಕ_ಒಳಹರಿವು[elf_ಶೀರ್ಷಿಕೆ.e_shnum - 1]
            ಆದರೆ ಕೊನೆಯ_shdr.sh_ಗಾತ್ರ == SWIV_SEG_ಗಾತ್ರ_MIN ಮತ್ತು verify_swiv_seg(elf_ಒಳ_fp, ಕೊನೆಯ_shdr.sh_ಸ್ಥಾನಾಂತರ):
                ಎತ್ತು RuntimeError("A SWIV section is detected in the input ELF.")

            ಆದರೆ is_elf64(elf_ಶೀರ್ಷಿಕೆ):
                swiv_shdr = Elf64_Shdr(b'\0' * Elf64_Shdr.s.ಗಾತ್ರ_2)
            ಇಲ್ಲದಿದ್ದರೆ:
                swiv_shdr = Elf32_Shdr(b'\0' * Elf32_Shdr.s.ಗಾತ್ರ_2)

            swiv_shdr.sh_name = 0
            swiv_shdr.sh_type = SWIV_SECTION_TYPE
            swiv_shdr.sh_flags = 0
            swiv_shdr.sh_addr = 0
            swiv_shdr.sh_ಗಾತ್ರ = SWIV_SEG_ಗಾತ್ರ_MIN
            swiv_shdr.sh_ಸ್ಥಾನಾಂತರ = elf_ಶೀರ್ಷಿಕೆ.e_shoff + (elf_ಶೀರ್ಷಿಕೆ.e_shnum + 1) * elf_ಶೀರ್ಷಿಕೆ.e_shentsize
            shdr_ಕೋಷ್ಟಕ_output.append(swiv_shdr)
            elf_ಶೀರ್ಷಿಕೆ.e_shnum += 1

            ಆದರೆ elf_ಶೀರ್ಷಿಕೆ.e_phoff != 0:
                ಗಾಗಿ phdr ಒಳಗೆ phdr_ಕೋಷ್ಟಕ:
                    file_copy_ಸ್ಥಾನಾಂತರ(elf_ಒಳ_fp, phdr.p_ಸ್ಥಾನಾಂತರ, elf_ಹೊರಹೋಗು_fp, phdr.p_ಸ್ಥಾನಾಂತರ, phdr.p_filesz)

            elf_ಹೊರಹೋಗು_fp.seek(0)
            elf_ಹೊರಹೋಗು_fp.ಲಿಖ(elf_ಶೀರ್ಷಿಕೆ.get_packed_ದತ್ತಾಂಶ())

            phdr_ಆರಂಭ = elf_ಶೀರ್ಷಿಕೆ.e_phoff
            ಗಾಗಿ phdr ಒಳಗೆ phdr_ಕೋಷ್ಟಕ:
                elf_ಹೊರಹೋಗು_fp.seek(phdr_ಆರಂಭ)
                elf_ಹೊರಹೋಗು_fp.ಲಿಖ(phdr.get_packed_ದತ್ತಾಂಶ())
                phdr_ಆರಂಭ += phdr_ಗಾತ್ರ

            ಗಾಗಿ shdr ಒಳಗೆ shdr_ಕೋಷ್ಟಕ_ಒಳಹರಿವು[:ಅಂಕೆ_shdrs]:
                ಆದರೆ shdr.sh_type == SHT_NOBITS ಅಥವಾ shdr.sh_ಗಾತ್ರ == 0:
                    ಮುಂದುವರಿ
                file_copy_ಸ್ಥಾನಾಂತರ(elf_ಒಳ_fp, shdr.sh_ಸ್ಥಾನಾಂತರ, elf_ಹೊರಹೋಗು_fp, shdr.sh_ಸ್ಥಾನಾಂತರ, shdr.sh_ಗಾತ್ರ)

            shdr_ಆರಂಭ = elf_ಶೀರ್ಷಿಕೆ.e_shoff
            ಗಾಗಿ shdr ಒಳಗೆ shdr_ಕೋಷ್ಟಕ_output:
                elf_ಹೊರಹೋಗು_fp.seek(shdr_ಆರಂಭ)
                elf_ಹೊರಹೋಗು_fp.ಲಿಖ(shdr.get_packed_ದತ್ತಾಂಶ())
                shdr_ಆರಂಭ += elf_ಶೀರ್ಷಿಕೆ.e_shentsize

            elf_ಹೊರಹೋಗು_fp.flush()

            swiv_seg = SWIV_Seg()
            swiv_seg.ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ = ಚಕ್ರೀಯ_ಪರೀಕ್ಷೆ_32_calculate(elf_ಹೊರಹೋಗು_fp, phdr_ಕೋಷ್ಟಕ)
            elf_ಹೊರಹೋಗು_fp.seek(swiv_shdr.sh_ಸ್ಥಾನಾಂತರ)
            elf_ಹೊರಹೋಗು_fp.ಲಿಖ(swiv_seg.get_packed_ದತ್ತಾಂಶ())

        ಆದರೆ ತಾತ್ಕಾಲಿಕ_output ಆಗಿರುವ ಅಲ್ಲ ಏನೂಇಲ್ಲ:
            os.replace(ತಾತ್ಕಾಲಿಕ_output, output_elf)
            ತಾತ್ಕಾಲಿಕ_output = ಏನೂಇಲ್ಲ

        ಹಿಂತಿರುಗಿ 0
    ಕೊನೆಗೆ:
        ಆದರೆ ತಾತ್ಕಾಲಿಕ_output ಆಗಿರುವ ಅಲ್ಲ ಏನೂಇಲ್ಲ ಮತ್ತು os.ಮಾರ್ಗ.exists(ತಾತ್ಕಾಲಿಕ_output):
            os.unlink(ತಾತ್ಕಾಲಿಕ_output)


ವ್ಯಾಖ್ಯಾನ ಮುಖ್ಯ(argv):
    parser = argparse.ವಾದ_Parser(description="Add a Software Image Validation (SWIV) section to an HTP ELF file.")
    parser.ಕೂಡಿಸು_ವಾದ("-i", "--input", dest="input_file", metavar="INPUT", required=ನಿಜ)
    parser.ಕೂಡಿಸು_ವಾದ("-o", "--output", dest="output_file", metavar="OUTPUT", required=ನಿಜ)
    parser.ಕೂಡಿಸು_ವಾದ("-v", "--verbose", action="store_true", help="show calculated CRC")
    ವಾದಗಳು_2 = parser.parse_ವಾದಗಳು(argv)

    logging.basicConfig(
        level=logging.INFO ಆದರೆ ವಾದಗಳು_2.verbose ಇಲ್ಲದಿದ್ದರೆ logging.WARNING,
        format="%(message)s",
    )

    ಪ್ರಯತ್ನಿಸು:
        ಹಿಂತಿರುಗಿ swiv_generate(ವಾದಗಳು_2.ಒಳಹರಿವು_file, ವಾದಗಳು_2.output_file)
    ಅಪವಾದ RuntimeError ಇವ exc:
        logger.error("ERROR: %s", exc)
        ಹಿಂತಿರುಗಿ 2


ಆದರೆ __name__ == "__main__":
    sys.exit(ಮುಖ್ಯ(sys.argv[1:]))
