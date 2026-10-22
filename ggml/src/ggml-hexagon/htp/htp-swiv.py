#!/usr/bin/env python3

आयात argparse
आयात logging
आयात os
आयात shutil
आयात struct
आयात sys
आयात tempfile
आयात zlib

ELF_MAGIC = b'\x7fELF'
ELF_CLASS_सूचकांक = 4
ELFCLASS32 = 1
ELFCLASS64 = 2
ELF_VERSION_सूचकांक = 6
EV_CURRENT = 1
MAX_PHDR_गिनती = 100

LOAD_TYPE = 0x1

SHT_NOBITS = 8

SWIV_MAGIC = b'SWIV'
SWIV_SEG_आकार_MIN = 16
# SHT_LOUSER plus ASCII "SWIV".
SWIV_SECTION_TYPE = 0xD3574956

logger = logging.getLogger("htp-swiv")


श्रेणी Elf_Ehdr_common:
    s = struct.Struct('16sHHI')

    def __init__(self, तथ्य):
        unpacked_तथ्य = (Elf_Ehdr_common.s).unpack(तथ्य)
        self.e_ident = unpacked_तथ्य[0]
        self.e_type = unpacked_तथ्य[1]
        self.e_machine = unpacked_तथ्य[2]
        self.e_version = unpacked_तथ्य[3]


श्रेणी Elf32_Ehdr:
    s = struct.Struct('16sHHIIIIIHHHHHH')

    def __init__(self, तथ्य):
        unpacked_तथ्य = (Elf32_Ehdr.s).unpack(तथ्य)
        self.e_ident = unpacked_तथ्य[0]
        self.e_type = unpacked_तथ्य[1]
        self.e_machine = unpacked_तथ्य[2]
        self.e_version = unpacked_तथ्य[3]
        self.e_entry = unpacked_तथ्य[4]
        self.e_phoff = unpacked_तथ्य[5]
        self.e_shoff = unpacked_तथ्य[6]
        self.e_flags = unpacked_तथ्य[7]
        self.e_ehsize = unpacked_तथ्य[8]
        self.e_phentsize = unpacked_तथ्य[9]
        self.e_phnum = unpacked_तथ्य[10]
        self.e_shentsize = unpacked_तथ्य[11]
        self.e_shnum = unpacked_तथ्य[12]
        self.e_shstrndx = unpacked_तथ्य[13]

    def get_packed_तथ्य(self):
        मूल्यावली = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_version,
                  self.e_entry,
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

        वापस (Elf32_Ehdr.s).pack(*मूल्यावली)


श्रेणी Elf32_Phdr:
    s = struct.Struct('I' * 8)

    def __init__(self, तथ्य):
        unpacked_तथ्य = (Elf32_Phdr.s).unpack(तथ्य)
        self.p_type = unpacked_तथ्य[0]
        self.p_offset = unpacked_तथ्य[1]
        self.p_vaddr = unpacked_तथ्य[2]
        self.p_paddr = unpacked_तथ्य[3]
        self.p_filesz = unpacked_तथ्य[4]
        self.p_memsz = unpacked_तथ्य[5]
        self.p_flags = unpacked_तथ्य[6]
        self.p_align = unpacked_तथ्य[7]

    def get_packed_तथ्य(self):
        मूल्यावली = [self.p_type,
                  self.p_offset,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_flags,
                  self.p_align
                  ]

        वापस (Elf32_Phdr.s).pack(*मूल्यावली)


श्रेणी Elf32_Shdr:
    s = struct.Struct('I' * 10)

    def __init__(self, तथ्य):
        unpacked_तथ्य = (Elf32_Shdr.s).unpack(तथ्य)
        self.sh_name = unpacked_तथ्य[0]
        self.sh_type = unpacked_तथ्य[1]
        self.sh_flags = unpacked_तथ्य[2]
        self.sh_addr = unpacked_तथ्य[3]
        self.sh_offset = unpacked_तथ्य[4]
        self.sh_आकार = unpacked_तथ्य[5]
        self.sh_link = unpacked_तथ्य[6]
        self.sh_info = unpacked_तथ्य[7]
        self.sh_addralign = unpacked_तथ्य[8]
        self.sh_entsize = unpacked_तथ्य[9]
        self.sh_name_str = ""

    def get_packed_तथ्य(self):
        मूल्यावली = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_offset,
                  self.sh_आकार,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        वापस (Elf32_Shdr.s).pack(*मूल्यावली)


श्रेणी Elf64_Ehdr:
    s = struct.Struct('16sHHIQQQIHHHHHH')

    def __init__(self, तथ्य):
        unpacked_तथ्य = (Elf64_Ehdr.s).unpack(तथ्य)
        self.e_ident = unpacked_तथ्य[0]
        self.e_type = unpacked_तथ्य[1]
        self.e_machine = unpacked_तथ्य[2]
        self.e_version = unpacked_तथ्य[3]
        self.e_entry = unpacked_तथ्य[4]
        self.e_phoff = unpacked_तथ्य[5]
        self.e_shoff = unpacked_तथ्य[6]
        self.e_flags = unpacked_तथ्य[7]
        self.e_ehsize = unpacked_तथ्य[8]
        self.e_phentsize = unpacked_तथ्य[9]
        self.e_phnum = unpacked_तथ्य[10]
        self.e_shentsize = unpacked_तथ्य[11]
        self.e_shnum = unpacked_तथ्य[12]
        self.e_shstrndx = unpacked_तथ्य[13]

    def get_packed_तथ्य(self):
        मूल्यावली = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_version,
                  self.e_entry,
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

        वापस (Elf64_Ehdr.s).pack(*मूल्यावली)


श्रेणी Elf64_Phdr:
    s = struct.Struct('IIQQQQQQ')

    def __init__(self, तथ्य):
        unpacked_तथ्य = (Elf64_Phdr.s).unpack(तथ्य)
        self.p_type = unpacked_तथ्य[0]
        self.p_flags = unpacked_तथ्य[1]
        self.p_offset = unpacked_तथ्य[2]
        self.p_vaddr = unpacked_तथ्य[3]
        self.p_paddr = unpacked_तथ्य[4]
        self.p_filesz = unpacked_तथ्य[5]
        self.p_memsz = unpacked_तथ्य[6]
        self.p_align = unpacked_तथ्य[7]

    def get_packed_तथ्य(self):
        मूल्यावली = [self.p_type,
                  self.p_flags,
                  self.p_offset,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_align
                  ]

        वापस (Elf64_Phdr.s).pack(*मूल्यावली)


श्रेणी Elf64_Shdr:
    s = struct.Struct('IIQQQQIIQQ')

    def __init__(self, तथ्य):
        unpacked_तथ्य = (Elf64_Shdr.s).unpack(तथ्य)
        self.sh_name = unpacked_तथ्य[0]
        self.sh_type = unpacked_तथ्य[1]
        self.sh_flags = unpacked_तथ्य[2]
        self.sh_addr = unpacked_तथ्य[3]
        self.sh_offset = unpacked_तथ्य[4]
        self.sh_आकार = unpacked_तथ्य[5]
        self.sh_link = unpacked_तथ्य[6]
        self.sh_info = unpacked_तथ्य[7]
        self.sh_addralign = unpacked_तथ्य[8]
        self.sh_entsize = unpacked_तथ्य[9]
        self.sh_name_str = ""

    def get_packed_तथ्य(self):
        मूल्यावली = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_offset,
                  self.sh_आकार,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        वापस (Elf64_Shdr.s).pack(*मूल्यावली)


श्रेणी SWIV_Seg:
    s = struct.Struct('4sIQ')

    def __init__(self):
        self.magic = SWIV_MAGIC
        self.crc = 0
        self.padding = 0

    def get_packed_तथ्य(self):
        मूल्यावली = [self.magic,
                  self.crc,
                  self.padding
                  ]

        वापस (SWIV_Seg.s).pack(*मूल्यावली)


def is_elf64(elf_header):
    वापस elf_header.e_ident[ELF_CLASS_सूचकांक] == ELFCLASS64


def verify_elf_header(elf_header):
    ident = elf_header.e_ident
    वापस (ident.startswith(ELF_MAGIC)
            and ident[ELF_CLASS_सूचकांक] in (ELFCLASS32, ELFCLASS64)
            and ident[ELF_VERSION_सूचकांक] == EV_CURRENT)


def file_copy_offset(in_fp, in_off, out_fp, out_off, अंक_bytes):
    in_fp.seek(in_off)
    read_in = in_fp.म_पढ़ो(अंक_bytes)
    out_fp.seek(out_off)
    out_fp.म_लिखो(read_in)

    वापस अंक_bytes


def preprocess_elf_file(elf_file_name):
    with open(elf_file_name, 'rb') as elf_fp:
        elf_header = Elf_Ehdr_common(elf_fp.म_पढ़ो(Elf_Ehdr_common.s.आकार))

        अगर verify_elf_header(elf_header) is False:
            raise RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        अगर is_elf64(elf_header):
            elf_header = Elf64_Ehdr(elf_fp.म_पढ़ो(Elf64_Ehdr.s.आकार))
        अन्यथा:
            elf_header = Elf32_Ehdr(elf_fp.म_पढ़ो(Elf32_Ehdr.s.आकार))

        phdr_table = []

        phdr_आकार = elf_header.e_phentsize

        file_offset = elf_header.e_phoff
        elf_fp.seek(file_offset)

        क्रम _ in range(elf_header.e_phnum):
            अगर is_elf64(elf_header):
                phdr_table.append(Elf64_Phdr(elf_fp.म_पढ़ो(phdr_आकार)))
            अन्यथा:
                phdr_table.append(Elf32_Phdr(elf_fp.म_पढ़ो(phdr_आकार)))

    वापस [elf_header, phdr_table]


def process_section_header(elf_file_name):
    with open(elf_file_name, 'rb') as elf_fp:
        elf_header = Elf_Ehdr_common(elf_fp.म_पढ़ो(Elf_Ehdr_common.s.आकार))

        अगर verify_elf_header(elf_header) is False:
            raise RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        अगर is_elf64(elf_header):
            elf_header = Elf64_Ehdr(elf_fp.म_पढ़ो(Elf64_Ehdr.s.आकार))
        अन्यथा:
            elf_header = Elf32_Ehdr(elf_fp.म_पढ़ो(Elf32_Ehdr.s.आकार))

        shdr_table = []

        shdr_आकार = elf_header.e_shentsize
        elf_fp.seek(elf_header.e_shoff)

        क्रम _ in range(elf_header.e_shnum):
            अगर is_elf64(elf_header):
                shdr_table.append(Elf64_Shdr(elf_fp.म_पढ़ो(shdr_आकार)))
            अन्यथा:
                shdr_table.append(Elf32_Shdr(elf_fp.म_पढ़ो(shdr_आकार)))

    वापस shdr_table


def crc32_update_file(elf_fp, offset, आकार, crc):
    elf_fp.seek(offset)
    remaining = आकार
    जबतक remaining > 0:
        chunk = elf_fp.म_पढ़ो(min(1024 * 1024, remaining))
        अगर not chunk:
            raise RuntimeError("Unexpected EOF while calculating CRC")
        crc = zlib.crc32(chunk, crc)
        remaining -= len(chunk)
    वापस crc


def crc32_update_zeros(आकार, crc):
    zero_chunk = b'\x00' * min(1024 * 1024, आकार)
    remaining = आकार
    जबतक remaining > 0:
        chunk = zero_chunk[:min(len(zero_chunk), remaining)]
        crc = zlib.crc32(chunk, crc)
        remaining -= len(chunk)
    वापस crc


def crc32_calculate(elf_fp, phdr_table):
    crc = 0

    # SWIV validates loadable segment contents, including BSS as zeros.
    क्रम curr_phdr in phdr_table:
        अगर curr_phdr.p_type != LOAD_TYPE:
            जारी

        crc = crc32_update_file(elf_fp, curr_phdr.p_offset, curr_phdr.p_filesz, crc)

        zero_आकार = curr_phdr.p_memsz - curr_phdr.p_filesz
        अगर zero_आकार > 0:
            crc = crc32_update_zeros(zero_आकार, crc)

    crc &= 0xffffffff
    logger.info("calculated CRC: %#x", crc)

    वापस crc


def verify_swiv_seg(elf_in_fp, offset):

    elf_in_fp.seek(offset)
    swiv_seg = SWIV_Seg()
    unpacked_तथ्य = (SWIV_Seg.s).unpack(elf_in_fp.म_पढ़ो(SWIV_SEG_आकार_MIN))
    swiv_seg.magic = unpacked_तथ्य[0]

    वापस swiv_seg.magic == SWIV_MAGIC


def same_output_file(निवेश_elf, output_elf):
    प्रयास:
        वापस os.path.exists(output_elf) and os.path.samefile(निवेश_elf, output_elf)
    except FileNotFoundError:
        वापस False


def make_output_target(निवेश_elf, output_elf):
    अगर not same_output_file(निवेश_elf, output_elf):
        वापस output_elf, None

    output_dir = os.path.dirname(os.path.abspath(output_elf))
    output_base = os.path.basename(output_elf)
    fd, अस्थायी_output = tempfile.mkstemp(prefix=output_base + ".", suffix=".tmp", dir=output_dir)
    os.बंद(fd)
    shutil.copymode(निवेश_elf, अस्थायी_output)
    वापस अस्थायी_output, अस्थायी_output


def swiv_generate(निवेश_elf, output_elf):
    output_target, अस्थायी_output = make_output_target(निवेश_elf, output_elf)

    प्रयास:
        with open(निवेश_elf, "rb") as elf_in_fp, open(output_target, "wb+") as elf_out_fp:
            [elf_header, phdr_table] = preprocess_elf_file(निवेश_elf)

            अंक_phdrs = elf_header.e_phnum
            अंक_shdrs = elf_header.e_shnum
            phdr_आकार = elf_header.e_phentsize

            अगर अंक_phdrs == 0 or अंक_phdrs > MAX_PHDR_गिनती:
                raise RuntimeError("Input ELF has invalid number of program headers.")

            अगर elf_header.e_shoff == 0 or अंक_shdrs == 0:
                raise RuntimeError("Input ELF does not contain a section header table.")

            shdr_table_निवेश = process_section_header(निवेश_elf)
            shdr_table_output = process_section_header(निवेश_elf)

            last_phdr = phdr_table[अंक_phdrs - 1]
            अगर last_phdr.p_memsz == SWIV_SEG_आकार_MIN and verify_swiv_seg(elf_in_fp, last_phdr.p_offset):
                raise RuntimeError("A SWIV segment is detected in the input ELF.")

            last_shdr = shdr_table_निवेश[elf_header.e_shnum - 1]
            अगर last_shdr.sh_आकार == SWIV_SEG_आकार_MIN and verify_swiv_seg(elf_in_fp, last_shdr.sh_offset):
                raise RuntimeError("A SWIV section is detected in the input ELF.")

            अगर is_elf64(elf_header):
                swiv_shdr = Elf64_Shdr(b'\0' * Elf64_Shdr.s.आकार)
            अन्यथा:
                swiv_shdr = Elf32_Shdr(b'\0' * Elf32_Shdr.s.आकार)

            swiv_shdr.sh_name = 0
            swiv_shdr.sh_type = SWIV_SECTION_TYPE
            swiv_shdr.sh_flags = 0
            swiv_shdr.sh_addr = 0
            swiv_shdr.sh_आकार = SWIV_SEG_आकार_MIN
            swiv_shdr.sh_offset = elf_header.e_shoff + (elf_header.e_shnum + 1) * elf_header.e_shentsize
            shdr_table_output.append(swiv_shdr)
            elf_header.e_shnum += 1

            अगर elf_header.e_phoff != 0:
                क्रम phdr in phdr_table:
                    file_copy_offset(elf_in_fp, phdr.p_offset, elf_out_fp, phdr.p_offset, phdr.p_filesz)

            elf_out_fp.seek(0)
            elf_out_fp.म_लिखो(elf_header.get_packed_तथ्य())

            phdr_start = elf_header.e_phoff
            क्रम phdr in phdr_table:
                elf_out_fp.seek(phdr_start)
                elf_out_fp.म_लिखो(phdr.get_packed_तथ्य())
                phdr_start += phdr_आकार

            क्रम shdr in shdr_table_निवेश[:अंक_shdrs]:
                अगर shdr.sh_type == SHT_NOBITS or shdr.sh_आकार == 0:
                    जारी
                file_copy_offset(elf_in_fp, shdr.sh_offset, elf_out_fp, shdr.sh_offset, shdr.sh_आकार)

            shdr_start = elf_header.e_shoff
            क्रम shdr in shdr_table_output:
                elf_out_fp.seek(shdr_start)
                elf_out_fp.म_लिखो(shdr.get_packed_तथ्य())
                shdr_start += elf_header.e_shentsize

            elf_out_fp.flush()

            swiv_seg = SWIV_Seg()
            swiv_seg.crc = crc32_calculate(elf_out_fp, phdr_table)
            elf_out_fp.seek(swiv_shdr.sh_offset)
            elf_out_fp.म_लिखो(swiv_seg.get_packed_तथ्य())

        अगर अस्थायी_output is not None:
            os.प्रतिस्थापन(अस्थायी_output, output_elf)
            अस्थायी_output = None

        वापस 0
    अन्तिमतः:
        अगर अस्थायी_output is not None and os.path.exists(अस्थायी_output):
            os.unlink(अस्थायी_output)


def मुख्य(argv):
    parser = argparse.ArgumentParser(description="Add a Software Image Validation (SWIV) section to an HTP ELF file.")
    parser.add_argument("-i", "--input", dest="input_file", metavar="INPUT", required=True)
    parser.add_argument("-o", "--output", dest="output_file", metavar="OUTPUT", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="show calculated CRC")
    तर्क = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO अगर तर्क.verbose अन्यथा logging.WARNING,
        format="%(message)s",
    )

    प्रयास:
        वापस swiv_generate(तर्क.निवेश_file, तर्क.output_file)
    except RuntimeError as exc:
        logger.error("ERROR: %s", exc)
        वापस 2


अगर __name__ == "__main__":
    sys.exit(मुख्य(sys.argv[1:]))
