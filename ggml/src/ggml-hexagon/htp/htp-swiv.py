#!/usr/bin/env python3

আমদানি argparse
আমদানি logging
আমদানি os
আমদানি shutil
আমদানি struct
আমদানি sys
আমদানি tempfile
আমদানি zlib

ELF_MAGIC = b'\x7fELF'
ELF_CLASS_সূচক = 4
ELFCLASS32 = 1
ELFCLASS64 = 2
ELF_VERSION_সূচক = 6
EV_CURRENT = 1
MAX_PHDR_গুনতি = 100

LOAD_TYPE = 0x1

SHT_NOBITS = 8

SWIV_MAGIC = b'SWIV'
SWIV_SEG_আকার_MIN = 16
# SHT_LOUSER plus ASCII "SWIV".
SWIV_SECTION_TYPE = 0xD3574956

logger = logging.getLogger("htp-swiv")


শ্রেণী Elf_Ehdr_common:
    s = struct.Struct('16sHHI')

    def __init__(self, তথ্য):
        unpacked_তথ্য = (Elf_Ehdr_common.s).unpack(তথ্য)
        self.e_ident = unpacked_তথ্য[0]
        self.e_type = unpacked_তথ্য[1]
        self.e_machine = unpacked_তথ্য[2]
        self.e_version = unpacked_তথ্য[3]


শ্রেণী Elf32_Ehdr:
    s = struct.Struct('16sHHIIIIIHHHHHH')

    def __init__(self, তথ্য):
        unpacked_তথ্য = (Elf32_Ehdr.s).unpack(তথ্য)
        self.e_ident = unpacked_তথ্য[0]
        self.e_type = unpacked_তথ্য[1]
        self.e_machine = unpacked_তথ্য[2]
        self.e_version = unpacked_তথ্য[3]
        self.e_entry = unpacked_তথ্য[4]
        self.e_phoff = unpacked_তথ্য[5]
        self.e_shoff = unpacked_তথ্য[6]
        self.e_flags = unpacked_তথ্য[7]
        self.e_ehsize = unpacked_তথ্য[8]
        self.e_phentsize = unpacked_তথ্য[9]
        self.e_phnum = unpacked_তথ্য[10]
        self.e_shentsize = unpacked_তথ্য[11]
        self.e_shnum = unpacked_তথ্য[12]
        self.e_shstrndx = unpacked_তথ্য[13]

    def get_packed_তথ্য(self):
        মানসমূহ = [self.e_ident,
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

        ফেরত (Elf32_Ehdr.s).pack(*মানসমূহ)


শ্রেণী Elf32_Phdr:
    s = struct.Struct('I' * 8)

    def __init__(self, তথ্য):
        unpacked_তথ্য = (Elf32_Phdr.s).unpack(তথ্য)
        self.p_type = unpacked_তথ্য[0]
        self.p_offset = unpacked_তথ্য[1]
        self.p_vaddr = unpacked_তথ্য[2]
        self.p_paddr = unpacked_তথ্য[3]
        self.p_filesz = unpacked_তথ্য[4]
        self.p_memsz = unpacked_তথ্য[5]
        self.p_flags = unpacked_তথ্য[6]
        self.p_align = unpacked_তথ্য[7]

    def get_packed_তথ্য(self):
        মানসমূহ = [self.p_type,
                  self.p_offset,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_flags,
                  self.p_align
                  ]

        ফেরত (Elf32_Phdr.s).pack(*মানসমূহ)


শ্রেণী Elf32_Shdr:
    s = struct.Struct('I' * 10)

    def __init__(self, তথ্য):
        unpacked_তথ্য = (Elf32_Shdr.s).unpack(তথ্য)
        self.sh_name = unpacked_তথ্য[0]
        self.sh_type = unpacked_তথ্য[1]
        self.sh_flags = unpacked_তথ্য[2]
        self.sh_addr = unpacked_তথ্য[3]
        self.sh_offset = unpacked_তথ্য[4]
        self.sh_আকার = unpacked_তথ্য[5]
        self.sh_link = unpacked_তথ্য[6]
        self.sh_info = unpacked_তথ্য[7]
        self.sh_addralign = unpacked_তথ্য[8]
        self.sh_entsize = unpacked_তথ্য[9]
        self.sh_name_str = ""

    def get_packed_তথ্য(self):
        মানসমূহ = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_offset,
                  self.sh_আকার,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        ফেরত (Elf32_Shdr.s).pack(*মানসমূহ)


শ্রেণী Elf64_Ehdr:
    s = struct.Struct('16sHHIQQQIHHHHHH')

    def __init__(self, তথ্য):
        unpacked_তথ্য = (Elf64_Ehdr.s).unpack(তথ্য)
        self.e_ident = unpacked_তথ্য[0]
        self.e_type = unpacked_তথ্য[1]
        self.e_machine = unpacked_তথ্য[2]
        self.e_version = unpacked_তথ্য[3]
        self.e_entry = unpacked_তথ্য[4]
        self.e_phoff = unpacked_তথ্য[5]
        self.e_shoff = unpacked_তথ্য[6]
        self.e_flags = unpacked_তথ্য[7]
        self.e_ehsize = unpacked_তথ্য[8]
        self.e_phentsize = unpacked_তথ্য[9]
        self.e_phnum = unpacked_তথ্য[10]
        self.e_shentsize = unpacked_তথ্য[11]
        self.e_shnum = unpacked_তথ্য[12]
        self.e_shstrndx = unpacked_তথ্য[13]

    def get_packed_তথ্য(self):
        মানসমূহ = [self.e_ident,
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

        ফেরত (Elf64_Ehdr.s).pack(*মানসমূহ)


শ্রেণী Elf64_Phdr:
    s = struct.Struct('IIQQQQQQ')

    def __init__(self, তথ্য):
        unpacked_তথ্য = (Elf64_Phdr.s).unpack(তথ্য)
        self.p_type = unpacked_তথ্য[0]
        self.p_flags = unpacked_তথ্য[1]
        self.p_offset = unpacked_তথ্য[2]
        self.p_vaddr = unpacked_তথ্য[3]
        self.p_paddr = unpacked_তথ্য[4]
        self.p_filesz = unpacked_তথ্য[5]
        self.p_memsz = unpacked_তথ্য[6]
        self.p_align = unpacked_তথ্য[7]

    def get_packed_তথ্য(self):
        মানসমূহ = [self.p_type,
                  self.p_flags,
                  self.p_offset,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_align
                  ]

        ফেরত (Elf64_Phdr.s).pack(*মানসমূহ)


শ্রেণী Elf64_Shdr:
    s = struct.Struct('IIQQQQIIQQ')

    def __init__(self, তথ্য):
        unpacked_তথ্য = (Elf64_Shdr.s).unpack(তথ্য)
        self.sh_name = unpacked_তথ্য[0]
        self.sh_type = unpacked_তথ্য[1]
        self.sh_flags = unpacked_তথ্য[2]
        self.sh_addr = unpacked_তথ্য[3]
        self.sh_offset = unpacked_তথ্য[4]
        self.sh_আকার = unpacked_তথ্য[5]
        self.sh_link = unpacked_তথ্য[6]
        self.sh_info = unpacked_তথ্য[7]
        self.sh_addralign = unpacked_তথ্য[8]
        self.sh_entsize = unpacked_তথ্য[9]
        self.sh_name_str = ""

    def get_packed_তথ্য(self):
        মানসমূহ = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_offset,
                  self.sh_আকার,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        ফেরত (Elf64_Shdr.s).pack(*মানসমূহ)


শ্রেণী SWIV_Seg:
    s = struct.Struct('4sIQ')

    def __init__(self):
        self.magic = SWIV_MAGIC
        self.crc = 0
        self.padding = 0

    def get_packed_তথ্য(self):
        মানসমূহ = [self.magic,
                  self.crc,
                  self.padding
                  ]

        ফেরত (SWIV_Seg.s).pack(*মানসমূহ)


def is_elf64(elf_header):
    ফেরত elf_header.e_ident[ELF_CLASS_সূচক] == ELFCLASS64


def verify_elf_header(elf_header):
    ident = elf_header.e_ident
    ফেরত (ident.startswith(ELF_MAGIC)
            এবং ident[ELF_CLASS_সূচক] মধ্যে (ELFCLASS32, ELFCLASS64)
            এবং ident[ELF_VERSION_সূচক] == EV_CURRENT)


def file_copy_offset(in_fp, in_off, out_fp, out_off, অঙ্ক_bytes):
    in_fp.seek(in_off)
    read_in = in_fp.পঠ(অঙ্ক_bytes)
    out_fp.seek(out_off)
    out_fp.লিখ(read_in)

    ফেরত অঙ্ক_bytes


def preprocess_elf_file(elf_file_name):
    with open(elf_file_name, 'rb') ইব elf_fp:
        elf_header = Elf_Ehdr_common(elf_fp.পঠ(Elf_Ehdr_common.s.আকার_2))

        যদি verify_elf_header(elf_header) is মিথ্যা:
            তোলো RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        যদি is_elf64(elf_header):
            elf_header = Elf64_Ehdr(elf_fp.পঠ(Elf64_Ehdr.s.আকার_2))
        অন্যথা:
            elf_header = Elf32_Ehdr(elf_fp.পঠ(Elf32_Ehdr.s.আকার_2))

        phdr_table = []

        phdr_আকার = elf_header.e_phentsize

        file_offset = elf_header.e_phoff
        elf_fp.seek(file_offset)

        জন্য _ মধ্যে range(elf_header.e_phnum):
            যদি is_elf64(elf_header):
                phdr_table.append(Elf64_Phdr(elf_fp.পঠ(phdr_আকার)))
            অন্যথা:
                phdr_table.append(Elf32_Phdr(elf_fp.পঠ(phdr_আকার)))

    ফেরত [elf_header, phdr_table]


def process_section_header(elf_file_name):
    with open(elf_file_name, 'rb') ইব elf_fp:
        elf_header = Elf_Ehdr_common(elf_fp.পঠ(Elf_Ehdr_common.s.আকার_2))

        যদি verify_elf_header(elf_header) is মিথ্যা:
            তোলো RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        যদি is_elf64(elf_header):
            elf_header = Elf64_Ehdr(elf_fp.পঠ(Elf64_Ehdr.s.আকার_2))
        অন্যথা:
            elf_header = Elf32_Ehdr(elf_fp.পঠ(Elf32_Ehdr.s.আকার_2))

        shdr_table = []

        shdr_আকার = elf_header.e_shentsize
        elf_fp.seek(elf_header.e_shoff)

        জন্য _ মধ্যে range(elf_header.e_shnum):
            যদি is_elf64(elf_header):
                shdr_table.append(Elf64_Shdr(elf_fp.পঠ(shdr_আকার)))
            অন্যথা:
                shdr_table.append(Elf32_Shdr(elf_fp.পঠ(shdr_আকার)))

    ফেরত shdr_table


def crc32_update_file(elf_fp, offset, আকার_2, crc):
    elf_fp.seek(offset)
    remaining = আকার_2
    যখন remaining > 0:
        chunk = elf_fp.পঠ(min(1024 * 1024, remaining))
        যদি নয় chunk:
            তোলো RuntimeError("Unexpected EOF while calculating CRC")
        crc = zlib.crc32(chunk, crc)
        remaining -= len(chunk)
    ফেরত crc


def crc32_update_zeros(আকার_2, crc):
    zero_chunk = b'\x00' * min(1024 * 1024, আকার_2)
    remaining = আকার_2
    যখন remaining > 0:
        chunk = zero_chunk[:min(len(zero_chunk), remaining)]
        crc = zlib.crc32(chunk, crc)
        remaining -= len(chunk)
    ফেরত crc


def crc32_calculate(elf_fp, phdr_table):
    crc = 0

    # SWIV validates loadable segment contents, including BSS as zeros.
    জন্য curr_phdr মধ্যে phdr_table:
        যদি curr_phdr.p_type != LOAD_TYPE:
            চালু

        crc = crc32_update_file(elf_fp, curr_phdr.p_offset, curr_phdr.p_filesz, crc)

        zero_আকার = curr_phdr.p_memsz - curr_phdr.p_filesz
        যদি zero_আকার > 0:
            crc = crc32_update_zeros(zero_আকার, crc)

    crc &= 0xffffffff
    logger.info("calculated CRC: %#x", crc)

    ফেরত crc


def verify_swiv_seg(elf_in_fp, offset):

    elf_in_fp.seek(offset)
    swiv_seg = SWIV_Seg()
    unpacked_তথ্য = (SWIV_Seg.s).unpack(elf_in_fp.পঠ(SWIV_SEG_আকার_MIN))
    swiv_seg.magic = unpacked_তথ্য[0]

    ফেরত swiv_seg.magic == SWIV_MAGIC


def same_output_file(প্রবেশ_elf, output_elf):
    চেষ্টা:
        ফেরত os.path.exists(output_elf) এবং os.path.samefile(প্রবেশ_elf, output_elf)
    ব্যতিক্রম FileNotFoundError:
        ফেরত মিথ্যা


def make_output_target(প্রবেশ_elf, output_elf):
    যদি নয় same_output_file(প্রবেশ_elf, output_elf):
        ফেরত output_elf, None

    output_dir = os.path.dirname(os.path.abspath(output_elf))
    output_base = os.path.basename(output_elf)
    fd, অস্থায়ী_output = tempfile.mkstemp(prefix=output_base + ".", suffix=".tmp", dir=output_dir)
    os.close(fd)
    shutil.copymode(প্রবেশ_elf, অস্থায়ী_output)
    ফেরত অস্থায়ী_output, অস্থায়ী_output


def swiv_generate(প্রবেশ_elf, output_elf):
    output_target, অস্থায়ী_output = make_output_target(প্রবেশ_elf, output_elf)

    চেষ্টা:
        with open(প্রবেশ_elf, "rb") ইব elf_in_fp, open(output_target, "wb+") ইব elf_out_fp:
            [elf_header, phdr_table] = preprocess_elf_file(প্রবেশ_elf)

            অঙ্ক_phdrs = elf_header.e_phnum
            অঙ্ক_shdrs = elf_header.e_shnum
            phdr_আকার = elf_header.e_phentsize

            যদি অঙ্ক_phdrs == 0 অথবা অঙ্ক_phdrs > MAX_PHDR_গুনতি:
                তোলো RuntimeError("Input ELF has invalid number of program headers.")

            যদি elf_header.e_shoff == 0 অথবা অঙ্ক_shdrs == 0:
                তোলো RuntimeError("Input ELF does not contain a section header table.")

            shdr_table_প্রবেশ = process_section_header(প্রবেশ_elf)
            shdr_table_output = process_section_header(প্রবেশ_elf)

            last_phdr = phdr_table[অঙ্ক_phdrs - 1]
            যদি last_phdr.p_memsz == SWIV_SEG_আকার_MIN এবং verify_swiv_seg(elf_in_fp, last_phdr.p_offset):
                তোলো RuntimeError("A SWIV segment is detected in the input ELF.")

            last_shdr = shdr_table_প্রবেশ[elf_header.e_shnum - 1]
            যদি last_shdr.sh_আকার == SWIV_SEG_আকার_MIN এবং verify_swiv_seg(elf_in_fp, last_shdr.sh_offset):
                তোলো RuntimeError("A SWIV section is detected in the input ELF.")

            যদি is_elf64(elf_header):
                swiv_shdr = Elf64_Shdr(b'\0' * Elf64_Shdr.s.আকার_2)
            অন্যথা:
                swiv_shdr = Elf32_Shdr(b'\0' * Elf32_Shdr.s.আকার_2)

            swiv_shdr.sh_name = 0
            swiv_shdr.sh_type = SWIV_SECTION_TYPE
            swiv_shdr.sh_flags = 0
            swiv_shdr.sh_addr = 0
            swiv_shdr.sh_আকার = SWIV_SEG_আকার_MIN
            swiv_shdr.sh_offset = elf_header.e_shoff + (elf_header.e_shnum + 1) * elf_header.e_shentsize
            shdr_table_output.append(swiv_shdr)
            elf_header.e_shnum += 1

            যদি elf_header.e_phoff != 0:
                জন্য phdr মধ্যে phdr_table:
                    file_copy_offset(elf_in_fp, phdr.p_offset, elf_out_fp, phdr.p_offset, phdr.p_filesz)

            elf_out_fp.seek(0)
            elf_out_fp.লিখ(elf_header.get_packed_তথ্য())

            phdr_start = elf_header.e_phoff
            জন্য phdr মধ্যে phdr_table:
                elf_out_fp.seek(phdr_start)
                elf_out_fp.লিখ(phdr.get_packed_তথ্য())
                phdr_start += phdr_আকার

            জন্য shdr মধ্যে shdr_table_প্রবেশ[:অঙ্ক_shdrs]:
                যদি shdr.sh_type == SHT_NOBITS অথবা shdr.sh_আকার == 0:
                    চালু
                file_copy_offset(elf_in_fp, shdr.sh_offset, elf_out_fp, shdr.sh_offset, shdr.sh_আকার)

            shdr_start = elf_header.e_shoff
            জন্য shdr মধ্যে shdr_table_output:
                elf_out_fp.seek(shdr_start)
                elf_out_fp.লিখ(shdr.get_packed_তথ্য())
                shdr_start += elf_header.e_shentsize

            elf_out_fp.flush()

            swiv_seg = SWIV_Seg()
            swiv_seg.crc = crc32_calculate(elf_out_fp, phdr_table)
            elf_out_fp.seek(swiv_shdr.sh_offset)
            elf_out_fp.লিখ(swiv_seg.get_packed_তথ্য())

        যদি অস্থায়ী_output is নয় None:
            os.replace(অস্থায়ী_output, output_elf)
            অস্থায়ী_output = None

        ফেরত 0
    অবশেষে:
        যদি অস্থায়ী_output is নয় None এবং os.path.exists(অস্থায়ী_output):
            os.unlink(অস্থায়ী_output)


def প্রধান(argv):
    parser = argparse.ArgumentParser(description="Add a Software Image Validation (SWIV) section to an HTP ELF file.")
    parser.add_argument("-i", "--input", dest="input_file", metavar="INPUT", required=সত্য)
    parser.add_argument("-o", "--output", dest="output_file", metavar="OUTPUT", required=সত্য)
    parser.add_argument("-v", "--verbose", action="store_true", help="show calculated CRC")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO যদি args.verbose অন্যথা logging.WARNING,
        format="%(message)s",
    )

    চেষ্টা:
        ফেরত swiv_generate(args.প্রবেশ_file, args.output_file)
    ব্যতিক্রম RuntimeError ইব exc:
        logger.error("ERROR: %s", exc)
        ফেরত 2


যদি __name__ == "__main__":
    sys.exit(প্রধান(sys.argv[1:]))
