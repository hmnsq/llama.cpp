#!/usr/bin/env python3

இறக்குமதி argparse
இறக்குமதி logging
இறக்குமதி os
இறக்குமதி shutil
இறக்குமதி struct
இறக்குமதி sys
இறக்குமதி tempfile
இறக்குமதி zlib

ELF_MAGIC = b'\x7fELF'
ELF_CLASS_INDEX = 4
ELFCLASS32 = 1
ELFCLASS64 = 2
ELF_VERSION_INDEX = 6
EV_CURRENT = 1
MAX_PHDR_மொத்தம் = 100

LOAD_TYPE = 0x1

SHT_NOBITS = 8

SWIV_MAGIC = b'SWIV'
SWIV_SEG_SIZE_MIN = 16
# SHT_LOUSER plus ASCII "SWIV".
SWIV_SECTION_TYPE = 0xD3574956

logger = logging.getLogger("htp-swiv")


வகுப்பு Elf_Ehdr_common:
    s = struct.Struct('16sHHI')

    def __init__(self, data):
        unpacked_data = (Elf_Ehdr_common.s).unpack(data)
        self.e_ident = unpacked_data[0]
        self.e_type = unpacked_data[1]
        self.e_machine = unpacked_data[2]
        self.e_version = unpacked_data[3]


வகுப்பு Elf32_Ehdr:
    s = struct.Struct('16sHHIIIIIHHHHHH')

    def __init__(self, data):
        unpacked_data = (Elf32_Ehdr.s).unpack(data)
        self.e_ident = unpacked_data[0]
        self.e_type = unpacked_data[1]
        self.e_machine = unpacked_data[2]
        self.e_version = unpacked_data[3]
        self.e_entry = unpacked_data[4]
        self.e_phoff = unpacked_data[5]
        self.e_shoff = unpacked_data[6]
        self.e_flags = unpacked_data[7]
        self.e_ehsize = unpacked_data[8]
        self.e_phentsize = unpacked_data[9]
        self.e_phnum = unpacked_data[10]
        self.e_shentsize = unpacked_data[11]
        self.e_shnum = unpacked_data[12]
        self.e_shstrndx = unpacked_data[13]

    def get_packed_data(self):
        மதிப்புகள் = [self.e_ident,
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

        திரும்பு (Elf32_Ehdr.s).pack(*மதிப்புகள்)


வகுப்பு Elf32_Phdr:
    s = struct.Struct('I' * 8)

    def __init__(self, data):
        unpacked_data = (Elf32_Phdr.s).unpack(data)
        self.p_type = unpacked_data[0]
        self.p_offset = unpacked_data[1]
        self.p_vaddr = unpacked_data[2]
        self.p_paddr = unpacked_data[3]
        self.p_filesz = unpacked_data[4]
        self.p_memsz = unpacked_data[5]
        self.p_flags = unpacked_data[6]
        self.p_align = unpacked_data[7]

    def get_packed_data(self):
        மதிப்புகள் = [self.p_type,
                  self.p_offset,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_flags,
                  self.p_align
                  ]

        திரும்பு (Elf32_Phdr.s).pack(*மதிப்புகள்)


வகுப்பு Elf32_Shdr:
    s = struct.Struct('I' * 10)

    def __init__(self, data):
        unpacked_data = (Elf32_Shdr.s).unpack(data)
        self.sh_name = unpacked_data[0]
        self.sh_type = unpacked_data[1]
        self.sh_flags = unpacked_data[2]
        self.sh_addr = unpacked_data[3]
        self.sh_offset = unpacked_data[4]
        self.sh_size = unpacked_data[5]
        self.sh_link = unpacked_data[6]
        self.sh_info = unpacked_data[7]
        self.sh_addralign = unpacked_data[8]
        self.sh_entsize = unpacked_data[9]
        self.sh_name_str = ""

    def get_packed_data(self):
        மதிப்புகள் = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_offset,
                  self.sh_size,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        திரும்பு (Elf32_Shdr.s).pack(*மதிப்புகள்)


வகுப்பு Elf64_Ehdr:
    s = struct.Struct('16sHHIQQQIHHHHHH')

    def __init__(self, data):
        unpacked_data = (Elf64_Ehdr.s).unpack(data)
        self.e_ident = unpacked_data[0]
        self.e_type = unpacked_data[1]
        self.e_machine = unpacked_data[2]
        self.e_version = unpacked_data[3]
        self.e_entry = unpacked_data[4]
        self.e_phoff = unpacked_data[5]
        self.e_shoff = unpacked_data[6]
        self.e_flags = unpacked_data[7]
        self.e_ehsize = unpacked_data[8]
        self.e_phentsize = unpacked_data[9]
        self.e_phnum = unpacked_data[10]
        self.e_shentsize = unpacked_data[11]
        self.e_shnum = unpacked_data[12]
        self.e_shstrndx = unpacked_data[13]

    def get_packed_data(self):
        மதிப்புகள் = [self.e_ident,
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

        திரும்பு (Elf64_Ehdr.s).pack(*மதிப்புகள்)


வகுப்பு Elf64_Phdr:
    s = struct.Struct('IIQQQQQQ')

    def __init__(self, data):
        unpacked_data = (Elf64_Phdr.s).unpack(data)
        self.p_type = unpacked_data[0]
        self.p_flags = unpacked_data[1]
        self.p_offset = unpacked_data[2]
        self.p_vaddr = unpacked_data[3]
        self.p_paddr = unpacked_data[4]
        self.p_filesz = unpacked_data[5]
        self.p_memsz = unpacked_data[6]
        self.p_align = unpacked_data[7]

    def get_packed_data(self):
        மதிப்புகள் = [self.p_type,
                  self.p_flags,
                  self.p_offset,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_align
                  ]

        திரும்பு (Elf64_Phdr.s).pack(*மதிப்புகள்)


வகுப்பு Elf64_Shdr:
    s = struct.Struct('IIQQQQIIQQ')

    def __init__(self, data):
        unpacked_data = (Elf64_Shdr.s).unpack(data)
        self.sh_name = unpacked_data[0]
        self.sh_type = unpacked_data[1]
        self.sh_flags = unpacked_data[2]
        self.sh_addr = unpacked_data[3]
        self.sh_offset = unpacked_data[4]
        self.sh_size = unpacked_data[5]
        self.sh_link = unpacked_data[6]
        self.sh_info = unpacked_data[7]
        self.sh_addralign = unpacked_data[8]
        self.sh_entsize = unpacked_data[9]
        self.sh_name_str = ""

    def get_packed_data(self):
        மதிப்புகள் = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_offset,
                  self.sh_size,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        திரும்பு (Elf64_Shdr.s).pack(*மதிப்புகள்)


வகுப்பு SWIV_Seg:
    s = struct.Struct('4sIQ')

    def __init__(self):
        self.magic = SWIV_MAGIC
        self.crc = 0
        self.padding = 0

    def get_packed_data(self):
        மதிப்புகள் = [self.magic,
                  self.crc,
                  self.padding
                  ]

        திரும்பு (SWIV_Seg.s).pack(*மதிப்புகள்)


def is_elf64(elf_header):
    திரும்பு elf_header.e_ident[ELF_CLASS_INDEX] == ELFCLASS64


def verify_elf_header(elf_header):
    ident = elf_header.e_ident
    திரும்பு (ident.startswith(ELF_MAGIC)
            மற்றும் ident[ELF_CLASS_INDEX] உள் (ELFCLASS32, ELFCLASS64)
            மற்றும் ident[ELF_VERSION_INDEX] == EV_CURRENT)


def file_copy_offset(in_fp, in_off, out_fp, out_off, இலக்கம்_bytes):
    in_fp.seek(in_off)
    read_in = in_fp.பட(இலக்கம்_bytes)
    out_fp.seek(out_off)
    out_fp.லிக(read_in)

    திரும்பு இலக்கம்_bytes


def preprocess_elf_file(elf_file_name):
    with open(elf_file_name, 'rb') இவ elf_fp:
        elf_header = Elf_Ehdr_common(elf_fp.பட(Elf_Ehdr_common.s.size))

        எனில் verify_elf_header(elf_header) is பொய்:
            எழுப்பு RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        எனில் is_elf64(elf_header):
            elf_header = Elf64_Ehdr(elf_fp.பட(Elf64_Ehdr.s.size))
        இல்லையெனில்:
            elf_header = Elf32_Ehdr(elf_fp.பட(Elf32_Ehdr.s.size))

        phdr_table = []

        phdr_size = elf_header.e_phentsize

        file_offset = elf_header.e_phoff
        elf_fp.seek(file_offset)

        க்கு _ உள் range(elf_header.e_phnum):
            எனில் is_elf64(elf_header):
                phdr_table.append(Elf64_Phdr(elf_fp.பட(phdr_size)))
            இல்லையெனில்:
                phdr_table.append(Elf32_Phdr(elf_fp.பட(phdr_size)))

    திரும்பு [elf_header, phdr_table]


def process_section_header(elf_file_name):
    with open(elf_file_name, 'rb') இவ elf_fp:
        elf_header = Elf_Ehdr_common(elf_fp.பட(Elf_Ehdr_common.s.size))

        எனில் verify_elf_header(elf_header) is பொய்:
            எழுப்பு RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        எனில் is_elf64(elf_header):
            elf_header = Elf64_Ehdr(elf_fp.பட(Elf64_Ehdr.s.size))
        இல்லையெனில்:
            elf_header = Elf32_Ehdr(elf_fp.பட(Elf32_Ehdr.s.size))

        shdr_table = []

        shdr_size = elf_header.e_shentsize
        elf_fp.seek(elf_header.e_shoff)

        க்கு _ உள் range(elf_header.e_shnum):
            எனில் is_elf64(elf_header):
                shdr_table.append(Elf64_Shdr(elf_fp.பட(shdr_size)))
            இல்லையெனில்:
                shdr_table.append(Elf32_Shdr(elf_fp.பட(shdr_size)))

    திரும்பு shdr_table


def crc32_update_file(elf_fp, offset, size, crc):
    elf_fp.seek(offset)
    remaining = size
    போது remaining > 0:
        chunk = elf_fp.பட(min(1024 * 1024, remaining))
        எனில் இல்லை chunk:
            எழுப்பு RuntimeError("Unexpected EOF while calculating CRC")
        crc = zlib.crc32(chunk, crc)
        remaining -= len(chunk)
    திரும்பு crc


def crc32_update_zeros(size, crc):
    zero_chunk = b'\x00' * min(1024 * 1024, size)
    remaining = size
    போது remaining > 0:
        chunk = zero_chunk[:min(len(zero_chunk), remaining)]
        crc = zlib.crc32(chunk, crc)
        remaining -= len(chunk)
    திரும்பு crc


def crc32_calculate(elf_fp, phdr_table):
    crc = 0

    # SWIV validates loadable segment contents, including BSS as zeros.
    க்கு curr_phdr உள் phdr_table:
        எனில் curr_phdr.p_type != LOAD_TYPE:
            தொடர்

        crc = crc32_update_file(elf_fp, curr_phdr.p_offset, curr_phdr.p_filesz, crc)

        zero_size = curr_phdr.p_memsz - curr_phdr.p_filesz
        எனில் zero_size > 0:
            crc = crc32_update_zeros(zero_size, crc)

    crc &= 0xffffffff
    logger.info("calculated CRC: %#x", crc)

    திரும்பு crc


def verify_swiv_seg(elf_in_fp, offset):

    elf_in_fp.seek(offset)
    swiv_seg = SWIV_Seg()
    unpacked_data = (SWIV_Seg.s).unpack(elf_in_fp.பட(SWIV_SEG_SIZE_MIN))
    swiv_seg.magic = unpacked_data[0]

    திரும்பு swiv_seg.magic == SWIV_MAGIC


def same_output_file(உள்ளீடு_elf, output_elf):
    முயல்:
        திரும்பு os.path.exists(output_elf) மற்றும் os.path.samefile(உள்ளீடு_elf, output_elf)
    விதிவிலக்கு FileNotFoundError:
        திரும்பு பொய்


def make_output_target(உள்ளீடு_elf, output_elf):
    எனில் இல்லை same_output_file(உள்ளீடு_elf, output_elf):
        திரும்பு output_elf, None

    output_dir = os.path.dirname(os.path.abspath(output_elf))
    output_base = os.path.basename(output_elf)
    fd, temp_output = tempfile.mkstemp(prefix=output_base + ".", suffix=".tmp", dir=output_dir)
    os.close(fd)
    shutil.copymode(உள்ளீடு_elf, temp_output)
    திரும்பு temp_output, temp_output


def swiv_generate(உள்ளீடு_elf, output_elf):
    output_target, temp_output = make_output_target(உள்ளீடு_elf, output_elf)

    முயல்:
        with open(உள்ளீடு_elf, "rb") இவ elf_in_fp, open(output_target, "wb+") இவ elf_out_fp:
            [elf_header, phdr_table] = preprocess_elf_file(உள்ளீடு_elf)

            இலக்கம்_phdrs = elf_header.e_phnum
            இலக்கம்_shdrs = elf_header.e_shnum
            phdr_size = elf_header.e_phentsize

            எனில் இலக்கம்_phdrs == 0 அல்லது இலக்கம்_phdrs > MAX_PHDR_மொத்தம்:
                எழுப்பு RuntimeError("Input ELF has invalid number of program headers.")

            எனில் elf_header.e_shoff == 0 அல்லது இலக்கம்_shdrs == 0:
                எழுப்பு RuntimeError("Input ELF does not contain a section header table.")

            shdr_table_உள்ளீடு = process_section_header(உள்ளீடு_elf)
            shdr_table_output = process_section_header(உள்ளீடு_elf)

            last_phdr = phdr_table[இலக்கம்_phdrs - 1]
            எனில் last_phdr.p_memsz == SWIV_SEG_SIZE_MIN மற்றும் verify_swiv_seg(elf_in_fp, last_phdr.p_offset):
                எழுப்பு RuntimeError("A SWIV segment is detected in the input ELF.")

            last_shdr = shdr_table_உள்ளீடு[elf_header.e_shnum - 1]
            எனில் last_shdr.sh_size == SWIV_SEG_SIZE_MIN மற்றும் verify_swiv_seg(elf_in_fp, last_shdr.sh_offset):
                எழுப்பு RuntimeError("A SWIV section is detected in the input ELF.")

            எனில் is_elf64(elf_header):
                swiv_shdr = Elf64_Shdr(b'\0' * Elf64_Shdr.s.size)
            இல்லையெனில்:
                swiv_shdr = Elf32_Shdr(b'\0' * Elf32_Shdr.s.size)

            swiv_shdr.sh_name = 0
            swiv_shdr.sh_type = SWIV_SECTION_TYPE
            swiv_shdr.sh_flags = 0
            swiv_shdr.sh_addr = 0
            swiv_shdr.sh_size = SWIV_SEG_SIZE_MIN
            swiv_shdr.sh_offset = elf_header.e_shoff + (elf_header.e_shnum + 1) * elf_header.e_shentsize
            shdr_table_output.append(swiv_shdr)
            elf_header.e_shnum += 1

            எனில் elf_header.e_phoff != 0:
                க்கு phdr உள் phdr_table:
                    file_copy_offset(elf_in_fp, phdr.p_offset, elf_out_fp, phdr.p_offset, phdr.p_filesz)

            elf_out_fp.seek(0)
            elf_out_fp.லிக(elf_header.get_packed_data())

            phdr_start = elf_header.e_phoff
            க்கு phdr உள் phdr_table:
                elf_out_fp.seek(phdr_start)
                elf_out_fp.லிக(phdr.get_packed_data())
                phdr_start += phdr_size

            க்கு shdr உள் shdr_table_உள்ளீடு[:இலக்கம்_shdrs]:
                எனில் shdr.sh_type == SHT_NOBITS அல்லது shdr.sh_size == 0:
                    தொடர்
                file_copy_offset(elf_in_fp, shdr.sh_offset, elf_out_fp, shdr.sh_offset, shdr.sh_size)

            shdr_start = elf_header.e_shoff
            க்கு shdr உள் shdr_table_output:
                elf_out_fp.seek(shdr_start)
                elf_out_fp.லிக(shdr.get_packed_data())
                shdr_start += elf_header.e_shentsize

            elf_out_fp.flush()

            swiv_seg = SWIV_Seg()
            swiv_seg.crc = crc32_calculate(elf_out_fp, phdr_table)
            elf_out_fp.seek(swiv_shdr.sh_offset)
            elf_out_fp.லிக(swiv_seg.get_packed_data())

        எனில் temp_output is இல்லை None:
            os.replace(temp_output, output_elf)
            temp_output = None

        திரும்பு 0
    இறுதியாக:
        எனில் temp_output is இல்லை None மற்றும் os.path.exists(temp_output):
            os.unlink(temp_output)


def முதன்மை(argv):
    parser = argparse.ArgumentParser(description="Add a Software Image Validation (SWIV) section to an HTP ELF file.")
    parser.add_argument("-i", "--input", dest="input_file", metavar="INPUT", required=உண்மை)
    parser.add_argument("-o", "--output", dest="output_file", metavar="OUTPUT", required=உண்மை)
    parser.add_argument("-v", "--verbose", action="store_true", help="show calculated CRC")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO எனில் args.verbose இல்லையெனில் logging.WARNING,
        format="%(message)s",
    )

    முயல்:
        திரும்பு swiv_generate(args.உள்ளீடு_file, args.output_file)
    விதிவிலக்கு RuntimeError இவ exc:
        logger.error("ERROR: %s", exc)
        திரும்பு 2


எனில் __name__ == "__main__":
    sys.exit(முதன்மை(sys.argv[1:]))
