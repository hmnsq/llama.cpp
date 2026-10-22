#!/usr/bin/env python3

आनयन argparse
आनयन logging
आनयन os
आनयन shutil
आनयन struct
आनयन sys
आनयन tempfile
आनयन zlib

ELF_MAGIC = b'\x7fELF'
ELF_CLASS_सूचकांक = 4
ELFCLASS32 = 1
ELFCLASS64 = 2
ELF_संस्करणम्_सूचकांक = 6
EV_CURRENT = 1
MAX_PHDR_गणनम् = 100

आरोपय_TYPE = 0x1

SHT_NOBITS = 8

SWIV_MAGIC = b'SWIV'
SWIV_SEG_आकार_MIN = 16
# SHT_LOUSER plus ASCII "SWIV".
SWIV_SECTION_TYPE = 0xD3574956

logger = logging.getLogger("htp-swiv")


वर्ग Elf_Ehdr_common:
    s = struct.Struct('16sHHI')

    परिभाषा __init__(self, दत्त):
        unpacked_दत्त = (Elf_Ehdr_common.s).unpack(दत्त)
        self.e_ident = unpacked_दत्त[0]
        self.e_type = unpacked_दत्त[1]
        self.e_machine = unpacked_दत्त[2]
        self.e_संस्करणम् = unpacked_दत्त[3]


वर्ग Elf32_Ehdr:
    s = struct.Struct('16sHHIIIIIHHHHHH')

    परिभाषा __init__(self, दत्त):
        unpacked_दत्त = (Elf32_Ehdr.s).unpack(दत्त)
        self.e_ident = unpacked_दत्त[0]
        self.e_type = unpacked_दत्त[1]
        self.e_machine = unpacked_दत्त[2]
        self.e_संस्करणम् = unpacked_दत्त[3]
        self.e_प्रविष्टिः = unpacked_दत्त[4]
        self.e_phoff = unpacked_दत्त[5]
        self.e_shoff = unpacked_दत्त[6]
        self.e_flags = unpacked_दत्त[7]
        self.e_ehsize = unpacked_दत्त[8]
        self.e_phentsize = unpacked_दत्त[9]
        self.e_phnum = unpacked_दत्त[10]
        self.e_shentsize = unpacked_दत्त[11]
        self.e_shnum = unpacked_दत्त[12]
        self.e_shstrndx = unpacked_दत्त[13]

    परिभाषा get_packed_दत्त(self):
        मानानि = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_संस्करणम्,
                  self.e_प्रविष्टिः,
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

        प्रत्यागम (Elf32_Ehdr.s).pack(*मानानि)


वर्ग Elf32_Phdr:
    s = struct.Struct('I' * 8)

    परिभाषा __init__(self, दत्त):
        unpacked_दत्त = (Elf32_Phdr.s).unpack(दत्त)
        self.p_type = unpacked_दत्त[0]
        self.p_विस्थापनम् = unpacked_दत्त[1]
        self.p_vaddr = unpacked_दत्त[2]
        self.p_paddr = unpacked_दत्त[3]
        self.p_filesz = unpacked_दत्त[4]
        self.p_memsz = unpacked_दत्त[5]
        self.p_flags = unpacked_दत्त[6]
        self.p_align = unpacked_दत्त[7]

    परिभाषा get_packed_दत्त(self):
        मानानि = [self.p_type,
                  self.p_विस्थापनम्,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_flags,
                  self.p_align
                  ]

        प्रत्यागम (Elf32_Phdr.s).pack(*मानानि)


वर्ग Elf32_Shdr:
    s = struct.Struct('I' * 10)

    परिभाषा __init__(self, दत्त):
        unpacked_दत्त = (Elf32_Shdr.s).unpack(दत्त)
        self.sh_name = unpacked_दत्त[0]
        self.sh_type = unpacked_दत्त[1]
        self.sh_flags = unpacked_दत्त[2]
        self.sh_addr = unpacked_दत्त[3]
        self.sh_विस्थापनम् = unpacked_दत्त[4]
        self.sh_आकार = unpacked_दत्त[5]
        self.sh_link = unpacked_दत्त[6]
        self.sh_info = unpacked_दत्त[7]
        self.sh_addralign = unpacked_दत्त[8]
        self.sh_entsize = unpacked_दत्त[9]
        self.sh_name_str = ""

    परिभाषा get_packed_दत्त(self):
        मानानि = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_विस्थापनम्,
                  self.sh_आकार,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        प्रत्यागम (Elf32_Shdr.s).pack(*मानानि)


वर्ग Elf64_Ehdr:
    s = struct.Struct('16sHHIQQQIHHHHHH')

    परिभाषा __init__(self, दत्त):
        unpacked_दत्त = (Elf64_Ehdr.s).unpack(दत्त)
        self.e_ident = unpacked_दत्त[0]
        self.e_type = unpacked_दत्त[1]
        self.e_machine = unpacked_दत्त[2]
        self.e_संस्करणम् = unpacked_दत्त[3]
        self.e_प्रविष्टिः = unpacked_दत्त[4]
        self.e_phoff = unpacked_दत्त[5]
        self.e_shoff = unpacked_दत्त[6]
        self.e_flags = unpacked_दत्त[7]
        self.e_ehsize = unpacked_दत्त[8]
        self.e_phentsize = unpacked_दत्त[9]
        self.e_phnum = unpacked_दत्त[10]
        self.e_shentsize = unpacked_दत्त[11]
        self.e_shnum = unpacked_दत्त[12]
        self.e_shstrndx = unpacked_दत्त[13]

    परिभाषा get_packed_दत्त(self):
        मानानि = [self.e_ident,
                  self.e_type,
                  self.e_machine,
                  self.e_संस्करणम्,
                  self.e_प्रविष्टिः,
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

        प्रत्यागम (Elf64_Ehdr.s).pack(*मानानि)


वर्ग Elf64_Phdr:
    s = struct.Struct('IIQQQQQQ')

    परिभाषा __init__(self, दत्त):
        unpacked_दत्त = (Elf64_Phdr.s).unpack(दत्त)
        self.p_type = unpacked_दत्त[0]
        self.p_flags = unpacked_दत्त[1]
        self.p_विस्थापनम् = unpacked_दत्त[2]
        self.p_vaddr = unpacked_दत्त[3]
        self.p_paddr = unpacked_दत्त[4]
        self.p_filesz = unpacked_दत्त[5]
        self.p_memsz = unpacked_दत्त[6]
        self.p_align = unpacked_दत्त[7]

    परिभाषा get_packed_दत्त(self):
        मानानि = [self.p_type,
                  self.p_flags,
                  self.p_विस्थापनम्,
                  self.p_vaddr,
                  self.p_paddr,
                  self.p_filesz,
                  self.p_memsz,
                  self.p_align
                  ]

        प्रत्यागम (Elf64_Phdr.s).pack(*मानानि)


वर्ग Elf64_Shdr:
    s = struct.Struct('IIQQQQIIQQ')

    परिभाषा __init__(self, दत्त):
        unpacked_दत्त = (Elf64_Shdr.s).unpack(दत्त)
        self.sh_name = unpacked_दत्त[0]
        self.sh_type = unpacked_दत्त[1]
        self.sh_flags = unpacked_दत्त[2]
        self.sh_addr = unpacked_दत्त[3]
        self.sh_विस्थापनम् = unpacked_दत्त[4]
        self.sh_आकार = unpacked_दत्त[5]
        self.sh_link = unpacked_दत्त[6]
        self.sh_info = unpacked_दत्त[7]
        self.sh_addralign = unpacked_दत्त[8]
        self.sh_entsize = unpacked_दत्त[9]
        self.sh_name_str = ""

    परिभाषा get_packed_दत्त(self):
        मानानि = [self.sh_name,
                  self.sh_type,
                  self.sh_flags,
                  self.sh_addr,
                  self.sh_विस्थापनम्,
                  self.sh_आकार,
                  self.sh_link,
                  self.sh_info,
                  self.sh_addralign,
                  self.sh_entsize]

        प्रत्यागम (Elf64_Shdr.s).pack(*मानानि)


वर्ग SWIV_Seg:
    s = struct.Struct('4sIQ')

    परिभाषा __init__(self):
        self.magic = SWIV_MAGIC
        self.crc = 0
        self.padding = 0

    परिभाषा get_packed_दत्त(self):
        मानानि = [self.magic,
                  self.crc,
                  self.padding
                  ]

        प्रत्यागम (SWIV_Seg.s).pack(*मानानि)


परिभाषा is_elf64(elf_शीर्षकम्):
    प्रत्यागम elf_शीर्षकम्.e_ident[ELF_CLASS_सूचकांक] == ELFCLASS64


परिभाषा verify_elf_शीर्षकम्(elf_शीर्षकम्):
    ident = elf_शीर्षकम्.e_ident
    प्रत्यागम (ident.startswith(ELF_MAGIC)
            च ident[ELF_CLASS_सूचकांक] अन्तः (ELFCLASS32, ELFCLASS64)
            च ident[ELF_संस्करणम्_सूचकांक] == EV_CURRENT)


परिभाषा file_copy_विस्थापनम्(अन्तर्_fp, अन्तर्_off, बहिर्गमनम्_fp, बहिर्गमनम्_off, अङ्कः_अष्टकानि):
    अन्तर्_fp.seek(अन्तर्_off)
    read_अन्तर् = अन्तर्_fp.पठ(अङ्कः_अष्टकानि)
    बहिर्गमनम्_fp.seek(बहिर्गमनम्_off)
    बहिर्गमनम्_fp.लिख(read_अन्तर्)

    प्रत्यागम अङ्कः_अष्टकानि


परिभाषा preprocess_elf_file(elf_file_name):
    सह open(elf_file_name, 'rb') इव elf_fp:
        elf_शीर्षकम् = Elf_Ehdr_common(elf_fp.पठ(Elf_Ehdr_common.s.आकार))

        यदि verify_elf_शीर्षकम्(elf_शीर्षकम्) अस्ति असत्य:
            उत्थापय RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        यदि is_elf64(elf_शीर्षकम्):
            elf_शीर्षकम् = Elf64_Ehdr(elf_fp.पठ(Elf64_Ehdr.s.आकार))
        अन्यथा:
            elf_शीर्षकम् = Elf32_Ehdr(elf_fp.पठ(Elf32_Ehdr.s.आकार))

        phdr_सारणी = []

        phdr_आकार = elf_शीर्षकम्.e_phentsize

        file_विस्थापनम् = elf_शीर्षकम्.e_phoff
        elf_fp.seek(file_विस्थापनम्)

        प्रति _ अन्तः परिसरः(elf_शीर्षकम्.e_phnum):
            यदि is_elf64(elf_शीर्षकम्):
                phdr_सारणी.append(Elf64_Phdr(elf_fp.पठ(phdr_आकार)))
            अन्यथा:
                phdr_सारणी.append(Elf32_Phdr(elf_fp.पठ(phdr_आकार)))

    प्रत्यागम [elf_शीर्षकम्, phdr_सारणी]


परिभाषा process_section_शीर्षकम्(elf_file_name):
    सह open(elf_file_name, 'rb') इव elf_fp:
        elf_शीर्षकम् = Elf_Ehdr_common(elf_fp.पठ(Elf_Ehdr_common.s.आकार))

        यदि verify_elf_शीर्षकम्(elf_शीर्षकम्) अस्ति असत्य:
            उत्थापय RuntimeError("Input ELF verification failed : " + elf_file_name)

        elf_fp.seek(0)

        यदि is_elf64(elf_शीर्षकम्):
            elf_शीर्षकम् = Elf64_Ehdr(elf_fp.पठ(Elf64_Ehdr.s.आकार))
        अन्यथा:
            elf_शीर्षकम् = Elf32_Ehdr(elf_fp.पठ(Elf32_Ehdr.s.आकार))

        shdr_सारणी = []

        shdr_आकार = elf_शीर्षकम्.e_shentsize
        elf_fp.seek(elf_शीर्षकम्.e_shoff)

        प्रति _ अन्तः परिसरः(elf_शीर्षकम्.e_shnum):
            यदि is_elf64(elf_शीर्षकम्):
                shdr_सारणी.append(Elf64_Shdr(elf_fp.पठ(shdr_आकार)))
            अन्यथा:
                shdr_सारणी.append(Elf32_Shdr(elf_fp.पठ(shdr_आकार)))

    प्रत्यागम shdr_सारणी


परिभाषा चक्रीयपरीक्षा_32_update_file(elf_fp, विस्थापनम्, आकार, crc):
    elf_fp.seek(विस्थापनम्)
    remaining = आकार
    यावत् remaining > 0:
        chunk = elf_fp.पठ(min(1024 * 1024, remaining))
        यदि न chunk:
            उत्थापय RuntimeError("Unexpected EOF while calculating CRC")
        crc = zlib.चक्रीयपरीक्षा_32(chunk, crc)
        remaining -= दीर्घता(chunk)
    प्रत्यागम crc


परिभाषा चक्रीयपरीक्षा_32_update_zeros(आकार, crc):
    zero_chunk = b'\x00' * min(1024 * 1024, आकार)
    remaining = आकार
    यावत् remaining > 0:
        chunk = zero_chunk[:min(दीर्घता(zero_chunk), remaining)]
        crc = zlib.चक्रीयपरीक्षा_32(chunk, crc)
        remaining -= दीर्घता(chunk)
    प्रत्यागम crc


परिभाषा चक्रीयपरीक्षा_32_calculate(elf_fp, phdr_सारणी):
    crc = 0

    # SWIV validates loadable segment contents, including BSS as zeros.
    प्रति curr_phdr अन्तः phdr_सारणी:
        यदि curr_phdr.p_type != आरोपय_TYPE:
            अनुवर्त

        crc = चक्रीयपरीक्षा_32_update_file(elf_fp, curr_phdr.p_विस्थापनम्, curr_phdr.p_filesz, crc)

        zero_आकार = curr_phdr.p_memsz - curr_phdr.p_filesz
        यदि zero_आकार > 0:
            crc = चक्रीयपरीक्षा_32_update_zeros(zero_आकार, crc)

    crc &= 0xffffffff
    logger.info("calculated CRC: %#x", crc)

    प्रत्यागम crc


परिभाषा verify_swiv_seg(elf_अन्तर्_fp, विस्थापनम्):

    elf_अन्तर्_fp.seek(विस्थापनम्)
    swiv_seg = SWIV_Seg()
    unpacked_दत्त = (SWIV_Seg.s).unpack(elf_अन्तर्_fp.पठ(SWIV_SEG_आकार_MIN))
    swiv_seg.magic = unpacked_दत्त[0]

    प्रत्यागम swiv_seg.magic == SWIV_MAGIC


परिभाषा same_output_file(निवेशः_elf, output_elf):
    यत्न:
        प्रत्यागम os.मार्गः.exists(output_elf) च os.मार्गः.samefile(निवेशः_elf, output_elf)
    अपवाद File_Not_प्राप्तम्_Error:
        प्रत्यागम असत्य


परिभाषा make_output_target(निवेशः_elf, output_elf):
    यदि न same_output_file(निवेशः_elf, output_elf):
        प्रत्यागम output_elf, किमपि_न

    output_dir = os.मार्गः.dirname(os.मार्गः.abspath(output_elf))
    output_आधारः = os.मार्गः.basename(output_elf)
    fd, अस्थायी_output = tempfile.mkstemp(उपसर्गः=output_आधारः + ".", suffix=".tmp", dir=output_dir)
    os.close(fd)
    shutil.copymode(निवेशः_elf, अस्थायी_output)
    प्रत्यागम अस्थायी_output, अस्थायी_output


परिभाषा swiv_generate(निवेशः_elf, output_elf):
    output_target, अस्थायी_output = make_output_target(निवेशः_elf, output_elf)

    यत्न:
        सह open(निवेशः_elf, "rb") इव elf_अन्तर्_fp, open(output_target, "wb+") इव elf_बहिर्गमनम्_fp:
            [elf_शीर्षकम्, phdr_सारणी] = preprocess_elf_file(निवेशः_elf)

            अङ्कः_phdrs = elf_शीर्षकम्.e_phnum
            अङ्कः_shdrs = elf_शीर्षकम्.e_shnum
            phdr_आकार = elf_शीर्षकम्.e_phentsize

            यदि अङ्कः_phdrs == 0 वा अङ्कः_phdrs > MAX_PHDR_गणनम्:
                उत्थापय RuntimeError("Input ELF has invalid number of program headers.")

            यदि elf_शीर्षकम्.e_shoff == 0 वा अङ्कः_shdrs == 0:
                उत्थापय RuntimeError("Input ELF does not contain a section header table.")

            shdr_सारणी_निवेशः = process_section_शीर्षकम्(निवेशः_elf)
            shdr_सारणी_output = process_section_शीर्षकम्(निवेशः_elf)

            चरमः_phdr = phdr_सारणी[अङ्कः_phdrs - 1]
            यदि चरमः_phdr.p_memsz == SWIV_SEG_आकार_MIN च verify_swiv_seg(elf_अन्तर्_fp, चरमः_phdr.p_विस्थापनम्):
                उत्थापय RuntimeError("A SWIV segment is detected in the input ELF.")

            चरमः_shdr = shdr_सारणी_निवेशः[elf_शीर्षकम्.e_shnum - 1]
            यदि चरमः_shdr.sh_आकार == SWIV_SEG_आकार_MIN च verify_swiv_seg(elf_अन्तर्_fp, चरमः_shdr.sh_विस्थापनम्):
                उत्थापय RuntimeError("A SWIV section is detected in the input ELF.")

            यदि is_elf64(elf_शीर्षकम्):
                swiv_shdr = Elf64_Shdr(b'\0' * Elf64_Shdr.s.आकार)
            अन्यथा:
                swiv_shdr = Elf32_Shdr(b'\0' * Elf32_Shdr.s.आकार)

            swiv_shdr.sh_name = 0
            swiv_shdr.sh_type = SWIV_SECTION_TYPE
            swiv_shdr.sh_flags = 0
            swiv_shdr.sh_addr = 0
            swiv_shdr.sh_आकार = SWIV_SEG_आकार_MIN
            swiv_shdr.sh_विस्थापनम् = elf_शीर्षकम्.e_shoff + (elf_शीर्षकम्.e_shnum + 1) * elf_शीर्षकम्.e_shentsize
            shdr_सारणी_output.append(swiv_shdr)
            elf_शीर्षकम्.e_shnum += 1

            यदि elf_शीर्षकम्.e_phoff != 0:
                प्रति phdr अन्तः phdr_सारणी:
                    file_copy_विस्थापनम्(elf_अन्तर्_fp, phdr.p_विस्थापनम्, elf_बहिर्गमनम्_fp, phdr.p_विस्थापनम्, phdr.p_filesz)

            elf_बहिर्गमनम्_fp.seek(0)
            elf_बहिर्गमनम्_fp.लिख(elf_शीर्षकम्.get_packed_दत्त())

            phdr_start = elf_शीर्षकम्.e_phoff
            प्रति phdr अन्तः phdr_सारणी:
                elf_बहिर्गमनम्_fp.seek(phdr_start)
                elf_बहिर्गमनम्_fp.लिख(phdr.get_packed_दत्त())
                phdr_start += phdr_आकार

            प्रति shdr अन्तः shdr_सारणी_निवेशः[:अङ्कः_shdrs]:
                यदि shdr.sh_type == SHT_NOBITS वा shdr.sh_आकार == 0:
                    अनुवर्त
                file_copy_विस्थापनम्(elf_अन्तर्_fp, shdr.sh_विस्थापनम्, elf_बहिर्गमनम्_fp, shdr.sh_विस्थापनम्, shdr.sh_आकार)

            shdr_start = elf_शीर्षकम्.e_shoff
            प्रति shdr अन्तः shdr_सारणी_output:
                elf_बहिर्गमनम्_fp.seek(shdr_start)
                elf_बहिर्गमनम्_fp.लिख(shdr.get_packed_दत्त())
                shdr_start += elf_शीर्षकम्.e_shentsize

            elf_बहिर्गमनम्_fp.flush()

            swiv_seg = SWIV_Seg()
            swiv_seg.crc = चक्रीयपरीक्षा_32_calculate(elf_बहिर्गमनम्_fp, phdr_सारणी)
            elf_बहिर्गमनम्_fp.seek(swiv_shdr.sh_विस्थापनम्)
            elf_बहिर्गमनम्_fp.लिख(swiv_seg.get_packed_दत्त())

        यदि अस्थायी_output अस्ति न किमपि_न:
            os.replace(अस्थायी_output, output_elf)
            अस्थायी_output = किमपि_न

        प्रत्यागम 0
    अन्ततः:
        यदि अस्थायी_output अस्ति न किमपि_न च os.मार्गः.exists(अस्थायी_output):
            os.unlink(अस्थायी_output)


परिभाषा मुख्यम्(argv):
    parser = argparse.तर्कः_Parser(description="Add a Software Image Validation (SWIV) section to an HTP ELF file.")
    parser.सङ्कलय_तर्कः("-i", "--input", dest="input_file", metavar="INPUT", required=सत्य)
    parser.सङ्कलय_तर्कः("-o", "--output", dest="output_file", metavar="OUTPUT", required=सत्य)
    parser.सङ्कलय_तर्कः("-v", "--verbose", action="store_true", help="show calculated CRC")
    तर्काः_2 = parser.parse_तर्काः(argv)

    logging.basicConfig(
        level=logging.INFO यदि तर्काः_2.verbose अन्यथा logging.WARNING,
        format="%(message)s",
    )

    यत्न:
        प्रत्यागम swiv_generate(तर्काः_2.निवेशः_file, तर्काः_2.output_file)
    अपवाद RuntimeError इव exc:
        logger.error("ERROR: %s", exc)
        प्रत्यागम 2


यदि __name__ == "__main__":
    sys.exit(मुख्यम्(sys.argv[1:]))
