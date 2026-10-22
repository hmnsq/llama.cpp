#

درآمد sys
درآمد logging
logger = logging.getLogger("opencl-embed-kernel")


تعریف مین():
    logging.basicConfig(level=logging.INFO)

    اگر لمبائی(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    برائے i میں ifile:
        ofile.اكتب('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


اگر __name__ == "__main__":
    مین()
