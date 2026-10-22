#

आनयन sys
आनयन logging
logger = logging.getLogger("opencl-embed-kernel")


def main():
    logging.basicConfig(level=logging.INFO)

    सचे len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    अत्थाय i अन्तः ifile:
        ofile.लिख('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


सचे __name__ == "__main__":
    main()
