#

आयात sys
आयात logging
logger = logging.getLogger("opencl-embed-kernel")


def main():
    logging.basicConfig(level=logging.INFO)

    जर len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    साठी i मध्ये ifile:
        ofile.लिख('R"({})"\n'.format(i))

    ifile.बंद()
    ofile.बंद()


जर __name__ == "__main__":
    main()
