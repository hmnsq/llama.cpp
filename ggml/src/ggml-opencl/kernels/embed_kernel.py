#

ಆನಯನ sys
ಆನಯನ logging
logger = logging.getLogger("opencl-embed-kernel")


def main():
    logging.basicConfig(level=logging.INFO)

    ಯದಿ len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    ಪ್ರತಿ i ಅನ್ತಃ ifile:
        ofile.ಲಿಖ('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


ಯದಿ __name__ == "__main__":
    main()
