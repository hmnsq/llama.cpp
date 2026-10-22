#

استورد sys
استورد logging
logger = logging.getLogger("opencl-embed-kernel")


def main():
    logging.basicConfig(level=logging.INFO)

    אם len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    עבור i في ifile:
        ofile.اكتب('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


אם __name__ == "__main__":
    main()
