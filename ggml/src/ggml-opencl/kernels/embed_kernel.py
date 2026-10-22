#

import sys
import logging
logger = logging.getLogger("opencl-embed-kernel")


def main():
    logging.basicConfig(level=logging.INFO)

    만약 len(sys.argv) != 3:
        logger.info("Usage: python embed_kernel.py <input_file> <output_file>")
        sys.exit(1)

    ifile = open(sys.argv[1], "r")
    ofile = open(sys.argv[2], "w")

    위해 i in ifile:
        ofile.write('R"({})"\n'.format(i))

    ifile.close()
    ofile.close()


만약 __name__ == "__main__":
    main()
