RECTO_SUBSTRING = '_recto.cr2'
RECTO_MD5_SUBSTRING = '_recto.md5'
VERSO_SUBSTRING = '_verso.cr2'
VERSO_MD5_SUBSTRING = '_verso.md5'
SAMPLES_DIR = 'samples'
VISITED_LOG_FILE_NAME = 'processed.txt'

RECTO_CARDBOARD_DEFAULT_FILENAME = 'cardboard-re.png'
VERSO_CARDBOARD_DEFAULT_FILENAME = 'cardboard-ve.png'
IMAGE_DEFAULT_FILENAME = 'image.png'
TEXT_SECTION_DEFAULT_FILENAME = 'text-section.png'

# IMAGE AND TEXT SECTION CROPPING
ACCEPTABLE_TEXT_SECTIONS_Y_RANGES = [(1100, 1370)]
RESIZE_HEIGHT = 1000.0
IMAGE_HEIGHT_LIMIT = 0.9 * RESIZE_HEIGHT
IMAGE_WIDTH_DELIMITER = 0.05 * RESIZE_HEIGHT
IMAGE_MASK_BORDER_WIDTH = 15

# CARDBOARD CROPPING
CARDBOARD_MIN_WIDTH = 4400
CARDBOARD_MAX_WIDTH = 4720

CARDBOARD_MIN_HEIGHT = 5000
CARDBOARD_MAX_HEIGHT = 5300

CARDBOARD_MIN_RATIO = 1.10
CARDBOARD_MAX_RATIO = 1.15

