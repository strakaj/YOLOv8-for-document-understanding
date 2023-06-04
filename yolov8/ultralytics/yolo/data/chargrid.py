import numpy as np

characters = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5',
              '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
              'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', 'a', 'b',
              'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
              'x', 'y', 'z', '|', '£', '°', 'À', 'Ç', 'È', 'É', 'Ô', 'à', 'â', 'ç', 'è', 'é', 'ê', 'ô', 'û', '€']


def create_encodings_three_digit_0(characters):
    # background filled with 0
    characters = ["", *characters]
    number_digits = 3
    base = np.ceil(np.cbrt(len(characters))).astype(int)
    print(f"base: {base}, number of digits {number_digits}, number of characters {len(characters)}")

    encodings = {c: np.base_repr(i, base=base).zfill(number_digits) for i, c in enumerate(characters)}
    encodings = {c: np.array([int(v) for v in e]) for c, e in encodings.items()}
    encodings = {c: e / (base - 1) for c, e in encodings.items()}

    return encodings


def create_encodings_three_digit_1(characters):
    # background filled with 1
    num_characters = len(characters) + 1
    number_digits = 3
    base = np.ceil(np.cbrt(num_characters)).astype(int)
    print(f"base: {base}, number of digits {number_digits}, number of characters {num_characters}")

    encodings = {c: np.base_repr(i, base=base).zfill(number_digits) for i, c in enumerate(characters)}
    encodings = {c: np.array([int(v) for v in e]) for c, e in encodings.items()}
    encodings = {c: e / (base - 1) for c, e in encodings.items()}
    encodings[""] = np.array([1.0, 1.0, 1.0])

    return encodings


def create_encodings_one_digit(characters):
    characters = ["", *characters]
    num_characters = len(characters)
    encodings = {c: i / num_characters for i, c in enumerate(characters)}

    return encodings


def get_char_grid(fields, shape, encodings):
    if list(encodings.keys())[0] == "":
        char_grid = np.zeros([*shape, 3])
    else:
        char_grid = np.ones([*shape, 3])

    if isinstance(list(encodings.values())[0], float):
        char_grid = np.zeros([*shape, 1])

    ih, iw = shape
    for field in fields:
        text = field.text
        if not text.strip():
            continue
        x0, y0, x1, y1 = field.bbox.to_absolute_coords(iw, ih).to_tuple()
        num_chars = len(text)
        w = x1 - x0
        step = w / num_chars

        x_start_idx = np.floor(np.arange(num_chars) * step + x0).astype(int)
        x_end_idx = np.ceil((np.arange(num_chars) + 1) * step + x0).astype(int)

        for i, c in enumerate(text):
            if c not in encodings:
                print(f"{c} not in encodings")
                enc = encodings[""]
            else:
                enc = encodings[c]
            char_grid[y0:y1, x_start_idx[i]:x_end_idx[i]] = enc
    return char_grid
