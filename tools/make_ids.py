"""The sweep's input: a prose prompt's token ids (u32 .npy) through the checkpoint's own tokenizer.
    python tools/make_ids.py MODEL_DIR OUT.npy
Needs `tokenizers` and `numpy`."""
import sys

import numpy as np
import tokenizers

TEXT = ("The printing press, introduced by Johannes Gutenberg around 1440, transformed European society in ways "
        "that few inventions before or since have matched. Before movable type, books were copied by hand, ") * 6

model_dir, out = sys.argv[1], sys.argv[2]
tok = tokenizers.Tokenizer.from_file(f"{model_dir}/tokenizer.json")
ids = np.array(tok.encode(TEXT).ids[:256], dtype=np.uint32)
np.save(out, ids)
print(f"{len(ids)} ids -> {out}")
