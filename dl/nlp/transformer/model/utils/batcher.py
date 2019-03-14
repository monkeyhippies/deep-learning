import numpy as np

def batcher(filepath, batch_size, max_sequence_length=150, offset=0):

    with open(filepath, "r") as file_obj:
 
        # Skip the first "offset" rows
        for i in range(offset):
            file_obj.readline()

        for batch in file_batcher(
            file_obj,
            batch_size,
            max_sequence_length
        ):
            yield batch
\
    while True:
        with open(filepath, "r") as file_obj:
            for batch in file_batcher(
                file_obj,
                batch_size,
                max_sequence_length
            ):
                yield batch
def file_batcher(file_obj, batch_size, max_sequence_length):

    batch=[]
    for line in file_obj:
        line = np.array(
            line.strip().split(),
            dtype="uint16"
        )[: max_sequence_length]
        line = np.pad(
            line,
            (0, max_sequence_length - len(line)),
            "constant"
        )
        batch.append(line)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
