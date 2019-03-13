import numpy as np

def batcher(filepath, batch_size, max_sequence_length=150):
    with open(filepath, "r") as file_obj:
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
