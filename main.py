from split_raw_data import *

import os
from pathlib import Path

import numpy as np
import time

def main():
    currentDir = Path(os.getcwd())
    datasetDir = str(Path(currentDir))

    data_path = Path('data')
    if not data_path.exists():
        data_path.mkdir()

    hemorrhage_diagnosis_df = pd.read_csv(
        Path(datasetDir, 'hemorrhage_diagnosis_raw_ct.csv'))

    result_path = data_path / 'hemorrhage_segment_results.csv'
    hemorrhage_diagnosis_df.to_csv(result_path, index=False)

    params = {1: [(512, 512), [40, 80]], 2: [(512, 512), [40, 120]], 3: [(256,
                                                                          256),
                                                                         [40,
                                                                          80]],
              4: [(256,
                   256),
                  [40,
                   120]]}

    for param in params.values():
        new_size = param[0]
        window_specs = param[1]
        read_ct_scans(new_size, window_specs)





if __name__ == '__main__':
    print('Process starting...')
    start_time = time.time()
    main()
    print('Process ending!')
    print("--- %s seconds ---" % (time.time() - start_time))
