from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os, sys, yaml, time
TOP_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_DIR = os.path.dirname(TOP_DIR)
sys.path.insert(0, TOP_DIR)


def test_loader():
    # import
    import numpy as np
    from mlreco.iotools.factories import loader_factory
    # find config file
    cfg_file = sys.argv[1]
    if not os.path.isfile(cfg_file): cfg_file = os.path.join(TOP_DIR, 'config', sys.argv[1])
    if not os.path.isfile(cfg_file):
        print(sys.argv[1],'not found...')
        sys.exit(1)

    # check if quiet mode
    quiet = 'quiet' in sys.argv
    # check if csv should be made
    csv   = 'csv' in sys.argv
    if csv:
        from mlreco.utils.utils import CSVData
        csv=CSVData('csv.txt')
    # check if batch is specified (1st integer value in sys.argv)
    MAX_BATCH_ID=20
    for argv in sys.argv:
        if not argv.isdigit(): continue
        MAX_BATCH_ID=int(argv)
        break

    # configure
    cfg = yaml.load(open(cfg_file,'r'),Loader=yaml.Loader)
    loader,data_keys = loader_factory(cfg)
    if not quiet: print(len(loader),'batches loaded')
    if not quiet: print('keys:',data_keys)

    # Loop
    tstart=time.time()
    tsum=0.
    t0=0.
    for batch_id,data in enumerate(loader):
        titer=time.time() - tstart
        if not quiet:
            print('Batch',batch_id)
            for data_id in range(len(data_keys)):
                key = data_keys[data_id]
                print('   ',key,np.shape(data[data_id]))
            print(data[-1])
            print('Duration',titer,'[s]')
        if batch_id < 1:
            t0 = titer
        tsum += (titer)
        if csv:
            csv.record(['iter','t'],[batch_id,titer])
            csv.write()
        if (batch_id+1) == MAX_BATCH_ID:
            break
        tstart=time.time()
    if not quiet:
        print('Total time:',tsum,'[s] ... Average time:',tsum/MAX_BATCH_ID,'[s]')
        if MAX_BATCH_ID>1:
            print('First iter:',t0,'[s] ... Average w/o first iter:',(tsum - t0)/(MAX_BATCH_ID-1),'[s]')
    if csv: csv.close()
    return True
