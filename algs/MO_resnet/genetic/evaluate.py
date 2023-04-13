import time
from comm.utils import CacheToResultFile
from algs.MO_resnet.utils import Utils
from compute.process import dispatch_to_do
from compute.gpu import gpus_all_available

class FitnessEvaluate(object):

    def __init__(self, individuals, log):
        self.individuals = individuals
        self.log = log

    def generate_to_python_file(self):
        self.log.info('Begin to generate python files')
        for indi in self.individuals:
            Utils.generate_pytorch_file(indi)
        self.log.info('Finish the generation of python files')

    def evaluate(self):
        """
        load fitness from cache file
        """
        self.log.info('Query fitness from cache')
        _map = Utils.load_cache_data()
        _count = 0
        for indi in self.individuals:
            _key, _str = indi.uuid()
            if _key in _map:
                _count += 1
                _error, _loss = _map[_key][0], _map[_key][1]
                self.log.info('Hit the cache for %s, key:%s, error:%.5f, loss:%.5f, assigned_error:%.5f'%(indi.id, _key, _error, _loss, indi.error_mean))
                CacheToResultFile.do(indi.id, _error, _loss)
                indi.error_mean = _error
                indi.loss_mean = _loss
        self.log.info('Total hit %d individuals for fitness'%(_count))

        for indi in self.individuals:
            if indi.error_mean < 0:
                _id = indi.id
                _uuid, _ = indi.uuid()
                dispatch_to_do(_id, _uuid)
                
        all_have_been_evaluated = False
        while all_have_been_evaluated is not True:
            #print('All have been evaluated flag ', all_have_been_evaluated)
            time.sleep(120)
            all_have_been_evaluated = gpus_all_available()
        
        # set the fitness values to each of individual  
