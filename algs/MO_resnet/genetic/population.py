import numpy as np
import hashlib
import copy
from algs.MO_resnet.genetic.statusupdatetool import StatusUpdateTool


class Unit(object):
    def __init__(self, number):
        self.number = number


class ConvUnit(Unit):
    def __init__(self, number=0, filter_width=0, filter_height=0, in_channel=0, out_channel=0, stride_width=0, stride_height=0,
                 conv_type=0, mean=0, std=0):
        super().__init__(number)
        self.type = 1
        self.filter_size = filter_width, filter_height
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride_size = stride_width, stride_height
        self.conv_type = conv_type  # 0 denotes VALID, 1 denotes SAME #
        self.mean = mean
        self.std = std

    def __parse__(self, a_list):
        self.type = a_list[0]
        self.filter_size = a_list[1][0], a_list[1][1]
        self.in_channel = a_list[2]
        self.out_channel = a_list[3]
        self.stride_size = a_list[4][0], a_list[4][1]
        self.mean = 0
        self.std = 1
        self.conv_type = 1

    def __list__(self):
        return [self.type, self.filter_size, self.in_channel, self.out_channel, self.stride_size]


class PoolUnit(Unit):
    def __init__(self, number=0, max_or_avg=0, kernel_width=0, kernel_height=0, stride_width=0, stride_height=0):
        super().__init__(number)
        self.type = 2
        self.kernel_size = kernel_width, kernel_height
        self.stride_size = stride_width, stride_height
        self.max_or_avg = max_or_avg  # max_pool for < 0.5 otherwise avg_pool

    def __parse__(self, a_list):
        self.type = a_list[0]
        self.kernel_size = a_list[1][0], a_list[1][1]
        self.max_or_avg = a_list[2]
        self.stride_size = a_list[3][0], a_list[3][1]

    def __list__(self):
        return [self.type, self.kernel_size, self.max_or_avg, self.stride_size]


class FcUnit(Unit):
    def __init__(self, number=0, input_neurons_number=0, output_neurons_number=0, mean=0, std=0):
        super().__init__(number)
        self.type = 3
        self.input_neurons_number = input_neurons_number
        self.output_neurons_number = output_neurons_number
        self.mean = mean
        self.std = std

    def __parse__(self, a_list):
        self.type = a_list[0]
        self.input_neurons_number = a_list[1]
        self.output_neurons_number = a_list[2]
        self.mean = 0
        self.std = 1

    def __list__(self):
        return [self.type, self.input_neurons_number, self.output_neurons_number]


class Individual(object):
    def __init__(self, params, indi_no):
        self.error_mean = -1.0
        self.loss_mean = -1.0
        self.acc_std = 0.0
        self.complexity = 0
        self.id = indi_no  # for record the id of current individual
        self.number_id = 0  # for record the latest number of basic unit

        self.image_channel = params['image_channel']
        self.min_layer = params['min_layer']  # minimal number of convolution units
        self.max_layer = params['max_layer']  # maximal number of convolution units
        self.min_conv = params['min_conv']  # minimal number of convolution units
        self.max_conv = params['max_conv']  # maximal number of convolution units
        self.min_pool = params['min_pool']  # minimal number of pool units
        self.max_pool = params['max_pool']  # maximal number of pool units
        self.min_fc = params['min_fc']  # minimal number of full connected units
        self.max_fc = params['max_fc']  # maximal number of full connected units
        self.min_std = params['min_std']  # minimal std
        self.max_std = params['max_std']  # maximal std
        self.min_mean = params['min_mean']  # minimal mean
        self.max_mean = params['max_mean']  # maximal mean

        self.min_conv_filter_size = params['conv_filter_size_min']  # the minimal filter size of convolution
        self.max_conv_filter_size = params['conv_filter_size_max']  # the maximal filter size of convolution
        self.min_channel = params['min_channel']  # the min out channel of the convolution unit
        self.max_channel = params['max_channel']  # the max out channel of the convolution unit
        self.pool_kernel_size_list = params['pool_kernel_size_list']  # the kernel size list of pool
        # this is not the true kernel size of the pool, and it's an exponent of 2
        self.min_hidden_neurons = params['min_hidden_neurons']  # the min number of hidden neurons
        self.max_hidden_neurons = params['max_hidden_neurons']  # the max number of hidden neurons

        self.units = []

    def __list__(self):
        if hasattr(self, 'ulist'):
            return self.ulist
        result_list = []
        for one in self.units:
            if isinstance(one,list):
                per = []
                for i in one:
                    per.append(i.__list__())
                result_list.append(per)
            else:
                result_list.append(one.__list__())
        return result_list

    def reset_acc(self):
        self.error_mean = -1.0
        self.loss_mean = -1.0
        self.acc_std = 0.0
        self.complexity = 0

    def initialize(self):
        # initialize how many resnet unit/pooling layer/densenet unit will be used
        num_layer = np.random.randint(self.min_layer, self.max_layer + 1)
        num_all_pool = np.random.randint(self.min_pool, self.max_pool + 1)
        while True:
            exact_pool = []
            for _ in range(num_all_pool):
                exact_pool.append(self.init_a_pool())
            img_size = StatusUpdateTool.get_input_size()
            flag = 1
            for pool in exact_pool:
                img_size = int(img_size / pool.kernel_size[0])
                if img_size < 1:
                    flag = 0
                    break
            if flag == 1:
                break
        wh_pool = []
        for _ in range(num_layer):
            wh_pool.append(0)
        for _ in range(num_all_pool):
            wh = np.random.randint(0, num_layer)
            wh_pool[wh] += 1
        now_pool = 0
        img_size = StatusUpdateTool.get_input_size()
        input_channel = self.image_channel
        for layer_index in range(num_layer):
            one_block = []
            num_conv = np.random.randint(self.min_conv, self.max_conv + 1)
            num_pool = wh_pool[layer_index]
            type_arr = []
            for i in range(num_conv + num_pool - 1):
                if i < num_conv - 1:
                    type_arr.append(0)
                else:
                    type_arr.append(1)
            import random
            random.shuffle(type_arr)
            type_arr.insert(0,0)
            flag = 1
            for t in type_arr:
                if 0 == t:
                    # generate a conv unit
                    conv = self.init_a_conv(_in_channel=input_channel)
                    input_channel = conv.out_channel
                    one_block.append(conv)
                if 1 == t and 1 == flag:
                    # generate a pool unit
                    pool = exact_pool[now_pool]
                    now_pool += 1
                    img_size = int(img_size / pool.kernel_size[0])
                    one_block.append(pool) # default kernel_size[0] = kernel_size[1]
            self.units.append(one_block)
        input_channel = input_channel * img_size ** 2
        num_fc = np.random.randint(self.min_fc - 1, self.max_fc)  # because the last unit must be a fc unit
        for _ in range(num_fc):
            fc = self.init_a_fc(_input_neurons_number=input_channel)
            input_channel = fc.output_neurons_number
            self.units.append(fc)
        last_fc = self.init_a_fc(_number=None, _input_neurons_number=input_channel, _output_neurons_number=None,
                                 is_last=True, _mean=None, _std=None)
        self.units.append(last_fc)

    def init_a_conv(self, _number=None, _filter_width=None, _filter_height=None, _in_channel=None, _out_channel=None,
                    _stride_width=None, _stride_height=None, _conv_type=None, _mean=None, _std=None):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if _filter_width:
            filter_width = _filter_width
        else:
            filter_width = np.random.randint(self.min_conv_filter_size, self.max_conv_filter_size + 1)
        if _filter_height:
            filter_height = _filter_height
        else:
            filter_height = filter_width
        if _out_channel:
            out_channel = _out_channel
        else:
            out_channel = np.random.randint(self.min_channel, self.max_channel)
        if _stride_width:
            stride_width = _stride_width
        else:
            stride_width = 1  # default stride is one
        if _stride_height:
            stride_height = _stride_height
        else:
            stride_height = 1  # default stride is one
        if _conv_type:
            conv_type = _conv_type
        else:
            conv_type = 1  # default SAME
        if _mean:
            mean = _mean
        else:
            mean = self.min_mean + np.random.random() * (self.max_mean - self.min_mean)
        if _std:
            std = _std
        else:
            std = self.min_std + np.random.random() * (self.max_std - self.min_std)
        conv = ConvUnit(number, filter_width, filter_height, _in_channel, out_channel, stride_width, stride_height,
                        conv_type, mean, std)
        return conv

    def init_a_pool(self, _number=None, _max_or_avg=None, _kernel_width=None, _kernel_height=None, _stride_width=None,
                    _stride_height=None):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if _max_or_avg:
            max_or_avg = _max_or_avg
        else:
            max_or_avg = np.random.rand()
        if _kernel_width:
            kernel_width = _kernel_width
        else:
            kernel_width = np.power(2, self.pool_kernel_size_list[np.random.randint(len(self.pool_kernel_size_list))])
        if _kernel_height:
            kernel_height = _kernel_height
        else:
            kernel_height = kernel_width  # by default
        if _stride_width:
            stride_width = _stride_width
        else:
            stride_width = kernel_width  # by default
        if _stride_height:
            stride_height = _stride_height
        else:
            stride_height = kernel_width  # by default
        pool = PoolUnit(number, max_or_avg, kernel_width, kernel_height, stride_width, stride_height)
        return pool

    def init_a_fc(self, _number=None, _input_neurons_number=None, _output_neurons_number=None, _mean=None, _std=None,
                  is_last=None):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if is_last:
            output_neurons_number = StatusUpdateTool.get_num_class()
        else:
            if _output_neurons_number:
                output_neurons_number = _output_neurons_number
            else:
                output_neurons_number = self.min_hidden_neurons + np.random.random() * (
                        self.max_hidden_neurons - self.min_hidden_neurons)
        if _mean:
            mean = _mean
        else:
            mean = self.min_mean + np.random.random() * (self.max_mean - self.min_mean)
        if _std:
            std = _std
        else:
            std = self.min_std + np.random.random() * (self.max_std - self.min_std)
        fc = FcUnit(number, _input_neurons_number, output_neurons_number, mean, std)
        return fc

    def get_conv_number(self):
        number = 0
        for unit in self.units:
            if isinstance(unit, list):
                for one in unit:
                    if one.type == 1:
                        number += 1
            else:
                break
        return number

    def get_pool_number(self):
        number = 0
        for unit in self.units:
            if isinstance(unit, list):
                for one in unit:
                    if one.type == 2:
                        number += 1
            else:
                break
        return number

    def get_fc_number(self):
        number = 0
        for unit in self.units:
            if isinstance(unit, list):
                continue
            else:
                number += 1
        return number

    @classmethod
    def get_last_output_channel(cls, pos, indi_units):
        '''
        the position varies between [0,len(indi_units)], and 0 denotes the input channel of the individual,
        1 denotes the output channel after one unit
        :param pos: position
        :param indi_units: a units list like self.units
        :return: last output channel before pos
        '''
        last_output_channel = 0
        if pos == 0:
            last_output_channel = StatusUpdateTool.get_input_channel()
        else:
            for i in range(pos - 1, -1, -1):
                if indi_units[i].type == 1 or indi_units[i].type == 3:
                    last_output_channel = indi_units[i].out_channel
                    break
        assert last_output_channel  # return not equal to 0
        return last_output_channel

    @classmethod
    def calculate_complexity(cls, indi_units):
        return 0
        '''
        current_img_size = StatusUpdateTool.get_input_size()
        num_connections = 0
        last_output_feature_map_size = StatusUpdateTool.get_input_channel()
        for i in range(len(indi_units)):
            if indi_units[i].type == 1:
                last_output_feature_map_size = indi_units[i].out_channel
                num_connections += indi_units[i].out_channel * current_img_size ** 2 + indi_units[i].out_channel
            elif indi_units[i].type == 2:
                num_connections += last_output_feature_map_size
                current_img_size = current_img_size / indi_units[i].kernel_size[0]
            else:  # indi_units[i].type == 3
                num_connections += indi_units[i].input_neurons_number * indi_units[i].output_neurons_number + \
                                   indi_units[i].output_neurons_number

        return num_connections
        '''
    @classmethod
    def update_all_channels(cls, indi_units, type, log):
        '''
        update the channels of all the units, and update the stride size of pool unit
        :param indi_units: a units list like self.units
        :param type: 0 denotes crossover, 1 denotes mutation
        :param log: log in crossover and mutation
        :return: updated indi_units
        '''
        input_channel = StatusUpdateTool.get_input_channel()
        is_the_first_fc = True
        shrink = 1
        for i in range(len(indi_units)):
            if isinstance(indi_units[i], list):
                for j in range(len(indi_units[i])):
                    if indi_units[i][j].type == 1:
                        indi_units[i][j].in_channel = input_channel
                        # generate log
                        if type == 0:
                            log.info('Due to the above crossover, unit at %d changes its input channel from %d to %d' % (
                                i, indi_units[i][j].in_channel, input_channel))
                        else:
                            log.info('Due to the above mutation, unit at %d changes its input channel from %d to %d' % (
                                i, indi_units[i][j].in_channel, input_channel))
                        input_channel = indi_units[i][j].out_channel
                    elif indi_units[i][j].type == 2:
                        shrink = shrink * indi_units[i][j].kernel_size[0]
                        indi_units[i][j].stride_size = indi_units[i][j].kernel_size
            else:
                if is_the_first_fc:
                    input_channel = input_channel * (int(StatusUpdateTool.get_input_size() / shrink) ** 2)
                    is_the_first_fc = False
                indi_units[i].input_neurons_number = input_channel
                # generate log
                if type == 0:
                    log.info('Due to the above crossover, unit at %d changes its input channel from %d to %d' % (
                        i, indi_units[i].input_neurons_number, input_channel))
                else:
                    log.info('Due to the above mutation, unit at %d changes its input channel from %d to %d' % (
                        i, indi_units[i].input_neurons_number, input_channel))
                input_channel = indi_units[i].output_neurons_number
        return indi_units

    @classmethod
    def update_all_channels_in_mutation(cls, indi_units, type, log):
        '''
        update the channels of all the units, and update the stride size of pool unit
        :param indi_units: a units list like self.units
        :param type: 0 denotes crossover, 1 denotes mutation
        :param log: log in crossover and mutation
        :return: updated indi_units
        '''
        input_channel = StatusUpdateTool.get_input_channel()
        is_the_first_fc = True
        shrink = 1
        for i in range(len(indi_units)):
            if indi_units[i].type == 1:
                indi_units[i].in_channel = input_channel
                # generate log
                if type == 0:
                    log.info('Due to the above crossover, unit at %d changes its input channel from %d to %d' % (
                        i, indi_units[i].in_channel, input_channel))
                else:
                    log.info('Due to the above mutation, unit at %d changes its input channel from %d to %d' % (
                        i, indi_units[i].in_channel, input_channel))
                input_channel = indi_units[i].out_channel
            elif indi_units[i].type == 2:
                shrink = shrink * indi_units[i].kernel_size[0]
                indi_units[i].stride_size = indi_units[i].kernel_size
            elif indi_units[i].type == 3:
                if is_the_first_fc:
                    input_channel = input_channel * (int(StatusUpdateTool.get_input_size() / shrink) ** 2)
                    is_the_first_fc = False
                indi_units[i].input_neurons_number = input_channel
                # generate log
                if type == 0:
                    log.info('Due to the above crossover, unit at %d changes its input channel from %d to %d' % (
                        i, indi_units[i].input_neurons_number, input_channel))
                else:
                    log.info('Due to the above mutation, unit at %d changes its input channel from %d to %d' % (
                        i, indi_units[i].input_neurons_number, input_channel))
                input_channel = indi_units[i].output_neurons_number
        return indi_units

    def uuid(self):
        _final_str_ = self.__str__()
        _final_utf8_str_ = _final_str_.encode('utf-8')
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()
        return _hash_key, _final_str_

    def __str__(self):
        _str = []
        _str.append('indi:%s' % (self.id))
        _str.append('Error_mean:%.5f' % (self.error_mean))
        _str.append('Loss_mean:%.5f' % (self.loss_mean))
        _str.append('Acc_std:%.5f' % (self.acc_std))
        _str.append('Complexity:%d' % (self.complexity))
        _str.append('%s' % self.__list__())
        return '\n'.join(_str)


class Population(object):
    def __init__(self, params, gen_no):
        self.gen_no = gen_no
        self.number_id = 0  # for record how many individuals have been generated
        self.pop_size = params['pop_size']
        self.params = params
        self.individuals = []
        self.offsprings = []
        self.parent_individuals = []

    def initialize(self, log):
        for _ in range(self.pop_size):
            indi_no = 'indi%05d_%05d' % (self.gen_no, self.number_id)
            self.number_id += 1
            indi = Individual(self.params, indi_no)
            indi.initialize()
            indi.complexity = Individual.calculate_complexity(indi.units)
            log.info('indi_no:%s, complexity=%d' % (indi_no, indi.complexity))
            self.individuals.append(indi)

    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%05d_%05d' % (self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id += 1
            indi.number_id = len(indi.units)
            self.individuals.append(indi)

    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append('-' * 100)
        return '\n'.join(_str)


def ttest_individual(params):
    ind = Individual(params, 0)
    ind.initialize()
    print(ind)
    print(ind.uuid())


def ttest_population(params):
    pop = Population(params, 0)
    from compute.redis import RedisLog
    import os
    pop.initialize(RedisLog(os.path.basename("test") + '.txt'))
    print(pop)


if __name__ == '__main__':
    # ttest_individual(StatusUpdateTool.get_init_params())
    ttest_population(StatusUpdateTool.get_init_params())
