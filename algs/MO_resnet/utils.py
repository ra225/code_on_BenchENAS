import os
import numpy as np
import time
from algs.MO_resnet.genetic.population import Population, Individual, ConvUnit, FcUnit, PoolUnit
from compute.file import get_algo_local_dir
from comm.log import Log
import platform
from algs.MO_resnet.genetic.statusupdatetool import StatusUpdateTool


class Utils(object):
    @classmethod
    def get_lock_for_write_fitness(cls):
        return cls._lock

    @classmethod
    def path_replace(cls, input_str):
        # input a str, replace '\\' with '/', because the os.path in windows return path with '\\' joining
        # please use it after creating a string with both os.path and string '/'
        if (platform.system() == 'Windows'):
            new_str = input_str.replace('\\', '/')
        else:  # Linux or Mac
            new_str = input_str
        return new_str

    @classmethod
    def load_EP(cls, prefix, gen_no):
        file_name = '%s/%s_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), prefix, np.min(gen_no))
        file_name = cls.path_replace(file_name)
        if os.path.exists(file_name):
            f = open(file_name, 'r')
            for each_line in f:
                return eval(each_line)
            f.close()


    @classmethod
    def load_cache_data(cls):
        file_name = '%s/cache.txt' % (os.path.join(get_algo_local_dir(), 'populations'))
        file_name = cls.path_replace(file_name)
        _map = {}
        if os.path.exists(file_name):
            f = open(file_name, 'r')
            for each_line in f:
                rs_ = each_line.strip().split(';')
                _map[rs_[0]] = []
                _map[rs_[0]].append(float(rs_[1]))
                _map[rs_[0]].append(float(rs_[2]))
            f.close()
        return _map

    @classmethod
    def save_fitness_to_cache(cls, individuals):
        _map = cls.load_cache_data()
        for indi in individuals:
            _key, _str = indi.uuid()
            _acc = indi.acc_mean
            if _key not in _map:
                Log.debug('Add record into cache, id:%s, acc:%.5f' % (_key, _acc))
                file_name = '%s/cache.txt' % (os.path.join(get_algo_local_dir(), 'populations'))
                file_name = cls.path_replace(file_name)
                f = open(file_name, 'a+')
                _str = '%s;%.5f;%s\n' % (_key, _acc, _str)
                f.write(_str)
                f.close()
                _map[_key] = _acc

    @classmethod
    def save_EP_after_evaluation(cls, _str, gen_no):
        file_name = '%s/EP_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), gen_no)
        file_name = cls.path_replace(file_name)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_at_begin(cls, _str, gen_no):
        file_name = '%s/begin_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), gen_no)
        # solve the path differences caused by different platforms
        file_name = cls.path_replace(file_name)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_after_crossover(cls, _str, gen_no):
        file_name = '%s/crossover_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), gen_no)
        file_name = cls.path_replace(file_name)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def save_population_after_mutation(cls, _str, gen_no):
        file_name = '%s/mutation_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), gen_no)
        file_name = cls.path_replace(file_name)
        with open(file_name, 'w') as f:
            f.write(_str)

    @classmethod
    def get_newest_file_based_on_prefix(cls, prefix):
        id_list = []
        for _, _, file_names in os.walk(os.path.join(get_algo_local_dir(), 'populations')):
            for file_name in file_names:
                if file_name.startswith(prefix):
                    number_index = len(prefix) + 1  # the first number index
                    id_list.append(int(file_name[number_index:number_index + 5]))
        if len(id_list) == 0:
            return None
        else:
            return np.max(id_list)

    @classmethod
    def load_population(cls, prefix, gen_no):
        file_name = '%s/%s_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), prefix, np.min(gen_no))
        file_name = cls.path_replace(file_name)
        params = StatusUpdateTool.get_init_params()
        pop = Population(params, gen_no)
        f = open(file_name)
        indi_start_line = f.readline().strip()
        while indi_start_line.startswith('indi'):
            indi_no = indi_start_line[5:]
            indi = Individual(params, indi_no)
            for line in f:
                line = line.strip()
                if line.startswith('--'):
                    indi_start_line = f.readline().strip()
                    break
                else:
                    if line.startswith('Error_mean'):
                        indi.error_mean = float(line[11:])
                    elif line.startswith('Loss_mean'):
                        indi.loss_mean = float(line[10:])
                    elif line.startswith('Acc_std'):
                        indi.acc_std = float(line[8:])
                    elif line.startswith('Complexity'):
                        indi.complexity = int(line[11:])
                    elif line.startswith('['):
                        indi.ulist = eval(line)
                        for one_block in indi.ulist:
                            sub_list = []
                            for one_unit in one_block:
                                if isinstance(one_unit, list):
                                    if one_unit[0] == 1:
                                        new_unit = ConvUnit()
                                        new_unit.__parse__(one_unit)
                                    else:
                                        new_unit = PoolUnit()
                                        new_unit.__parse__(one_unit)
                                    sub_list.append(new_unit)
                                else:
                                    new_unit = FcUnit()
                                    new_unit.__parse__(one_block)
                                    sub_list = new_unit
                                    break
                            indi.units.append(sub_list)
                    else:
                        print('Unknown key for load unit type, line content:%s' % (line))
            pop.individuals.append(indi)
        f.close()

        return pop

    @classmethod
    def read_template(cls):
        _path = os.path.join(os.path.dirname(__file__), 'template', 'model_template.py')
        part1 = []
        part2 = []
        part3 = []

        f = open(_path)
        f.readline()  # skip this comment
        line = f.readline().rstrip()
        while line.strip() != '# generated_init':
            part1.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part1))

        line = f.readline().rstrip()  # skip the comment '#generated_init'
        while line.strip() != '# generate_forward':
            part2.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part2))

        line = f.readline().rstrip()  # skip the comment '#generate_forward'
        while line.strip() != '"""':
            part3.append(line)
            line = f.readline().rstrip()
        # print('\n'.join(part3))
        return part1, part2, part3

    @classmethod
    def generate_pytorch_file(cls, indi):
        # query resnet and densenet unit
        unit_list = []
        indi_list = indi.__list__()
        index = -1
        image_output_size = StatusUpdateTool.get_input_size()
        last_size = image_output_size
        num_conv = 0
        last_channels = 0
        first_block = True
        for u0 in indi_list: #a resnet block/a full layer
            next_size = last_size
            first_channels = last_channels
            for u in u0: #a conv or pool layer in resnet block/ a number in full layer
                index += 1
                if isinstance(u, list):
                    if u[0] == 1:
                        num_conv += 1
                        pad = 'self.pad%d = SamePad2d(kernel_size=(%d, %d), stride=(%d, %d))' % (
                            index, u[1][0], u[1][1], u[4][0], u[4][1])
                        unit_list.append(pad)
                        layer = 'self.op%d = nn.Conv2d(in_channels=%d, out_channels=%d, kernel_size=(%d, %d), stride=(%d, %d), padding=0)' % (
                            index, u[2], u[3], u[1][0], u[1][1], u[4][0],
                            u[4][1])
                        last_channels = u[3]
                        unit_list.append(layer)
                        norm = 'self.norm%d = nn.BatchNorm2d(%d)' % (index, u[3])
                        unit_list.append(norm)
                        #relu = 'self.relu%d = nn.ReLU(inplace=False)' % (index)
                        #unit_list.append(relu)
                    if u[0] == 2:
                        next_size = int(next_size / u[1][0])
                else:
                   # unit_list.pop()
                    layer = 'self.op%d = nn.Linear(in_features=%d, out_features=%d)' % (
                        index, u0[1], u0[2])
                    unit_list.append(layer)
                    norm = 'self.norm%d = nn.BatchNorm1d(%d)' % (index, u0[2])
                    unit_list.append(norm)
                    last_size = next_size
                    break
            if first_block:
                first_channels = u0[0][2]
                first_block = False
            if isinstance(u, list) and (first_channels!=last_channels or last_size != next_size):
                downsample = 'self.downsample%d = nn.Sequential( \
                      nn.Conv2d(%d, %d, kernel_size=1, stride=%d, bias=False), \
                      nn.BatchNorm2d(%d), \
                  )' % (index, first_channels, last_channels, int(last_size/next_size), last_channels)
                unit_list.append(downsample)

        # print('\n'.join(unit_list))

        # query fully-connect layer
        '''
        out_channel_list = []
        out_channel_list.append(StatusUpdateTool.get_input_channel())
        image_output_size = StatusUpdateTool.get_input_size()
        for u0 in indi_list:
            for u in u0:
                if isinstance(u, list):
                    if u[0] == 1:
                        out_channel_list.append(u[3])
                elif u.type == 3:
                    out_channel_list.append(u.output_neurons_number)
                else:
                    out_channel_list.append(out_channel_list[-1])
                    image_output_size = int(image_output_size / u.kernel_size[
                        0])  # default kernel_size = stride_size, and kernel_size[0] = kernel_size[1]
                        '''
        # print(fully_layer_name, out_channel_list, image_output_size)

        # generate the forward part
        forward_list = []

        i = -1
        is_first_fc = True
        for u0 in indi_list:#a resnet block/a full layer
            first = 1
            has_pool = False
            for u in u0:#a conv or pool layer in resnet block/ a number in full layer
                i += 1
                if i == 0:
                    last_out_put = 'x'
                else:
                    last_out_put = 'out_%d' % (i - 1)
                if isinstance(u, list):
                    if first == 1:
                        _str = 'identity_%d = %s' % (i, last_out_put)
                        identity_id = i
                        forward_list.append(_str)
                        last_downsample_out = last_out_put
                        first = 0
                    if u[0] == 1:
                        _str = 'out_%d = self.pad%d(%s)' % (i, i, last_out_put)
                        forward_list.append(_str)
                        last_out_put = 'out_%d' % i
                        _str = 'out_%d = self.op%d(%s)' % (i, i, last_out_put)
                        forward_list.append(_str)
                        _str = 'out_%d = self.norm%d(out_%d)' % (i, i, i)
                        forward_list.append(_str)
                        _str = 'out_%d = self.relu(out_%d)' % (i, i)
                        forward_list.append(_str)
                        last_channels = u[3]
                    else:
                        has_pool = True
                        if u[2] < 0.5:
                            _str = 'out_%d = F.max_pool2d(%s, %d)' % (
                                i, last_out_put, u[1][0])  # default kernel_size[0]=kernel_size[1]
                        else:
                            _str = 'out_%d = F.avg_pool2d(%s, %d)' % (i, last_out_put, u[1][0])
                        forward_list.append(_str)
                        last_out_put = 'out_%d' % i
                else:
                    if is_first_fc:
                        _str = 'out_%d = out_%d.view(out_%d.size(0), -1)' % (i - 1, i - 1, i - 1)
                        forward_list.append(_str)
                        is_first_fc = False
                    _str = 'out_%d = self.op%d(%s)' % (i, i, last_out_put)
                    forward_list.append(_str)
                    _str = 'out_%d = self.norm%d(out_%d)' % (i, i, i)
                    forward_list.append(_str)
                    break
            if isinstance(u, list):
                if u0[0][2]!=last_channels or has_pool:
                    _str = 'identity_%d = self.downsample%d(%s)' % (identity_id, i, last_downsample_out)
                    forward_list.append(_str)
                _str = 'out_%d = out_%d + identity_%d' % (i, i, identity_id)
                forward_list.append(_str)

        forward_list.append('return out_%d' % (i))
        # print('\n'.join(forward_list))

        part1, part2, part3 = cls.read_template()
        _str = []
        current_time = time.strftime("%Y-%m-%d  %H:%M:%S")
        _str.append('"""')
        _str.append(current_time)
        _str.append('"""')
        _str.extend(part1)
        _str.append('        %s' % ('# all unit'))
        for s in unit_list:
            _str.append('        %s' % (s))

        _str.extend(part2)
        for s in forward_list:
            _str.append('        %s' % (s))
        _str.extend(part3)
        # print('\n'.join(_str))
        file_name = '%s/%s.py' % (os.path.join(get_algo_local_dir(), 'scripts'), indi.id)
        file_name = cls.path_replace(file_name)
        script_file_handler = open(file_name, 'w')
        script_file_handler.write('\n'.join(_str))
        script_file_handler.flush()
        script_file_handler.close()

    @classmethod
    def write_to_file(cls, _str, _file):
        f = open(_file, 'w')
        f.write(_str)
        f.flush()
        f.close()


if __name__ == '__main__':
    print()
