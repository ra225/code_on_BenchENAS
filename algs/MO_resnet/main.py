import numpy as np
import copy
import os
from compute.file import get_algo_local_dir
from comm.log import Log
from comm.utils import GPUFitness
from algs.MO_resnet.utils import Utils
from algs.MO_resnet.genetic.statusupdatetool import StatusUpdateTool
from algs.MO_resnet.genetic.population import Population
from algs.MO_resnet.genetic.evaluate import FitnessEvaluate
from algs.MO_resnet.genetic.crossover_and_mutation import CrossoverAndMutation
from algs.MO_resnet.genetic.selection_operator import Selection


class EvolveRESNET(object):
    def __init__(self, params):
        self.params = params
        self.pops = None
        self.pop_size = params['pop_size']
        self.M = 2  #the number of targets is 2
        self.lamda = np.zeros((self.pop_size, self.M))
        for i in range(self.pop_size):
            self.lamda[i][0] = i / self.pop_size
            self.lamda[i][1] = (self.pop_size - i) / self.pop_size
        self.T = int(self.pop_size / 5)
        self.B = np.zeros((self.pop_size, self.pop_size))
        self.EP = []
        for i in range(self.pop_size):
            for j in range(self.pop_size):
                self.B[i][j] = np.linalg.norm(self.lamda[i, :] - self.lamda[j, :]);
            self.B[i, :] = np.argsort(self.B[i, :])
        self.z = np.zeros(self.M)
        for i in range(self.M):
            self.z[i] = 100

    def dominate(self, x, y):
        lte = 0
        lt = 0
        gte = 0
        gt = 0
        eq = 0
        for i in range(self.M):
            if x[i] <= y[i]:
                lte = lte + 1
            if x[i] < y[i]:
                lt = lt + 1
            if x[i] >= y[i]:
                gte = gte + 1
            if x[i] > y[i]:
                gt = gt + 1
            if x[i] == y[i]:
                eq = eq + 1
        if lte == self.M and lt > 0:
            return 1
        elif gte == self.M and gt > 0:
            return -1
        elif eq == self.M:
            return -2
        else:
            return 0

    def gte(self, f, lamda, z):
        return max(lamda * abs(f - z))

    def initialize_population(self):
        StatusUpdateTool.begin_evolution()
        pops = Population(self.params, 0)
        pops.initialize(Log)
        self.pops = pops
        Utils.save_population_at_begin(str(pops), 0)

    def modify_EP(self, indi):
        flag = 1
        j = 0
        candidate = [indi.error_mean, indi.loss_mean]
        while j < len(self.EP):
            if j >= len(self.EP):
                break
            r = self.dominate(candidate, self.EP[j][0:self.M])
            if -2 == r:
                return -1
            if 1 == r:
                del self.EP[j]
                j -= 1
            elif -1 == r:
                flag = 0
            j += 1
        if flag == 1:
            candidate.append(indi.id)
            self.EP.append(candidate)
        return 0

    def fitness_evaluate(self, isFirst):
        fitness = FitnessEvaluate(self.pops.individuals, Log)
        fitness.generate_to_python_file()
        # for indi in self.pops.individuals:
        #     if indi.acc_mean == -1:
        #         indi.acc_mean = np.random.random()
        fitness.evaluate()
        import time
        time.sleep(3)
        fitness_map = GPUFitness.read()
        for indi in self.pops.individuals:
            if indi.error_mean == -1 and indi.id in fitness_map.keys():
                indi.error_mean, indi.loss_mean = fitness_map[indi.id][0], fitness_map[indi.id][1]
        for indi in self.pops.individuals:
            if indi.error_mean != -1:
                if self.z[0] > indi.error_mean:
                    self.z[0] = indi.error_mean
                if self.z[1] > indi.loss_mean:
                    self.z[1] = indi.loss_mean
                if -1 == self.modify_EP(indi):
                    Log.info('%s has duplicate' % (indi.id))
        Utils.save_EP_after_evaluation(str(self.EP), self.pops.gen_no)
        if isFirst == False:
            self.pops.offsprings = copy.deepcopy(self.pops.individuals)
            self.pops.individuals = copy.deepcopy(self.pops.parent_individuals)

    def crossover_and_mutation(self):
        params = {}
        params['crossover_eta'] = StatusUpdateTool.get_crossover_eta()
        params['mutation_eta'] = StatusUpdateTool.get_mutation_eta()
        params['acc_mean_threshold'] = StatusUpdateTool.get_acc_mean_threshold()
        params['complexity_threshold'] = StatusUpdateTool.get_complexity_threshold()
        cm = CrossoverAndMutation(self.params['genetic_prob'][0], self.params['genetic_prob'][1], Log,
                                  self.pops.individuals, self.B, self.T, self.pops.gen_no, params)
        offspring = cm.process()
        self.pops.parent_individuals = copy.deepcopy(self.pops.individuals)
        self.pops.individuals = copy.deepcopy(offspring)

    def environment_selection(self):
        v_list = []
        indi_list = []
        _str = []
        for indi in self.pops.individuals:
            indi_list.append(indi)
            v_list.append(indi.error_mean)
            _t_str = 'Indi-%s-%.5f-%.5f-%s' % (indi.id, indi.error_mean, indi.loss_mean, indi.uuid()[0])
            _str.append(_t_str)
        for indi in self.pops.offsprings:
            indi_list.append(indi)
            v_list.append(indi.error_mean)
            _t_str = 'Offs-%s-%.5f-%.5f-%s' % (indi.id, indi.error_mean, indi.loss_mean, indi.uuid()[0])
            _str.append(_t_str)

        i = 0
        while i < len(self.pops.offsprings):
            indi = copy.deepcopy(self.pops.offsprings[i])
            for j in range(self.T):
                p = int(self.B[i, j])
                o = copy.deepcopy(self.pops.individuals[p])
                value_fj = self.gte([indi.error_mean, indi.loss_mean], self.lamda[p, :], self.z)
                value_p = self.gte([o.error_mean, o.loss_mean], self.lamda[p, :], self.z)
                if value_fj < value_p:
                    self.pops.individuals[p] = copy.deepcopy(indi)
            i += 1

        _file = '%s/ENVI_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), self.pops.gen_no)
        Utils.write_to_file('\n'.join(_str), _file)

        """Here, the population information should be updated, such as the gene no and then to the individual id"""
        next_gen_pops = Population(self.pops.params, self.pops.gen_no + 1)
        next_gen_pops.create_from_offspring(self.pops.individuals)
        self.pops = next_gen_pops
        for _, indi in enumerate(self.pops.individuals):
            _t_str = 'new -%s-%.5f-%.5f-%s' % (indi.id, indi.error_mean, indi.loss_mean, indi.uuid()[0])
            _str.append(_t_str)
        _file = '%s/ENVI_%05d.txt' % (os.path.join(get_algo_local_dir(), 'populations'), self.pops.gen_no - 1)
        Utils.write_to_file('\n'.join(_str), _file)

        Utils.save_population_at_begin(str(self.pops), self.pops.gen_no)

    def create_necessary_folders(self):
        sub_folders = [os.path.join(get_algo_local_dir(), v) for v in ['populations', 'log', 'scripts']]
        if not os.path.exists(get_algo_local_dir()):
            os.mkdir(get_algo_local_dir())
        for each_sub_folder in sub_folders:
            if not os.path.exists(each_sub_folder):
                os.mkdir(each_sub_folder)

    def do_work(self, max_gen):
        # create the corresponding fold under runtime
        self.create_necessary_folders()

        # the step 1
        if StatusUpdateTool.is_evolution_running():
            Log.info('Initialize from existing population data')
            gen_no = Utils.get_newest_file_based_on_prefix('begin')
            if gen_no is not None:
                Log.info('Initialize from %d-th generation' % (gen_no))
                pops = Utils.load_population('begin', gen_no)
                self.pops = pops
                if gen_no > 0:
                    EP = Utils.load_EP('EP', gen_no-1)
                    self.EP = EP
            else:
                raise ValueError('The running flag is set to be running, but there is no generated population stored')
        else:
            gen_no = 0
            Log.info('Initialize...')
            self.initialize_population()
        Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (gen_no))
        self.fitness_evaluate(True)
        Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (gen_no))



        for curr_gen in range(gen_no, max_gen):
            self.params['gen_no'] = curr_gen
            self.pops.gen_no = curr_gen
            # step 3
            Log.info('EVOLVE[%d-gen]-Begin to crossover and mutation' % (self.pops.gen_no))
            self.crossover_and_mutation()
            Log.info('EVOLVE[%d-gen]-Finish crossover and mutation' % (self.pops.gen_no))

            self.pops.gen_no += 1
            Log.info('EVOLVE[%d-gen]-Begin to evaluate the fitness' % (self.pops.gen_no))
            self.fitness_evaluate(False)
            Log.info('EVOLVE[%d-gen]-Finish the evaluation' % (self.pops.gen_no))

            self.environment_selection()
            Log.info('EVOLVE[%d-gen]-Finish the environment selection' % (
                    self.pops.gen_no - 1))  # in environment_selection, gen_no increase 1
        StatusUpdateTool.end_evolution()


class Run():
    def do(self):
        params = StatusUpdateTool.get_init_params()
        evoCNN = EvolveRESNET(params)
        evoCNN.do_work(params['max_gen'])


if __name__ == '__main__':
    r = Run()
    r.do()
    # params = StatusUpdateTool.get_init_params()
    # evoCNN = EvolveCNN(params)
    # evoCNN.create_necessary_folders()
    # evoCNN.initialize_population()
    # evoCNN.pops = Utils.load_population('begin', 0)
    # evoCNN.fitness_evaluate()
    # evoCNN.crossover_and_mutation()
