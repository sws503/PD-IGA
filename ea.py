from src.gan import get_model, gan_get_model, get_dataset, get_initial_model
from src.gui import GUI, PLT_GUI, GUI_initialization
from src.utils import set_seed, torch_to_pil, torch_to_np, visualize_population
import torch
import tqdm
import os
import random
from PIL import Image
from scipy.spatial.distance import hamming
from sklearn.metrics import jaccard_score
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
import time
import pandas as pd
from scipy.stats import entropy
import math

class new_Evolutionary_algorithm(object):
    def __init__(self, gui_args={}, n_population=10, p_mutation=0.5, batch_size=1, use_gui=True, seed=time.time()):


        #GUI 설정
        self.use_gui = use_gui
        self.gui = GUI(**gui_args) if use_gui else PLT_GUI(**gui_args)
        self.gui_init = GUI_initialization()

        #1개의 population 개수
        self.n_population = 30#n_population

        #total_population 개수
        self.dataset_dict = get_dataset()
        self.d_population = 5000 #len(self.dataset_dict)-1 # self.model.config.latentVectorDim

        #mutation 확률
        self.p_mutation = p_mutation
        self.batch_size = batch_size

        #GA
        self.fit_dict = {}
        self.selected_dict = {}
        self.preference_list = []

        if seed:
            #set_seed(seed)
            pass

    def init_population(self, preference_indices, title):
        #n_population size길이의 randint(0,d_population을 만듦)
        population_list = []
        img_indices = list(range(0, self.d_population))


        if title == str('-remove1') or title == str('-remove12'):
            pass
        else:
            #print('preference init')
            for i in range(int(self.d_population/5)):
                img_indices = list(np.concatenate((img_indices, preference_indices)))


        for i in range(100):
            #non unique list경우에서도 중복을 허용하지 않도록 수정
            population = []
            tmp_indices = img_indices.copy()
            for j in range(self.n_population):
                random_num = random.choice(tmp_indices)
                population.append(random_num)
                tmp_indices = [elem for elem in tmp_indices if elem != random_num]

            population.sort()
            population_list.append(population)

        """for i in range(100):
            #unique list인 경우에서만 중복을 허용하지 않음
            population = random.sample(img_indices, self.n_population)
            population.sort()
            population_list.append(population)
        """
        """for i in population_list:
            print('\n',i)"""

        return population_list

    def crossover_uniform(self, tmp_population,preference_indices):
        N = len(tmp_population)
        L = len(tmp_population[0])
        prob_crossover = 0.5

        children = []

        for i in range(int(N / 2)):
            mask = torch.rand(L, ) > 0.5
            child = np.where(mask, tmp_population[i], tmp_population[int(i + (N / 2))])  # mask의 확률로 0이면 a 1이면 b를 넣은 값을 child로 만들겠다.
            children.append(child.T.tolist())

        return children

    def crossover(self, tmp_population,preference_indices):
        N = len(tmp_population)
        L = len(tmp_population[0])
        prob_crossover = 0.5

        for i in range(int(N / 2)):
            if tmp_population[i] == preference_indices and tmp_population[int(i + (N / 2))] == preference_indices:
                continue

            if random.random() <= prob_crossover:
                for i in range(1):  # 3point_crossover이므로
                    pos = random.randint(1, L - 1)
                    # print(pos)
                    for k in range(pos + 1, L - 1):
                        aux = tmp_population[i][k]
                        tmp_population[i][k] = tmp_population[int(i + (N / 2))][k]
                        tmp_population[int(i + (N / 2))][k] = aux
        return tmp_population

    def old_crossover(self, population, n_children, point):
        if n_children == 0:
            return torch.empty([0, self.d_population], dtype=population.dtype)

        children = []
        n_parents = population.shape[0] #선택된 old_members 개수

        for _ in range(n_children):
            parent_idxs = torch.multinomial( #parent_idxs = oldmember를 확률로 2개를 샘플링한 결과
                torch.ones(n_parents), 2, replacement=False)
            parent_a, parent_b = torch.unbind(population[parent_idxs], 0) #parent a 와 b로 parent_idx의 값을 쪼갠다

            if point == 0: #uniform
                mask = torch.rand(self.d_population, ) > 0.5 #population과 동일한 크기의 50%확률로 0과 1로 이루어진 mask 생성
            elif point == 1:
                position = random.randint(1, self.d_population)
                ones = torch.ones(position)
                zeros = torch.zeros(self.d_population - position)
                mask = torch.cat((ones, zeros)) > 0.5
            elif point == 2:
                position1 = random.randint(1, self.d_population)
                position2 = random.randint(1, self.d_population)
                position = [position1, position2]
                position.sort()
                diff = position[1] - position[0]
                ones1 = torch.ones(position[0])
                zeros = torch.zeros(diff)
                ones2 = torch.ones(self.d_population - diff - position[0])
                mask = torch.cat((ones1, zeros, ones2)) > 0.5
            else:
                raise Exception('Point has 0 1 2')

            child = torch.where(mask, parent_a, parent_b) #mask의 확률로 0이면 a 1이면 b를 넣은 값을 child로 만들겠다.
            children.append(child)

        children = torch.stack(children, dim=0)
        return children



    def jaccard(self, list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    def Simulator_Evaluation(self, sample_population, reference_individual):
        score_list = []

        #population의 individual을 평가
        for individual in sample_population:
            # 평가기준 : reference와의 유사도
            hamming_distance = 1 - hamming(individual, reference_individual)
            score_list.append(hamming_distance*10)

        score_list = np.array(score_list)
        mask = np.where(score_list>0, True, False)

        # selected_population에 score_list만큼의 점수를 global dict에 입력
        for idx, individual in enumerate(sample_population):
            self.selected_dict[tuple(individual)] = score_list[idx]

        return score_list, mask

    def Simulator_Evaluation_entropy(self, sample_population, reference_individual):
        score_list = []

        #population의 individual을 평가
        for individual in sample_population:
            # 평가기준 : reference와의 유사도
            distance = 0
            for idx in range(len(individual)):
                #distance += abs(individual[idx] - reference_individual[idx])  # 각 index 별 차이. 0일수록 높아야함.
                difference = abs(individual[idx] - reference_individual[idx])
                if difference == 0:
                    pass
                else:
                    distance += min(np.log(difference) / (np.log(self.d_population-self.n_population)), 1)

            #score_list.append(10 * (self.d_population - distance) / self.d_population)
            score_list.append(self.n_population - distance)

        score_list = np.array(score_list)
        mask = np.where(score_list>0, True, False)

        # selected_population에 score_list만큼의 점수를 global dict에 입력
        for idx, individual in enumerate(sample_population):
            self.selected_dict[tuple(individual)] = score_list[idx]

        return score_list, mask

    def User_Evaluation(self, sample_population, reference_individual):
        images_list = []

        # Feed the latents into the gan to obtain images
        for individual in tqdm.tqdm(sample_population):
            images_batch = get_model().test(individual)
            images_list.append(images_batch)

        # Receive mask and flags from gui
        reset_flag, exit_flag, score_list = self.gui.render(images_list, reference_individual)
        score_list = np.array(score_list)*2
        mask = np.where(score_list>0, True, False)

        # selected_population에 score_list만큼의 점수를 global dict에 입력
        for idx, individual in enumerate(sample_population):
            self.selected_dict[tuple(individual)] = score_list[idx]

        return reset_flag, exit_flag, score_list, mask

    def mutation(self, tmp_population, preference_indices):  # bitwise mutation 함수입니다.
        N = len(tmp_population)
        individual_length = len(tmp_population[0])

        for individual in tmp_population:
            if individual == preference_indices:
                continue


            target_idx_candidate = list(range(0, individual_length))
            while True:
                target_idx = random.sample(target_idx_candidate, 1).pop()
                low = min([individual[target_idx] + 1, self.d_population])
                high = max([individual[target_idx] - 1, 0])
                mutation_candidate = [low, high]
                for i in individual:
                    if i in mutation_candidate:
                        mutation_candidate.remove(i)

                #mutate = random.sample(mutation_candidate, 1) if mutation_candidate else [random.randint(0, self.d_population)]
                if mutation_candidate:
                    mutate = random.sample(mutation_candidate, 1)
                    break
                elif target_idx_candidate:
                    #mutate = [random.randint(0, self.d_population)]
                    pass
                else:
                    #print('there is no mutation candidate')
                    break

            individual[target_idx] = mutate.pop()
        return tmp_population

    def mutation_archieve(self, tmp_population, preference_indices):  # bitwise mutation 함수입니다.
        N = len(tmp_population)
        L = len(tmp_population[0])

        for individual in tmp_population:
            if individual == preference_indices:
                continue
            for k in range(L):
                if random.random() < 0.1:
                    mutation_candidate = preference_indices.copy()
                    #mutation_candidate = list(range(0, self.d_population))
                    for i in individual:
                        if i in mutation_candidate:
                            mutation_candidate.remove(i)

                    mutate = random.sample(mutation_candidate, 1) if mutation_candidate else [random.randint(0, self.d_population)]
                    #mutate = [random.randint(0, self.d_population)]
                    individual[k] = mutate.pop()
                    #individual.sort()
        return tmp_population

    def old_mutation(self, population):
        mask1 = torch.rand(population.shape[0])[:, None] < self.p_mutation
        mask2 = torch.rand(population.shape) < self.p_mutation

        mask = mask1 & mask2

        noise = self.init_population(population.shape[0])
        mutated_population = torch.where(mask, noise, population)

        return mutated_population

    def Preference_based_initializer(self):
        population = list(range(0,self.d_population))

        # Feed the latents into the gan to obtain images
        images_batch = get_initial_model().mergeimg_init(population)

        # Receive mask and flags from gui
        reset_flag, exit_flag, mask = self.gui_init.render(images_batch)
        score_list = np.array(mask)
        mask = np.where(score_list > 0, True, False)

        arglist = np.argwhere(score_list > 0)
        preference_indices = np.concatenate(arglist)
        preference_indices = list(preference_indices)

        return score_list, mask, preference_indices

    def make_preference_simulator(self, reference_individual):
        score_list = np.zeros(self.d_population)
        for chromosome in reference_individual:
            score_list[chromosome] = 1

        score_list = np.array(score_list)
        mask = np.where(score_list > 0, True, False)

        arglist = np.argwhere(score_list > 0)
        preference_indices = np.concatenate(arglist)
        preference_indices = list(preference_indices)

        return score_list, mask, preference_indices

    def fitness_evaluation_archieve(self, individual):
        individual.sort()
        #이미 평가한 값이라면,
        if tuple(individual) in self.selected_dict:
            self.fit_dict[tuple(individual)] = self.selected_dict[tuple(individual)]
            return self.selected_dict[tuple(individual)]
        else:
            hdistance_list = []
            fitval_list = []
            hamming_distance = 0
            for selected_individual, fitval in self.selected_dict.items():
                hamming_distance = 1 - hamming(individual, selected_individual)
                fitness_approximation = hamming_distance # TODO 밖으로 빼기

                hdistance_list.append(fitness_approximation)
                fitval_list.append(fitness_approximation * fitval)

            max_distance = fitval_list[np.argmax(hdistance_list)] if self.selected_dict else 0
            self.fit_dict[tuple(individual)] = max_distance
            return max_distance

    def fitness_evaluation(self, individual):
        individual.sort()
        #이미 평가한 값이라면,
        if tuple(individual) in self.selected_dict:
            self.fit_dict[tuple(individual)] = self.selected_dict[tuple(individual)]
            return self.selected_dict[tuple(individual)]
        else:
            hdistance_list = []
            fitval_list = []
            hamming_distance = 0
            for selected_individual, fitval in self.selected_dict.items():
                distance = 0
                for idx in range(len(individual)):
                    distance += 1 - (abs(individual[idx] - selected_individual[idx]) / (self.d_population)) #각 index 별 차이. 0일수록 높아야함.

                hamming_distance = distance / self.n_population
                fitness_approximation = hamming_distance # TODO 밖으로 빼기


                hdistance_list.append(fitness_approximation)
                fitval_list.append(fitness_approximation * fitval)

            max_distance = fitval_list[np.argmax(hdistance_list)] if self.selected_dict else 0
            self.fit_dict[tuple(individual)] = max_distance
            return max_distance

    def fitness_evaluation_jaccard(self, individual):
        individual.sort()
        #이미 평가한 값이라면,
        if tuple(individual) in self.selected_dict:
            self.fit_dict[tuple(individual)] = self.selected_dict[tuple(individual)]
            return self.selected_dict[tuple(individual)]
        else:
            hdistance_list = []
            fitval_list = []
            entropy_distance = 0
            for selected_individual, fitval in self.selected_dict.items():
                jaccard_distance = self.jaccard(individual, selected_individual)
                hamming_distance = 0 #hamming(individual, selected_individual)
                fitness_approximation = (jaccard_distance + hamming_distance)

                hdistance_list.append(fitness_approximation)
                fitval_list.append(fitness_approximation * fitval)

            max_distance = fitval_list[np.argmax(hdistance_list)] if self.selected_dict else 0
            self.fit_dict[tuple(individual)] = max_distance
            return max_distance

    def uniform_sampling(self, total_population, title):
        rankdict = {}
        sorted_ranklist = []
        #전체 population의 값들에 전부 fitness값을 매김
        for idx, individual in enumerate(total_population):
            fitval = self.fitness_evaluation(individual)
            #TODO 중복값들이 합쳐지면서 값이 줄고 있다
            #rankdict[tuple(individual)] = fitval
            sorted_ranklist.append((individual,fitval))

        #sorted_ranklist = list(sorted(rankdict.items(), key = lambda item: item[1], reverse=True))
        sorted_ranklist.sort(key=lambda x: x[1], reverse=True)
        sample_population = []
        sample_scorelist = []
        if title == str('-remove2') or title == str('-remove12'):
            # fitness값 기준으로 sorted_ranklist를 만들고 0 1 2 3 4 5 6 7 8 9 을 리턴함
            for uniform_idx in range(10):
                sample_population.append(sorted_ranklist[uniform_idx][0])
                sample_scorelist.append(sorted_ranklist[uniform_idx][1])
        else:
            # fitness값 기준으로 sorted_ranklist를 만들고 0 10 20 30 40 50 ~ 90 100 을 리턴함
            for uniform_idx in range(0,len(sorted_ranklist), 10):
                sample_population.append(sorted_ranklist[uniform_idx][0])
                sample_scorelist.append(sorted_ranklist[uniform_idx][1])

        return sample_population, sample_scorelist, sorted_ranklist

    def tournament_selection(self, total_population):  # Tournament selection 함수입니다.
        N = len(total_population)
        tmp_representation = []

        #1부터 100까지 random_idx와 경쟁시킨다
        for i in range(N):
            random_idx = random.randint(0, N - 1)
            if self.fitness_evaluation(total_population[i]) < self.fitness_evaluation(total_population[random_idx]):
                tmp_representation.append(total_population[random_idx].copy())
            else:
                tmp_representation.append(total_population[i].copy())
        return tmp_representation

    def soft_tournament_selection(self, total_population1, sorted_ranklist):  # Tournament selection 함수입니다.
        N = len(total_population1)
        tmp_representation = []

        #1부터 100까지 random_idx와 경쟁시킨다
        for _ in range(2):
            for i in range(int(N/2)):
                target_idx = i + int((N/2))
                if random.random() <= 0.7:
                    if self.fitness_evaluation(list(sorted_ranklist[i][0])) < self.fitness_evaluation(list(sorted_ranklist[target_idx][0])):
                        tmp_representation.append(list(sorted_ranklist[target_idx][0]).copy())
                    else:
                        tmp_representation.append(list(sorted_ranklist[i][0]).copy())
                else:
                    if self.fitness_evaluation(list(sorted_ranklist[i][0])) > self.fitness_evaluation(list(sorted_ranklist[target_idx][0])):
                        tmp_representation.append(list(sorted_ranklist[target_idx][0]).copy())
                    else:
                        tmp_representation.append(list(sorted_ranklist[i][0]).copy())
        return tmp_representation

    def save_to_csv(self,fitness_values, file_name):
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(fitness_values)

    def export_fitness_values(self, fitness_values, file_name):
        df = pd.DataFrame(fitness_values)
        df.to_csv(file_name, index=False)

    def graph(self, max_fit, avg_fit, current_time):
        plt.clf()
        plt.plot(max_fit, label="Maximum Fitness Values")
        plt.plot(avg_fit, label="Average Fitness Values")
        plt.title("Maximum and Average Fitness Values", fontsize=12)
        plt.xlabel('GENERATIONS', fontsize=12)
        plt.ylabel('FITNESS VALUES', fontsize=12)
        plt.legend(loc=4)
        #plt.ylim(0, 11)
        #plt.ylim(9,10)
        #plt.yscale('log')
        plt.savefig("./result/fitness_value" + current_time + ".png")

        #plt.show()

    def subgraph(self, max_fit, avg_fit, current_time):
        # plot data
        fig, ax = plt.subplots()
        ax.plot(max_fit, label="Maximum Fitness Values")
        ax.plot(avg_fit, label="Average Fitness Values")
        # set y-axis limits and scale
        ax.set_ylim([0, 10])
        ax.set_yscale('symlog', linthresh=9)

        plt.title("Maximum and Average Fitness Values", fontsize=12)
        plt.xlabel('GENERATIONS', fontsize=12)
        plt.ylabel('FITNESS VALUES', fontsize=12)
        plt.legend(loc=4)
        #plt.ylim(0, 11)
        plt.savefig("./result/fitness_value" + current_time + ".png")

        plt.show()

    def repair_chromosome(self, tmp_population):
        #중복 체크
        for individual in tmp_population:
            unique_set = set(individual)
            if len(unique_set) != len(individual):
                for item in unique_set:
                    count = individual.count(item)
                    if count > 1:
                        indices = [i for i, x in enumerate(individual) if x == item]
                        for index in indices:
                            tick = 1
                            while True:
                                low = min([individual[index] + tick, self.d_population-1])
                                high = max([individual[index] - tick, 0])
                                mutation_candidate = [low, high]

                                for i in individual:
                                    if i in mutation_candidate:
                                        mutation_candidate.remove(i)

                                if mutation_candidate:
                                    mutate = random.sample(mutation_candidate, 1)
                                    individual[index] = mutate.pop()
                                    break
                                else:
                                    tick += 1
                                    #print(individual, tick)


        # sort 체크
        for individual in tmp_population:
            is_sorted = all(x <= y for x, y in zip(individual[:-1], individual[1:]))
            if is_sorted:
                pass
            else:
                individual.sort()

        return tmp_population

    def evolve_archieve(self, preference_indices, reference_individual):
        title = str('default')
        title = str('remove2')
        total_population = self.init_population(preference_indices, title)

        iteration = 1
        while True:
            iteration += 1

            #10배수 순위의 sample population을 생성 후 평가
            sample_population, sample_scorelist, sorted_ranklist = self.uniform_sampling(total_population, title)
            reset_flag, exit_flag, score_list, mask = self.User_Evaluation(sample_population, reference_individual)

            # selected_population에 score_list만큼의 점수를 global dict에 입력
            for idx, individual in enumerate(sample_population):
                self.selected_dict[tuple(individual)] = score_list[idx]

            #100개짜리 total population으로 작업
            tmp_population = self.tournament_selection(total_population)

            tmp_population = self.crossover(tmp_population, preference_indices)

            tmp_population = self.mutation(tmp_population, preference_indices)

            total_population = tmp_population

    def reference_hamming(self, individual, reference_individual):
        # 평가기준 : reference와의 유사도
        distance = 0
        # 평가기준 : reference와의 유사도
        hamming_distance = 1 - hamming(individual, reference_individual)

        return (hamming_distance * self.n_population)

    def reference_hammingsimilarity(self, individual, reference_individual):

        # 평가기준 : reference와의 유사도
        distance = 0
        for idx in range(len(individual)):
            difference = abs(individual[idx] - reference_individual[idx])
            if difference < self.d_population / self.n_population:
                pass
            else:
                distance += 1

        return (self.n_population - distance)

    def reference_similarity(self, individual, reference_individual):

        # 평가기준 : reference와의 유사도
        distance = 0
        for idx in range(len(individual)):
            difference = abs(individual[idx] - reference_individual[idx])
            if difference == 0:
                pass
            else:
                distance += 1/2 + np.log(difference)/(np.log(self.d_population)*2)

        return (self.n_population - distance)

    def reference_similarity_archieve(self, individual, reference_individual):
        # 평가기준 : reference와의 유사도
        distance = 0
        for idx in range(len(individual)):
            distance += abs(individual[idx] - reference_individual[idx])#(self.d_population-10)  # 각 index 별 차이. 0일수록 높아야함.
        return self.n_population*(self.d_population - distance)/self.d_population#max(10 - distance, 0)

    def evolve_simulation(self, preference_indices, reference_individual, title):
        total_population = self.init_population(preference_indices, title)
        max_fit = []
        avg_fit = []

        max_fit2 = []
        avg_fit2 = []
        iteration = 0
        max_iteration = 50
        while iteration < max_iteration:
            iteration += 1

            current_scores = []
            current_scores2 = []
            for individual in total_population:
                #\current_scores.append(self.fitness_evaluation(individual))
                current_scores.append(self.reference_hammingsimilarity(individual, reference_individual))
                #current_scores2.append(self.reference_similarity(individual, reference_individual))
                current_scores2.append(self.reference_hamming(individual, reference_individual))


            #10배수 순위의 sample population을 생성 후 평가
            sample_population, sample_scorelist, sorted_ranklist = self.uniform_sampling(total_population, title)
            #Fitness value를 평가할 수 있는 기준을 준다.
            #self.Simulator_Evaluation(sample_population, reference_individual)
            self.Simulator_Evaluation_entropy(sample_population, reference_individual)

            elitelist = []
            for idx in range(1):
                elitelist.append(sorted_ranklist[idx][0])
                total_population.remove(sorted_ranklist[idx][0])
            #print(elite)

            #100개짜리 total population으로 작업
            tmp_population = self.tournament_selection(total_population)
            #tmp_population = self.soft_tournament_selection(total_population, sorted_ranklist)
            tmp_population = self.crossover(tmp_population, preference_indices)

            tmp_population = self.mutation(tmp_population, preference_indices)

            tmp_population = self.repair_chromosome(tmp_population)

            for elite in elitelist:
                tmp_population.append(elite)
            total_population = tmp_population

            max_fit.append(max(current_scores))
            avg_fit.append(sum(current_scores)/len(current_scores))

            max_fit2.append(max(current_scores2))
            avg_fit2.append(sum(current_scores2) / len(current_scores2))


            """if iteration % 1 == 0:
                print(iteration)
                print("max : ", max(current_scores))
                print("avg : ", sum(current_scores) / len(current_scores))

                print("**************************************")
                for i in sorted_ranklist:
                    pass
                    #print(i[0])"""

        current_time = title + datetime.datetime.now().strftime("-%H-%M-%S")

        self.save_to_csv(avg_fit, "./result/data"+current_time+title+"avg.csv")
        self.save_to_csv(max_fit, "./result/data" + current_time + title + "max.csv")

        self.save_to_csv(avg_fit2, "./result/data" + current_time + title + "avg2.csv")
        self.save_to_csv(max_fit2, "./result/data" + current_time + title + "max2.csv")

        self.graph(max_fit, avg_fit, current_time)
        self.graph(max_fit2, avg_fit2, current_time+'v2')
        #self.export_fitness_values(avg_fit, "./result/data"+current_time+"pdavgfit.csv")

    def evolve(self, preference_indices, reference_individual, title):
        total_population = self.init_population(preference_indices, title)

        iteration = 1
        while True:
            iteration += 1

            #10배수 순위의 sample population을 생성 후 평가
            sample_population, sample_scorelist, sorted_ranklist = self.uniform_sampling(total_population, title)
            reset_flag, exit_flag, score_list, mask = self.User_Evaluation(sample_population, reference_individual)

            elitelist = []
            for idx in range(10):
                elitelist.append(sorted_ranklist[idx][0])
                total_population.remove(sorted_ranklist[idx][0])

            #100개짜리 total population으로 작업
            tmp_population = self.tournament_selection(total_population)

            tmp_population = self.crossover(tmp_population, preference_indices)

            tmp_population = self.mutation(tmp_population, preference_indices)

            tmp_population = self.repair_chromosome(tmp_population)

            for elite in elitelist:
                tmp_population.append(elite)

            total_population = tmp_population


    def Population_to_ImgList(self, population):
        for idx, order in enumerate(population):
            merged_image = Image.new('RGB', (256, 256), (250, 250, 250))
            img_file = self.dataset_dict[order]['image']
            img = Image.open(img_file)
            img = img.resize((256, 256))
            merged_image.paste(img, (256, 0))


        return merged_image

        images_tr = []
        for step in tqdm.tqdm(range(0, population.shape[0])):
            images_tr.append(images_batch)

        """images_tr = torch.cat(images_tr)
        #print(images_tr)
        # Transform torch.tensor into PIL or numpy
        if self.use_gui:
            images = torch_to_pil(images_tr)
        else:
            images = torch_to_np(images_tr)"""
        Image_List = images_tr
        return Image_List

    def Make_Reference(self, simulation, candidate=None):
        reference_individual = random.sample(range(0,self.d_population), self.n_population)
        if candidate != None:
            reference_individual = candidate

        reference_individual.sort()

        if simulation == True:
            pass
        else:
            reference_imglist = []
            #images_batch = get_model().test(reference_individual)

            images_batch = Image.new('RGB', (10 * 256, 256), (250, 250, 250))

            for idx, order in enumerate(reference_individual):
                img_file = self.dataset_dict[order]['image']
                img = Image.open(img_file)
                img = img.resize((256, 256))
                img_size = img.size
                images_batch.paste(img, (idx * 256, 0))

            images_batch.show()
            reference_imglist.append(images_batch)

            # Receive mask and flags from gui
            #reset_flag, exit_flag, mask = self.gui.render(images_list)

        return reference_individual

    def run(self, title, simulation):
        #Make Reference
        ref_candidate0 = list(range(10))
        ref_candidate1 = [0,1, 12, 23, 24, 25, 26, 31, 32, 48]
        reference_individual = self.Make_Reference(simulation)


        if simulation == True:
            # GET preference
            preference_score, preference_list, preference_indices = self.make_preference_simulator(reference_individual)

            self.evolve_simulation(preference_indices, reference_individual, title)
        else:
            # GET preference
            preference_score, preference_list, preference_indices = self.Preference_based_initializer()

            self.evolve(preference_indices, reference_individual, title)