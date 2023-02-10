import argparse
import os
#from src.ea import Evolutionary_algorithm
from src.ea import new_Evolutionary_algorithm
from src.original_ea import Evolutionary_algorithm2
import yaml
import csv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config', '-c', default='configs/config.yaml',
        help='Path to the config file')

    args = parser.parse_args()

    with open(args.config, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    img_size = (config['gui_args']['img_width'],
                config['gui_args']['img_height'])

    config['gui_args'].pop('img_width', None)
    config['gui_args'].pop('img_height', None)
    config['gui_args']['img_size'] = img_size

    try:
        import google.colab
        colab = True
    except:
        colab = False

    config['gui_args']['colab'] = colab
    if colab:
        config['ea_args']['use_gui'] = False

    return config


def main(title, simulation):
    config = parse_args()
    new_ea = new_Evolutionary_algorithm(
        gui_args=config['gui_args'])

    #ea.run()
    new_ea.run(title, simulation=simulation)

def import_fitness_values(directory, end):
    fitness_values = []
    num = 0
    for file_name in os.listdir(directory):
        if file_name.endswith(end):
            num += 1
            with open(os.path.join(directory, file_name)) as file:
                reader = csv.reader(file)
                values = []
                for row in reader:
                    values.append([float(val) for val in row])
                fitness_values.append(values)
    print(num)
    return fitness_values


import matplotlib.pyplot as plt
import numpy as np

def test(values, title=None):
    # Get the number of lists and their length
    num_lists = len(values)
    list_len = len(values[0][0])

    # Initialize the average values list
    avg_values = [0] * list_len

    # Calculate the average value for each index
    for i in range(list_len):
        for j in range(num_lists):
            avg_values[i] += values[j][0][i]
        avg_values[i] /= num_lists

    # Draw the graph
    index = range(list_len)
    plt.plot(index, avg_values)
    plt.xlabel("Index")
    plt.ylabel("Average Value")
    plt.title("Graph with average values for each index")

    plt.legend(loc=4)
    plt.ylim(0, 11)
    plt.savefig("./result/fitness_value" + title + ".png")
    plt.show()

def get_avg_values(values):
    # Get the number of lists and their length
    num_lists = len(values)
    list_len = len(values[0][0])

    # Initialize the average values list
    avg_values = [0] * list_len

    # Calculate the average value for each index
    for i in range(list_len):
        for j in range(num_lists):
            avg_values[i] += values[j][0][i]
        avg_values[i] /= num_lists
    return avg_values

def draw2graph(avgvalues1, avgvalues2, title):
    # Draw the graph
    index = range(len(avgvalues1))
    plt.plot(avgvalues1, label='average')
    plt.plot(avgvalues2, label="max")
    plt.xlabel("Index")
    plt.ylabel("Average Value")
    plt.title("Graph with average values for each index")
    plt.legend(loc=4)
    plt.ylim(0, 11)
    plt.savefig("./result/fitness_value" + title + ".png")
    plt.show()

def draw4graph(avgvalues1, avgvalues2,avgvalues3, avgvalues4, title):
    # Draw the graph
    plt.clf()

    index = range(len(avgvalues1))
    plt.plot(avgvalues1, label='Preference based Initialization + Diversity Preserving')
    plt.plot(avgvalues2, label="Diversity Preserving")
    plt.plot(avgvalues3, label='Preference based Population Initialization')
    plt.plot(avgvalues4, label="GA")
    plt.xlabel("Index")
    plt.ylabel("Average Value")
    plt.title("Graph with average values for each index")
    plt.legend(loc=4)
    #plt.ylim(0, 11)
    plt.xlim(1, 20)
    #plt.ylim(4,6)

    #plt.yscale('log')
    plt.savefig("./result/fitness_value" + title + "-2.png")
    #plt.show()

def postprocess():
    title0 = str('-default')
    # TODO remove1
    title1 = str('-remove1')
    # TODO remove2
    title2 = str('-remove2')
    # TODO remove12
    title12 = str('-remove12')

    title = title0
    """fitness_avgvalue = import_fitness_values('./0131result/', title + 'avg.csv')
    fitness_maxvalue = import_fitness_values('./0131result/', title + 'max.csv')
    fitness_sim = import_fitness_values('./0131result/', title + 'avg2.csv')
    fitness_maxsim = import_fitness_values('./0131result/', title + 'max2.csv')
    
    avg1 = get_avg_values(fitness_avgvalue)
    avg2 = get_avg_values(fitness_maxvalue)
    avg3 = get_avg_values(fitness_sim)
    avg4 = get_avg_values(fitness_maxsim)
    # draw2graph(avg1, avg2, title+ 'avgmax')
    # draw2graph(avg3, avg4, title + 'sim')"""

    # fitness_values2 = import_fitness_values('./result/', 'remove2.csv')
    # print(fitness_values)
    # draw_graph(fitness_values)

    # test(fitness_avgvalue, title + 'avg')
    # test(fitness_maxvalue, title + 'max')
    # test(fitness_sim, title + 'avg2')
    # test(fitness_maxsim, title + 'max2')


    result_dir = './result/'
    # 4개 그래프 비교
    fitness_avgvalue1 = import_fitness_values(result_dir, title0 + 'avg.csv')
    fitness_avgvalue2 = import_fitness_values(result_dir, title1 + 'avg.csv')
    fitness_avgvalue3 = import_fitness_values(result_dir, title2 + 'avg.csv')
    fitness_avgvalue4 = import_fitness_values(result_dir, title12 + 'avg.csv')

    avg1 = get_avg_values(fitness_avgvalue1)
    avg2 = get_avg_values(fitness_avgvalue2)
    avg3 = get_avg_values(fitness_avgvalue3)
    avg4 = get_avg_values(fitness_avgvalue4)

    draw4graph(avg1, avg2, avg3, avg4, 'total_avg')

    fitness_avgvalue1 = import_fitness_values(result_dir, title0 + 'max.csv')
    fitness_avgvalue2 = import_fitness_values(result_dir, title1 + 'max.csv')
    fitness_avgvalue3 = import_fitness_values(result_dir, title2 + 'max.csv')
    fitness_avgvalue4 = import_fitness_values(result_dir, title12 + 'max.csv')

    avg1 = get_avg_values(fitness_avgvalue1)
    avg2 = get_avg_values(fitness_avgvalue2)
    avg3 = get_avg_values(fitness_avgvalue3)
    avg4 = get_avg_values(fitness_avgvalue4)

    draw4graph(avg1, avg2, avg3, avg4, 'total_max')

    fitness_avgvalue1 = import_fitness_values(result_dir, title0 + 'avg2.csv')
    fitness_avgvalue2 = import_fitness_values(result_dir, title1 + 'avg2.csv')
    fitness_avgvalue3 = import_fitness_values(result_dir, title2 + 'avg2.csv')
    fitness_avgvalue4 = import_fitness_values(result_dir, title12 + 'avg2.csv')

    avg1 = get_avg_values(fitness_avgvalue1)
    avg2 = get_avg_values(fitness_avgvalue2)
    avg3 = get_avg_values(fitness_avgvalue3)
    avg4 = get_avg_values(fitness_avgvalue4)

    draw4graph(avg1, avg2, avg3, avg4, 'total_avg2')

    fitness_avgvalue1 = import_fitness_values(result_dir, title0 + 'max2.csv')
    fitness_avgvalue2 = import_fitness_values(result_dir, title1 + 'max2.csv')
    fitness_avgvalue3 = import_fitness_values(result_dir, title2 + 'max2.csv')
    fitness_avgvalue4 = import_fitness_values(result_dir, title12 + 'max2.csv')

    avg1 = get_avg_values(fitness_avgvalue1)
    avg2 = get_avg_values(fitness_avgvalue2)
    avg3 = get_avg_values(fitness_avgvalue3)
    avg4 = get_avg_values(fitness_avgvalue4)

    draw4graph(avg1, avg2, avg3, avg4, 'total_max2')

if __name__ == '__main__':
    title0 = str('-default')
    # TODO remove1
    title1 = str('-remove1')
    # TODO remove2
    title2 = str('-remove2')
    # TODO remove12
    title12 = str('-remove12')

    #main(title0, simulation=True)
    #main(title1, simulation=True)
    #main(title2, simulation=True)
    #main(title12, simulation=True)


    """for i in range(1000):
        print('0')
        for a in range(1):
            main(title0, simulation=True)
            #pass
        print('1')
        for b in range(1):
            main(title1, simulation=True)
            #pass
        print('2')
        for c in range(1):
            main(title2, simulation=True)
            #pass
        print('12')
        for d in range(1):
            main(title12, simulation=True)
            #pass"""

    postprocess()

