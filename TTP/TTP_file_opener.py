import csv
import numpy as np


class TTP_file_opener:
    def __init__(self, filename):

        print("创建解包器, 从文件{}".format(filename))
        self.filename = filename
        self.content = None
        self.rowCounter = None
        self.hyperparameter = None
        self.cities = None

    # file_scanner 函数用于转化并导入数据集文件
    def file_scanner(self):
        # 用open() 函数 打开目标文件 并读取
        with open(self.filename, 'r') as file:
            reader = csv.reader(file)
            # 输入文件行数检测 + 行内容记录
            self.rowCounter = 0
            self.content = []
            for row in reader:
                # 每次遍历， 记录其行数 和 他的内容
                self.content.append(str(row[0]))
                print("所在行：{}, 内容是：{}".format(self.rowCounter, row))
                self.rowCounter += 1
        return self.rowCounter, self.content

    def set_hyperparameter(self):
        problem_name = ""
        knapsack_data_type = ""
        dimension = 0
        number_of_item = 0
        capacity_of_knapsack = 0
        min_speed = 0.0
        max_speed = 0.0
        renting_ratio = 0.0
        edge_weight_type = ""
        self.hyperparameter = [problem_name,
                               knapsack_data_type,
                               dimension,  # number of city
                               number_of_item,
                               capacity_of_knapsack,
                               min_speed,
                               max_speed,
                               renting_ratio,
                               edge_weight_type]
        for index in range(9):
            if index != 1:
                if index == 2 or index == 3 or index == 4:
                    self.hyperparameter[index] = int((self.content[index].split('\t'))[1])
                elif index == 5 or index == 6 or index == 7:
                    self.hyperparameter[index] = float((self.content[index].split('\t'))[1])
                else:
                    self.hyperparameter[index] = (self.content[index].split('\t'))[1]
            else:
                self.hyperparameter[index] = (self.content[index].split(': '))[1]
        return self.hyperparameter

    def set_cities(self):
        # 引入 城市矩阵 n个城市 (index, position_x, position_y)
        self.cities = np.zeros((self.hyperparameter[2], 3))
        for index in range(self.hyperparameter[2]):
            temp_info = self.content[index+10].split('\t')
            for j in range(3):
                temp_info[j] = int(temp_info[j])
            self.cities[temp_info[0]-1] = np.array(temp_info)
        return self.cities
