from TTP_file_opener import TTP_file_opener
import matplotlib.pyplot as plt
import networkx as nx

filename1 = '../TTP_sample/a280-ttp/a280_n279_bounded-strongly-corr_01.ttp'
filename2 = '../TTP_sample/a280-ttp/a280_n279_bounded-strongly-corr_02.ttp'
files = [filename1, filename2]
openers = []
# 创建类的实例
# opener = TTP_file_opener(filename1)
# row_counter, content = opener.file_scanner()
# print(content)
# hyper = opener.set_hyperparameter()
# print(hyper)

for i in range(len(files)):
    opener = TTP_file_opener(files[i])
    opener.file_scanner()
    opener.set_hyperparameter()
    openers.append(opener)

for i in range(len(files)):
    print(openers[i].hyperparameter)

cities_1 = openers[1].set_cities()

position_x = cities_1[:, 1]
position_y = cities_1[:, 2]
names = cities_1[:, 0]

G= nx.Graph()
for i in range(openers[0].hyperparameter[2]):
    G.add_node(int(names[i]), pos=(position_x[i], position_y[i]))

pos = {node: (x, y) for node, (x, y) in nx.get_node_attributes(G, "pos").items()}
nx.draw(G, pos, with_labels=True, node_size=30, node_color='skyblue', font_size=8, font_color='black')

plt.title("Cities Position Graph")
plt.show()