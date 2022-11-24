import numpy as np
from parser import parameter_parser
from models.CGE import CGEConv
from models.CGE_Variants import CGEVariant
from models.FFNN import FFNN
from preprocessing import get_graph_feature, get_pattern_feature

args = parameter_parser()


def main():
    print("CGEVulDetector")
    graph_train, graph_test, graph_experts_train, graph_experts_test = get_graph_feature()
    pattern_train, pattern_test, _, _ = get_pattern_feature()

    graph_train = np.array(graph_train)  # The training set of graph feature
    graph_test = np.array(graph_test)  # The testing set of graph feature

    pattern_train = np.array(pattern_train)
    pattern_test = np.array(pattern_test)

    # The ground truth label of certain contract in training set
    y_train = []
    for i in range(len(graph_experts_train)):
        y_train.append(int(graph_experts_train[i]))
    y_train = np.array(y_train)
    print("Y_Train:")
    print(y_train)
    # The label of certain contract in testing set
    y_test = []
    for i in range(len(graph_experts_test)):
        y_test.append(int(graph_experts_test[i]))
    y_test = np.array(y_test)
#     model = FFNN(pattern_train, pattern_test, y_train, y_test)
    if args.model == 'CGE':  # Conv layer, maxpooling layer, dense layer
        model = CGEConv(graph_train, graph_test, pattern_train, pattern_test, y_train, y_test)
    elif args.model == 'CGEVariant':  # Conv layer and dense layer
        model = CGEVariant(graph_train, graph_test, pattern_train, pattern_test, y_train, y_test)

#     model.train()  # training
#     model.test()  # testing


if __name__ == "__main__":
    main()
