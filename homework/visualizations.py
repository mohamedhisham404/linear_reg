from matplotlib import pyplot as plt

def visualize_iter(cost_history):
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    # plt.xlim(0, 100)
    # plt.ylim(0, 0.08)
    plt.grid()
    plt.plot(list(range(len(cost_history))), cost_history, '-r')
    plt.show()    

def visualize_columns(data):
    import seaborn as sns
    sns.pairplot(data, x_vars=['Feat1', 'Feat2', 'Feat3'], y_vars='Target', height=4, aspect=1, kind='scatter')

    plt.show()

def visualize_p1(x,t,pred):
    plt.scatter(x, t, marker='o', color='red')    
    plt.plot(x, pred, color = 'blue')              
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()    