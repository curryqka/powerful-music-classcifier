import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_curve, auc,  roc_auc_score

from utlis.check_utlis import codes


def show_confusion_matrix(true_values, preds, name='./results/confusion_matrix.png'):
    """
    Displays a confusion matrix
    :param true_values: true values
    :param preds: predictions
    :return: None
    """
    classlist = list(codes.keys())

    # Plotting confusion matrices for small and big data
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))  # Set up the figure
    # fig.clf()

    y_ = [true_values]
    preds_lst = [preds]
    titles = ['Test Data confusion matrix']
    for i in range(1):
        actual = y_[i]
        predicted =preds_lst[i]  # get the prediction for each model (each loop)

        confusion_matrix = metrics.confusion_matrix(np.array(actual), np.array(predicted))
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        cm_display.plot(ax=ax)
        ax.set_xticklabels(classlist, rotation=45)
        ax.set_yticklabels(classlist)
        ax.set_title(titles[i])

    fig.suptitle('Model prediction on test data')
    #设置图框线粗细
    bwith = 2 #边框宽度设置为2
    ax.spines['bottom'].set_linewidth(bwith)#图框下边
    ax.spines['left'].set_linewidth(bwith)#图框左边
    ax.spines['top'].set_linewidth(bwith)#图框上边
    ax.spines['right'].set_linewidth(bwith)#图框右边
    
    # plt.rcParams['figure.figsize']=(6.0,4.0) #图形大小
    plt.rcParams['savefig.dpi'] = 300 #图片像素
    plt.rcParams['figure.dpi'] = 300 #分辨率
    # 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
    # 指定dpi=200，图片尺寸为 1200*800
    # 指定dpi=300，图片尺寸为 1800*1200
    # 设置figsize可以在不改变分辨率情况下改变比例
    plt.savefig(name, bbox_inches='tight')
    # plt.show()
    # plt.close()
    


def show_results(data):
    """
    Displays the results of the classification
    :param data: data to display
    :return: None
    """
    x_data =['blues',
        'classical',
        'country',
        'disco',
        'hiphop',
        'jazz',
        'metal',
        'pop',
        'reggae',
        'rock']
    y_data = data[0]


    for i in range(len(x_data)):
        plt.bar(x_data[i], y_data[i])

    plt.title("show results")
    # x
    plt.xlabel("genres")
    # y
    plt.ylabel("scores")

    plt.show()


def show_roc(true_values, preds):
    """
    Displays ROC curve
    :param y_test: true values
    :param y_pred_prob: predictions
    :return: None
    """
    # 计算ROC曲线下的面积
    roc_auc = auc(true_values, preds)

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(true_values, preds)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def get_roc_auc_score(y_true, y_probs):
    '''
    Uses roc_auc_score function from sklearn.metrics to calculate the micro ROC AUC score for a given y_true and y_probs.
    '''

    # NoFindingIndex = list(codes.keys()).index('No Finding')

    class_roc_auc_list = []
    useful_classes_roc_auc_list = []

    for i in range(y_true.shape[1]):
        class_roc_auc = roc_auc_score(y_true[:, i], y_probs[:, i])
        class_roc_auc_list.append(class_roc_auc)
        # if i != NoFindingIndex:
        useful_classes_roc_auc_list.append(class_roc_auc)
    print('class_roc_auc_list: ', class_roc_auc_list)
    # print('useful_classes_roc_auc_list', {data.all_classes[i]: x for i, x in enumerate(useful_classes_roc_auc_list)})
    useful_classes = [x for x in codes if x !='No Finding']
    print('useful_classes_roc_auc_list', {useful_classes[i]: x for i, x in enumerate(useful_classes_roc_auc_list)})
    return np.mean(np.array(useful_classes_roc_auc_list))


def visual_loss(train_loss, vali_loss, name):
    
    '''
    Displays loss
    :param train_loss: training loss
    :param vail_loss: validation loss
    :return: None
    '''
    
    fig = plt.figure( figsize = (6,4))
    fig.clf()
    ax = fig.subplots()
    ax.plot(range(0, len(train_loss)), train_loss, label = 'train loss', color = 'tab:blue',
            linewidth = 2.5, marker = 'D')
    ax.plot(range(0, len(vali_loss)), vali_loss, label = 'vail loss', color = 'tab:red',
            linewidth = 2.5, marker = 's')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_xlim(xmin=0, xmax=len(train_loss))
    ax.legend()
    ax.grid(linestyle="--",alpha=0.3)
    #设置图框线粗细
    bwith = 2 #边框宽度设置为2
    ax.spines['bottom'].set_linewidth(bwith)#图框下边
    ax.spines['left'].set_linewidth(bwith)#图框左边
    ax.spines['top'].set_linewidth(bwith)#图框上边
    ax.spines['right'].set_linewidth(bwith)#图框右边
    
    plt.rcParams['figure.figsize']=(6.0,4.0) #图形大小
    plt.rcParams['savefig.dpi'] = 300 #图片像素
    plt.rcParams['figure.dpi'] = 300 #分辨率
    # 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
    # 指定dpi=200，图片尺寸为 1200*800
    # 指定dpi=300，图片尺寸为 1800*1200
    # 设置figsize可以在不改变分辨率情况下改变比例
    plt.savefig(name, bbox_inches='tight')
    # plt.show()
    # plt.close()


def visual_lr(lr, name):
    """
    Displays learning rate
    :param lr: learning rate
    :return: None
    """
    fig = plt.figure( figsize = (6,4))
    fig.clf()
    ax = fig.subplots()
    ax.plot(range(0, len(lr)), lr, label = 'learning rate with E method', color = 'tab:blue',
            linewidth = 2.5, marker = 'D')
    ax.set_xlabel('epoch')
    ax.set_ylabel('learning rate')
    ax.set_xlim(xmin=0, xmax=len(lr))
    ax.legend()
    ax.grid(linestyle="--",alpha=0.3)
    #设置图框线粗细
    bwith = 2 #边框宽度设置为2
    ax.spines['bottom'].set_linewidth(bwith)#图框下边
    ax.spines['left'].set_linewidth(bwith)#图框左边
    ax.spines['top'].set_linewidth(bwith)#图框上边
    ax.spines['right'].set_linewidth(bwith)#图框右边
    
    plt.rcParams['figure.figsize']=(6.0,4.0) #图形大小
    plt.rcParams['savefig.dpi'] = 300 #图片像素
    plt.rcParams['figure.dpi'] = 300 #分辨率
    # 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600&400
    # 指定dpi=200，图片尺寸为 1200*800
    # 指定dpi=300，图片尺寸为 1800*1200
    # 设置figsize可以在不改变分辨率情况下改变比例
    plt.savefig(name, bbox_inches='tight')
    # plt.show()

def visual_GA(weights_history, fitness_history, population_history, NGEN, evaluate):
        # 可视化结果
    # generations = list(range(NGEN))
    # w1_history = [w[0] for w in weights_history]
    # w2_history = [w[1] for w in weights_history]

    # plt.figure(figsize=(18, 5))

    # # 绘制适应度曲线
    # plt.subplot(1, 3, 1)
    # plt.plot(generations, fitness_history, label='Fitness')
    # plt.xlabel('Generation')
    # plt.ylabel('Fitness')
    # plt.title('Fitness over Generations')
    # plt.legend()

    # # 绘制权重变化曲线
    # plt.subplot(1, 3, 2)
    # plt.plot(generations, w1_history, label='w1')
    # plt.plot(generations, w2_history, label='w2')
    # plt.xlabel('Generation')
    # plt.ylabel('Weights')
    # plt.title('Weights over Generations')
    # plt.legend()

    # # 绘制三维网格图和种群位置
    # ax = plt.subplot(1, 3, 3, projection='3d')

    # # 生成三维网格数据
    # w1_vals = np.linspace(0, 1, 50)
    # w2_vals = 1 - w1_vals
    # W1, W2 = np.meshgrid(w1_vals, w2_vals)
    # Z = np.zeros_like(W1)

    # for i in range(W1.shape[0]):
    #     for j in range(W1.shape[1]):
    #         weights = (W1[i, j], W2[i, j])
    #         Z[i, j] = evaluate(weights)[0]

    # # 绘制三维表面
    # ax.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.6)
    # ax.set_xlabel('w1')
    # ax.set_ylabel('w2')
    # ax.set_zlabel('Fitness')
    # ax.set_title('Fitness Landscape')

    # # 绘制种群位置
    # colors = plt.cm.jet(np.linspace(0, 1, NGEN))
    # for gen in range(NGEN):
    #     pop = population_history[gen]
    #     w1_pop = [ind[0] for ind in pop]
    #     w2_pop = [ind[1] for ind in pop]
    #     fitness_pop = [evaluate(ind)[0] for ind in pop]
    #     ax.scatter(w1_pop, w2_pop, fitness_pop, color=colors[gen], label=f'Gen {gen}' if gen % 10 == 0 else "", s=10)

    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # 设置图形参数
    plt.rcParams['figure.figsize'] = (20.0, 16.0)  # 图形大小
    plt.rcParams['savefig.dpi'] = 300  # 图片像素
    plt.rcParams['figure.dpi'] = 300  # 分辨率
    plt.rcParams.update({'font.size': 8})  # 设置全局字体大小

    # 创建图形
    fig = plt.figure(figsize=(10, 3))

    # 绘制适应度曲线
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(range(NGEN), fitness_history, label='Fitness')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness over Generations')
    ax1.legend()

    # 设置图框线粗细
    bwith = 2
    ax1.spines['bottom'].set_linewidth(bwith)
    ax1.spines['left'].set_linewidth(bwith)
    ax1.spines['top'].set_linewidth(bwith)
    ax1.spines['right'].set_linewidth(bwith)

    # 绘制权重变化曲线
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(range(NGEN), [w[0] for w in weights_history], label='w1')
    ax2.plot(range(NGEN), [w[1] for w in weights_history], label='w2')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Weights')
    ax2.set_title('Weights over Generations')
    ax2.legend()

    # 设置图框线粗细
    ax2.spines['bottom'].set_linewidth(bwith)
    ax2.spines['left'].set_linewidth(bwith)
    ax2.spines['top'].set_linewidth(bwith)
    ax2.spines['right'].set_linewidth(bwith)

    # 绘制三维网格图和种群位置
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    # 生成三维网格数据
    w1_vals = np.linspace(0, 1, 50)
    w2_vals = 1 - w1_vals
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    Z = np.zeros_like(W1)

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            Z[i, j] = evaluate([W1[i, j], W2[i, j]])[0]

    # 绘制三维表面
    ax3.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.6)
    ax3.set_xlabel('w1')
    ax3.set_ylabel('w2')
    ax3.set_zlabel('Fitness')
    ax3.set_title('Fitness Landscape')

    # 绘制种群位置
    colors = plt.cm.jet(np.linspace(0, 1, NGEN))

    # 初始化图例句柄列表
    legend_handles = []

    for gen in range(NGEN):
        pop = population_history[gen]
        w1_pop = [ind[0] for ind in pop]
        w2_pop = [ind[1] for ind in pop]
        fitness_pop = [evaluate(ind)[0] for ind in pop]
        scatter = ax3.scatter(w1_pop, w2_pop, fitness_pop, color=colors[gen], s=(gen+1)*2, label=f'Gen {gen}' if gen % 10 == 0 else "")
        
        # 仅每隔10代添加一次图例句柄
        if gen % 10 == 0:
            legend_handles.append(scatter)

    # 添加图例
    ax3.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1))
    # 设置图框线粗细
    ax3.spines['bottom'].set_linewidth(bwith)
    ax3.spines['left'].set_linewidth(bwith)
    ax3.spines['top'].set_linewidth(bwith)
    ax3.spines['right'].set_linewidth(bwith)
    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.4)
    # 保存图形
    plt.tight_layout()
    plt.savefig('optimization_results.png', bbox_inches='tight')
    # plt.show()s