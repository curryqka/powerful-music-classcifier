from deap import base, creator, tools, algorithms
from sklearn import metrics
from data.data_loader import data_provider, data_provider_ML

# 定义适应度函数
def evaluate(weights):
    
    batch_x, batch_y = 
    w1, w2 = weights
    # 归一化权重
    w1, w2 = w1 / (w1 + w2), w2 / (w1 + w2)
    
    # 生成预测
    y_pred1 = model1.predict_proba(X_test)
    y_pred2 = model2.predict_proba(X_test)
    y_pred = w1 * y_pred1 + w2 * y_pred2
    y_pred = np.argmax(y_pred, axis=1)
    
    # 计算准确率
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return accuracy,

# 创建遗传算法工具箱
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)  # 初始化权重为随机浮点数
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# 初始化种群
population = toolbox.population(n=20)

# 遗传算法参数
NGEN = 50
CXPB = 0.5
MUTPB = 0.2

# 运行遗传算法
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
    fits = map(toolbox.evaluate, offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))
    
    # 记录并输出当前最优个体
    best_ind = tools.selBest(population, 1)[0]
    print(f"Generation {gen}: Best accuracy = {best_ind.fitness.values[0]}, Weights = {best_ind}")

# 输出最终结果
best_ind = tools.selBest(population, 1)[0]
print(f"Final Best accuracy = {best_ind.fitness.values[0]}, Weights = {best_ind}")