import train_e2sn2
import statistics

results = []
for _ in range(5):
    train_metric, val_metric, test_metric = train_e2sn2.main()
    results.append((train_metric, val_metric, test_metric))

# 打印结果
print()
for i, (train_metric, val_metric, test_metric) in enumerate(results):
    print(f"Run {i}: Train RMSE: {train_metric:.3f}, Val RMSE: {val_metric:.3f}, Test RMSE: {test_metric:.3f}")

# 计算平均值
avg_train_metric = sum(train_metric for train_metric, _, _ in results) / len(results)
avg_val_metric = sum(val_metric for _, val_metric, _ in results) / len(results)
avg_test_metric = sum(test_metric for _, _, test_metric in results) / len(results)

# 计算标准差
std_dev_train = statistics.stdev(train_metric for train_metric, _, _ in results)
std_dev_val = statistics.stdev(val_metric for _, val_metric, _ in results)
std_dev_test = statistics.stdev(test_metric for _, _, test_metric in results)

print(f"\nAverage Train RMSE: {avg_train_metric:.3f} ± {std_dev_train:.3f}")
print(f"Average Val RMSE: {avg_val_metric:.3f} ± {std_dev_val:.3f}")
print(f"Average Test RMSE: {avg_test_metric:.3f} ± {std_dev_test:.3f}")
