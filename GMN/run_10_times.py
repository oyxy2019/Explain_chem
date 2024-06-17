import train_e2sn2


results = []
for _ in range(10):
    val_metric, test_metric = train_e2sn2.main()
    results.append((val_metric, test_metric))

# 打印结果
print()
for i, (val_metric, test_metric) in enumerate(results):
    print(f"Run {i}: Validation RMSE: {val_metric:.5f}, Test RMSE: {test_metric:.5f}")

# 计算平均值
avg_val_metric = sum(val_metric for val_metric, _ in results) / len(results)
avg_test_metric = sum(test_metric for _, test_metric in results) / len(results)
print(f"\nAverage Validation RMSE: {avg_val_metric:.5f}")
print(f"Average Test RMSE: {avg_test_metric:.5f}")
