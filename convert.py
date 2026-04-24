import os
import glob
import numpy as np
import pandas as pd

data_dir = "./dataset/Express4D/data"
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

if len(csv_files) == 0:
    print(f"错误：在 {data_dir} 没找到任何 CSV 文件！请检查路径。")
else:
    print(f"一共找到了 {len(csv_files)} 个 CSV 文件。开始转换...")

for i, csv_file in enumerate(csv_files):
    try:
        df = pd.read_csv(csv_file)
        # 丢弃 Timecode 和 BlendshapeCount，保留 61 个特征
        features = df.iloc[:, 2:].values.astype(np.float32)
        
        npy_file = csv_file.replace('.csv', '.npy')
        np.save(npy_file, features)
        
    except Exception as e:
        print(f"处理 {csv_file} 时出错: {e}")
        
    if (i + 1) % 500 == 0:
        print(f"已处理 {i + 1} / {len(csv_files)} 个文件...")

print("🎉 转换彻底完成！所有的 .npy 文件已经准备就绪。")
