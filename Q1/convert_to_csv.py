import os
import pandas as pd


def main():
    q1_dir = r"D:\Project\Model\Q1"

    print("开始扫描 Q1 目录下的 xlsx 文件并转换为 csv...")

    for root, dirs, files in os.walk(q1_dir):
        for file in files:
            if file.endswith(".xlsx") and not file.startswith("~"):
                excel_path = os.path.join(root, file)
                print(f"\n[读取] {excel_path}")
                try:
                    # 读取所有 sheet
                    xls = pd.read_excel(excel_path, sheet_name=None)
                    for sheet_name, df in xls.items():
                        # 生成 csv 文件名：原文件名_sheet名.csv
                        csv_name = f"{os.path.splitext(file)[0]}_{sheet_name}.csv"
                        csv_path = os.path.join(root, csv_name)
                        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                        print(f"  -> [保存] {csv_path}")
                except Exception as e:
                    print(f"  [失败] {file}: {e}")


if __name__ == "__main__":
    main()
