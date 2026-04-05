# 取消跟踪 Q2 结果目录
git rm --cached -r Q2_Results/

# 取消跟踪 flow 数据目录
git rm --cached -r flow/data/

# 取消跟踪 Q1 的临时 Excel 锁文件
git rm --cached "Q1/degree_distribution/~$degree_distribution.xlsx" 2>$null

# 取消跟踪其他可能残留的结果文件（按需）
git rm --cached -r Q1/**/*.xlsx 2>$null
git rm --cached -r Q1/**/*.csv 2>$null