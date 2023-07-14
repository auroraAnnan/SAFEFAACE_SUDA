import pyodbc
import numpy as np

from interface import database_interface

# 连接到数据库
conn = pyodbc.connect(database_interface('{SQL Server}',
                                         r'LEGION-Y9000P\SQLEXPRESS', 'SafeFace', 'sa', ''))

# 创建一个游标对象，用于执行 SQL 语句
cursor = conn.cursor()

# 执行查询语句
cursor.execute("SELECT name, info FROM [characters]")

# 获取查询结果
rows = cursor.fetchall()

# 遍历结果并处理数据
for row in rows:
    name = row.name
    info_bytes = row.info
    # 将 varbinary 数据转换为矩阵
    matrix = np.frombuffer(info_bytes, dtype=np.float64)

    # 处理数据...
    print("Name:", name)
    print("Matrix:")
    print(matrix)
    print(len(matrix))

# 关闭数据库连接
conn.close()
